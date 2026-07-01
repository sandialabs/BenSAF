from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from bensaf.model.workflow import run_analysis

project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "case-studies" / "ord"
calibration_file = project_root / "data" / "aermod_calibration_coefficients.json"

print("Loading data files...")
tracts_gdf = gpd.read_file(data_dir / "tracts_geometries.geojson")

# Each CSV: col 1 = GEOID index, col 2 = value.  Column names are ignored.
demographics_df = pd.read_csv(data_dir / "demographics_df.csv")
mortality_df = pd.read_csv(data_dir / "mortality_df.csv")

# East flow: 1/3 weight, West flow: 2/3 weight
east_weight = 1 / 3
west_weight = 2 / 3

landing_files = [
    (data_dir / "landing_eastflow.ADO", east_weight),
    (data_dir / "landing_westflow.ADO", west_weight),
]
takeoff_files = None  # include takeoff if available

print(f"Loaded {len(tracts_gdf)} tracts")

exposure_data = {
    'landing_files': landing_files,
    'takeoff_files': takeoff_files,
    'calibration_file': calibration_file,
    'aermod_crs': 'EPSG:32616',  # UTM Zone 16N
}

print("\nGenerating exposure from AERMOD files and running analysis...")
results = run_analysis(
    tracts_gdf=tracts_gdf,
    demographics_df=demographics_df,
    exposure_source='aermod_workflow',
    exposure_data=exposure_data,
    incidence_df=mortality_df,
    scenarios=[25, 50],
)

print("Analysis complete!")

# Summary output
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

pop_series = None
dc = results.inputs.demographics_core
if dc is not None and 'population' in dc.columns:
    pop_series = dc['population']

for scenario_id, scenario_result in sorted(results.scenarios.items()):
    print(f"\nScenario: {scenario_result.spec.scenario_label}")
    agg = scenario_result.get_aggregated_results(population=pop_series)
    if 'mortality' in agg:
        tac = agg['mortality']['total_attributable_cases']
        print(f"  Cases avoided (mean):  {tac.mean:.1f}")
        print(f"  Cases avoided (95% CI): {tac.lower:.1f} – {tac.upper:.1f}")
    if 'economic_benefits' in agg and 'mortality_economic_value' in agg['economic_benefits']:
        ev = agg['economic_benefits']['mortality_economic_value']
        print(f"  Economic value: ${ev.mean / 1e6:.2f} million")

# Basic choropleth plots
merged = results.get_merged_data()

n_scenarios = len(results.scenarios)
fig, axes = plt.subplots(1, n_scenarios + 1, figsize=(6 * (n_scenarios + 1), 5))
if n_scenarios + 1 == 1:
    axes = [axes]

# Baseline exposure
ax0 = axes[0]
if 'ufp' in merged.columns:
    merged.plot(column='ufp', ax=ax0, legend=True, cmap='YlOrRd')
    ax0.set_title('Baseline UFP Exposure (pt/cm³)')
else:
    ax0.text(0.5, 0.5, 'Exposure data not available', ha='center', va='center',
             transform=ax0.transAxes)
    ax0.set_title('Baseline UFP Exposure')
ax0.axis('off')

for i, (scenario_id, scenario_result) in enumerate(sorted(results.scenarios.items())):
    ax = axes[i + 1]
    col = f"{scenario_result.spec.scenario_label}_mortality_attributable_cases_mean"
    if col in merged.columns:
        merged.plot(column=col, ax=ax, legend=True, cmap='RdYlGn_r')
        ax.set_title(f'Cases Avoided\n({scenario_result.spec.scenario_label})')
    else:
        ax.text(0.5, 0.5, 'Results not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(f'{scenario_result.spec.scenario_label}')
    ax.axis('off')

plt.tight_layout()
output_path = project_root / 'examples' / 'ord_analysis_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()
