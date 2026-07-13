import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from aermod_parser import AermodFile
import bensaf
from bensaf.model.workflow import Workflow

data_dir = bensaf.DATA_DIR / "ord"
calibration_file = bensaf.CALIBRATION_FILE

# East flow: 1/3 weight, West flow: 2/3 weight
landing_files = [
    (data_dir / "AERMOD" / "landing_eastflow.ADO", 1 / 3),
    (data_dir / "AERMOD" / "landing_westflow.ADO", 2 / 3),
]

## Inspect AERMOD files

print("Inspecting AERMOD files with aermod_parser...")
for path, weight in landing_files:
    f = AermodFile.from_path(path)
    print(f"\n  {path.name}  (weight={weight:.3f})")
    print(f"    Pollutant : {f.metadata.pollutant}")
    print(f"    Periods   : {', '.join(f.metadata.averaging_periods)}")
    print(f"    Sections  : {', '.join(f.section_types)}")
    for nid, net in f.metadata.networks.items():
        print(f"    Network   : {nid}  type={net.network_type}  "
              f"origin=({net.origin_x}, {net.origin_y})")
    aa = f.annual_average
    if not aa.empty:
        lo, hi = aa['concentration'].min(), aa['concentration'].max()
        print(f"    Annual avg: {len(aa)} receptors, conc range [{lo:.4g}, {hi:.4g}]")

## Load spatial and tabular data

print("\nLoading data files...")
tracts_gdf = gpd.read_file(data_dir / "tracts_geometries.geojson")
# Each CSV: col 1 = GEOID index, col 2 = value.  Column names are ignored.
demographics_df = pd.read_csv(data_dir / "demographics_df.csv")
mortality_df = pd.read_csv(data_dir / "mortality_df.csv")
print(f"  {len(tracts_gdf)} census tracts")

## Build Workflow and load inputs

workflow = Workflow()

workflow.load_inputs(
    tracts_gdf=tracts_gdf,
    demographics_df=demographics_df,
    exposure_source='aermod_workflow',
    exposure_data={
        'landing_files': landing_files,
        'takeoff_files': None,
        'calibration_file': calibration_file,
        'aermod_crs': 'EPSG:32616',  # UTM Zone 16N
    },
    incidence_df=mortality_df,
)

print(f"\nExposure loaded: {len(workflow.inputs.baseline_exposure)} tracts with UFP values")

## Run scenarios

results = workflow.run_scenarios(scenarios=[25, 50])

## Print summary

print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

pop_series = None
dc = workflow.inputs.demographics_core
if dc is not None and 'population' in dc.columns:
    pop_series = dc['population']

for scenario_id, scenario_result in sorted(results.scenarios.items()):
    print(f"\nScenario: {scenario_result.spec.scenario_label}")
    agg = scenario_result.get_aggregated_results(population=pop_series)
    if 'mortality' in agg:
        tac = agg['mortality']['total_attributable_cases']
        print(f"  Cases avoided (mean):   {tac.mean:.1f}")
        print(f"  Cases avoided (95% CI): {tac.lower:.1f} – {tac.upper:.1f}")
    if 'economic_benefits' in agg and 'mortality_economic_value' in agg['economic_benefits']:
        ev = agg['economic_benefits']['mortality_economic_value']
        print(f"  Economic value: ${ev.mean / 1e6:.2f} million")

## Choropleth plots

merged = results.get_merged_data()

n_scenarios = len(results.scenarios)
fig, axes = plt.subplots(1, n_scenarios + 1, figsize=(6 * (n_scenarios + 1), 5))
if n_scenarios + 1 == 1:
    axes = [axes]

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
        merged.plot(column=col, ax=ax, legend=True, cmap='Blues')
        ax.set_title(f'Cases Avoided\n({scenario_result.spec.scenario_label})')
    else:
        ax.text(0.5, 0.5, 'Results not available', ha='center', va='center',
                transform=ax0.transAxes)
        ax.set_title(f'{scenario_result.spec.scenario_label}')
    ax.axis('off')

plt.tight_layout()
output_path = bensaf.DATA_DIR.parent / 'examples' / 'ord_analysis_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()
