"""
Minimal script to run ORD example through BenSAF workflow.

This script demonstrates the most parsimonious API for running an analysis.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from bensaf.workflow import run_analysis

# Set up paths
project_root = Path(__file__).parent.parent
data_dir = project_root / "data" / "dash_examples" / "ord"

# Load data files
print("Loading data files...")
tracts_gdf = gpd.read_file(data_dir / "tracts_geometries.geojson")
demographics_df = pd.read_csv(data_dir / "demographics_df.csv")
exposure_df = pd.read_csv(data_dir / "exposure_df.csv")
mortality_df = pd.read_csv(data_dir / "mortality_df.csv")

# Rename exposure column if needed
if 'pollutant_concentration' in exposure_df.columns:
    exposure_df = exposure_df.rename(columns={'pollutant_concentration': 'ufp'})

print(f"Loaded {len(tracts_gdf)} tracts")

# Run analysis using the most parsimonious API
print("\nRunning analysis...")
results = run_analysis(
    tracts_gdf=tracts_gdf,
    demographics_df=demographics_df,
    exposure_source='csv',
    exposure_data=exposure_df,
    incidence_df=mortality_df,
    scenarios=[25, 50],  # 25% and 50% SAF blend scenarios
    pollutant_name='ufp'
)

print("Analysis complete!")

# Extract results
scenario_results = results['scenario_results']
tract_level = results['tract_level']

# Create basic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ORD Example Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Baseline exposure map
ax1 = axes[0, 0]
# Ensure GEOID types match
exposure_df['GEOID'] = exposure_df['GEOID'].astype(str)
tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(str)
tracts_with_data = tracts_gdf.merge(
    exposure_df[['GEOID', 'ufp']],
    on='GEOID',
    how='left'
)
tracts_with_data.plot(column='ufp', ax=ax1, legend=True, cmap='YlOrRd')
ax1.set_title('Baseline UFP Exposure (pt/cm³)')
ax1.axis('off')

# Plot 2: Attributable cases avoided (25% SAF)
ax2 = axes[0, 1]
scenario_25 = scenario_results[25]
if scenario_25 and 'mortality' in scenario_25.health_impacts:
    impacts = scenario_25.health_impacts['mortality']
    if 'attributable_cases_mean' in impacts:
        cases = impacts['attributable_cases_mean']
        cases_df = cases.to_frame('cases').reset_index()
        cases_df['GEOID'] = cases_df['GEOID'].astype(str)
        tracts_with_cases = tracts_gdf.merge(
            cases_df[['GEOID', 'cases']],
            on='GEOID',
            how='left'
        )
        tracts_with_cases.plot(column='cases', ax=ax2, legend=True, cmap='RdYlGn_r')
        ax2.set_title('Attributable Cases Avoided (25% SAF)')
ax2.axis('off')

# Plot 3: Scenario comparison - Total attributable cases
ax3 = axes[1, 0]
scenario_ids = []
total_cases = []
for scenario_id, scenario in scenario_results.items():
    if scenario and 'mortality' in scenario.health_impacts:
        impacts = scenario.health_impacts['mortality']
        if 'attributable_cases_mean' in impacts:
            total = impacts['attributable_cases_mean'].sum()
            scenario_ids.append(f"{scenario_id}%")
            total_cases.append(total)

if scenario_ids:
    ax3.bar(scenario_ids, total_cases, color=['#2ecc71', '#3498db'])
    ax3.set_title('Total Attributable Cases Avoided by Scenario')
    ax3.set_ylabel('Cases Avoided')
    ax3.set_xlabel('SAF Blend Percentage')
    ax3.grid(axis='y', alpha=0.3)

# Plot 4: Economic benefits (if available)
ax4 = axes[1, 1]
scenario_ids_econ = []
economic_values = []
for scenario_id, scenario in scenario_results.items():
    if scenario and 'mortality_economic_value_mean' in scenario.economic_benefits:
        value = scenario.economic_benefits['mortality_economic_value_mean']
        scenario_ids_econ.append(f"{scenario_id}%")
        economic_values.append(value / 1e6)  # Convert to millions

if scenario_ids_econ:
    ax4.bar(scenario_ids_econ, economic_values, color=['#2ecc71', '#3498db'])
    ax4.set_title('Mortality Economic Benefits')
    ax4.set_ylabel('Value (Millions $)')
    ax4.set_xlabel('SAF Blend Percentage')
    ax4.grid(axis='y', alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Economic benefits not calculated\n(per_capita_consumption not configured)',
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Mortality Economic Benefits')
    ax4.axis('off')

plt.tight_layout()
plt.savefig(project_root / 'examples' / 'ord_analysis_results.png', dpi=150, bbox_inches='tight')
print(f"\nPlots saved to: {project_root / 'examples' / 'ord_analysis_results.png'}")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
for scenario_id, scenario in scenario_results.items():
    print(f"\nScenario: {scenario.scenario_label}")
    if 'mortality' in scenario.health_impacts:
        impacts = scenario.health_impacts['mortality']
        if 'attributable_cases_mean' in impacts:
            total_cases = impacts['attributable_cases_mean'].sum()
            print(f"  Total attributable cases avoided: {total_cases:.1f}")
    if 'mortality_economic_value_mean' in scenario.economic_benefits:
        value = scenario.economic_benefits['mortality_economic_value_mean']
        print(f"  Economic value: ${value/1e6:.2f} million")
