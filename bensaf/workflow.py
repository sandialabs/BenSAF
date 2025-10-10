"""
SAF Toolkit Workflow

This module provides a generalized workflow for Sustainable Aviation Fuel (SAF) health impact
assessment. It builds on the existing HealthImpactWorkflow class but provides a more
flexible and generalized interface.

The workflow follows the methodology from the paper:
"Quantifying health benefits of sustainable aviation fuels: Modeling decreased ultrafine
particle emissions and associated impacts on communities near the Seattle-Tacoma
International Airport"
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bensaf.health_impacts import (
    transform_relative_risk,
    calculate_attributable_fraction,
    calculate_attributable_cases,
    calculate_attributable_mortality
)
from bensaf.utils import bin_tracts_by_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Workflow:
    """
    Generalized workflow for SAF health impact assessment.
    
    This class provides a simplified interface to the health impact assessment workflow,
    focusing on the core steps and making minimal assumptions about the data.
    
    The workflow consists of the following steps:
    1. Load and validate data
    2. Calculate population-weighted exposures
    3. Apply control scenarios
    4. Calculate health impacts
    5. Generate results and visualizations
    
    Example:
        ```python
        # Initialize workflow
        workflow = Workflow()
        
        # Load data
        workflow.load_tract_data(tracts_gdf)
        workflow.load_exposure_data(exposure_df)
        workflow.load_mortality_data(mortality_df)
        
        # Run analysis
        workflow.calculate_population_weighted_exposure()
        workflow.apply_control_scenarios([25, 50, 75])
        workflow.calculate_health_impacts()
        
        # Get results
        results = workflow.get_results()
        workflow.create_visualizations("results")
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SAF workflow.
        
        Args:
            config: Optional configuration dictionary with parameters like hazard ratios,
                control scenarios, etc.
        """
        self.config = config or {}
        
        # Initialize data containers
        self.tracts_gdf = None
        self.exposure_df = None
        self.mortality_df = None
        self.results = {}
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    
    def load_tract_data(self, tracts_gdf: gpd.GeoDataFrame) -> None:
        """
        Load census tract data with demographics.
        
        Args:
            tracts_gdf: GeoDataFrame with census tract geometries and demographic data.
                Must contain columns:
                - GEOID: Census tract identifier
                - geometry: Tract geometry
                - population: Total population
                - Additional demographic columns (optional)
        """
        self.logger.info("Loading tract data")
        
        # Validate required columns
        required_columns = ['GEOID', 'geometry']
        for col in required_columns:
            if col not in tracts_gdf.columns:
                raise ValueError(f"Missing required column in tract data: {col}")
            
        # Cast GEOID to int
        tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(int)
        
        # Check if population column exists
        if 'population' not in tracts_gdf.columns and 'Population' not in tracts_gdf.columns:
            self.logger.warning("No 'population' column found in tract data")
        
        # Store data
        self.tracts_gdf = tracts_gdf.copy()
        
        # Standardize column names
        if 'Population' in self.tracts_gdf.columns and 'population' not in self.tracts_gdf.columns:
            self.tracts_gdf['population'] = self.tracts_gdf['Population']
        
        # Check and project to EPSG:4326 if necessary
        if self.tracts_gdf.crs is None:
            self.logger.warning("Tract data has no CRS defined, assuming EPSG:4326")
            self.tracts_gdf.set_crs("EPSG:4326", inplace=True)
        elif self.tracts_gdf.crs != "EPSG:4326":
            self.logger.info(f"Reprojecting tract data from {self.tracts_gdf.crs} to EPSG:4326")
            self.tracts_gdf = self.tracts_gdf.to_crs("EPSG:4326")
        else:
            self.logger.info("Tract data is already in EPSG:4326")
        
        self.logger.info(f"Loaded {len(self.tracts_gdf)} census tracts")
    
    def load_exposure_data(self, exposure_df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> None:
        """
        Load exposure data with pollutant concentrations.
        
        Args:
            exposure_df: DataFrame or GeoDataFrame with pollutant concentrations.
                Must contain columns:
                - GEOID: Census tract identifier
                - pollutant_concentration: Baseline pollutant concentration
                  (column name can be specified in config)
        """
        self.logger.info("Loading exposure data")
        
        # Get pollutant column name from config or use default
        pollutant_col = self.config.get('pollutant_column', 'pollutant_concentration')
        
        # Validate required columns
        required_columns = ['GEOID', pollutant_col]
        for col in required_columns:
            if col not in exposure_df.columns:
                raise ValueError(f"Missing required column in exposure data: {col}")
            
        # cast GEOID to int
        exposure_df['GEOID'] = exposure_df['GEOID'].astype(int)
        
        # Store data
        self.exposure_df = exposure_df.copy()
        
        # Rename pollutant column to standard name if needed
        if pollutant_col != 'pollutant_concentration':
            self.exposure_df['pollutant_concentration'] = self.exposure_df[pollutant_col]
        
        self.logger.info(f"Loaded exposure data with {len(self.exposure_df)} records")
    
    def load_mortality_data(self, mortality_df: pd.DataFrame) -> None:
        """
        Load mortality data with baseline rates.
        
        Args:
            mortality_df: DataFrame with mortality rates.
                Must contain columns:
                - GEOID: Census tract identifier
                - mortality_rate: Baseline mortality rate (deaths per person per year)
        """
        self.logger.info("Loading mortality data")
        
        # Validate required columns
        required_columns = ['GEOID', 'mortality_rate']
        for col in required_columns:
            if col not in mortality_df.columns:
                raise ValueError(f"Missing required column in mortality data: {col}")
        
        # cast GEOID to int
        mortality_df['GEOID'] = mortality_df['GEOID'].astype(int)
        
        # Store data
        self.mortality_df = mortality_df.copy()
        
        self.logger.info(f"Loaded mortality data with {len(self.mortality_df)} records")
    
    def load_health_impact_function(self, 
                                   mean_rr: float, 
                                   lower_rr: float, 
                                   upper_rr: float,
                                   unit_increase: float) -> None:
        """
        Load health impact function parameters.
        
        Args:
            mean_rr: Mean relative risk
            lower_rr: Lower bound of relative risk (95% CI)
            upper_rr: Upper bound of relative risk (95% CI)
            unit_increase: Unit increase in pollutant concentration for the relative risk
        """
        self.logger.info("Loading health impact function")
        
        # Calculate beta and SE from relative risk
        z = 1.96  # 95% confidence interval
        
        mean_log = np.log(mean_rr)
        lower_log = np.log(lower_rr)
        upper_log = np.log(upper_rr)
        
        se_log = ((upper_log - mean_log) + (mean_log - lower_log)) / (2 * z)
        
        mean_log_one_unit = mean_log / unit_increase
        se_log_one_unit = se_log / unit_increase
        
        # Store parameters
        self.health_function = {
            'mean_rr': mean_rr,
            'lower_rr': lower_rr,
            'upper_rr': upper_rr,
            'unit_increase': unit_increase,
            'mean_log_one_unit': mean_log_one_unit,
            'se_log_one_unit': se_log_one_unit
        }
        
        self.logger.info("Loaded health impact function")
    
    def prepare_data(self) -> None:
        """
        Prepare data for analysis by integrating tract, exposure, and mortality data.
        
        This method joins the tract, exposure, and mortality data into a single
        GeoDataFrame for analysis.
        """
        self.logger.info("Preparing data for analysis")
        
        # Validate that all required data is loaded
        if self.tracts_gdf is None:
            raise ValueError("Tract data must be loaded first")
        if self.exposure_df is None:
            raise ValueError("Exposure data must be loaded first")
        if self.mortality_df is None:
            raise ValueError("Mortality data must be loaded first")
        
        # Start with tract data
        analysis_data = self.tracts_gdf.copy()
        
        # Join exposure data
        if isinstance(self.exposure_df, gpd.GeoDataFrame):
            # If exposure data is a GeoDataFrame, use spatial join
            analysis_data = gpd.sjoin(
                analysis_data,
                self.exposure_df[['GEOID', 'pollutant_concentration', 'geometry']],
                how='left',
                predicate='intersects'
            )
        else:
            # Otherwise, use attribute join
            analysis_data = analysis_data.merge(
                self.exposure_df[['GEOID', 'pollutant_concentration']],
                on='GEOID',
                how='left'
            )
        
        # Join mortality data
        analysis_data = analysis_data.merge(
            self.mortality_df[['GEOID', 'mortality_rate']],
            on='GEOID',
            how='left'
        )
        
        # Check for missing values
        missing_exposure = analysis_data['pollutant_concentration'].isna().sum()
        missing_mortality = analysis_data['mortality_rate'].isna().sum()
        
        # Fill missing values with mean
        if missing_exposure > 0:
            self.logger.warning(f"{missing_exposure} tracts missing exposure data")
            # Fill with mean value
            mean_exposure = analysis_data['pollutant_concentration'].mean()
            analysis_data['pollutant_concentration'] = analysis_data['pollutant_concentration'].fillna(mean_exposure)
        
        if missing_mortality > 0:
            self.logger.warning(f"{missing_mortality} tracts missing mortality data")
            # Fill with mean value
            mean_mortality = analysis_data['mortality_rate'].mean()
            analysis_data['mortality_rate'] = analysis_data['mortality_rate'].fillna(mean_mortality)
        
        # Store integrated data
        self.analysis_data = analysis_data
        
        # Compute natural mortality rate per capita
        analysis_data['nmr per 100k'] = np.array(analysis_data['mortality_rate']*100000)
        
        self.logger.info("Data preparation complete")
    
    def apply_control_scenarios(self, scenarios: Optional[List[float]] = None) -> Dict[str, gpd.GeoDataFrame]:
        """
        Apply control scenarios to calculate reduced exposures.
        
        Args:
            scenarios: List of control scenarios (percent reductions).
                If None, uses scenarios from config.
                
        Returns:
            Dictionary of GeoDataFrames with reduced exposures by scenario
        """
        self.logger.info("Applying control scenarios")
        
        # Use provided scenarios or get from config
        if scenarios is None:
            scenarios = self.config.get('control_scenarios', [25, 50, 75])
        
        # Prepare data if not already done
        if not hasattr(self, 'analysis_data'):
            self.prepare_data()
            
        control_scenarios = {}

        for scenario in scenarios:
            # Calculate reduced exposure
            control_data = pd.DataFrame(index=self.analysis_data.index)
            
            # Calculate reduced exposure
            control_data[f'reduced_concentration_{scenario}'] = (
                self.analysis_data['pollutant_concentration'] * (1 - scenario / 100)
            )
            
            # Calculate delta exposure
            control_data[f'delta_concentration_{scenario}'] = (
                self.analysis_data['pollutant_concentration'] - control_data[f'reduced_concentration_{scenario}']
            )

            control_scenarios[scenario] = {'data': control_data}

        self.logger.info(f"Applied {len(scenarios) if scenarios else 0} control scenarios")
        self.control_scenarios = control_scenarios
        return control_scenarios
    
    def calculate_health_impacts(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate health impacts for all control scenarios.
        
        Returns:
            Dictionary of health impact results by scenario
        """
        self.logger.info("Calculating health impacts")

        # Check if control scenarios have been applied
        if not hasattr(self, 'control_scenarios'):
            self.logger.warning("Control scenarios not applied, applying default scenarios")
            self.apply_control_scenarios()
        
        # Check if health impact function is loaded
        if not hasattr(self, 'health_function'):
            self.logger.warning("Health impact function not loaded, using default from Bouma et al.")
            # Default values from Bouma et al.
            self.load_health_impact_function(
                mean_rr=1.012,
                lower_rr=1.010,
                upper_rr=1.015,
                unit_increase=2723  # pt/cm3
            )
        
        # Calculate health impacts for each scenario
        for scenario, scenario_dict in self.control_scenarios.items():
            scenario_df = scenario_dict['data']
            # Calculate attributable fraction
            delta_ap = scenario_df[f'delta_concentration_{scenario}']
            
            # Transform relative risk
            mean_rr, lower_rr, upper_rr = transform_relative_risk(
                self.health_function['mean_log_one_unit'],
                self.health_function['se_log_one_unit'],
                delta_ap
            )
            
            # Add to scenario dictionary
            scenario_dict[f'relative_risk_{scenario}'] = mean_rr
            scenario_dict[f'lower_relative_risk_{scenario}'] = lower_rr
            scenario_dict[f'upper_relative_risk_{scenario}'] = upper_rr
            
            # Calculate attributable fraction
            mean_af = calculate_attributable_fraction(mean_rr)
            lower_af = calculate_attributable_fraction(lower_rr)
            upper_af = calculate_attributable_fraction(upper_rr)
            
            # Add to scenario dictionary
            scenario_dict[f'attributable_fraction_{scenario}'] = mean_af
            scenario_dict[f'lower_attributable_fraction_{scenario}'] = lower_af
            scenario_dict[f'upper_attributable_fraction_{scenario}'] = upper_af
            
            # Calculate attributable cases
            mean_ac = calculate_attributable_cases(
                mean_af,
                self.analysis_data['mortality_rate'],
                self.analysis_data['population']
            )
            lower_ac = calculate_attributable_cases(
                lower_af,
                self.analysis_data['mortality_rate'],
                self.analysis_data['population']
            )
            upper_ac = calculate_attributable_cases(
                upper_af,
                self.analysis_data['mortality_rate'],
                self.analysis_data['population']
            )
            
            # Calculate attributable mortality rate
            mean_amr = calculate_attributable_mortality(
                mean_af,
                self.analysis_data['mortality_rate']
            )
            lower_amr = calculate_attributable_mortality(
                lower_af,
                self.analysis_data['mortality_rate']
            )
            upper_amr = calculate_attributable_mortality(
                upper_af,
                self.analysis_data['mortality_rate']
            )
            
            # Add to scenario dictionary
            scenario_dict[f'attributable_cases_{scenario}'] = mean_ac
            scenario_dict[f'lower_attributable_cases_{scenario}'] = lower_ac
            scenario_dict[f'upper_attributable_cases_{scenario}'] = upper_ac
            scenario_dict[f'attributable_mortality_rate_{scenario}'] = mean_amr
            scenario_dict[f'lower_attributable_mortality_rate_{scenario}'] = lower_amr
            scenario_dict[f'upper_attributable_mortality_rate_{scenario}'] = upper_amr

        
        self.logger.info("Health impact calculation complete")
        return
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all analysis results.
        
        Returns:
            Dictionary containing all analysis results
        """
        return self.results
    
    def create_visualizations(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create visualization plots for the analysis.
        
        Args:
            output_dir: Optional directory to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        self.logger.info("Creating visualizations")
        
        # Check if health impacts have been calculated
        if 'health_impacts' not in self.results:
            self.logger.warning("Health impacts not calculated, calculating now")
            self.calculate_health_impacts()
        
        # Create output directory if specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures
        figures = {}
        
        # 1. Map of baseline pollutant concentration
        fig, ax = plt.subplots(figsize=(10, 8))
        self.analysis_data.plot(
            column='pollutant_concentration',
            ax=ax,
            legend=True,
            cmap='viridis',
            legend_kwds={'label': 'Pollutant Concentration'}
        )
        ax.set_title('Baseline Pollutant Concentration')
        figures['baseline_concentration'] = fig
        
        if output_dir:
            fig.savefig(str(output_dir / 'baseline_concentration.png'), dpi=300, bbox_inches='tight')
        
        # 2. Map of attributable cases for first scenario
        first_scenario = list(self.results['health_impacts'].keys())[0]
        scenario_data = self.results['health_impacts'][first_scenario]['data']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scenario_data.plot(
            column=f'attributable_cases_{first_scenario}',
            ax=ax,
            legend=True,
            cmap='viridis',
            legend_kwds={'label': 'Attributable Cases'}
        )
        ax.set_title(f'Attributable Cases ({first_scenario}% Reduction)')
        figures['attributable_cases'] = fig
        
        if output_dir:
            fig.savefig(str(output_dir / 'attributable_cases.png'), dpi=300, bbox_inches='tight')
        
        # 3. Bar chart of total attributable cases by scenario
        fig, ax = plt.subplots(figsize=(10, 6))
        scenarios = list(self.results['health_impacts'].keys())
        total_cases = [self.results['health_impacts'][s]['total_attributable_cases'] for s in scenarios]
        
        ax.bar(scenarios, total_cases)
        ax.set_xlabel('Control Scenario (% Reduction)')
        ax.set_ylabel('Total Attributable Cases')
        ax.set_title('Health Impacts by Control Scenario')
        figures['scenario_comparison'] = fig
        
        if output_dir:
            fig.savefig(str(output_dir / 'scenario_comparison.png'), dpi=300, bbox_inches='tight')
        
        # 4. Demographic comparison if available
        for group_col in self.config.get('demographic_columns', []):
            if group_col in self.analysis_data.columns:
                # Check if demographic results are available
                if f'by_{group_col}' in self.results['health_impacts'][first_scenario]:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    group_results = self.results['health_impacts'][first_scenario][f'by_{group_col}']
                    groups = list(group_results.keys())
                    group_cases = [group_results[g]['attributable_cases'] for g in groups]
                    
                    ax.bar(groups, group_cases)
                    ax.set_xlabel(group_col.capitalize())
                    ax.set_ylabel('Attributable Cases')
                    ax.set_title(f'Health Impacts by {group_col.capitalize()} ({first_scenario}% Reduction)')
                    ax.tick_params(axis='x', rotation=45)
                    
                    figures[f'demographic_{group_col}'] = fig
                    
                    if output_dir:
                        fig.savefig(str(output_dir / f'demographic_{group_col}.png'), dpi=300, bbox_inches='tight')
        
        self.logger.info(f"Created {len(figures)} visualization plots")
        return figures
    
    def export_results(self, output_dir: Union[str, Path]) -> None:
        """
        Export all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        self.logger.info("Exporting results")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export analysis data
        if hasattr(self, 'analysis_data'):
            self.analysis_data.to_file(
                output_dir / 'analysis_data.gpkg',
                driver='GPKG'
            )
        
        # Export population-weighted exposure results
        if 'population_weighted_exposure' in self.results:
            pd.DataFrame(self.results['population_weighted_exposure'].items(), 
                        columns=['group', 'weighted_exposure']).to_csv(
                output_dir / 'population_weighted_exposure.csv',
                index=False
            )
        
        # Export health impact results
        if 'health_impacts' in self.results:
            # Create health impacts directory
            health_dir = output_dir / 'health_impacts'
            health_dir.mkdir(exist_ok=True)
            
            # Export summary
            summary_data = []
            for scenario, results in self.results['health_impacts'].items():
                summary_data.append({
                    'scenario': scenario,
                    'total_attributable_cases': results['total_attributable_cases'],
                    'overall_attributable_rate': results['overall_attributable_rate']
                })
            
            pd.DataFrame(summary_data).to_csv(
                health_dir / 'summary.csv',
                index=False
            )
            
            # Export detailed results for each scenario
            for scenario, results in self.results['health_impacts'].items():
                scenario_dir = health_dir / f'scenario_{scenario}'
                scenario_dir.mkdir(exist_ok=True)
                
                # Export spatial data
                results['data'].to_file(
                    scenario_dir / 'spatial_results.gpkg',
                    driver='GPKG'
                )
                
                # Export demographic results if available
                for group_col in self.config.get('demographic_columns', []):
                    if f'by_{group_col}' in results:
                        group_results = results[f'by_{group_col}']
                        pd.DataFrame([
                            {
                                'group': group,
                                'attributable_cases': data['attributable_cases'],
                                'population': data['population'],
                                'attributable_rate': data['attributable_rate']
                            }
                            for group, data in group_results.items()
                        ]).to_csv(
                            scenario_dir / f'by_{group_col}.csv',
                            index=False
                        )
        
        # Create visualizations and save them
        self.create_visualizations(output_dir / 'figures')
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def run_complete_analysis(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Run the complete analysis workflow.
        
        Args:
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Running complete analysis")
        
        # Step 1: Prepare data
        self.prepare_data()
        
        # Step 2: Calculate population-weighted exposure
        self.calculate_population_weighted_exposure()
        
        # Step 3: Apply control scenarios
        self.apply_control_scenarios()
        
        # Step 4: Calculate health impacts
        self.calculate_health_impacts()
        
        # Step 5: Create visualizations and export results
        if output_dir:
            self.export_results(output_dir)
        
        self.logger.info("Complete analysis finished")
        return self.results
    
    def bin_tracts_by_distance(self, distance_bins: Optional[List[float]] = None) -> None:
        """
        Bin tracts by distance from a point.
        """
        self.logger.info("Binning tracts by distance")
        self.analysis_data = bin_tracts_by_distance(self.analysis_data, self.config['airport_coordinates'], distance_bins)
        self.logger.info("Tracts binned by distance")
