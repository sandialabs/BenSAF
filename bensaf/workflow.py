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
from bensaf.data_model import AnalysisData, AnalysisConfig
from bensaf.utils import bin_tracts_by_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Workflow:
    """
    Generalized workflow for SAF health impact assessment.
    
    This class provides a simplified interface to the health impact assessment workflow,
    focusing on computation and orchestration while delegating data storage to AnalysisData.
    
    The workflow consists of the following steps:
    1. Load and validate data (using AnalysisData)
    2. Apply control scenarios
    3. Calculate health impacts
    4. Generate results and visualizations
    
    Example:
        ```python
        # Initialize workflow
        config = AnalysisConfig(saf_scenarios=[5, 25, 50])
        workflow = Workflow(config)
        
        # Load data
        workflow.data.load_tract_geometries(tracts_gdf)
        workflow.data.load_demographics(demographics_df)
        workflow.data.load_baseline_exposure_data(exposure_df)
        workflow.data.load_mortality_data(mortality_df)
        
        # Set health impact function
        workflow.set_health_impact_function(1.012, 1.010, 1.015, 2723)
        
        # Run analysis
        workflow.apply_control_scenarios()
        workflow.calculate_health_impacts()
        
        # Get results
        results = workflow.get_results()
        workflow.create_visualizations("results")
        ```
    """
    
    def __init__(self, config: Optional[Union[AnalysisConfig, Dict[str, Any]]] = None):
        """
        Initialize the SAF workflow.
        
        Args:
            config: Optional AnalysisConfig instance or dictionary with parameters.
                If dict, will be converted to AnalysisConfig.
        """
        # Convert dict to AnalysisConfig if needed
        if isinstance(config, dict):
            self.config = AnalysisConfig(**config)
        elif config is None:
            self.config = AnalysisConfig()
        else:
            self.config = config
        
        # Initialize data model
        self.data = AnalysisData(crs=self.config.crs)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Aggregated results (separate from tract-level outputs in AnalysisData)
        self.aggregated_results: Dict[str, Any] = {}
    
    def set_health_impact_function(self, 
                                   mean_rr: float, 
                                   lower_rr: float, 
                                   upper_rr: float,
                                   unit_increase: float) -> None:
        """
        Set health impact function parameters.
        
        Args:
            mean_rr: Mean relative risk
            lower_rr: Lower bound of relative risk (95% CI)
            upper_rr: Upper bound of relative risk (95% CI)
            unit_increase: Unit increase in pollutant concentration for the relative risk
        """
        self.logger.info("Setting health impact function")
        
        # Calculate beta and SE from relative risk
        z = 1.96  # 95% confidence interval
        
        mean_log = np.log(mean_rr)
        lower_log = np.log(lower_rr)
        upper_log = np.log(upper_rr)
        
        se_log = ((upper_log - mean_log) + (mean_log - lower_log)) / (2 * z)
        
        mean_log_one_unit = mean_log / unit_increase
        se_log_one_unit = se_log / unit_increase
        
        # Store parameters in config
        self.config.health_impact_function = {
            'mean_rr': mean_rr,
            'lower_rr': lower_rr,
            'upper_rr': upper_rr,
            'unit_increase': unit_increase,
            'mean_log_one_unit': mean_log_one_unit,
            'se_log_one_unit': se_log_one_unit
        }
        
        self.logger.info("Health impact function set")
    
    def calculate_pollutant_reduction_from_saf(self, saf_percentage: float) -> float:
        """
        Calculate pollutant reduction percentage from SAF blend percentage using polynomial fit.
        
        Args:
            saf_percentage: SAF blend percentage (0-100)
            
        Returns:
            Pollutant reduction percentage (0-100)
        """
        coeffs = self.config.saf_polynomial_coeffs
        
        # Calculate polynomial: reduction = a0 + a1*SAF + a2*SAF^2 + ...
        reduction = 0.0
        for i, coeff in enumerate(coeffs):
            reduction += coeff * (saf_percentage ** i)
        
        # Ensure reduction is within valid range
        reduction = max(0.0, min(100.0, reduction))
        
        return reduction
    
    def get_saf_reduction_curve(self, saf_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the SAF to pollutant reduction curve for visualization.
        
        Args:
            saf_range: Array of SAF percentages to evaluate. If None, uses 0-100 in steps of 1.
            
        Returns:
            Tuple of (saf_percentages, pollutant_reductions)
        """
        if saf_range is None:
            saf_range = np.linspace(0, 100, 101)
        
        reductions = np.array([self.calculate_pollutant_reduction_from_saf(saf) for saf in saf_range])
        
        return saf_range, reductions
    
    def apply_control_scenarios(self, scenarios: Optional[List[float]] = None, 
                               use_saf_polynomial: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Apply control scenarios to calculate reduced exposures.
        
        Args:
            scenarios: List of SAF blend percentages (0-100).
                If None, uses scenarios from config.
            use_saf_polynomial: If True, calculates pollutant reduction from SAF percentage
                using polynomial fit. If False, treats scenarios as direct pollutant reductions.
                
        Returns:
            Dictionary of scenario results with metadata
        """
        self.logger.info("Applying control scenarios")
        
        # Validate data is ready
        self.data.validate()
        
        # Use provided scenarios or get from config
        if scenarios is None:
            scenarios = self.config.saf_scenarios
        
        # Get baseline exposure
        baseline_exposure = self.data.baseline_exposure['baseline_pollutant_concentration']
        mortality_rate = self.data.mortality['mortality_rate']
        
        scenario_results = {}
        
        for saf_percentage in scenarios:
            # Calculate pollutant reduction from SAF percentage
            if use_saf_polynomial:
                pollutant_reduction = self.calculate_pollutant_reduction_from_saf(saf_percentage)
                self.logger.info(f"SAF {saf_percentage}% -> Pollutant reduction {pollutant_reduction:.2f}%")
            else:
                pollutant_reduction = saf_percentage
            
            # Calculate reduced exposure
            reduced_concentration = baseline_exposure * (1 - pollutant_reduction / 100)
            
            # Calculate delta exposure
            delta_concentration = baseline_exposure - reduced_concentration
            
            # Create scenario name
            scenario_name = f"{saf_percentage}% SAF Usage"
            
            # Prepare outputs dictionary
            outputs = {
                'saf_percentage': pd.Series(saf_percentage, index=baseline_exposure.index),
                'pollutant_reduction': pd.Series(pollutant_reduction, index=baseline_exposure.index),
                'reduced_concentration': reduced_concentration,
                'delta_concentration': delta_concentration
            }
            
            # Add to AnalysisData
            self.data.add_scenario_output(scenario_name, outputs)
            
            # Store metadata
            scenario_results[scenario_name] = {
                'saf_percentage': saf_percentage,
                'pollutant_reduction': pollutant_reduction
            }
        
        self.logger.info(f"Applied {len(scenarios)} control scenarios")
        return scenario_results
    
    def calculate_health_impacts(self) -> Dict[str, Any]:
        """
        Calculate health impacts for all control scenarios.
        
        Returns:
            Dictionary of health impact results by scenario
        """
        self.logger.info("Calculating health impacts")
        
        # Check if health impact function is set
        if self.config.health_impact_function is None:
            self.logger.warning("Health impact function not set, using default from Bouma et al.")
            self.set_health_impact_function(
                mean_rr=1.012,
                lower_rr=1.010,
                upper_rr=1.015,
                unit_increase=2723  # pt/cm3
            )
        
        # Get required data
        baseline_exposure = self.data.baseline_exposure['baseline_pollutant_concentration']
        mortality_rate = self.data.mortality['mortality_rate']
        
        # Get population if available
        if self.data.demographics is not None and 'population' in self.data.demographics.columns:
            population = self.data.demographics['population']
        else:
            self.logger.warning("No population data found, using 1.0 for all tracts")
            population = pd.Series(1.0, index=baseline_exposure.index)
        
        health_function = self.config.health_impact_function
        
        # Calculate health impacts for each scenario
        for scenario_name, scenario_df in self.data.scenario_outputs.items():
            delta_concentration = scenario_df['delta_concentration']
            
            # Transform relative risk
            mean_rr, lower_rr, upper_rr = transform_relative_risk(
                health_function['mean_log_one_unit'],
                health_function['se_log_one_unit'],
                delta_concentration.values
            )
            
            # Convert to Series
            mean_rr = pd.Series(mean_rr, index=delta_concentration.index)
            lower_rr = pd.Series(lower_rr, index=delta_concentration.index)
            upper_rr = pd.Series(upper_rr, index=delta_concentration.index)
            
            # Calculate attributable fraction
            mean_af = calculate_attributable_fraction(mean_rr)
            lower_af = calculate_attributable_fraction(lower_rr)
            upper_af = calculate_attributable_fraction(upper_rr)
            
            # Calculate attributable cases (per tract)
            mean_ac = pd.Series(mean_af * mortality_rate * population, index=delta_concentration.index)
            lower_ac = pd.Series(lower_af * mortality_rate * population, index=delta_concentration.index)
            upper_ac = pd.Series(upper_af * mortality_rate * population, index=delta_concentration.index)
            
            # Calculate attributable mortality rate (per tract)
            mean_amr = pd.Series(mean_af * mortality_rate, index=delta_concentration.index)
            lower_amr = pd.Series(lower_af * mortality_rate, index=delta_concentration.index)
            upper_amr = pd.Series(upper_af * mortality_rate, index=delta_concentration.index)
            
            # Add to scenario outputs
            new_outputs = {
                'relative_risk_mean': mean_rr,
                'relative_risk_lower': lower_rr,
                'relative_risk_upper': upper_rr,
                'attributable_fraction_mean': pd.Series(mean_af, index=delta_concentration.index),
                'attributable_fraction_lower': pd.Series(lower_af, index=delta_concentration.index),
                'attributable_fraction_upper': pd.Series(upper_af, index=delta_concentration.index),
                'attributable_cases_mean': mean_ac,
                'attributable_cases_lower': lower_ac,
                'attributable_cases_upper': upper_ac,
                'attributable_mortality_rate_mean': mean_amr,
                'attributable_mortality_rate_lower': lower_amr,
                'attributable_mortality_rate_upper': upper_amr
            }
            
            # Update scenario outputs
            for key, value in new_outputs.items():
                scenario_df[key] = value
        
        # Calculate aggregated results
        self._calculate_aggregated_results()
        
        self.logger.info("Health impact calculation complete")
        return self.aggregated_results
    
    def _calculate_aggregated_results(self) -> None:
        """Calculate aggregated results across all scenarios."""
        self.aggregated_results = {}
        
        for scenario_name, scenario_df in self.data.scenario_outputs.items():
            # Sum attributable cases
            total_ac_mean = scenario_df['attributable_cases_mean'].sum()
            total_ac_lower = scenario_df['attributable_cases_lower'].sum()
            total_ac_upper = scenario_df['attributable_cases_upper'].sum()
            
            # Calculate overall attributable mortality rate (weighted average)
            if self.data.demographics is not None and 'population' in self.data.demographics.columns:
                population = self.data.demographics['population']
                total_pop = population.sum()
                
                overall_amr_mean = (
                    (scenario_df['attributable_mortality_rate_mean'] * population).sum() / total_pop
                )
                overall_amr_lower = (
                    (scenario_df['attributable_mortality_rate_lower'] * population).sum() / total_pop
                )
                overall_amr_upper = (
                    (scenario_df['attributable_mortality_rate_upper'] * population).sum() / total_pop
                )
            else:
                overall_amr_mean = scenario_df['attributable_mortality_rate_mean'].mean()
                overall_amr_lower = scenario_df['attributable_mortality_rate_lower'].mean()
                overall_amr_upper = scenario_df['attributable_mortality_rate_upper'].mean()
            
            self.aggregated_results[scenario_name] = {
                'total_attributable_cases': {
                    'mean': total_ac_mean,
                    'lower': total_ac_lower,
                    'upper': total_ac_upper
                },
                'overall_attributable_mortality_rate': {
                    'mean': overall_amr_mean,
                    'lower': overall_amr_lower,
                    'upper': overall_amr_upper
                }
            }
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all analysis results.
        
        Returns:
            Dictionary containing aggregated results and access to tract-level data
        """
        return {
            'aggregated': self.aggregated_results,
            'tract_level': self.data
        }
    
    def create_visualizations(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create visualization plots for the analysis.
        
        Args:
            output_dir: Optional directory to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        self.logger.info("Creating visualizations")
        
        # Get merged data
        merged_data = self.data.get_merged_data()
        
        # Create output directory if specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figures
        figures = {}
        
        # 1. Map of baseline pollutant concentration
        fig, ax = plt.subplots(figsize=(10, 8))
        merged_data.plot(
            column='baseline_pollutant_concentration',
            ax=ax,
            legend=True,
            cmap='viridis',
            legend_kwds={'label': 'Baseline Pollutant Concentration'}
        )
        ax.set_title('Baseline Pollutant Concentration')
        figures['baseline_concentration'] = fig
        
        if output_dir:
            fig.savefig(str(output_dir / 'baseline_concentration.png'), dpi=300, bbox_inches='tight')
        
        # 2. Map of attributable cases for first scenario
        if self.data.scenario_outputs:
            first_scenario = list(self.data.scenario_outputs.keys())[0]
            scenario_col = f"{first_scenario}_attributable_cases_mean"
            
            if scenario_col in merged_data.columns:
                fig, ax = plt.subplots(figsize=(10, 8))
                merged_data.plot(
                    column=scenario_col,
                    ax=ax,
                    legend=True,
                    cmap='viridis',
                    legend_kwds={'label': 'Attributable Cases'}
                )
                ax.set_title(f'Attributable Cases ({first_scenario})')
                figures['attributable_cases'] = fig
                
                if output_dir:
                    fig.savefig(str(output_dir / 'attributable_cases.png'), dpi=300, bbox_inches='tight')
        
        # 3. Bar chart of total attributable cases by scenario
        if self.aggregated_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            scenarios = list(self.aggregated_results.keys())
            total_cases = [
                self.aggregated_results[s]['total_attributable_cases']['mean'] 
                for s in scenarios
            ]
            
            ax.bar(scenarios, total_cases)
            ax.set_xlabel('Control Scenario')
            ax.set_ylabel('Total Attributable Cases')
            ax.set_title('Health Impacts by Control Scenario')
            ax.tick_params(axis='x', rotation=45)
            figures['scenario_comparison'] = fig
            
            if output_dir:
                fig.savefig(str(output_dir / 'scenario_comparison.png'), dpi=300, bbox_inches='tight')
        
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
        
        # Export merged data
        merged_data = self.data.get_merged_data()
        merged_data.to_file(
            output_dir / 'analysis_data.gpkg',
            driver='GPKG'
        )
        
        # Export aggregated results
        if self.aggregated_results:
            summary_data = []
            for scenario, results in self.aggregated_results.items():
                summary_data.append({
                    'scenario': scenario,
                    'total_attributable_cases_mean': results['total_attributable_cases']['mean'],
                    'total_attributable_cases_lower': results['total_attributable_cases']['lower'],
                    'total_attributable_cases_upper': results['total_attributable_cases']['upper'],
                    'overall_amr_mean': results['overall_attributable_mortality_rate']['mean'],
                    'overall_amr_lower': results['overall_attributable_mortality_rate']['lower'],
                    'overall_amr_upper': results['overall_attributable_mortality_rate']['upper']
                })
            
            pd.DataFrame(summary_data).to_csv(
                output_dir / 'aggregated_results.csv',
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
        
        # Step 1: Validate data
        self.data.validate()
        
        # Step 2: Apply control scenarios
        self.apply_control_scenarios()
        
        # Step 3: Calculate health impacts
        self.calculate_health_impacts()
        
        # Step 4: Create visualizations and export results
        if output_dir:
            self.export_results(output_dir)
        
        self.logger.info("Complete analysis finished")
        return self.get_results()
