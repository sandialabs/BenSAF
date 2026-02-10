"""
Scenario data structure.

This module defines the Scenario class which holds both inputs and outputs
for a specific scenario analysis. The Scenario is initialized with input data
and then passed to pipeline functions that populate its output attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from bensaf.data_model import AnalysisData, AnalysisConfig


@dataclass
class Scenario:
    """
    Container for scenario inputs and outputs.
    
    This class holds both the input data needed for a scenario analysis and
    the computed outputs. It is initialized with inputs, then passed to pipeline
    functions that populate the output attributes.
    
    Inputs:
    - saf_percentage: SAF blend percentage (0-100)
    - pollutant_name: Name of pollutant (e.g., "ufp", "pm25", "nox")
    - baseline_exposure: Baseline pollutant concentration per tract (Series)
    - data: Reference to AnalysisData with shared data (demographics, incidence, etc.)
    - config: Reference to AnalysisConfig with analysis parameters
    
    Outputs (populated by pipelines):
    - reduced_concentration: Reduced pollutant concentration per tract (Series)
    - delta_concentration: Change in pollutant concentration per tract (Series)
    - pollutant_reduction: Pollutant reduction percentage (negative, e.g., -30 for 30% reduction)
    - health_impacts: Dictionary of health impact metrics by endpoint
    - economic_benefits: Dictionary of economic benefit Series
    
    Metadata:
    - scenario_id: Numeric identifier (e.g., 25 for 25% SAF)
    - scenario_label: Human-readable label (e.g., "25% SAF Usage")
    """
    
    # Inputs
    scenario_id: int
    scenario_label: str
    saf_percentage: float
    pollutant_name: str
    baseline_exposure: pd.Series
    data: 'AnalysisData'
    
    # Outputs (initially None, populated by pipelines)
    reduced_concentration: Optional[pd.Series] = None
    delta_concentration: Optional[pd.Series] = None
    pollutant_reduction: Optional[float] = None
    health_impacts: Dict[str, Dict[str, pd.Series]] = field(default_factory=dict)
    economic_benefits: Dict[str, pd.Series] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert scenario to a single DataFrame.
        
        Returns:
            DataFrame with GEOID as index and all scenario outputs as columns
        """
        if self.delta_concentration is None:
            raise ValueError("Scenario outputs not yet computed. Run pipeline functions first.")
        
        # Start with exposure data
        data = {
            'reduced_concentration': self.reduced_concentration,
            'delta_concentration': self.delta_concentration,
        }
        
        # Add scalar metadata as constant columns
        index = self.delta_concentration.index
        data['scenario_id'] = pd.Series(self.scenario_id, index=index)
        data['scenario_label'] = pd.Series(self.scenario_label, index=index)
        data['saf_percentage'] = pd.Series(self.saf_percentage, index=index)
        if self.pollutant_reduction is not None:
            data['pollutant_reduction'] = pd.Series(self.pollutant_reduction, index=index)
        data['pollutant_name'] = pd.Series(self.pollutant_name, index=index)
        
        # Add health impacts
        for endpoint_name, endpoint_data in self.health_impacts.items():
            for metric_name, metric_series in endpoint_data.items():
                col_name = f"{endpoint_name}_{metric_name}"
                data[col_name] = metric_series
        
        # Add economic benefits
        for benefit_name, benefit_series in self.economic_benefits.items():
            data[benefit_name] = benefit_series
        
        return pd.DataFrame(data)
    
    def get_aggregated_results(self, population: Optional[pd.Series] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregated results across all tracts.
        
        Args:
            population: Optional population Series for weighted averages
            
        Returns:
            Dictionary with aggregated metrics by endpoint
        """
        results = {}
        
        for endpoint_name, endpoint_data in self.health_impacts.items():
            endpoint_results = {}
            
            # Sum attributable cases
            if 'attributable_cases_mean' in endpoint_data:
                ac_mean = endpoint_data['attributable_cases_mean'].sum()
                ac_lower = endpoint_data['attributable_cases_lower'].sum()
                ac_upper = endpoint_data['attributable_cases_upper'].sum()
                endpoint_results['total_attributable_cases'] = {
                    'mean': float(ac_mean),
                    'lower': float(ac_lower),
                    'upper': float(ac_upper)
                }
            
            # Weighted average attributable mortality rate
            if 'attributable_mortality_rate_mean' in endpoint_data:
                if population is not None:
                    total_pop = population.sum()
                    amr_mean = (endpoint_data['attributable_mortality_rate_mean'] * population).sum() / total_pop
                    amr_lower = (endpoint_data['attributable_mortality_rate_lower'] * population).sum() / total_pop
                    amr_upper = (endpoint_data['attributable_mortality_rate_upper'] * population).sum() / total_pop
                else:
                    amr_mean = endpoint_data['attributable_mortality_rate_mean'].mean()
                    amr_lower = endpoint_data['attributable_mortality_rate_lower'].mean()
                    amr_upper = endpoint_data['attributable_mortality_rate_upper'].mean()
                
                endpoint_results['overall_attributable_mortality_rate'] = {
                    'mean': float(amr_mean),
                    'lower': float(amr_lower),
                    'upper': float(amr_upper)
                }
            
            results[endpoint_name] = endpoint_results
        
        # Add economic benefits
        economic_results = {}
        for benefit_name, benefit_series in self.economic_benefits.items():
            if 'mean' in benefit_name:
                base_name = benefit_name.replace('_mean', '')
                mean_val = benefit_series.sum()
                lower_val = self.economic_benefits.get(benefit_name.replace('_mean', '_lower'), pd.Series([0])).sum()
                upper_val = self.economic_benefits.get(benefit_name.replace('_mean', '_upper'), pd.Series([0])).sum()
                economic_results[base_name] = {
                    'mean': float(mean_val),
                    'lower': float(lower_val),
                    'upper': float(upper_val)
                }
        
        if economic_results:
            results['economic_benefits'] = economic_results
        
        return results
