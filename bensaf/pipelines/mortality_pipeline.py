"""
Mortality pipeline for scenario analysis.

This pipeline computes mortality health impacts and economic benefits.
It loads its own static configuration from JSON files.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import json
import numpy as np
from scipy import stats

import pandas as pd

from bensaf.scenario import Scenario
from bensaf.health_impacts import calculate_health_impacts
from bensaf.economic_benefits import calculate_mortality_economic_value
from bensaf.mortality_functions import MortalityFunctionLibrary

logger = logging.getLogger(__name__)


def _load_mortality_function_config(function_id: Optional[int] = None) -> Dict:
    """
    Load mortality function configuration from JSON.
    
    Args:
        function_id: Optional function ID. If None, uses first available.
        
    Returns:
        Dictionary with mortality function parameters
    """
    library = MortalityFunctionLibrary()
    
    if function_id is None:
        functions = library.list_functions()
        if not functions:
            raise ValueError("No mortality functions available")
        function_id = functions[0]['id']
    
    function_data = library.get_function(function_id)
    if function_data is None:
        raise ValueError(f"Mortality function {function_id} not found")
    
    return function_data


def _load_economic_parameters() -> Dict:
    """
    Load economic parameters from JSON.
    
    Returns:
        Dictionary with economic parameters
    """
    project_root = Path(__file__).parent.parent.parent
    econ_params_file = project_root / 'data' / 'economic_parameters.json'
    
    default_params = {
        'per_capita_consumption': None,
        'life_years_gained': 10.0
    }
    
    if not econ_params_file.exists():
        logger.debug(f"Economic parameters file not found, using defaults")
        return default_params
    
    try:
        with open(econ_params_file, 'r') as f:
            data = json.load(f)
        return {**default_params, **data}
    except Exception as e:
        logger.warning(f"Error loading economic parameters: {e}, using defaults")
        return default_params


def run_mortality_pipeline(scenario: Scenario, mortality_function_id: Optional[int] = None) -> None:
    """
    Run mortality pipeline to compute mortality health impacts and economic benefits.
    
    This pipeline:
    1. Loads mortality function configuration from JSON
    2. Computes mortality health impacts (attributable cases, rates, etc.)
    3. Computes mortality economic benefits (if per_capita_consumption is configured)
    
    Args:
        scenario: Scenario object with delta_concentration populated
        mortality_function_id: Optional mortality function ID. If None, uses first available.
    """
    logger.debug(f"Running mortality pipeline for scenario {scenario.scenario_id}")
    
    if scenario.delta_concentration is None:
        raise ValueError("Exposure pipeline must be run before mortality pipeline")
    
    # Load mortality function configuration
    function_data = _load_mortality_function_config(mortality_function_id)
    
    # Convert to log-transformed parameters
    z = stats.norm.ppf(0.975)  # 95% confidence interval
    mean_log = np.log(function_data['mean_rr'])
    lower_log = np.log(function_data['lower_rr'])
    upper_log = np.log(function_data['upper_rr'])
    se_log = ((upper_log - mean_log) + (mean_log - lower_log)) / (2 * z)
    mean_log_one_unit = mean_log / function_data['unit_increase']
    se_log_one_unit = se_log / function_data['unit_increase']
    
    # Get mortality rate from incidence data
    if scenario.data.incidence is None or 'mortality_rate' not in scenario.data.incidence.columns:
        logger.warning("Mortality rate column not found, skipping mortality pipeline")
        return
    
    mortality_rate = scenario.data.incidence['mortality_rate']
    
    # Get population from core demographics
    if scenario.data.demographics_core is not None and 'population' in scenario.data.demographics_core.columns:
        population = scenario.data.demographics_core['population']
    else:
        logger.warning("No population data found, using 1.0 for all tracts")
        population = pd.Series(1.0, index=scenario.delta_concentration.index)
    
    # Ensure indices align
    common_index = scenario.delta_concentration.index.intersection(mortality_rate.index)
    if len(common_index) == 0:
        logger.error("No common index between delta_concentration and mortality_rate")
        return
    
    delta_concentration_aligned = scenario.delta_concentration.reindex(common_index, fill_value=0.0)
    mortality_rate_aligned = mortality_rate.reindex(common_index)
    population_aligned = population.reindex(common_index, fill_value=0.0)
    
    # Calculate health impacts
    impacts = calculate_health_impacts(
        delta_concentration=delta_concentration_aligned,
        mortality_rate=mortality_rate_aligned,
        population=population_aligned,
        mean_log_one_unit=mean_log_one_unit,
        se_log_one_unit=se_log_one_unit
    )
    
    scenario.health_impacts['mortality'] = impacts
    logger.debug("Computed mortality impacts")
    
    # Calculate economic benefits if configured
    econ_params = _load_economic_parameters()
    if econ_params.get('per_capita_consumption') is not None:
        mean_mort_value = calculate_mortality_economic_value(
            impacts['attributable_cases_mean'],
            econ_params['per_capita_consumption'],
            econ_params['life_years_gained']
        )
        lower_mort_value = calculate_mortality_economic_value(
            impacts['attributable_cases_lower'],
            econ_params['per_capita_consumption'],
            econ_params['life_years_gained']
        )
        upper_mort_value = calculate_mortality_economic_value(
            impacts['attributable_cases_upper'],
            econ_params['per_capita_consumption'],
            econ_params['life_years_gained']
        )
        
        scenario.economic_benefits['mortality_economic_value_mean'] = mean_mort_value
        scenario.economic_benefits['mortality_economic_value_lower'] = lower_mort_value
        scenario.economic_benefits['mortality_economic_value_upper'] = upper_mort_value
        
        logger.debug("Computed mortality economic benefits")
    else:
        logger.debug("Economic parameters not configured, skipping mortality economic benefits")
