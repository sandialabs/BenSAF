"""
Health impacts calculation module.

This module provides functions to calculate health impacts from pollutant exposure changes.
All functions operate on pandas Series for tract-level calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def calculate_health_impacts(
    delta_concentration: pd.Series,
    mortality_rate: pd.Series,
    population: pd.Series,
    mean_log_one_unit: float,
    se_log_one_unit: float
) -> Dict[str, pd.Series]:
    """
    Calculate health impacts from pollutant concentration changes.
    
    This function takes tract-level inputs as Series and returns tract-level
    health impact metrics as Series. All Series must have the same index (GEOID).
    
    Args:
        delta_concentration: Change in pollutant concentration per tract (Series)
        mortality_rate: Baseline mortality rate per tract (Series)
        population: Population per tract (Series)
        mean_log_one_unit: Mean log-transformed relative risk for one unit change
        se_log_one_unit: Standard error of log-transformed relative risk for one unit change
        
    Returns:
        Dictionary with the following Series (all indexed by GEOID):
        - relative_risk_mean: Mean relative risk
        - relative_risk_lower: Lower bound (95% CI)
        - relative_risk_upper: Upper bound (95% CI)
        - attributable_fraction_mean: Mean attributable fraction
        - attributable_fraction_lower: Lower attributable fraction (95% CI)
        - attributable_fraction_upper: Upper attributable fraction (95% CI)
        - attributable_cases_mean: Mean attributable cases per tract
        - attributable_cases_lower: Lower attributable cases per tract (95% CI)
        - attributable_cases_upper: Upper attributable cases per tract (95% CI)
        - attributable_mortality_rate_mean: Mean attributable mortality rate per tract
        - attributable_mortality_rate_lower: Lower attributable mortality rate (95% CI)
        - attributable_mortality_rate_upper: Upper attributable mortality rate (95% CI)
    """
    # Validate inputs have same index
    if not (delta_concentration.index.equals(mortality_rate.index) and 
            mortality_rate.index.equals(population.index)):
        raise ValueError("All input Series must have the same index")
    
    z = stats.norm.ppf(0.975)  # 95% confidence interval
    
    # Transform relative risk
    mean_log_trans = mean_log_one_unit * delta_concentration.values
    se_log_trans = se_log_one_unit * delta_concentration.values
    
    mean_rr = pd.Series(np.exp(mean_log_trans), index=delta_concentration.index)
    lower_rr = pd.Series(np.exp(mean_log_trans - z * se_log_trans), index=delta_concentration.index)
    upper_rr = pd.Series(np.exp(mean_log_trans + z * se_log_trans), index=delta_concentration.index)
    
    # Calculate attributable fraction
    mean_af = (mean_rr - 1) / mean_rr
    lower_af = (lower_rr - 1) / lower_rr
    upper_af = (upper_rr - 1) / upper_rr
    
    # Calculate attributable cases (per tract)
    mean_ac = mean_af * mortality_rate * population
    lower_ac = lower_af * mortality_rate * population
    upper_ac = upper_af * mortality_rate * population
    
    # Calculate attributable mortality rate (per tract)
    mean_amr = mean_af * mortality_rate
    lower_amr = lower_af * mortality_rate
    upper_amr = upper_af * mortality_rate
    
    return {
        'relative_risk_mean': mean_rr,
        'relative_risk_lower': lower_rr,
        'relative_risk_upper': upper_rr,
        'attributable_fraction_mean': mean_af,
        'attributable_fraction_lower': lower_af,
        'attributable_fraction_upper': upper_af,
        'attributable_cases_mean': mean_ac,
        'attributable_cases_lower': lower_ac,
        'attributable_cases_upper': upper_ac,
        'attributable_mortality_rate_mean': mean_amr,
        'attributable_mortality_rate_lower': lower_amr,
        'attributable_mortality_rate_upper': upper_amr
    }
