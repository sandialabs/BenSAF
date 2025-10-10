import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Dict, Optional, Tuple

def transform_relative_risk(
    mean_log_one_unit: float,
    se_log_one_unit: float,
    delta_ap: Union[float, np.ndarray]
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Transform relative risk parameters to calculate relative risk for a given change in air pollution.
    Includes confidence interval calculations.
    
    Args:
        mean_log_one_unit: Mean of log-transformed relative risk for one unit change
        se_log_one_unit: Standard error of log-transformed relative risk
        delta_ap: Change in air pollution concentration
        
    Returns:
        Tuple of (mean_trans, lower_trans, upper_trans) relative risk values with confidence intervals
    """
    z = stats.norm.ppf(0.975)  # 95% confidence interval

    mean_log_trans = mean_log_one_unit * delta_ap
    se_log_trans = se_log_one_unit * delta_ap

    mean_trans = np.exp(mean_log_trans)
    lower_trans = np.exp(mean_log_trans - z * se_log_trans)
    upper_trans = np.exp(mean_log_trans + z * se_log_trans)

    return mean_trans, lower_trans, upper_trans

def calculate_delta_ap(
    baseline_ufp: Union[float, np.ndarray],
    pct_reduction: float
) -> Union[float, np.ndarray]:
    """
    Calculate change in air pollution concentration.
    
    Args:
        baseline_ufp: Baseline UFP concentration
        pct_reduction: Percentage reduction in UFP
        
    Returns:
        Change in air pollution concentration
    """
    return baseline_ufp * (pct_reduction / 100)

def calculate_health_impacts(
    baseline_ufp: Union[float, np.ndarray],
    pct_reduction: float,
    incidence_rates: Union[float, np.ndarray],
    population: Union[float, np.ndarray],
    rr_params: Dict[str, float]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate health impacts of UFP reduction.
    
    Args:
        baseline_ufp: Baseline UFP concentration
        pct_reduction: Percentage reduction in UFP
        incidence_rates: Disease incidence rates
        population: Population size
        rr_params: Dictionary containing relative risk parameters
        
    Returns:
        Dictionary containing health impact metrics
    """
    # Calculate delta AP
    delta_ap = calculate_delta_ap(baseline_ufp, pct_reduction)
    
    # Transform relative risk
    rr_mean, rr_lower, rr_upper = transform_relative_risk(
        rr_params['mean_log_one_unit'],
        rr_params['se_log_one_unit'],
        delta_ap
    )
    
    # Calculate attributable fraction
    af_mean = calculate_attributable_fraction(rr_mean)
    af_lower = calculate_attributable_fraction(rr_lower)
    af_upper = calculate_attributable_fraction(rr_upper)
    
    # Calculate attributable cases
    ac_mean = calculate_attributable_cases(af_mean, incidence_rates, population)
    ac_lower = calculate_attributable_cases(af_lower, incidence_rates, population)
    ac_upper = calculate_attributable_cases(af_upper, incidence_rates, population)
    
    # Calculate attributable mortality
    am_mean = calculate_attributable_mortality(af_mean, incidence_rates)
    am_lower = calculate_attributable_mortality(af_lower, incidence_rates)
    am_upper = calculate_attributable_mortality(af_upper, incidence_rates)
    
    return {
        'delta_ap': delta_ap,
        'relative_risk': {
            'mean': rr_mean,
            'lower': rr_lower,
            'upper': rr_upper
        },
        'attributable_fraction': {
            'mean': af_mean,
            'lower': af_lower,
            'upper': af_upper
        },
        'attributable_cases': {
            'mean': ac_mean,
            'lower': ac_lower,
            'upper': ac_upper
        },
        'attributable_mortality': {
            'mean': am_mean,
            'lower': am_lower,
            'upper': am_upper
        }
    }

def calculate_attributable_fraction(rr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate attributable fraction from relative risk.
    
    Args:
        rr: Relative risk value(s)
        
    Returns:
        Attributable fraction
    """
    return (rr - 1) / rr

def calculate_attributable_cases(
    af: Union[float, np.ndarray],
    incidence_rates: Union[float, np.ndarray],
    population: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate attributable cases.
    
    Args:
        af: Attributable fraction
        incidence_rates: Disease incidence rates
        population: Population size
        
    Returns:
        Number of attributable cases
    """
    return sum(af * incidence_rates * population)

def calculate_attributable_mortality(
    af: Union[float, np.ndarray],
    incidence_rates: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate attributable mortality rate.
    
    Args:
        af: Attributable fraction
        incidence_rates: Disease incidence rates
        
    Returns:
        Attributable mortality rate
    """
    return af * incidence_rates 