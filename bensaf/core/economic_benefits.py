"""
Economic benefits calculation module.

This module provides functions to convert health impact quantifications into
economic benefits, following the methodology from the economic benefits paper.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def calculate_mortality_economic_value(
    attributable_cases: Union[float, np.ndarray, pd.Series],
    per_capita_consumption: float,
    life_years_gained: float = 10.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate economic value of mortality reduction.
    
    Uses the expenditure function model: life value = per_capita_consumption x life_years_gained,
    then multiplies by attributable cases to get total economic value.
    
    Args:
        attributable_cases: Number of deaths avoided (can be mean, lower, or upper bound)
        per_capita_consumption: Per capita consumption for the region (e.g., LA County, King County)
        life_years_gained: Years of life gained per adult death avoided (default: 10)
        
    Returns:
        Economic value of mortality reduction
    """
    life_value = per_capita_consumption * life_years_gained
    economic_value = life_value * attributable_cases
    
    return economic_value


def calculate_preterm_birth_reduction(
    baseline_ptb: Union[float, np.ndarray, pd.Series],
    delta_ufp: Union[float, np.ndarray, pd.Series],
    odds_ratio: float
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate reduction in preterm births due to UFP reduction.
    
    Formula: delta_ptb = baseline_ptb x (1 - odds_ratio ** delta_ufp)
    
    Args:
        baseline_ptb: Baseline number of preterm births
        delta_ufp: Change in UFP concentration (marginal change from SAF blending)
        odds_ratio: Odds ratio for preterm birth associated with UFP exposure (typically 1.4)
        
    Returns:
        Reduction in preterm births
    """
    delta_ptb = baseline_ptb * (1 - odds_ratio ** delta_ufp)
    
    return delta_ptb


def calculate_preterm_birth_economic_value(
    delta_ptb: Union[float, np.ndarray, pd.Series],
    monetary_value_per_ptb: float
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate economic value of preterm birth reduction.
    
    Args:
        delta_ptb: Reduction in preterm births (can be mean, lower, or upper bound)
        monetary_value_per_ptb: Total cost per preterm birth avoided (includes immediate
            medical costs and lifetime costs)
            
    Returns:
        Economic value of preterm birth reduction
    """
    economic_value = delta_ptb * monetary_value_per_ptb
    
    return economic_value
