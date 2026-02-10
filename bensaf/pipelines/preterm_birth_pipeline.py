"""
Preterm birth pipeline for scenario analysis.

This pipeline computes preterm birth reduction and economic benefits.
It loads its own static configuration from JSON files.
"""

import logging
from pathlib import Path
import json

import pandas as pd

from bensaf.scenario import Scenario
from bensaf.economic_benefits import (
    calculate_preterm_birth_reduction,
    calculate_preterm_birth_economic_value
)

logger = logging.getLogger(__name__)


def _load_preterm_birth_parameters() -> dict:
    """
    Load preterm birth parameters from JSON.
    
    Returns:
        Dictionary with preterm_birth_odds_ratio and monetary_value_per_ptb
    """
    project_root = Path(__file__).parent.parent.parent
    econ_params_file = project_root / 'data' / 'economic_parameters.json'
    
    default_params = {
        'preterm_birth_odds_ratio': None,
        'monetary_value_per_ptb': None
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


def run_preterm_birth_pipeline(scenario: Scenario) -> None:
    """
    Run preterm birth pipeline to compute preterm birth reduction and economic benefits.
    
    This pipeline:
    1. Loads preterm birth parameters from JSON
    2. Computes reduction in preterm births due to UFP reduction
    3. Computes economic benefits
    
    Args:
        scenario: Scenario object with delta_concentration populated
    """
    logger.debug(f"Running preterm birth pipeline for scenario {scenario.scenario_id}")
    
    if scenario.delta_concentration is None:
        raise ValueError("Exposure pipeline must be run before preterm birth pipeline")
    
    # Load parameters
    params = _load_preterm_birth_parameters()
    
    # Check if required data and parameters are available
    if (params['preterm_birth_odds_ratio'] is None or
        params['monetary_value_per_ptb'] is None or
        scenario.data.preterm_birth_core is None):
        logger.debug("Preterm birth data or parameters not configured, skipping preterm birth pipeline")
        return
    
    baseline_ptb = scenario.data.preterm_birth_core['baseline_preterm_births']
    delta_concentration = scenario.delta_concentration
    
    # Ensure indices align
    common_index = delta_concentration.index.intersection(baseline_ptb.index)
    if len(common_index) == 0:
        logger.error("No common index between delta_concentration and baseline_preterm_births")
        return
    
    delta_concentration_aligned = delta_concentration.reindex(common_index, fill_value=0.0)
    baseline_ptb_aligned = baseline_ptb.reindex(common_index)
    
    # Calculate preterm birth reduction
    mean_ptb_reduction = calculate_preterm_birth_reduction(
        baseline_ptb_aligned,
        delta_concentration_aligned,
        params['preterm_birth_odds_ratio']
    )
    
    # For lower/upper bounds, use mean for now (could be enhanced with uncertainty propagation)
    lower_ptb_reduction = mean_ptb_reduction * 0.9
    upper_ptb_reduction = mean_ptb_reduction * 1.1
    
    # Calculate economic value
    mean_ptb_value = calculate_preterm_birth_economic_value(
        mean_ptb_reduction,
        params['monetary_value_per_ptb']
    )
    lower_ptb_value = calculate_preterm_birth_economic_value(
        lower_ptb_reduction,
        params['monetary_value_per_ptb']
    )
    upper_ptb_value = calculate_preterm_birth_economic_value(
        upper_ptb_reduction,
        params['monetary_value_per_ptb']
    )
    
    # Populate scenario outputs
    scenario.economic_benefits['preterm_birth_reduction_mean'] = mean_ptb_reduction
    scenario.economic_benefits['preterm_birth_reduction_lower'] = lower_ptb_reduction
    scenario.economic_benefits['preterm_birth_reduction_upper'] = upper_ptb_reduction
    scenario.economic_benefits['preterm_birth_economic_value_mean'] = mean_ptb_value
    scenario.economic_benefits['preterm_birth_economic_value_lower'] = lower_ptb_value
    scenario.economic_benefits['preterm_birth_economic_value_upper'] = upper_ptb_value
    
    logger.debug(
        f"Preterm birth pipeline complete: mean reduction={mean_ptb_reduction.sum():.2f}, "
        f"mean value=${mean_ptb_value.sum():,.0f}"
    )
