"""
Exposure pipeline for scenario analysis.

This pipeline computes exposure changes (reduced concentration, delta concentration,
and pollutant reduction) from SAF blend percentage.
"""

import logging
from typing import List, Tuple
import pandas as pd

from bensaf.scenario import Scenario

logger = logging.getLogger(__name__)


def calculate_pollutant_reduction_from_saf(
    saf_percentage: float,
    polynomial_coeffs: List[float]
) -> float:
    """
    Calculate pollutant reduction percentage from SAF blend percentage using polynomial fit.
    
    The polynomial produces a negative percentage (0 to -100) representing the percentage
    of pollutant to reduce. For example, -30 means reduce by 30% (multiply by 0.70).
    
    Args:
        saf_percentage: SAF blend percentage (0-100)
        polynomial_coeffs: Polynomial coefficients [a0, a1, a2, ...] for reduction = a0 + a1*SAF + a2*SAF^2 + ...
        
    Returns:
        Pollutant reduction percentage as negative value (0 to -100)
        e.g., -30 means reduce pollutant by 30%
    """
    reduction = 0.0
    for i, coeff in enumerate(polynomial_coeffs):
        reduction += coeff * (saf_percentage ** i)
    
    # Convert from decimal to percentage (e.g., -0.3 -> -30)
    reduction = reduction * 100
    
    # Ensure reduction is within valid range (0 to -100)
    # The polynomial should produce negative values, so we clamp to [-100, 0]
    reduction = max(-100.0, min(0.0, reduction))
    
    return reduction


def apply_control_scenario(
    baseline_exposure: pd.Series,
    saf_percentage: float,
    polynomial_coeffs: List[float],
    pollutant_name: str
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Apply a control scenario to calculate reduced exposures.
    
    Args:
        baseline_exposure: Baseline pollutant concentration per tract (Series)
        saf_percentage: SAF blend percentage (0-100)
        polynomial_coeffs: Polynomial coefficients for SAF to pollutant reduction
        pollutant_name: Name of the pollutant
        
    Returns:
        Tuple of (reduced_concentration, delta_concentration, pollutant_reduction)
        where pollutant_reduction is negative (e.g., -30 means 30% reduction)
    """
    pollutant_reduction = calculate_pollutant_reduction_from_saf(
        saf_percentage,
        polynomial_coeffs
    )
    
    # Apply reduction: reduction is negative (e.g., -30 means multiply by 0.70)
    # Formula: reduced = baseline * (1 + reduction/100) where reduction is negative
    reduced_concentration = baseline_exposure * (1 + pollutant_reduction / 100)
    delta_concentration = baseline_exposure - reduced_concentration
    
    logger.debug(
        f"apply_control_scenario: baseline_exposure index type={type(baseline_exposure.index)}, "
        f"length={len(baseline_exposure.index)}, dtype={baseline_exposure.index.dtype}, "
        f"name={baseline_exposure.index.name}"
    )
    
    return reduced_concentration, delta_concentration, pollutant_reduction


def run_exposure_pipeline(scenario: Scenario, polynomial_coeffs: List[float]) -> None:
    """
    Run exposure pipeline to compute exposure changes.
    
    This is the first pipeline that all scenarios need. It computes:
    - pollutant_reduction: Negative percentage reduction (e.g., -30 for 30% reduction)
    - reduced_concentration: Baseline exposure after reduction
    - delta_concentration: Change in concentration (baseline - reduced)
    
    Args:
        scenario: Scenario object with baseline_exposure populated
        polynomial_coeffs: Polynomial coefficients for SAF to pollutant reduction
    """
    logger.debug(f"Running exposure pipeline for scenario {scenario.scenario_id}")
    
    # Apply control scenario to get reduced and delta concentrations
    reduced_concentration, delta_concentration, pollutant_reduction = apply_control_scenario(
        scenario.baseline_exposure,
        scenario.saf_percentage,
        polynomial_coeffs,
        scenario.pollutant_name
    )
    
    # Populate scenario outputs
    scenario.pollutant_reduction = pollutant_reduction
    scenario.reduced_concentration = reduced_concentration
    scenario.delta_concentration = delta_concentration
    
    logger.debug(
        f"Exposure pipeline complete: reduction={pollutant_reduction:.2f}%, "
        f"delta_concentration range=[{delta_concentration.min():.2f}, {delta_concentration.max():.2f}]"
    )
