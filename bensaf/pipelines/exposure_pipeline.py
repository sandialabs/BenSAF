"""
Exposure pipeline for scenario analysis.

Pure functions: no Scenario mutation, no side effects.
"""

import logging
from typing import List, Tuple
import pandas as pd

from bensaf.model.domain import ScenarioSpec

logger = logging.getLogger(__name__)


def calculate_pollutant_reduction_from_saf(
    saf_percentage: float,
    polynomial_coeffs: List[float],
) -> float:
    """
    Calculate pollutant reduction percentage from SAF blend percentage.

    Uses a polynomial fit. Returns a negative value in [−100, 0] representing
    the percentage reduction (e.g. −30 means 30% reduction).
    """
    reduction = sum(coeff * (saf_percentage ** i) for i, coeff in enumerate(polynomial_coeffs))
    reduction *= 100
    return max(-100.0, min(0.0, reduction))


def apply_control_scenario(
    baseline_exposure: pd.Series,
    saf_percentage: float,
    polynomial_coeffs: List[float],
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Compute reduced and delta concentrations for a SAF blend percentage.

    Returns:
        (reduced_concentration, delta_concentration, pollutant_reduction)
        where pollutant_reduction is negative (e.g. -30 = 30% reduction).
    """
    pollutant_reduction = calculate_pollutant_reduction_from_saf(saf_percentage, polynomial_coeffs)
    reduced_concentration = baseline_exposure * (1 + pollutant_reduction / 100)
    delta_concentration = baseline_exposure - reduced_concentration
    return reduced_concentration, delta_concentration, pollutant_reduction


def run_exposure_pipeline(
    spec: ScenarioSpec,
    polynomial_coeffs: List[float],
) -> Tuple[pd.Series, pd.Series, float]:
    """
    Compute exposure changes for a scenario.

    Args:
        spec: ScenarioSpec with saf_percentage and baseline_exposure
        polynomial_coeffs: Polynomial coefficients for SAF → pollutant reduction

    Returns:
        (reduced_concentration, delta_concentration, pollutant_reduction)
    """
    logger.debug(f"Running exposure pipeline for scenario {spec.scenario_id}")

    reduced_concentration, delta_concentration, pollutant_reduction = apply_control_scenario(
        spec.baseline_exposure,
        spec.saf_percentage,
        polynomial_coeffs,
    )

    logger.debug(
        f"Exposure pipeline complete: reduction={pollutant_reduction:.2f}%, "
        f"delta range=[{delta_concentration.min():.2f}, {delta_concentration.max():.2f}]"
    )

    return reduced_concentration, delta_concentration, pollutant_reduction
