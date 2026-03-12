"""
Workflow computation functions.

Pure orchestration: loads parameters once, delegates to pipeline functions,
assembles and returns a complete ScenarioResult.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from bensaf.data_model import AnalysisInputs
from bensaf.domain import EconomicBenefit, HealthImpact, ScenarioResult, ScenarioSpec
from bensaf.params import (
    load_economic_parameters,
    load_mortality_function_config,
    load_saf_blend_parameters,
)
from bensaf.pipelines.exposure_pipeline import run_exposure_pipeline
from bensaf.pipelines.mortality_pipeline import run_mortality_pipeline
from bensaf.pipelines.preterm_birth_pipeline import run_preterm_birth_pipeline

logger = logging.getLogger(__name__)


def run_scenario(
    spec: ScenarioSpec,
    inputs: AnalysisInputs,
    mortality_function_id: Optional[int] = None,
) -> ScenarioResult:
    """
    Run a single scenario and return a complete, immutable ScenarioResult.

    Loads all parameters internally, then delegates to pipeline functions.
    No lazy imports are needed because ScenarioSpec holds no reference to
    AnalysisInputs, breaking the prior circular dependency.

    Args:
        spec: ScenarioSpec describing the scenario inputs
        inputs: AnalysisInputs with shared data (demographics, incidence, etc.)
        mortality_function_id: Optional mortality function ID; uses first available if None.
    """
    logger.debug(f"Running scenario {spec.scenario_id} ({spec.scenario_label})")

    polynomial_coeffs = load_saf_blend_parameters()
    econ_params = load_economic_parameters()
    mortality_params = load_mortality_function_config(mortality_function_id)

    # Exposure pipeline
    reduced_concentration, delta_concentration, pollutant_reduction = run_exposure_pipeline(
        spec, polynomial_coeffs
    )

    # Mortality pipeline
    health_impacts: Dict[str, HealthImpact] = {}
    economic_benefits: List[EconomicBenefit] = []

    mortality_result = run_mortality_pipeline(
        spec=spec,
        delta_concentration=delta_concentration,
        inputs=inputs,
        mortality_function_params=mortality_params,
        econ_params=econ_params,
    )
    if mortality_result is not None:
        impact, econ_benefit = mortality_result
        health_impacts['mortality'] = impact
        if econ_benefit is not None:
            economic_benefits.append(econ_benefit)

    # Preterm birth pipeline
    ptb_benefits = run_preterm_birth_pipeline(
        spec=spec,
        delta_concentration=delta_concentration,
        inputs=inputs,
        params=econ_params,
    )
    if ptb_benefits is not None:
        economic_benefits.extend(ptb_benefits)

    return ScenarioResult(
        spec=spec,
        reduced_concentration=reduced_concentration,
        delta_concentration=delta_concentration,
        pollutant_reduction=pollutant_reduction,
        health_impacts=health_impacts,
        economic_benefits=economic_benefits,
    )
