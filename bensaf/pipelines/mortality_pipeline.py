"""
Mortality pipeline for scenario analysis.

Pure functions: receives all data and parameters as arguments,
returns typed domain objects rather than mutating a Scenario.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from bensaf.model.domain import EconomicBenefit, HealthImpact, ScenarioSpec, TractEstimate
from bensaf.model.data_model import AnalysisInputs
from bensaf.core.health_impacts import calculate_health_impacts
from bensaf.core.economic_benefits import calculate_mortality_economic_value

logger = logging.getLogger(__name__)


def run_mortality_pipeline(
    spec: ScenarioSpec,
    delta_concentration: pd.Series,
    inputs: AnalysisInputs,
    mortality_function_params: Dict[str, Any],
    econ_params: Dict[str, Any],
) -> Optional[Tuple[HealthImpact, Optional[EconomicBenefit]]]:
    """
    Compute mortality health impacts and optional economic benefit.

    Args:
        spec: ScenarioSpec (used for logging only)
        delta_concentration: Change in pollutant concentration per tract
        inputs: AnalysisInputs providing incidence and demographics
        mortality_function_params: Dict with mean_rr, lower_rr, upper_rr, unit_increase
        econ_params: Dict with per_capita_consumption, life_years_gained

    Returns:
        (HealthImpact, EconomicBenefit or None), or None if mortality_rate is unavailable.
    """
    logger.debug(f"Running mortality pipeline for scenario {spec.scenario_id}")

    if inputs.incidence is None or 'mortality_rate' not in inputs.incidence.columns:
        logger.warning("mortality_rate column not found, skipping mortality pipeline")
        return None

    mortality_rate = inputs.incidence['mortality_rate']

    if inputs.demographics_core is not None and 'population' in inputs.demographics_core.columns:
        population = inputs.demographics_core['population']
    else:
        logger.warning("No population data found, using 1.0 for all tracts")
        population = pd.Series(1.0, index=delta_concentration.index)

    common_index = delta_concentration.index.intersection(mortality_rate.index)
    if len(common_index) == 0:
        logger.error("No common index between delta_concentration and mortality_rate")
        return None

    delta_conc = delta_concentration.reindex(common_index, fill_value=0.0)
    mort_rate = mortality_rate.reindex(common_index)
    pop = population.reindex(common_index, fill_value=0.0)

    z = stats.norm.ppf(0.975)
    mean_log = np.log(mortality_function_params['mean_rr'])
    lower_log = np.log(mortality_function_params['lower_rr'])
    upper_log = np.log(mortality_function_params['upper_rr'])
    se_log = ((upper_log - mean_log) + (mean_log - lower_log)) / (2 * z)
    unit = mortality_function_params['unit_increase']
    mean_log_one_unit = mean_log / unit
    se_log_one_unit = se_log / unit

    impact = calculate_health_impacts(
        delta_concentration=delta_conc,
        mortality_rate=mort_rate,
        population=pop,
        mean_log_one_unit=mean_log_one_unit,
        se_log_one_unit=se_log_one_unit,
        endpoint='mortality',
    )
    logger.debug("Computed mortality impacts")

    econ_benefit: Optional[EconomicBenefit] = None
    per_capita = econ_params.get('per_capita_consumption')
    life_years = econ_params.get('life_years_gained', 10.0)

    if per_capita is not None:
        mean_val = calculate_mortality_economic_value(
            impact.attributable_cases.mean, per_capita, life_years
        )
        lower_val = calculate_mortality_economic_value(
            impact.attributable_cases.lower, per_capita, life_years
        )
        upper_val = calculate_mortality_economic_value(
            impact.attributable_cases.upper, per_capita, life_years
        )
        econ_benefit = EconomicBenefit(
            name='mortality_economic_value',
            value=TractEstimate(mean=mean_val, lower=lower_val, upper=upper_val),
        )
        logger.debug("Computed mortality economic benefit")
    else:
        logger.debug("per_capita_consumption not configured, skipping mortality economic benefit")

    return impact, econ_benefit
