"""
Preterm birth pipeline for scenario analysis.

Pure functions: receives all data and parameters as arguments,
returns typed domain objects rather than mutating a Scenario.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from bensaf.domain import EconomicBenefit, ScenarioSpec, TractEstimate
from bensaf.data_model import AnalysisInputs
from bensaf.economic_benefits import (
    calculate_preterm_birth_reduction,
    calculate_preterm_birth_economic_value,
)

logger = logging.getLogger(__name__)


def run_preterm_birth_pipeline(
    spec: ScenarioSpec,
    delta_concentration: pd.Series,
    inputs: AnalysisInputs,
    params: Dict[str, Any],
) -> Optional[List[EconomicBenefit]]:
    """
    Compute preterm birth reduction and economic benefit.

    Args:
        spec: ScenarioSpec (used for logging only)
        delta_concentration: Change in pollutant concentration per tract
        inputs: AnalysisInputs providing preterm birth baseline data
        params: Dict with preterm_birth_odds_ratio and monetary_value_per_ptb

    Returns:
        List of EconomicBenefit objects, or None if data/params are unavailable.
    """
    logger.debug(f"Running preterm birth pipeline for scenario {spec.scenario_id}")

    odds_ratio = params.get('preterm_birth_odds_ratio')
    monetary_value = params.get('monetary_value_per_ptb')

    if odds_ratio is None or monetary_value is None or inputs.preterm_birth_core is None:
        logger.debug(
            "Preterm birth data or parameters not configured, skipping preterm birth pipeline"
        )
        return None

    baseline_ptb = inputs.preterm_birth_core['baseline_preterm_births']
    common_index = delta_concentration.index.intersection(baseline_ptb.index)
    if len(common_index) == 0:
        logger.error("No common index between delta_concentration and baseline_preterm_births")
        return None

    delta_conc = delta_concentration.reindex(common_index, fill_value=0.0)
    baseline = baseline_ptb.reindex(common_index)

    mean_reduction = calculate_preterm_birth_reduction(baseline, delta_conc, odds_ratio)
    lower_reduction = mean_reduction * 0.9
    upper_reduction = mean_reduction * 1.1

    mean_value = calculate_preterm_birth_economic_value(mean_reduction, monetary_value)
    lower_value = calculate_preterm_birth_economic_value(lower_reduction, monetary_value)
    upper_value = calculate_preterm_birth_economic_value(upper_reduction, monetary_value)

    logger.debug(
        f"Preterm birth pipeline complete: mean reduction={mean_reduction.sum():.2f}, "
        f"mean value=${mean_value.sum():,.0f}"
    )

    return [
        EconomicBenefit(
            name='preterm_birth_reduction',
            value=TractEstimate(mean=mean_reduction, lower=lower_reduction, upper=upper_reduction),
        ),
        EconomicBenefit(
            name='preterm_birth_economic_value',
            value=TractEstimate(mean=mean_value, lower=lower_value, upper=upper_value),
        ),
    ]
