"""
Health impacts calculation module.

Provides calculate_health_impacts which takes tract-level Series inputs
and returns a typed HealthImpact value object.
"""

import numpy as np
import pandas as pd
from scipy import stats

from bensaf.model.domain import HealthImpact, TractEstimate


def calculate_health_impacts(
    delta_concentration: pd.Series,
    mortality_rate: pd.Series,
    population: pd.Series,
    mean_log_one_unit: float,
    se_log_one_unit: float,
    endpoint: str = 'mortality',
) -> HealthImpact:
    """
    Calculate health impacts from pollutant concentration changes.

    All Series must share the same index (GEOID). Returns a HealthImpact
    with tract-level TractEstimate fields for each metric.

    Args:
        delta_concentration: Change in pollutant concentration per tract
        mortality_rate: Baseline mortality rate per tract
        population: Population per tract
        mean_log_one_unit: Mean log-transformed relative risk per unit change
        se_log_one_unit: Standard error of log-transformed relative risk per unit change
        endpoint: Name for this health endpoint (e.g. 'mortality')
    """
    if not (
        delta_concentration.index.equals(mortality_rate.index)
        and mortality_rate.index.equals(population.index)
    ):
        raise ValueError("All input Series must have the same index")

    z = stats.norm.ppf(0.975)

    mean_log_trans = mean_log_one_unit * delta_concentration.values
    se_log_trans = se_log_one_unit * delta_concentration.values

    idx = delta_concentration.index
    mean_rr = pd.Series(np.exp(mean_log_trans), index=idx)
    lower_rr = pd.Series(np.exp(mean_log_trans - z * se_log_trans), index=idx)
    upper_rr = pd.Series(np.exp(mean_log_trans + z * se_log_trans), index=idx)

    mean_af = (mean_rr - 1) / mean_rr
    lower_af = (lower_rr - 1) / lower_rr
    upper_af = (upper_rr - 1) / upper_rr

    mean_ac = mean_af * mortality_rate * population
    lower_ac = lower_af * mortality_rate * population
    upper_ac = upper_af * mortality_rate * population

    mean_amr = mean_af * mortality_rate
    lower_amr = lower_af * mortality_rate
    upper_amr = upper_af * mortality_rate

    return HealthImpact(
        endpoint=endpoint,
        relative_risk=TractEstimate(mean=mean_rr, lower=lower_rr, upper=upper_rr),
        attributable_fraction=TractEstimate(mean=mean_af, lower=lower_af, upper=upper_af),
        attributable_cases=TractEstimate(mean=mean_ac, lower=lower_ac, upper=upper_ac),
        attributable_mortality_rate=TractEstimate(mean=mean_amr, lower=lower_amr, upper=upper_amr),
    )
