"""
Domain value objects and result types for SAF health impact assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """Scalar confidence interval (mean + 95% CI bounds)."""
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class TractEstimate:
    """Tract-level confidence interval — each bound is a Series indexed by GEOID."""
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series

    def sum(self) -> Estimate:
        return Estimate(
            mean=float(self.mean.sum()),
            lower=float(self.lower.sum()),
            upper=float(self.upper.sum()),
        )

    def weighted_mean(self, weights: pd.Series) -> Estimate:
        total = weights.sum()
        return Estimate(
            mean=float((self.mean * weights).sum() / total),
            lower=float((self.lower * weights).sum() / total),
            upper=float((self.upper * weights).sum() / total),
        )


@dataclass(frozen=True)
class HealthImpact:
    """Tract-level health impacts for a single endpoint."""
    endpoint: str
    relative_risk: TractEstimate
    attributable_fraction: TractEstimate
    attributable_cases: TractEstimate
    attributable_mortality_rate: TractEstimate


@dataclass(frozen=True)
class EconomicBenefit:
    """Tract-level economic benefit for a single named benefit."""
    name: str
    value: TractEstimate


@dataclass(frozen=True)
class ScenarioSpec:
    """Pure input description for a single scenario (no outputs)."""
    scenario_id: int
    scenario_label: str
    saf_percentage: float
    pollutant_name: str
    baseline_exposure: pd.Series


@dataclass(frozen=True)
class ScenarioResult:
    """Complete, immutable outputs for a single scenario."""
    spec: ScenarioSpec
    reduced_concentration: pd.Series
    delta_concentration: pd.Series
    pollutant_reduction: float
    health_impacts: Dict[str, HealthImpact] = field(default_factory=dict)
    economic_benefits: List[EconomicBenefit] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scenario result to a single DataFrame indexed by GEOID."""
        index = self.delta_concentration.index
        data: Dict = {
            'reduced_concentration': self.reduced_concentration,
            'delta_concentration': self.delta_concentration,
            'scenario_id': pd.Series(self.spec.scenario_id, index=index),
            'scenario_label': pd.Series(self.spec.scenario_label, index=index),
            'saf_percentage': pd.Series(self.spec.saf_percentage, index=index),
            'pollutant_reduction': pd.Series(self.pollutant_reduction, index=index),
            'pollutant_name': pd.Series(self.spec.pollutant_name, index=index),
        }

        for impact in self.health_impacts.values():
            ep = impact.endpoint
            data[f'{ep}_relative_risk_mean'] = impact.relative_risk.mean
            data[f'{ep}_relative_risk_lower'] = impact.relative_risk.lower
            data[f'{ep}_relative_risk_upper'] = impact.relative_risk.upper
            data[f'{ep}_attributable_fraction_mean'] = impact.attributable_fraction.mean
            data[f'{ep}_attributable_fraction_lower'] = impact.attributable_fraction.lower
            data[f'{ep}_attributable_fraction_upper'] = impact.attributable_fraction.upper
            data[f'{ep}_attributable_cases_mean'] = impact.attributable_cases.mean
            data[f'{ep}_attributable_cases_lower'] = impact.attributable_cases.lower
            data[f'{ep}_attributable_cases_upper'] = impact.attributable_cases.upper
            data[f'{ep}_attributable_mortality_rate_mean'] = impact.attributable_mortality_rate.mean
            data[f'{ep}_attributable_mortality_rate_lower'] = impact.attributable_mortality_rate.lower
            data[f'{ep}_attributable_mortality_rate_upper'] = impact.attributable_mortality_rate.upper

        for benefit in self.economic_benefits:
            data[f'{benefit.name}_mean'] = benefit.value.mean
            data[f'{benefit.name}_lower'] = benefit.value.lower
            data[f'{benefit.name}_upper'] = benefit.value.upper

        return pd.DataFrame(data)

    def get_aggregated_results(self, population: Optional[pd.Series] = None) -> Dict:
        """
        Aggregate tract-level results to scalar summaries.

        Returns a dict keyed by endpoint name (and 'economic_benefits'),
        each containing Estimate values for key metrics.
        """
        results: Dict = {}

        for endpoint_name, impact in self.health_impacts.items():
            results[endpoint_name] = {
                'total_attributable_cases': impact.attributable_cases.sum(),
            }
            if population is not None:
                results[endpoint_name]['overall_attributable_mortality_rate'] = (
                    impact.attributable_mortality_rate.weighted_mean(population)
                )
            else:
                total = len(impact.attributable_mortality_rate.mean)
                results[endpoint_name]['overall_attributable_mortality_rate'] = Estimate(
                    mean=float(impact.attributable_mortality_rate.mean.mean()),
                    lower=float(impact.attributable_mortality_rate.lower.mean()),
                    upper=float(impact.attributable_mortality_rate.upper.mean()),
                )

        if self.economic_benefits:
            results['economic_benefits'] = {
                b.name: b.value.sum() for b in self.economic_benefits
            }

        return results
