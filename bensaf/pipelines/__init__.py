"""
Pipeline functions for scenario analysis.

Each pipeline function is a pure function that takes explicit arguments
and returns typed domain objects rather than mutating a shared Scenario.
"""

from bensaf.pipelines.exposure_pipeline import run_exposure_pipeline
from bensaf.pipelines.mortality_pipeline import run_mortality_pipeline
from bensaf.pipelines.preterm_birth_pipeline import run_preterm_birth_pipeline

__all__ = [
    'run_exposure_pipeline',
    'run_mortality_pipeline',
    'run_preterm_birth_pipeline',
]
