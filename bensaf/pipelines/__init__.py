"""
Pipeline functions for scenario analysis.

Each pipeline function takes a Scenario object, performs computations,
and populates the scenario's output attributes.
"""

from bensaf.pipelines.exposure_pipeline import run_exposure_pipeline
from bensaf.pipelines.mortality_pipeline import run_mortality_pipeline
from bensaf.pipelines.preterm_birth_pipeline import run_preterm_birth_pipeline

__all__ = [
    'run_exposure_pipeline',
    'run_mortality_pipeline',
    'run_preterm_birth_pipeline',
]
