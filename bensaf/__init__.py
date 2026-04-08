"""
BenSAF - SAF health impact assessment toolkit.
"""

from bensaf.workflow import Workflow, run_analysis
from bensaf.data_model import AnalysisConfig, AnalysisInputs, AnalysisResults
from bensaf.domain import (
    Estimate,
    TractEstimate,
    HealthImpact,
    EconomicBenefit,
    ScenarioSpec,
    ScenarioResult,
)

__version__ = "0.1.0"

__all__ = [
    'Workflow',
    'run_analysis',
    'AnalysisConfig',
    'AnalysisInputs',
    'AnalysisResults',
    'Estimate',
    'TractEstimate',
    'HealthImpact',
    'EconomicBenefit',
    'ScenarioSpec',
    'ScenarioResult',
]
