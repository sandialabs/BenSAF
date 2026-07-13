"""
BenSAF - SAF health impact assessment toolkit.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = _PROJECT_ROOT / "data"
CALIBRATION_FILE = DATA_DIR / "aermod_calibration_coefficients.json"
CASE_STUDIES_FILE = DATA_DIR / "case_studies.json"
SAF_BLEND_PARAMETERS_FILE = DATA_DIR / "saf_blend_parameters.json"
MORTALITY_FUNCTIONS_FILE = DATA_DIR / "mortality_functions.json"

from bensaf.model.workflow import Workflow, run_analysis
from bensaf.model.data_model import AnalysisConfig, AnalysisInputs, AnalysisResults
from bensaf.model.domain import (
    Estimate,
    TractEstimate,
    HealthImpact,
    EconomicBenefit,
    ScenarioSpec,
    ScenarioResult,
)

__version__ = "0.1.0"

__all__ = [
    'DATA_DIR',
    'CALIBRATION_FILE',
    'CASE_STUDIES_FILE',
    'SAF_BLEND_PARAMETERS_FILE',
    'MORTALITY_FUNCTIONS_FILE',
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
