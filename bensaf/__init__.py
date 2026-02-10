"""
SAF Toolkit - A toolkit for sustainable aviation fuel analysis.
"""

from bensaf.workflow import Workflow, run_analysis
from bensaf.data_model import AnalysisData, AnalysisConfig
from bensaf.scenario import Scenario
from bensaf.scenario_results import ScenarioResults

__version__ = "0.1.0"

__all__ = [
    'Workflow',
    'run_analysis',
    'AnalysisData',
    'AnalysisConfig',
    'Scenario',
    'ScenarioResults',
]
