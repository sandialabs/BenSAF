"""
Workflow computation functions.

This module contains pure computation functions for the workflow.
All logic is separated from the Workflow class for better modularity.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from bensaf.data_model import AnalysisData
from bensaf.scenario import Scenario
from bensaf.utils import load_saf_blend_parameters

logger = logging.getLogger(__name__)


def run_scenario(
    scenario_id: int,
    saf_percentage: float,
    baseline_exposure: pd.Series,
    data: AnalysisData,
    pollutant_name: str,
    mortality_function_id: Optional[int] = None
) -> Scenario:
    """
    Run a single scenario and return Scenario.
    
    This function creates a Scenario object, runs the necessary pipelines,
    and returns the populated scenario.
    
    Args:
        scenario_id: Numeric identifier for the scenario
        saf_percentage: SAF blend percentage (0-100)
        baseline_exposure: Baseline pollutant concentration per tract (Series)
        data: AnalysisData with required data
        pollutant_name: Name of the pollutant
        mortality_function_id: Optional mortality function ID. If None, uses first available.
        
    Returns:
        Scenario object with outputs populated
    """
    import traceback
    
    try:
        # Validate indices match before proceeding
        if data.tract_geometries is not None:
            tract_index = data.tract_geometries.index
            exposure_index = baseline_exposure.index
            
            logger.debug(
                f"run_scenario: tract_index type={type(tract_index)}, length={len(tract_index)}, "
                f"dtype={tract_index.dtype}, name={tract_index.name}"
            )
            logger.debug(
                f"run_scenario: exposure_index type={type(exposure_index)}, length={len(exposure_index)}, "
                f"dtype={exposure_index.dtype}, name={exposure_index.name}"
            )
            
            if not exposure_index.equals(tract_index):
                error_msg = (
                    f"Index mismatch in run_scenario:\n"
                    f"  baseline_exposure index: type={type(exposure_index)}, length={len(exposure_index)}, "
                    f"dtype={exposure_index.dtype}, name={exposure_index.name}, sample={list(exposure_index[:5])}\n"
                    f"  tract_geometries index: type={type(tract_index)}, length={len(tract_index)}, "
                    f"dtype={tract_index.dtype}, name={tract_index.name}, sample={list(tract_index[:5])}"
                )
                logger.error(error_msg)
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise ValueError(error_msg)
        
        # Create scenario with inputs
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_label=f"{saf_percentage}% SAF Usage",
            saf_percentage=saf_percentage,
            pollutant_name=pollutant_name,
            baseline_exposure=baseline_exposure,
            data=data
        )
        
        # Load SAF polynomial coefficients
        polynomial_coeffs = load_saf_blend_parameters()
        
        # Import pipelines locally to avoid circular import
        from bensaf.pipelines import (
            run_exposure_pipeline,
            run_mortality_pipeline,
            run_preterm_birth_pipeline
        )
        
        # Run exposure pipeline (required for all scenarios)
        run_exposure_pipeline(scenario, polynomial_coeffs)
        
        # Run mortality pipeline
        run_mortality_pipeline(scenario, mortality_function_id)
        
        # Run preterm birth pipeline
        run_preterm_birth_pipeline(scenario)
        
        return scenario
        
    except Exception as e:
        logger.error(f"Error in run_scenario: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
