"""
SAF Toolkit Workflow

This module provides a simplified workflow interface for Sustainable Aviation Fuel (SAF) 
health impact assessment. The Workflow class orchestrates data loading and scenario execution,
while computation logic is handled by functions in workflow_compute module.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import pandas as pd
import geopandas as gpd

from bensaf.data_model import AnalysisData, AnalysisConfig
from bensaf.workflow_compute import run_scenario
from bensaf.exposure_generation import (
    generate_exposure_from_aermod,
    load_exposure_from_aermod
)

logger = logging.getLogger(__name__)


class Workflow:
    """
    Simplified workflow for SAF health impact assessment.
    
    This class provides a data-in, data-out interface:
    - Load data (tracts, demographics, exposure, incidence)
    - Run scenarios (health endpoints and economic parameters loaded from JSON)
    - Get results
    
    All computation logic is in workflow_compute module.
    Visualization and export should be done separately on the results.
    """
    
    def __init__(self, config: Optional[Union[AnalysisConfig, Dict[str, Any]]] = None):
        """
        Initialize the SAF workflow.
        
        Args:
            config: Optional AnalysisConfig instance or dictionary with parameters.
                If dict, will be converted to AnalysisConfig.
        """
        # Convert dict to AnalysisConfig if needed
        if isinstance(config, dict):
            self.config = AnalysisConfig(**config)
        elif config is None:
            self.config = AnalysisConfig()
        else:
            self.config = config
        
        # Initialize data model
        self.data = AnalysisData(crs=self.config.crs)
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_inputs(
        self,
        tracts_gdf: gpd.GeoDataFrame,
        demographics_df: pd.DataFrame,
        exposure_source: str,
        exposure_data: Union[pd.DataFrame, Dict[str, Any]],
        incidence_df: pd.DataFrame,
        preterm_birth_df: Optional[pd.DataFrame] = None,
        pollutant_name: str = 'ufp'
    ) -> None:
        """
        Load all input data for the analysis.
        
        This is a high-level convenience method that handles different exposure sources.
        
        Args:
            tracts_gdf: GeoDataFrame with census tract geometries
            demographics_df: DataFrame with demographic data
            exposure_source: Source of exposure data ('csv', 'aermod', 'aermod_workflow')
            exposure_data: Exposure data, format depends on exposure_source:
                - 'csv': DataFrame with pollutant columns
                - 'aermod': Dict with 'file_paths', optional 'section_types', 'center_location', etc.
                - 'aermod_workflow': Dict with 'landing_files', 'takeoff_files', 'calibration_file', etc.
            incidence_df: DataFrame with incidence rates by endpoint
            preterm_birth_df: Optional DataFrame with preterm birth data
            pollutant_name: Name of pollutant for single-pollutant exposure (default: 'ufp')
        """
        self.logger.info("Loading input data")
        
        # Load tract geometries
        self.data.load_tract_geometries(tracts_gdf)
        
        # Load demographics
        self.data.load_demographics(demographics_df)
        
        # Load exposure based on source
        if exposure_source == 'csv':
            if isinstance(exposure_data, pd.DataFrame):
                self.data.load_baseline_exposure(exposure_data)
            else:
                raise ValueError("exposure_data must be DataFrame for 'csv' source")
        
        elif exposure_source == 'aermod':
            if not isinstance(exposure_data, dict):
                raise ValueError("exposure_data must be dict for 'aermod' source")
            
            file_paths = exposure_data.get('file_paths')
            if file_paths is None:
                raise ValueError("exposure_data must contain 'file_paths' for 'aermod' source")
            
            exposure_df = load_exposure_from_aermod(
                aermod_file_path=file_paths,
                tracts_gdf=self.data.tract_geometries.reset_index(),
                section_types=exposure_data.get('section_types'),
                center_location=exposure_data.get('center_location'),
                center_crs=exposure_data.get('center_crs'),
                aermod_crs=exposure_data.get('aermod_crs', 'EPSG:32616'),
                pollutant_name=pollutant_name
            )
            self.data.load_baseline_exposure(exposure_df, pollutant_columns=[pollutant_name])
        
        elif exposure_source == 'aermod_workflow':
            if not isinstance(exposure_data, dict):
                raise ValueError("exposure_data must be dict for 'aermod_workflow' source")
            
            landing_files = exposure_data.get('landing_files')
            takeoff_files = exposure_data.get('takeoff_files')
            calibration_file = exposure_data.get('calibration_file')
            
            if landing_files is None and takeoff_files is None:
                raise ValueError(
                    "exposure_data must contain at least one of 'landing_files' or 'takeoff_files' "
                    "for 'aermod_workflow' source"
                )
            
            if calibration_file is None:
                raise ValueError(
                    "exposure_data must contain 'calibration_file' for 'aermod_workflow' source"
                )
            
            exposure_df = generate_exposure_from_aermod(
                landing_files=landing_files,
                takeoff_files=takeoff_files,
                tracts_gdf=self.data.tract_geometries.reset_index(),
                calibration_file=calibration_file,
                aermod_crs=exposure_data.get('aermod_crs', 'EPSG:32616'),
                aggregation_method=exposure_data.get('aggregation_method', 'spatial_join'),
                **exposure_data.get('aggregation_kwargs', {})
            )
            # Rename 'ufp' to pollutant name if different
            if pollutant_name != 'ufp' and 'ufp' in exposure_df.columns:
                exposure_df = exposure_df.rename(columns={'ufp': pollutant_name})
            self.data.load_baseline_exposure(exposure_df, pollutant_columns=[pollutant_name])
        
        else:
            raise ValueError(f"Unknown exposure_source: {exposure_source}")
        
        # Load incidence data
        self.data.load_incidence_data(incidence_df)
        
        # Load preterm birth if provided
        if preterm_birth_df is not None:
            self.data.load_preterm_birth_data(preterm_birth_df)
        
        self.logger.info("Input data loaded successfully")
    
    def run_scenarios(
        self,
        scenarios: Optional[List[float]] = None,
        pollutant_name: Optional[str] = None
    ) -> Dict[int, Any]:
        """
        Run scenarios and calculate health impacts and economic benefits.
        
        Args:
            scenarios: List of SAF blend percentages (0-100). If None, uses config.saf_scenarios
            pollutant_name: Name of pollutant to use. If None, uses first pollutant in baseline_exposure
            
        Returns:
            Dictionary of scenario_id -> aggregated results
        """
        self.logger.info("Running scenarios")
        
        # Validate data is ready
        self.data.validate()
        
        # Determine scenarios
        if scenarios is None:
            scenarios = self.config.saf_scenarios
        
        # Determine pollutant
        if pollutant_name is None:
            if self.data.baseline_exposure is None or len(self.data.baseline_exposure.columns) == 0:
                raise ValueError("No baseline exposure data available")
            pollutant_name = self.data.baseline_exposure.columns[0]
        
        if pollutant_name not in self.data.baseline_exposure.columns:
            raise ValueError(f"Pollutant '{pollutant_name}' not found in baseline exposure")
        
        baseline_exposure = self.data.baseline_exposure[pollutant_name]
        
        # Log index information for debugging
        self.logger.debug(
            f"run_scenarios: baseline_exposure index type={type(baseline_exposure.index)}, "
            f"length={len(baseline_exposure.index)}, dtype={baseline_exposure.index.dtype}, "
            f"name={baseline_exposure.index.name}"
        )
        if self.data.tract_geometries is not None:
            self.logger.debug(
                f"run_scenarios: tract_geometries index type={type(self.data.tract_geometries.index)}, "
                f"length={len(self.data.tract_geometries.index)}, dtype={self.data.tract_geometries.index.dtype}, "
                f"name={self.data.tract_geometries.index.name}"
            )
            # Ensure indices match
            if not baseline_exposure.index.equals(self.data.tract_geometries.index):
                self.logger.warning(
                    f"Index mismatch detected. Aligning baseline_exposure index to tract_geometries index."
                )
                baseline_exposure = baseline_exposure.reindex(self.data.tract_geometries.index)
        
        # Run each scenario
        aggregated_results = {}
        for idx, saf_percentage in enumerate(scenarios):
            scenario_id = int(saf_percentage)  # Use SAF percentage as ID
            
            try:
                scenario_result = run_scenario(
                    scenario_id=scenario_id,
                    saf_percentage=saf_percentage,
                    baseline_exposure=baseline_exposure,
                    data=self.data,
                    pollutant_name=pollutant_name
                )
                
                # Store scenario result
                self.data.add_scenario_result(scenario_result)
            except Exception as e:
                import traceback
                self.logger.error(f"Error running scenario {scenario_id}: {str(e)}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise
            
            # Get aggregated results
            population = None
            if self.data.demographics_core is not None and 'population' in self.data.demographics_core.columns:
                population = self.data.demographics_core['population']
            
            aggregated_results[scenario_id] = scenario_result.get_aggregated_results(population)
        
        self.logger.info(f"Completed {len(scenarios)} scenarios")
        return aggregated_results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all analysis results.
        
        Returns:
            Dictionary containing scenario results and access to tract-level data
        """
        return {
            'scenario_results': self.data.scenario_results,
            'tract_level': self.data
        }


def run_analysis(
    tracts_gdf: gpd.GeoDataFrame,
    demographics_df: pd.DataFrame,
    exposure_source: str,
    exposure_data: Union[pd.DataFrame, Dict[str, Any]],
    incidence_df: pd.DataFrame,
    config: Optional[Union[AnalysisConfig, Dict[str, Any]]] = None,
    scenarios: Optional[List[float]] = None,
    pollutant_name: str = 'ufp',
    preterm_birth_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Top-level function to run complete analysis.
    
    This is a convenience function that creates a Workflow, loads data, and runs scenarios.
    
    Args:
        tracts_gdf: GeoDataFrame with census tract geometries
        demographics_df: DataFrame with demographic data
        exposure_source: Source of exposure data ('csv', 'aermod', 'aermod_workflow')
        exposure_data: Exposure data (format depends on exposure_source)
        incidence_df: DataFrame with incidence rates by endpoint
        config: Optional AnalysisConfig or dict with configuration
        scenarios: Optional list of SAF percentages. If None, uses default from config
        pollutant_name: Name of pollutant (default: 'ufp')
        preterm_birth_df: Optional DataFrame with preterm birth data
        
    Returns:
        Dictionary with 'scenario_results' and 'tract_level' data
    """
    workflow = Workflow(config)
    
    workflow.load_inputs(
        tracts_gdf=tracts_gdf,
        demographics_df=demographics_df,
        exposure_source=exposure_source,
        exposure_data=exposure_data,
        incidence_df=incidence_df,
        preterm_birth_df=preterm_birth_df,
        pollutant_name=pollutant_name
    )
    
    workflow.run_scenarios(scenarios=scenarios, pollutant_name=pollutant_name)
    
    return workflow.get_results()
