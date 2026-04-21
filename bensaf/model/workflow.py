"""
SAF Toolkit Workflow

Simplified workflow interface for SAF health impact assessment.
Workflow orchestrates data loading and scenario execution; all computation
is delegated to workflow_compute.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import geopandas as gpd

from bensaf.model.data_model import AnalysisConfig, AnalysisInputs, AnalysisResults
from bensaf.model.domain import ScenarioSpec, ScenarioResult
from bensaf.model.workflow_compute import run_scenario
from bensaf.core.exposure_generation import (
    generate_exposure_from_aermod,
    load_exposure_from_aermod,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exposure loading strategies
# ---------------------------------------------------------------------------

def _load_exposure_csv(
    exposure_data: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(exposure_data, pd.DataFrame):
        raise ValueError("exposure_data must be a DataFrame for 'csv' source")
    return exposure_data


def _load_exposure_aermod(
    exposure_data: Dict[str, Any],
    tracts_gdf: gpd.GeoDataFrame,
    pollutant_name: str,
) -> pd.DataFrame:
    if not isinstance(exposure_data, dict):
        raise ValueError("exposure_data must be a dict for 'aermod' source")
    file_paths = exposure_data.get('file_paths')
    if file_paths is None:
        raise ValueError("exposure_data must contain 'file_paths' for 'aermod' source")
    return load_exposure_from_aermod(
        aermod_file_path=file_paths,
        tracts_gdf=tracts_gdf,
        section_types=exposure_data.get('section_types'),
        center_location=exposure_data.get('center_location'),
        center_crs=exposure_data.get('center_crs'),
        aermod_crs=exposure_data.get('aermod_crs', 'EPSG:32616'),
        pollutant_name=pollutant_name,
    )


def _load_exposure_aermod_workflow(
    exposure_data: Dict[str, Any],
    tracts_gdf: gpd.GeoDataFrame,
    pollutant_name: str,
) -> pd.DataFrame:
    if not isinstance(exposure_data, dict):
        raise ValueError("exposure_data must be a dict for 'aermod_workflow' source")
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
        tracts_gdf=tracts_gdf,
        calibration_file=calibration_file,
        aermod_crs=exposure_data.get('aermod_crs', 'EPSG:32616'),
        aggregation_method=exposure_data.get('aggregation_method', 'spatial_join'),
        **exposure_data.get('aggregation_kwargs', {}),
    )
    if pollutant_name != 'ufp' and 'ufp' in exposure_df.columns:
        exposure_df = exposure_df.rename(columns={'ufp': pollutant_name})
    return exposure_df


_EXPOSURE_LOADERS = {
    'csv': _load_exposure_csv,
    'aermod': _load_exposure_aermod,
    'aermod_workflow': _load_exposure_aermod_workflow,
}


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class Workflow:
    """
    Simplified workflow for SAF health impact assessment.

    Usage:
        workflow = Workflow(config)
        workflow.load_inputs(tracts_gdf, demographics_df, ...)
        results = workflow.run_scenarios()
    """

    def __init__(self, config: Optional[Union[AnalysisConfig, Dict[str, Any]]] = None):
        if isinstance(config, dict):
            self.config = AnalysisConfig(**config)
        elif config is None:
            self.config = AnalysisConfig()
        else:
            self.config = config

        self.inputs = AnalysisInputs(crs=self.config.crs)
        self.results: Optional[AnalysisResults] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_inputs(
        self,
        tracts_gdf: gpd.GeoDataFrame,
        demographics_df: pd.DataFrame,
        exposure_source: str,
        exposure_data: Union[pd.DataFrame, Dict[str, Any]],
        incidence_df: pd.DataFrame,
        preterm_birth_df: Optional[pd.DataFrame] = None,
        mortality_economic_df: Optional[pd.DataFrame] = None,
        pollutant_name: str = 'ufp',
    ) -> None:
        """
        Load all input data for the analysis.

        Args:
            tracts_gdf: GeoDataFrame with census tract geometries
            demographics_df: DataFrame with demographic data
            exposure_source: One of 'csv', 'aermod', 'aermod_workflow'
            exposure_data: Exposure data; format depends on exposure_source
            incidence_df: DataFrame with incidence rates by endpoint
            preterm_birth_df: Optional DataFrame with preterm birth data
            mortality_economic_df: Optional tract table with at least GEOID and
                per_capita_consumption (and optionally life_years_gained)
            pollutant_name: Name of the pollutant column (default: 'ufp')
        """
        self.logger.info("Loading input data")

        self.inputs.load_tract_geometries(tracts_gdf)
        self.inputs.load_demographics(demographics_df)

        if exposure_source not in _EXPOSURE_LOADERS:
            raise ValueError(
                f"Unknown exposure_source: {exposure_source!r}. "
                f"Must be one of: {list(_EXPOSURE_LOADERS)}"
            )

        loader = _EXPOSURE_LOADERS[exposure_source]
        if exposure_source == 'csv':
            exposure_df = loader(exposure_data)
        else:
            tracts_reset = self.inputs.tract_geometries.reset_index()
            exposure_df = loader(exposure_data, tracts_reset, pollutant_name)

        self.inputs.load_baseline_exposure(exposure_df, pollutant_columns=[pollutant_name])
        self.inputs.load_incidence_data(incidence_df)

        if preterm_birth_df is not None:
            self.inputs.load_preterm_birth_data(preterm_birth_df)

        if mortality_economic_df is not None:
            self.inputs.load_mortality_economic_tract_data(mortality_economic_df)

        self.logger.info("Input data loaded successfully")

    def run_scenarios(
        self,
        scenarios: Optional[List[float]] = None,
        pollutant_name: Optional[str] = None,
        mortality_function_id: Optional[int] = None,
    ) -> AnalysisResults:
        """
        Run all scenarios and return an AnalysisResults container.

        Args:
            scenarios: SAF blend percentages to evaluate. Defaults to config.saf_scenarios.
            pollutant_name: Pollutant to use. Defaults to first column in baseline_exposure.
            mortality_function_id: Optional mortality function ID.

        Returns:
            AnalysisResults with all computed ScenarioResult objects.
        """
        self.logger.info("Running scenarios")
        self.inputs.validate()

        if scenarios is None:
            scenarios = self.config.saf_scenarios

        if pollutant_name is None:
            if self.inputs.baseline_exposure is None or self.inputs.baseline_exposure.empty:
                raise ValueError("No baseline exposure data available")
            pollutant_name = self.inputs.baseline_exposure.columns[0]

        if pollutant_name not in self.inputs.baseline_exposure.columns:
            raise ValueError(f"Pollutant '{pollutant_name}' not found in baseline exposure")

        baseline_series = self.inputs.baseline_exposure[pollutant_name]

        if (
            self.inputs.tract_geometries is not None
            and not baseline_series.index.equals(self.inputs.tract_geometries.index)
        ):
            self.logger.warning("Aligning baseline_exposure index to tract_geometries index")
            baseline_series = baseline_series.reindex(self.inputs.tract_geometries.index)

        analysis_results = AnalysisResults(self.inputs)

        for saf_percentage in scenarios:
            scenario_id = int(saf_percentage)
            spec = ScenarioSpec(
                scenario_id=scenario_id,
                scenario_label=f"{saf_percentage}% SAF Usage",
                saf_percentage=saf_percentage,
                pollutant_name=pollutant_name,
                baseline_exposure=baseline_series,
            )
            result = run_scenario(spec, self.inputs, mortality_function_id)
            analysis_results.add_scenario(result)

        self.results = analysis_results
        self.logger.info(f"Completed {len(scenarios)} scenarios")
        return analysis_results

    def get_results(self) -> Optional[AnalysisResults]:
        return self.results


def run_analysis(
    tracts_gdf: gpd.GeoDataFrame,
    demographics_df: pd.DataFrame,
    exposure_source: str,
    exposure_data: Union[pd.DataFrame, Dict[str, Any]],
    incidence_df: pd.DataFrame,
    config: Optional[Union[AnalysisConfig, Dict[str, Any]]] = None,
    scenarios: Optional[List[float]] = None,
    pollutant_name: str = 'ufp',
    preterm_birth_df: Optional[pd.DataFrame] = None,
    mortality_economic_df: Optional[pd.DataFrame] = None,
) -> AnalysisResults:
    """
    Convenience function: create a Workflow, load data, run scenarios, return results.
    """
    workflow = Workflow(config)
    workflow.load_inputs(
        tracts_gdf=tracts_gdf,
        demographics_df=demographics_df,
        exposure_source=exposure_source,
        exposure_data=exposure_data,
        incidence_df=incidence_df,
        preterm_birth_df=preterm_birth_df,
        mortality_economic_df=mortality_economic_df,
        pollutant_name=pollutant_name,
    )
    return workflow.run_scenarios(scenarios=scenarios, pollutant_name=pollutant_name)
