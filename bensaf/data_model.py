"""
Data model for SAF health impact assessment.

This module defines the core data structures used in the analysis:
- AnalysisData: Holds all geospatial data (inputs, derived inputs, outputs)
- AnalysisConfig: Holds analysis configuration parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
import geopandas as gpd
import pandas as pd
import logging

from bensaf.scenario_results import ScenarioResults

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bensaf.scenario import Scenario


def validate_geoid_alignment(df: pd.DataFrame, tracts_gdf: gpd.GeoDataFrame, 
                             data_name: str = "data") -> pd.DataFrame:
    """
    Validate and align DataFrame GEOIDs with tract GeoDataFrame.
    
    Args:
        df: DataFrame with GEOID column or index
        tracts_gdf: GeoDataFrame with GEOID as index (int)
        data_name: Name of the data for error messages
        
    Returns:
        DataFrame aligned with tracts_gdf (extra GEOIDs dropped, missing raise exception)
        Index will be int (GEOID)
    """
    # Get GEOIDs from tracts (should be int)
    tract_geoids = set(tracts_gdf.index)
    
    # Get GEOIDs from df and convert to int
    if 'GEOID' in df.columns:
        df_geoids = set(df['GEOID'].astype(int))
    elif df.index.name == 'GEOID' or isinstance(df.index, pd.Index):
        df_geoids = set(df.index.astype(int))
    else:
        raise ValueError(f"{data_name} must have GEOID as column or index")
    
    # Check for missing GEOIDs
    missing = tract_geoids - df_geoids
    if missing:
        raise ValueError(f"{data_name} is missing GEOIDs: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
    
    # Drop extra GEOIDs
    extra = df_geoids - tract_geoids
    if extra:
        logger.warning(f"{data_name} has {len(extra)} extra GEOIDs, dropping them")
        if 'GEOID' in df.columns:
            df = df[df['GEOID'].isin(tract_geoids)].copy()
        else:
            df = df[df.index.isin(tract_geoids)].copy()
    
    # Set GEOID as index if not already
    if 'GEOID' in df.columns:
        df = df.set_index('GEOID')
    
    # Ensure index is int and matches tracts
    df.index = df.index.astype(int)
    df = df.reindex(tract_geoids)
    
    return df


@dataclass
class AnalysisConfig:
    """
    Configuration parameters for the analysis.
    
    This class holds minimal configuration that controls how the analysis is run.
    Health endpoints and economic parameters are loaded from JSON by pipelines.
    """
    # SAF scenarios
    saf_scenarios: List[float] = field(default_factory=lambda: [5, 25, 50])
    
    # Study area
    airport_coordinates: Optional[tuple] = None
    
    # Other configuration
    crs: str = 'EPSG:4326'


class AnalysisData:
    """
    Core data model for the analysis.
    
    This class holds all geospatial data organized by category, with a distinction
    between core columns (required for analysis) and covariate columns (for future
    benefit distribution analysis).
    
    Core data (required for analysis):
    - Tract geometries: GeoDataFrame with GEOID (int) as index, geometry as only column
    - Demographics core: DataFrame with GEOID (int) as index, only 'population' column
    - Baseline exposure: DataFrame with GEOID (int) as index, multiple pollutant columns
    - Mortality/Incidence: DataFrame with GEOID (int) as index, incidence rates by endpoint
    - Preterm birth core: DataFrame with GEOID (int) as index, only 'baseline_preterm_births' column
    - Derived inputs: Values derived from inputs (e.g., % low income)
    - Scenario results: Dict[int, Scenario] keyed by scenario_id
    
    Covariate data (for future analysis):
    - Demographics covariates: DataFrame with GEOID (int) as index, all columns except 'population'
    - Preterm birth covariates: DataFrame with GEOID (int) as index, all columns except 'baseline_preterm_births'
    
    Study area: Optional GeoDataFrame with study area boundary
    
    Properties:
    - demographics: Returns merged demographics (core + covariates) for backward compatibility
    - preterm_birth: Returns merged preterm birth (core + covariates) for backward compatibility
    """
    
    def __init__(self, crs: str = 'EPSG:4326'):
        """
        Initialize AnalysisData.
        
        Args:
            crs: Coordinate reference system for geospatial data
        """
        self.crs = crs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tract geometries: GEOID (int) as index, geometry as only column
        self.tract_geometries: Optional[gpd.GeoDataFrame] = None
        
        # Core data (required for analysis)
        # Demographics core: GEOID (int) as index, only 'population' column
        self.demographics_core: Optional[pd.DataFrame] = None
        
        # Baseline exposure: GEOID (int) as index, columns are pollutant names
        # e.g., columns: ['ufp', 'pm25', 'nox']
        self.baseline_exposure: Optional[pd.DataFrame] = None
        
        # Mortality/Incidence: GEOID (int) as index
        # Columns: incidence rates by endpoint (e.g., 'mortality_rate', 'asthma_rate')
        self.incidence: Optional[pd.DataFrame] = None
        
        # Preterm birth core: GEOID (int) as index, only 'baseline_preterm_births' column
        self.preterm_birth_core: Optional[pd.DataFrame] = None
        
        # Derived inputs: GEOID (int) as index
        self.derived_inputs: Optional[pd.DataFrame] = None
        
        # Covariate data (for future analysis of benefit distribution)
        # Demographics covariates: GEOID (int) as index, all columns except 'population'
        self.demographics_covariates: Optional[pd.DataFrame] = None
        
        # Preterm birth covariates: GEOID (int) as index, all columns except 'baseline_preterm_births'
        self.preterm_birth_covariates: Optional[pd.DataFrame] = None
        
        # Backward compatibility: demographics property returns core + covariates merged
        self._demographics: Optional[pd.DataFrame] = None
        self._preterm_birth: Optional[pd.DataFrame] = None
        
        # Scenario results: Dict[int, Scenario] keyed by scenario_id
        self.scenario_results: Dict[int, 'Scenario'] = {}
        
        # Study area boundary (optional)
        self.study_area: Optional[gpd.GeoDataFrame] = None
    
    @property
    def demographics(self) -> Optional[pd.DataFrame]:
        """
        Get merged demographics (core + covariates) for backward compatibility.
        
        Returns:
            DataFrame with all demographic columns merged
        """
        if self._demographics is not None:
            return self._demographics
        
        if self.demographics_core is None:
            return None
        
        if self.demographics_covariates is not None:
            return pd.concat([self.demographics_core, self.demographics_covariates], axis=1)
        
        return self.demographics_core
    
    @property
    def preterm_birth(self) -> Optional[pd.DataFrame]:
        """
        Get merged preterm birth data (core + covariates) for backward compatibility.
        
        Returns:
            DataFrame with all preterm birth columns merged
        """
        if self._preterm_birth is not None:
            return self._preterm_birth
        
        if self.preterm_birth_core is None:
            return None
        
        if self.preterm_birth_covariates is not None:
            return pd.concat([self.preterm_birth_core, self.preterm_birth_covariates], axis=1)
        
        return self.preterm_birth_core
    
    @property
    def is_ready(self) -> bool:
        """Check if data is ready for analysis."""
        return (self.tract_geometries is not None and 
                self.baseline_exposure is not None and 
                self.incidence is not None)
    
    def load_tract_geometries(self, tracts_gdf: gpd.GeoDataFrame) -> None:
        """
        Load census tract geometries.
        
        Args:
            tracts_gdf: GeoDataFrame with census tract geometries.
                Must contain columns:
                - GEOID: Census tract identifier (will be converted to int)
                - geometry: Tract geometry
        """
        self.logger.info("Loading tract geometries")
        
        required_columns = ['GEOID', 'geometry']
        for col in required_columns:
            if col not in tracts_gdf.columns:
                raise ValueError(f"Missing required column in tract data: {col}")
        
        tracts_gdf = tracts_gdf.copy()
        tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(int)
        
        # Ensure correct CRS
        if tracts_gdf.crs is None:
            self.logger.warning("Tract data has no CRS defined, assuming EPSG:4326")
            tracts_gdf.set_crs(self.crs, inplace=True)
        elif tracts_gdf.crs != self.crs:
            self.logger.info(f"Reprojecting tract data from {tracts_gdf.crs} to {self.crs}")
            tracts_gdf = tracts_gdf.to_crs(self.crs)
        
        # Set GEOID as index and keep only geometry
        self.tract_geometries = tracts_gdf[['GEOID', 'geometry']].set_index('GEOID')
        
        self.logger.info(f"Loaded {len(self.tract_geometries)} census tract geometries")
    
    def load_demographics(self, demographics_df: pd.DataFrame) -> None:
        """
        Load demographic data and split into core and covariates.
        
        Core columns: 'population' (required for analysis)
        Covariate columns: All other columns (for future benefit distribution analysis)
        
        Args:
            demographics_df: DataFrame with demographic data.
                Must contain GEOID as column or index.
                Must contain 'population' column (or 'Population' which will be renamed).
                Can include race, ethnicity, income, etc. as covariates.
        """
        self.logger.info("Loading demographic data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        demographics_df = demographics_df.copy()
        
        # Validate and align GEOIDs
        demographics_df = validate_geoid_alignment(
            demographics_df, 
            self.tract_geometries,
            "demographic data"
        )
        
        # Standardize population column name if needed
        if 'Population' in demographics_df.columns and 'population' not in demographics_df.columns:
            demographics_df['population'] = demographics_df['Population']
        
        # Check that population column exists
        if 'population' not in demographics_df.columns:
            raise ValueError("Demographics data must contain 'population' column")
        
        # Split into core and covariates
        core_columns = ['population']
        covariate_columns = [col for col in demographics_df.columns if col not in core_columns]
        
        self.demographics_core = demographics_df[core_columns].copy()
        
        if covariate_columns:
            self.demographics_covariates = demographics_df[covariate_columns].copy()
            self.logger.info(f"Loaded {len(covariate_columns)} demographic covariate columns: {covariate_columns}")
        else:
            self.demographics_covariates = None
        
        # Store merged version for backward compatibility
        self._demographics = demographics_df
        
        # Compute derived inputs automatically
        self.compute_derived_inputs()
        
        self.logger.info(f"Loaded demographic data: {len(core_columns)} core column(s), {len(covariate_columns)} covariate column(s)")
    
    def load_baseline_exposure(self, exposure_df: pd.DataFrame, 
                              pollutant_columns: Optional[List[str]] = None) -> None:
        """
        Load baseline exposure data with pollutant concentrations.
        
        Args:
            exposure_df: DataFrame with pollutant concentrations.
                Must contain GEOID as column or index.
                Columns should be pollutant names (e.g., 'ufp', 'pm25', 'nox').
            pollutant_columns: Optional list of column names to use as pollutants.
                If None, uses all numeric columns (excluding GEOID if present).
        """
        self.logger.info("Loading baseline exposure data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        exposure_df = exposure_df.copy()
        
        # Validate and align GEOIDs
        exposure_df = validate_geoid_alignment(
            exposure_df,
            self.tract_geometries,
            "baseline exposure data"
        )
        
        # Determine pollutant columns
        if pollutant_columns is None:
            # Use all numeric columns
            numeric_cols = exposure_df.select_dtypes(include=[float, int]).columns.tolist()
            pollutant_columns = numeric_cols
        
        # Validate pollutant columns exist
        missing_cols = [col for col in pollutant_columns if col not in exposure_df.columns]
        if missing_cols:
            raise ValueError(f"Missing pollutant columns: {missing_cols}")
        
        # Store only pollutant columns
        self.baseline_exposure = exposure_df[pollutant_columns].copy()
        
        self.logger.info(f"Loaded baseline exposure data with {len(pollutant_columns)} pollutants: {pollutant_columns}")
    
    def load_incidence_data(self, incidence_df: pd.DataFrame, 
                           endpoint_columns: Optional[List[str]] = None) -> None:
        """
        Load incidence rate data for health endpoints.
        
        Args:
            incidence_df: DataFrame with incidence rates.
                Must contain GEOID as column or index.
                Columns should be endpoint names (e.g., 'mortality_rate', 'asthma_rate').
            endpoint_columns: Optional list of column names to use as endpoints.
                If None, uses all numeric columns (excluding GEOID if present).
        """
        self.logger.info("Loading incidence data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        incidence_df = incidence_df.copy()
        
        # Validate and align GEOIDs
        incidence_df = validate_geoid_alignment(
            incidence_df,
            self.tract_geometries,
            "incidence data"
        )
        
        # Determine endpoint columns
        if endpoint_columns is None:
            # Use all numeric columns
            numeric_cols = incidence_df.select_dtypes(include=[float, int]).columns.tolist()
            endpoint_columns = numeric_cols
        
        # Validate endpoint columns exist
        missing_cols = [col for col in endpoint_columns if col not in incidence_df.columns]
        if missing_cols:
            raise ValueError(f"Missing endpoint columns: {missing_cols}")
        
        # Store only endpoint columns
        self.incidence = incidence_df[endpoint_columns].copy()
        
        self.logger.info(f"Loaded incidence data with {len(endpoint_columns)} endpoints: {endpoint_columns}")
    
    def load_preterm_birth_data(self, preterm_birth_df: pd.DataFrame) -> None:
        """
        Load baseline preterm birth data and split into core and covariates.
        
        Core columns: 'baseline_preterm_births' (required for analysis)
        Covariate columns: All other columns (for future benefit distribution analysis)
        
        Args:
            preterm_birth_df: DataFrame with baseline preterm birth counts.
                Must contain GEOID as column or index.
                Must contain 'baseline_preterm_births' or 'preterm_births' column.
                Other columns will be stored as covariates.
        """
        self.logger.info("Loading preterm birth data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        preterm_birth_df = preterm_birth_df.copy()
        
        # Check for required column
        if 'baseline_preterm_births' not in preterm_birth_df.columns:
            if 'preterm_births' in preterm_birth_df.columns:
                preterm_birth_df['baseline_preterm_births'] = preterm_birth_df['preterm_births']
            else:
                raise ValueError("Missing required column 'baseline_preterm_births' or 'preterm_births' in preterm birth data")
        
        # Validate and align GEOIDs
        preterm_birth_df = validate_geoid_alignment(
            preterm_birth_df,
            self.tract_geometries,
            "preterm birth data"
        )
        
        # Split into core and covariates
        core_columns = ['baseline_preterm_births']
        covariate_columns = [col for col in preterm_birth_df.columns if col not in core_columns]
        
        self.preterm_birth_core = preterm_birth_df[core_columns].copy()
        
        if covariate_columns:
            self.preterm_birth_covariates = preterm_birth_df[covariate_columns].copy()
            self.logger.info(f"Loaded {len(covariate_columns)} preterm birth covariate columns: {covariate_columns}")
        else:
            self.preterm_birth_covariates = None
        
        # Store merged version for backward compatibility
        self._preterm_birth = preterm_birth_df
        
        self.logger.info(f"Loaded preterm birth data: {len(core_columns)} core column(s), {len(covariate_columns)} covariate column(s)")
    
    def compute_derived_inputs(self) -> None:
        """
        Compute derived input values from raw inputs.
        
        This method automatically computes derived metrics such as:
        - Percentage of low-income population
        - Distance from airport (if airport_coordinates provided)
        - Other demographic-derived metrics
        
        This is called automatically when demographics are loaded.
        """
        # Use merged demographics for derived inputs (may include covariates)
        demographics_merged = self.demographics
        if demographics_merged is None:
            return
        
        self.logger.info("Computing derived inputs")
        
        derived = {}
        
        # Compute percentage of low-income population if income data available
        if 'households_below_poverty' in demographics_merged.columns and 'total_households' in demographics_merged.columns:
            derived['pct_low_income'] = (
                demographics_merged['households_below_poverty'] / 
                demographics_merged['total_households'] * 100
            )
        elif 'poverty_rate' in demographics_merged.columns:
            derived['pct_low_income'] = demographics_merged['poverty_rate'] * 100
        
        if derived:
            if self.derived_inputs is None:
                self.derived_inputs = pd.DataFrame(index=demographics_merged.index)
            
            for name, values in derived.items():
                self.derived_inputs[name] = values
                self.logger.info(f"Computed derived input: {name}")
    
    def add_scenario_result(self, scenario: 'Scenario') -> None:
        """
        Add a scenario result.
        
        Args:
            scenario: Scenario object
        """
        import traceback
        from bensaf.scenario import Scenario as ScenarioType
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        # Validate scenario has correct index
        if scenario.delta_concentration is None:
            raise ValueError("Scenario outputs not yet computed. Run pipeline functions first.")
        
        scenario_index = scenario.delta_concentration.index
        tract_index = self.tract_geometries.index
        
        if not scenario_index.equals(tract_index):
            # Provide detailed error information
            error_msg = (
                f"Scenario result index must match tract geometries index.\n"
                f"  Scenario index type: {type(scenario_index)}, length: {len(scenario_index)}\n"
                f"  Tract index type: {type(tract_index)}, length: {len(tract_index)}\n"
                f"  Scenario index sample (first 5): {list(scenario_index[:5])}\n"
                f"  Tract index sample (first 5): {list(tract_index[:5])}\n"
                f"  Index names match: {scenario_index.name == tract_index.name}\n"
                f"  Index dtypes match: {scenario_index.dtype == tract_index.dtype}\n"
            )
            self.logger.error(error_msg)
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise ValueError(error_msg)
        
        self.scenario_results[scenario.scenario_id] = scenario
        self.logger.info(f"Added scenario result: {scenario.scenario_label} (ID: {scenario.scenario_id})")
    
    def get_scenario_result(self, scenario_id: int) -> Optional['Scenario']:
        """Get scenario result by ID."""
        return self.scenario_results.get(scenario_id)
    
    def get_merged_data(self) -> gpd.GeoDataFrame:
        """
        Get merged GeoDataFrame with all data (including covariates).
        
        Returns:
            GeoDataFrame with geometries and all other data merged
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")
        
        merged = self.tract_geometries.copy()
        
        # Join core demographics
        if self.demographics_core is not None:
            merged = merged.join(self.demographics_core, how='left')
        
        # Join covariate demographics
        if self.demographics_covariates is not None:
            merged = merged.join(self.demographics_covariates, how='left')
        
        if self.baseline_exposure is not None:
            merged = merged.join(self.baseline_exposure, how='left')
        
        if self.incidence is not None:
            merged = merged.join(self.incidence, how='left')
        
        # Join core preterm birth
        if self.preterm_birth_core is not None:
            merged = merged.join(self.preterm_birth_core, how='left')
        
        # Join covariate preterm birth
        if self.preterm_birth_covariates is not None:
            merged = merged.join(self.preterm_birth_covariates, how='left')
        
        if self.derived_inputs is not None:
            merged = merged.join(self.derived_inputs, how='left')
        
        # Add scenario results
        for scenario_id, scenario in self.scenario_results.items():
            scenario_df = scenario.to_dataframe()
            # Prefix with scenario label
            scenario_df_prefixed = scenario_df.add_prefix(f"{scenario.scenario_label}_")
            merged = merged.join(scenario_df_prefixed, how='left')
        
        return merged
    
    def get_merged_core_data(self) -> gpd.GeoDataFrame:
        """
        Get merged GeoDataFrame with only core data (excluding covariates).
        
        Returns:
            GeoDataFrame with geometries and core data only
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")
        
        merged = self.tract_geometries.copy()
        
        # Join core demographics only
        if self.demographics_core is not None:
            merged = merged.join(self.demographics_core, how='left')
        
        if self.baseline_exposure is not None:
            merged = merged.join(self.baseline_exposure, how='left')
        
        if self.incidence is not None:
            merged = merged.join(self.incidence, how='left')
        
        # Join core preterm birth only
        if self.preterm_birth_core is not None:
            merged = merged.join(self.preterm_birth_core, how='left')
        
        if self.derived_inputs is not None:
            merged = merged.join(self.derived_inputs, how='left')
        
        # Add scenario results
        for scenario_id, scenario in self.scenario_results.items():
            scenario_df = scenario.to_dataframe()
            # Prefix with scenario label
            scenario_df_prefixed = scenario_df.add_prefix(f"{scenario.scenario_label}_")
            merged = merged.join(scenario_df_prefixed, how='left')
        
        return merged
    
    def validate(self) -> bool:
        """
        Validate that all required data is present.
        
        Returns:
            True if valid, raises ValueError if not
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")
        
        if self.baseline_exposure is None:
            raise ValueError("Baseline exposure data must be loaded")
        
        if self.incidence is None:
            raise ValueError("Incidence data must be loaded")
        
        return True
