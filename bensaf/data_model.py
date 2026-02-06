"""
Data model for SAF health impact assessment.

This module defines the core data structures used in the analysis:
- AnalysisData: Holds all geospatial data (inputs, derived inputs, outputs)
- AnalysisConfig: Holds analysis configuration parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import logging

from bensaf.aermod_parser import AermodParser

logger = logging.getLogger(__name__)


def validate_geoid_alignment(df: pd.DataFrame, tracts_gdf: gpd.GeoDataFrame, 
                             data_name: str = "data") -> pd.DataFrame:
    """
    Validate and align DataFrame GEOIDs with tract GeoDataFrame.
    
    Args:
        df: DataFrame with GEOID column or index
        tracts_gdf: GeoDataFrame with GEOID as index
        data_name: Name of the data for error messages
        
    Returns:
        DataFrame aligned with tracts_gdf (extra GEOIDs dropped, missing raise exception)
    """
    # Get GEOIDs from tracts
    tract_geoids = set(tracts_gdf.index)
    
    # Get GEOIDs from df
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
    
    # Ensure index matches tracts
    df = df.reindex(tract_geoids)
    
    return df


@dataclass
class AnalysisConfig:
    """
    Configuration parameters for the analysis.
    
    This class holds all parameters that control how the analysis is run,
    but not the data itself.
    """
    # SAF scenarios
    saf_scenarios: List[float] = field(default_factory=lambda: [5, 25, 50])
    
    # SAF to pollutant reduction relationship
    saf_polynomial_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    
    # Health impact function parameters (stored directly)
    # Expected keys: mean_rr, lower_rr, upper_rr, unit_increase, 
    #                mean_log_one_unit, se_log_one_unit
    health_impact_function: Optional[Dict[str, float]] = None
    
    # Column names
    pollutant_column: str = 'baseline_pollutant_concentration'
    demographic_columns: List[str] = field(default_factory=list)
    
    # Study area
    airport_coordinates: Optional[tuple] = None
    
    # Other configuration
    crs: str = 'EPSG:4326'


class AnalysisData:
    """
    Core data model for the analysis.
    
    This class holds all geospatial data organized by category:
    - Tract geometries: GeoDataFrame with GEOID as index, geometry as only column
    - Demographics: DataFrame with GEOID as index, demographic attributes
    - Baseline exposure: DataFrame with GEOID as index, baseline pollutant concentrations
    - Mortality: DataFrame with GEOID as index, mortality rates
    - Derived inputs: Values derived from inputs (e.g., % low income)
    - Outputs: Analysis results for each scenario (tract-level)
    - Study area: Optional GeoDataFrame with study area boundary
    """
    
    def __init__(self, crs: str = 'EPSG:4326'):
        """
        Initialize AnalysisData.
        
        Args:
            crs: Coordinate reference system for geospatial data
        """
        self.crs = crs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tract geometries: GEOID as index, geometry as only column
        self.tract_geometries: Optional[gpd.GeoDataFrame] = None
        
        # Demographics: GEOID as index
        self.demographics: Optional[pd.DataFrame] = None
        
        # Baseline exposure: GEOID as index
        self.baseline_exposure: Optional[pd.DataFrame] = None
        
        # Mortality: GEOID as index
        self.mortality: Optional[pd.DataFrame] = None
        
        # Derived inputs: GEOID as index
        self.derived_inputs: Optional[pd.DataFrame] = None
        
        # Scenario outputs: Dict[str, DataFrame] where key is scenario name
        self.scenario_outputs: Dict[str, pd.DataFrame] = {}
        
        # Study area boundary (optional)
        self.study_area: Optional[gpd.GeoDataFrame] = None
        
        # Raw AERMOD data (stored when loading from AERMOD file)
        self._aermod_data: Optional[pd.DataFrame] = None
    
    @property
    def is_ready(self) -> bool:
        """Check if data is ready for analysis."""
        return (self.tract_geometries is not None and 
                self.baseline_exposure is not None and 
                self.mortality is not None)
    
    def load_tract_geometries(self, tracts_gdf: gpd.GeoDataFrame) -> None:
        """
        Load census tract geometries.
        
        Args:
            tracts_gdf: GeoDataFrame with census tract geometries.
                Must contain columns:
                - GEOID: Census tract identifier
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
        Load demographic data.
        
        Args:
            demographics_df: DataFrame with demographic data.
                Must contain GEOID as column or index.
                Can include population, race, ethnicity, income, etc.
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
        
        self.demographics = demographics_df
        
        # Compute derived inputs automatically
        self.compute_derived_inputs()
        
        self.logger.info(f"Loaded demographic data with {len(demographics_df.columns)} columns")
    
    def load_baseline_exposure_data(self, exposure_df: Union[pd.DataFrame, gpd.GeoDataFrame],
                                    pollutant_column: str = 'baseline_pollutant_concentration') -> None:
        """
        Load baseline exposure data with pollutant concentrations.
        
        Args:
            exposure_df: DataFrame or GeoDataFrame with pollutant concentrations.
                Must contain GEOID as column or index.
            pollutant_column: Name of the column containing baseline pollutant concentration
        """
        self.logger.info("Loading baseline exposure data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        exposure_df = exposure_df.copy()
        
        if pollutant_column not in exposure_df.columns:
            raise ValueError(f"Missing required column in baseline exposure data: {pollutant_column}")
        
        # Validate and align GEOIDs
        exposure_df = validate_geoid_alignment(
            exposure_df,
            self.tract_geometries,
            "baseline exposure data"
        )
        
        # Standardize column name
        if pollutant_column != 'baseline_pollutant_concentration':
            exposure_df['baseline_pollutant_concentration'] = exposure_df[pollutant_column]
        
        # Keep only the baseline concentration column
        self.baseline_exposure = exposure_df[['baseline_pollutant_concentration']]
        
        self.logger.info(f"Loaded baseline exposure data with {len(exposure_df)} records")
    
    def load_baseline_exposure_from_AERMOD(self, aermod_file_path: Union[str, Path, List[Union[str, Path]]],
                                          section_types: Optional[List[str]] = None,
                                          center_location: Optional[Tuple[float, float]] = None,
                                          center_crs: Optional[str] = None,
                                          aermod_crs: Optional[str] = None) -> None:
        """
        Load baseline exposure data from one or more AERMOD files.
        
        This method:
        1. Parses AERMOD file(s) to extract concentration data
        2. Handles coordinate transformations for polar coordinates (if center provided)
        3. Creates point geometries from x_coord, y_coord
        4. Intersects points with tracts
        5. Averages concentrations within each tract
        6. If multiple files, sums the annual average values across all files
        
        Args:
            aermod_file_path: Path to AERMOD .OUT or .ADO file, or list of paths.
                           If multiple files, annual averages are summed.
            section_types: Optional list of section types to extract.
                          If None, extracts 'ANNUAL_AVERAGE' only.
                          Options: 'ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST'
            center_location: Optional tuple (x, y) of center point in center_crs.
                           Required for polar coordinate systems to properly convert coordinates.
            center_crs: CRS of the center_location (e.g., 'EPSG:4326' for lat/lon).
                       If None, uses self.crs.
            aermod_crs: CRS of the AERMOD coordinate data (e.g., 'EPSG:32616' for UTM).
                       If None, assumes same as center_crs or self.crs.
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        # Normalize to list of paths
        if isinstance(aermod_file_path, (str, Path)):
            file_paths = [aermod_file_path]
        else:
            file_paths = list(aermod_file_path)
        
        if len(file_paths) == 0:
            raise ValueError("At least one AERMOD file path must be provided")
        
        self.logger.info(f"Loading baseline exposure from {len(file_paths)} AERMOD file(s)")
        
        # Set default CRS values
        if center_crs is None:
            center_crs = self.crs
        if aermod_crs is None:
            aermod_crs = center_crs
        
        if section_types is None:
            section_types = ['ANNUAL_AVERAGE']
        
        # Process each file and collect tract-level exposures
        all_tract_exposures = []
        all_aermod_data = []
        
        for file_idx, file_path in enumerate(file_paths):
            self.logger.info(f"Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
            
            # Parse AERMOD file using comprehensive parser
            parser = AermodParser(str(file_path))
            
            # Parse and extract concentration data
            results = parser.parse(section_types=section_types)
            
            # Extract DataFrame from results dictionary
            aermod_df = pd.DataFrame()
            for section_type in section_types:
                if section_type in results and isinstance(results[section_type], pd.DataFrame):
                    if len(aermod_df) == 0:
                        aermod_df = results[section_type]
                    else:
                        aermod_df = pd.concat([aermod_df, results[section_type]], ignore_index=True)
            
            if len(aermod_df) == 0:
                self.logger.warning(f"No concentration data found in {file_path}, skipping")
                continue
            
            # Store raw AERMOD data
            aermod_df['source_file'] = str(file_path)
            all_aermod_data.append(aermod_df.copy())
            
            # Check if we have coordinates
            if 'x_coord' not in aermod_df.columns or 'y_coord' not in aermod_df.columns:
                self.logger.warning(f"Missing coordinates in {file_path}, skipping")
                continue
            
            # Handle coordinate transformations for polar coordinates
            if center_location is not None and 'network_id' in aermod_df.columns:
                # Transform center location to AERMOD CRS
                center_point = gpd.GeoDataFrame(
                    [1], 
                    geometry=gpd.points_from_xy([center_location[0]], [center_location[1]]),
                    crs=center_crs
                )
                center_point_aermod = center_point.to_crs(aermod_crs)
                center_x_aermod = center_point_aermod.geometry.iloc[0].x
                center_y_aermod = center_point_aermod.geometry.iloc[0].y
                
                # Check if we have polar coordinates (need to recompute with proper center)
                network_ids = aermod_df['network_id'].dropna().unique()
                
                for network_id in network_ids:
                    if network_id in parser.network_info:
                        network_info = parser.network_info[network_id]
                        if network_info.get('network_type') == 'GRIDPOLR':
                            # Get network mask
                            network_mask = aermod_df['network_id'] == network_id
                            network_data = aermod_df[network_mask].copy()
                            
                            # Get origin from network info (the origin used by parser)
                            origin_x = network_info.get('origin_x', 0.0)
                            origin_y = network_info.get('origin_y', 0.0)
                            
                            # Parser already converted polar to cartesian using file origin
                            # We need to adjust coordinates to use user-provided center instead
                            # Calculate offset: new_coord = old_coord - origin + center
                            offset_x = center_x_aermod - origin_x
                            offset_y = center_y_aermod - origin_y
                            
                            # Adjust coordinates in AERMOD CRS
                            aermod_df.loc[network_mask, 'x_coord'] = (
                                network_data['x_coord'].values + offset_x
                            )
                            aermod_df.loc[network_mask, 'y_coord'] = (
                                network_data['y_coord'].values + offset_y
                            )
                            
                            self.logger.info(
                                f"Adjusted polar coordinates for network {network_id}: "
                                f"origin ({origin_x:.2f}, {origin_y:.2f}) -> "
                                f"center ({center_x_aermod:.2f}, {center_y_aermod:.2f})"
                            )
                
                # Transform all coordinates from AERMOD CRS to center CRS
                points_aermod = gpd.GeoDataFrame(
                    aermod_df,
                    geometry=gpd.points_from_xy(aermod_df['x_coord'], aermod_df['y_coord']),
                    crs=aermod_crs
                )
                points_center = points_aermod.to_crs(center_crs)
                
                # Update coordinates in dataframe to center CRS
                aermod_df['x_coord'] = points_center.geometry.x.values
                aermod_df['y_coord'] = points_center.geometry.y.values
                
                self.logger.info(f"Transformed coordinates from {aermod_crs} to {center_crs}")
            
            # Create point geometries from coordinates
            # If center_location was provided, coordinates are in center_crs
            # Otherwise, coordinates are in aermod_crs (from parser)
            if center_location is not None:
                # Coordinates already transformed to center_crs above
                points_crs = center_crs
            else:
                # Coordinates are in aermod_crs, transform to self.crs if different
                points_crs = aermod_crs
                if aermod_crs != self.crs:
                    points_aermod = gpd.GeoDataFrame(
                        aermod_df,
                        geometry=gpd.points_from_xy(aermod_df['x_coord'], aermod_df['y_coord']),
                        crs=aermod_crs
                    )
                    points_self_crs = points_aermod.to_crs(self.crs)
                    aermod_df['x_coord'] = points_self_crs.geometry.x.values
                    aermod_df['y_coord'] = points_self_crs.geometry.y.values
                    points_crs = self.crs
                    self.logger.info(f"Transformed coordinates from {aermod_crs} to {self.crs}")
            
            points_gdf = gpd.GeoDataFrame(
                aermod_df,
                geometry=gpd.points_from_xy(aermod_df['x_coord'], aermod_df['y_coord']),
                crs=points_crs
            )
            
            # Transform to tract CRS if needed
            if points_gdf.crs != self.tract_geometries.crs:
                self.logger.info(f"Transforming AERMOD points from {points_gdf.crs} to {self.tract_geometries.crs}")
                points_gdf = points_gdf.to_crs(self.tract_geometries.crs)
            
            # Spatial join with tracts
            # Use sjoin to find which points fall within each tract
            points_in_tracts = gpd.sjoin(
                points_gdf[['concentration', 'geometry']],
                self.tract_geometries.reset_index(),
                how='inner',
                predicate='within'
            )
            
            # Group by GEOID and average concentrations for this file
            tract_exposure = points_in_tracts.groupby('GEOID')['concentration'].mean().reset_index()
            tract_exposure.columns = ['GEOID', 'baseline_pollutant_concentration']
            all_tract_exposures.append(tract_exposure)
        
        # Combine all tract exposures
        if len(all_tract_exposures) == 0:
            raise ValueError("No concentration data found in any AERMOD file")
        
        # Store combined raw AERMOD data
        if len(all_aermod_data) > 0:
            self._aermod_data = pd.concat(all_aermod_data, ignore_index=True)
        
        # Sum concentrations across all files by GEOID
        combined_exposure = all_tract_exposures[0].set_index('GEOID')
        for tract_exposure in all_tract_exposures[1:]:
            tract_exposure_idx = tract_exposure.set_index('GEOID')
            combined_exposure = combined_exposure.add(
                tract_exposure_idx, 
                fill_value=0.0
            )
        
        combined_exposure = combined_exposure.reset_index()
        
        # Align with tract geometries (allow missing GEOIDs, will fill with mean)
        tract_geoids = set(self.tract_geometries.index)
        exposure_geoids = set(combined_exposure['GEOID'].astype(int))
        
        # Check for missing GEOIDs
        missing_geoids = tract_geoids - exposure_geoids
        if missing_geoids:
            self.logger.warning(
                f"{len(missing_geoids)} tracts missing exposure data from AERMOD, "
                f"will fill with mean"
            )
        
        # Set GEOID as index for alignment
        if 'GEOID' in combined_exposure.columns:
            combined_exposure = combined_exposure.set_index('GEOID')
        
        # Reindex to match all tract GEOIDs
        exposure_df = combined_exposure.reindex(tract_geoids)
        
        # Fill missing tracts with mean
        missing = exposure_df['baseline_pollutant_concentration'].isna().sum()
        if missing > 0:
            self.logger.warning(f"{missing} tracts missing exposure data from AERMOD, filling with mean")
            mean_exposure = exposure_df['baseline_pollutant_concentration'].mean()
            exposure_df['baseline_pollutant_concentration'] = (
                exposure_df['baseline_pollutant_concentration'].fillna(mean_exposure)
            )
        
        self.baseline_exposure = exposure_df[['baseline_pollutant_concentration']]
        
        self.logger.info(
            f"Loaded baseline exposure from {len(file_paths)} AERMOD file(s): "
            f"{len(exposure_df)} tracts, summed concentrations"
        )
    
    def load_baseline_exposure_from_aermod_workflow(
        self,
        landing_files: List[Tuple[Union[str, Path], float]],
        takeoff_files: List[Tuple[Union[str, Path], float]],
        calibration_file: Union[str, Path],
        aermod_crs: str = 'EPSG:32616',
        aggregation_method: str = 'spatial_join',
        **kwargs
    ) -> None:
        """
        Load baseline exposure data from AERMOD files using the expert-defined workflow.
        
        This method uses the full pipeline:
        1. Extract annual averages from AERMOD files
        2. Weighted combination of flows (e.g., east/west) for landing and takeoff
        3. Convert to percentiles and apply log-linear calibration to obtain UFP
        4. Aggregate UFP values at receptor locations to census tracts
        5. Sum landing and takeoff UFP for total exposure
        
        Args:
            landing_files: List of (file_path, weight) tuples for landing flows.
                          Weights should sum to approximately 1.0.
            takeoff_files: List of (file_path, weight) tuples for takeoff flows.
                          Weights should sum to approximately 1.0.
            calibration_file: Path to JSON file with calibration coefficients.
                            Should contain 'intercept', 'coef_landing', 'coef_takeoff'.
            aermod_crs: Coordinate reference system for AERMOD data (default: EPSG:32616)
            aggregation_method: Method for aggregating to tracts.
                              'spatial_join' (default): spatial join with IDW fallback
                              'idw_interpolation': pure IDW interpolation
            **kwargs: Additional parameters passed to generate_exposure_from_aermod:
                     - idw_power: Power parameter for IDW (default: 2)
                     - idw_max_distance: Max distance for IDW (default: None)
                     - idw_num_neighbors: Number of neighbors for IDW (default: None)
                     - receptor_match_tolerance: Tolerance for matching receptors (default: 1.0)
                     - airport_location: Optional tuple (lon, lat) of airport coordinates in WGS84
                     - airport_threshold_distance: Distance threshold in meters for airport proximity (default: 3000)
                     - airport_source_multiplier: Multiplier for max receptor value near airport (default: 1.15)
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        from bensaf.exposure_generation import generate_exposure_from_aermod
        
        self.logger.info(
            f"Loading baseline exposure from AERMOD workflow: "
            f"{len(landing_files)} landing file(s), {len(takeoff_files)} takeoff file(s)"
        )
        
        # Generate exposure data using the workflow
        exposure_df = generate_exposure_from_aermod(
            landing_files=landing_files,
            takeoff_files=takeoff_files,
            tracts_gdf=self.tract_geometries.reset_index(),
            calibration_file=calibration_file,
            aermod_crs=aermod_crs,
            aggregation_method=aggregation_method,
            **kwargs
        )
        
        # Validate and align GEOIDs
        exposure_df = validate_geoid_alignment(
            exposure_df,
            self.tract_geometries,
            "AERMOD workflow exposure data"
        )
        
        # Set as baseline exposure
        self.baseline_exposure = exposure_df[['baseline_pollutant_concentration']]
        
        self.logger.info(
            f"Loaded baseline exposure from AERMOD workflow: "
            f"{len(exposure_df)} tracts with UFP concentrations"
        )
    
    def load_mortality_data(self, mortality_df: pd.DataFrame) -> None:
        """
        Load mortality data with baseline rates.
        
        Args:
            mortality_df: DataFrame with mortality rates.
                Must contain GEOID as column or index.
                Must contain 'mortality_rate' column.
        """
        self.logger.info("Loading mortality data")
        
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        mortality_df = mortality_df.copy()
        
        if 'mortality_rate' not in mortality_df.columns:
            raise ValueError("Missing required column 'mortality_rate' in mortality data")
        
        # Validate and align GEOIDs
        mortality_df = validate_geoid_alignment(
            mortality_df,
            self.tract_geometries,
            "mortality data"
        )
        
        # Keep only mortality_rate column
        self.mortality = mortality_df[['mortality_rate']].copy()
        
        # Compute natural mortality rate per 100k
        self.mortality['nmr_per_100k'] = self.mortality['mortality_rate'] * 100000
        
        self.logger.info(f"Loaded mortality data with {len(mortality_df)} records")
    
    def compute_derived_inputs(self) -> None:
        """
        Compute derived input values from raw inputs.
        
        This method automatically computes derived metrics such as:
        - Percentage of low-income population
        - Distance from airport (if airport_coordinates provided)
        - Other demographic-derived metrics
        
        This is called automatically when demographics are loaded.
        """
        if self.demographics is None:
            return
        
        self.logger.info("Computing derived inputs")
        
        derived = {}
        
        # Compute percentage of low-income population if income data available
        if 'households_below_poverty' in self.demographics.columns and 'total_households' in self.demographics.columns:
            derived['pct_low_income'] = (
                self.demographics['households_below_poverty'] / 
                self.demographics['total_households'] * 100
            )
        elif 'poverty_rate' in self.demographics.columns:
            derived['pct_low_income'] = self.demographics['poverty_rate'] * 100
        
        # Compute distance from airport if coordinates provided
        # This would require airport_coordinates in config, so we'll skip for now
        # Can be added later if needed
        
        if derived:
            if self.derived_inputs is None:
                self.derived_inputs = pd.DataFrame(index=self.demographics.index)
            
            for name, values in derived.items():
                self.derived_inputs[name] = values
                self.logger.info(f"Computed derived input: {name}")
    
    def add_scenario_output(self, scenario_name: str, outputs: Dict[str, Union[pd.Series, np.ndarray]]) -> None:
        """
        Add analysis outputs for a specific scenario.
        
        Args:
            scenario_name: Name/identifier for the scenario (e.g., "5% SAF Usage")
            outputs: Dictionary of output columns to add, e.g.:
                {
                    'reduced_concentration': pd.Series(...),
                    'delta_concentration': pd.Series(...),
                    'attributable_fraction': pd.Series(...),
                    'attributable_cases': pd.Series(...),
                    ...
                }
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")
        
        scenario_df = pd.DataFrame(index=self.tract_geometries.index)
        
        for col_name, values in outputs.items():
            if isinstance(values, (list, np.ndarray)):
                values = pd.Series(values, index=self.tract_geometries.index)
            
            if len(values) != len(self.tract_geometries):
                raise ValueError(f"Length of {col_name} ({len(values)}) must match number of tracts ({len(self.tract_geometries)})")
            
            if not values.index.equals(self.tract_geometries.index):
                values = values.reindex(self.tract_geometries.index)
            
            scenario_df[col_name] = values
        
        self.scenario_outputs[scenario_name] = scenario_df
        self.logger.info(f"Added outputs for scenario: {scenario_name}")
    
    def get_scenario_output(self, scenario_name: str) -> Optional[pd.DataFrame]:
        """Get outputs for a specific scenario."""
        return self.scenario_outputs.get(scenario_name)
    
    def get_merged_data(self) -> gpd.GeoDataFrame:
        """
        Get merged GeoDataFrame with all data.
        
        Returns:
            GeoDataFrame with geometries and all other data merged
        """
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")
        
        merged = self.tract_geometries.copy()
        
        if self.demographics is not None:
            merged = merged.join(self.demographics, how='left')
        
        if self.baseline_exposure is not None:
            merged = merged.join(self.baseline_exposure, how='left')
        
        if self.mortality is not None:
            merged = merged.join(self.mortality, how='left')
        
        if self.derived_inputs is not None:
            merged = merged.join(self.derived_inputs, how='left')
        
        for scenario_name, scenario_df in self.scenario_outputs.items():
            # Prefix scenario columns with scenario name
            scenario_df_prefixed = scenario_df.add_prefix(f"{scenario_name}_")
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
        
        if self.mortality is None:
            raise ValueError("Mortality data must be loaded")
        
        return True
