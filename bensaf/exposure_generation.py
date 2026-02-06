"""
AERMOD Exposure Generation Module

This module provides functions to generate exposure data from AERMOD files
following the expert-defined pipeline:
1. Extract annual averages from AERMOD files
2. Weighted combination of flows (e.g., east/west)
3. Convert to percentiles and apply log-linear calibration to obtain UFP
4. Aggregate UFP values at receptor locations to census tracts
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from bensaf.aermod_parser import AermodParser

logger = logging.getLogger(__name__)


def extract_annual_average(ado_file_path: Union[str, Path], aermod_crs: str = 'EPSG:32616') -> Optional[gpd.GeoDataFrame]:
    """
    Extract annual average concentration data from an AERMOD .ADO file.
    
    Args:
        ado_file_path: Path to AERMOD .ADO file
        aermod_crs: Coordinate reference system for AERMOD data
        
    Returns:
        GeoDataFrame with annual average concentrations, or None if no data found
    """
    parser = AermodParser(str(ado_file_path))
    results = parser.parse(section_types=['ANNUAL_AVERAGE'])
    
    annual_avg = results.get('ANNUAL_AVERAGE', pd.DataFrame())
    if len(annual_avg) == 0:
        return None
    
    # Filter valid coordinates
    valid_annual = annual_avg.dropna(subset=['x_coord', 'y_coord'])
    valid_annual = valid_annual[
        (valid_annual['x_coord'].abs() < 1e10) & 
        (valid_annual['y_coord'].abs() < 1e10)
    ]
    
    if len(valid_annual) == 0:
        return None
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(valid_annual['x_coord'], valid_annual['y_coord'])]
    gdf = gpd.GeoDataFrame(valid_annual, geometry=geometry, crs=aermod_crs)
    
    return gdf


def match_receptor_points(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                         tolerance: float = 1.0) -> pd.DataFrame:
    """
    Match receptor points between two GeoDataFrames based on coordinates.
    
    Args:
        gdf1: First GeoDataFrame with receptor points
        gdf2: Second GeoDataFrame with receptor points
        tolerance: Maximum distance for matching (meters)
        
    Returns:
        DataFrame with matched points and concentrations
    """
    coords1 = np.array([[g.x, g.y] for g in gdf1.geometry])
    coords2 = np.array([[g.x, g.y] for g in gdf2.geometry])
    
    # Build KDTree for gdf2
    tree = cKDTree(coords2)
    
    # Find nearest neighbors
    distances, indices = tree.query(coords1, k=1)
    
    # Filter by tolerance
    mask = distances <= tolerance
    
    # Create matched dataframe
    matched = pd.DataFrame({
        'x_coord': coords1[mask, 0],
        'y_coord': coords1[mask, 1],
        'conc1': gdf1.iloc[mask]['concentration'].values,
        'conc2': gdf2.iloc[indices[mask]]['concentration'].values
    })
    
    return matched


def weighted_combine_flows(flow_gdfs: List[gpd.GeoDataFrame], 
                          weights: List[float],
                          aermod_crs: str = 'EPSG:32616',
                          tolerance: float = 1.0) -> gpd.GeoDataFrame:
    """
    Combine multiple flow GeoDataFrames with weights.
    
    This is a generalization of the notebook's east/west flow combination
    to support any number of files with weights. Uses the first flow as the
    reference and matches all other flows to it.
    
    Args:
        flow_gdfs: List of GeoDataFrames with flow data
        weights: List of weights (should sum to ~1.0)
        aermod_crs: Coordinate reference system
        tolerance: Receptor matching tolerance (meters)
        
    Returns:
        Combined GeoDataFrame with weighted concentrations
    """
    if len(flow_gdfs) == 0:
        raise ValueError("At least one flow GeoDataFrame must be provided")
    
    if len(flow_gdfs) != len(weights):
        raise ValueError(f"Number of flow GeoDataFrames ({len(flow_gdfs)}) must match number of weights ({len(weights)})")
    
    # Normalize weights
    total_weight = sum(weights)
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
        weights = [w / total_weight for w in weights]
    
    # Use first flow as reference
    reference_gdf = flow_gdfs[0].copy()
    combined_conc = reference_gdf['concentration'].values * weights[0]
    
    # Match and add remaining flows
    for i in range(1, len(flow_gdfs)):
        matched = match_receptor_points(reference_gdf, flow_gdfs[i], tolerance=tolerance)
        
        if len(matched) == 0:
            logger.warning(f"No matching receptors found between reference flow and flow {i}")
            continue
        
        # Create lookup dictionary for matched points
        matched_dict = {}
        for _, row in matched.iterrows():
            coord_key = (row['x_coord'], row['y_coord'])
            matched_dict[coord_key] = row['conc2']
        
        # Update combined concentrations for matched points
        for idx, point in enumerate(reference_gdf.geometry):
            coord_key = (point.x, point.y)
            if coord_key in matched_dict:
                combined_conc[idx] += matched_dict[coord_key] * weights[i]
    
    # Create combined GeoDataFrame
    combined = reference_gdf.copy()
    combined['concentration'] = combined_conc
    
    return combined


def load_calibration_coefficients(calibration_file: Union[str, Path]) -> Tuple[float, float, float]:
    """
    Load calibration coefficients from JSON file.
    
    Args:
        calibration_file: Path to JSON file with calibration coefficients
        
    Returns:
        Tuple of (intercept, coef_landing, coef_takeoff)
    """
    calibration_file = Path(calibration_file)
    if not calibration_file.exists():
        raise FileNotFoundError(
            f"Calibration coefficients file not found: {calibration_file}\n"
            f"Run scripts/extract_calibration_coefficients.py first to generate this file."
        )
    
    with open(calibration_file, 'r') as f:
        coeffs = json.load(f)
    
    intercept = coeffs['intercept']
    coef_landing = coeffs['coef_landing']
    coef_takeoff = coeffs['coef_takeoff']
    
    return intercept, coef_landing, coef_takeoff


def apply_calibration(landing_combined: gpd.GeoDataFrame, 
                     takeoff_combined: gpd.GeoDataFrame,
                     calibration_file: Union[str, Path]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Apply log-linear calibration to convert CO concentrations to UFP concentrations.
    
    Args:
        landing_combined: GeoDataFrame with combined landing flow concentrations
        takeoff_combined: GeoDataFrame with combined takeoff flow concentrations
        calibration_file: Path to JSON file with calibration coefficients
        
    Returns:
        Tuple of (landing_ufp, takeoff_ufp) GeoDataFrames with UFP concentrations
    """
    # Load calibration coefficients
    intercept, coef_landing, coef_takeoff = load_calibration_coefficients(calibration_file)
    
    # Convert to percentiles (rank-based, separately for landing and takeoff)
    landing_percentile = landing_combined['concentration'].rank(pct=True)
    takeoff_percentile = takeoff_combined['concentration'].rank(pct=True)
    
    # Apply log-linear model components separately
    log_ufp_landing = intercept + coef_landing * landing_percentile
    log_ufp_takeoff = intercept + coef_takeoff * takeoff_percentile
    
    # Exponentiate to get UFP concentrations
    landing_ufp = landing_combined.copy()
    landing_ufp['ufp_concentration'] = np.exp(log_ufp_landing)
    
    takeoff_ufp = takeoff_combined.copy()
    takeoff_ufp['ufp_concentration'] = np.exp(log_ufp_takeoff)
    
    return landing_ufp, takeoff_ufp


def idw_interpolation(source_points: np.ndarray, 
                     source_values: np.ndarray,
                     target_points: np.ndarray,
                     power: int = 2,
                     max_distance: Optional[float] = None,
                     num_neighbors: Optional[int] = None) -> np.ndarray:
    """
    Perform Inverse Distance Weighting (IDW) interpolation.
    
    Args:
        source_points: Array of source point coordinates (N x 2)
        source_values: Array of source values (N,)
        target_points: Array of target point coordinates (M x 2)
        power: Power parameter for IDW (default: 2)
        max_distance: Maximum distance for interpolation (None = no limit)
        num_neighbors: Number of neighbors to use (None = use all)
        
    Returns:
        Array of interpolated values (M,)
    """
    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(source_points)
    
    # Determine number of neighbors
    if num_neighbors is None:
        num_neighbors = len(source_points)
    
    # Find nearest neighbors for each target point
    distances, indices = tree.query(target_points, k=min(num_neighbors, len(source_points)))
    
    # Handle case where only one neighbor is requested
    if num_neighbors == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    
    # Calculate interpolated values
    interpolated = np.zeros(len(target_points))
    
    for i in range(len(target_points)):
        # Get distances and values for neighbors
        neighbor_distances = distances[i]
        neighbor_indices = indices[i]
        
        # Filter by max_distance if specified
        if max_distance is not None:
            mask = neighbor_distances <= max_distance
            neighbor_distances = neighbor_distances[mask]
            neighbor_indices = neighbor_indices[mask]
        
        if len(neighbor_distances) == 0:
            interpolated[i] = np.nan
            continue
        
        # Avoid division by zero (exact matches)
        neighbor_distances = np.maximum(neighbor_distances, 1e-10)
        
        # Calculate weights: 1 / distance^power
        weights = 1.0 / (neighbor_distances ** power)
        
        # Get neighbor values
        neighbor_values = source_values[neighbor_indices]
        
        # IDW formula: sum(zi / di^p) / sum(1 / di^p)
        numerator = np.sum(neighbor_values * weights)
        denominator = np.sum(weights)
        
        if denominator > 0:
            interpolated[i] = numerator / denominator
        else:
            interpolated[i] = np.nan
    
    return interpolated


def aggregate_to_tracts(receptor_gdf: gpd.GeoDataFrame,
                       value_column: str,
                       tracts_gdf: gpd.GeoDataFrame,
                       method: str = 'spatial_join',
                       idw_power: int = 2,
                       idw_max_distance: Optional[float] = None,
                       idw_num_neighbors: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate receptor point values to census tracts.
    
    For method='spatial_join':
        - Uses spatial join (average) for tracts with direct receptor intersections
        - Uses IDW interpolation for tracts without intersections
        - Uses nearest neighbor as fallback for isolated tracts
    
    For method='idw_interpolation':
        - Uses pure IDW interpolation to all tract centroids
    
    Args:
        receptor_gdf: GeoDataFrame with receptor points and values
        value_column: Name of column containing values to aggregate
        tracts_gdf: GeoDataFrame with census tract geometries (must have GEOID)
        method: Aggregation method ('spatial_join' or 'idw_interpolation')
        idw_power: Power parameter for IDW (if used)
        idw_max_distance: Maximum distance for IDW (if used)
        idw_num_neighbors: Number of neighbors for IDW (if used)
        
    Returns:
        DataFrame with GEOID and aggregated values
    """
    # Ensure same CRS
    if receptor_gdf.crs != tracts_gdf.crs:
        tracts_gdf = tracts_gdf.to_crs(receptor_gdf.crs)
    
    # Start with all tracts
    if 'GEOID' in tracts_gdf.columns:
        all_tracts = tracts_gdf[['GEOID', 'geometry']].copy()
    else:
        all_tracts = tracts_gdf.reset_index()[['GEOID', 'geometry']].copy()
    
    if method == 'spatial_join':
        # Step 1: Spatial join for tracts with direct intersections
        points_in_tracts = gpd.sjoin(
            receptor_gdf[[value_column, 'geometry']],
            all_tracts,
            how='inner',
            predicate='within'
        )
        
        # Group by GEOID and average values
        tracts_with_data = points_in_tracts.groupby('GEOID')[value_column].mean().reset_index()
        
        # Step 2: Identify tracts without data
        tracts_without_data = all_tracts[~all_tracts['GEOID'].isin(tracts_with_data['GEOID'])]
        
        if len(tracts_without_data) > 0:
            # Step 3: Use IDW interpolation for tracts without direct intersections
            receptor_coords = np.array([[g.x, g.y] for g in receptor_gdf.geometry])
            receptor_values = receptor_gdf[value_column].values
            
            # Get centroids of tracts without data
            tract_centroids = tracts_without_data.geometry.centroid
            target_coords = np.array([[g.x, g.y] for g in tract_centroids])
            
            # Perform IDW interpolation
            interpolated_values = idw_interpolation(
                receptor_coords,
                receptor_values,
                target_coords,
                power=idw_power,
                max_distance=idw_max_distance,
                num_neighbors=idw_num_neighbors
            )
            
            # Step 4: For any remaining NaN values (very isolated tracts), use nearest neighbor
            nan_mask = np.isnan(interpolated_values)
            if nan_mask.any():
                # Find nearest receptor for isolated tracts
                tree = cKDTree(receptor_coords)
                isolated_coords = target_coords[nan_mask]
                distances, indices = tree.query(isolated_coords, k=1)
                interpolated_values[nan_mask] = receptor_values[indices]
            
            # Create DataFrame for interpolated tracts
            tracts_interpolated = pd.DataFrame({
                'GEOID': tracts_without_data['GEOID'].values,
                value_column: interpolated_values
            })
            
            # Combine results
            tract_exposure = pd.concat([tracts_with_data, tracts_interpolated], ignore_index=True)
        else:
            tract_exposure = tracts_with_data
        
        return tract_exposure
    
    elif method == 'idw_interpolation':
        # Pure IDW interpolation to all tract centroids
        receptor_coords = np.array([[g.x, g.y] for g in receptor_gdf.geometry])
        receptor_values = receptor_gdf[value_column].values
        
        tract_centroids = tracts_gdf.geometry.centroid
        target_coords = np.array([[g.x, g.y] for g in tract_centroids])
        
        interpolated_values = idw_interpolation(
            receptor_coords,
            receptor_values,
            target_coords,
            power=idw_power,
            max_distance=idw_max_distance,
            num_neighbors=idw_num_neighbors
        )
        
        # Fallback to nearest neighbor for any NaN values
        nan_mask = np.isnan(interpolated_values)
        if nan_mask.any():
            tree = cKDTree(receptor_coords)
            isolated_coords = target_coords[nan_mask]
            distances, indices = tree.query(isolated_coords, k=1)
            interpolated_values[nan_mask] = receptor_values[indices]
        
        if 'GEOID' in tracts_gdf.columns:
            geoids = tracts_gdf['GEOID'].values
        else:
            geoids = tracts_gdf.reset_index()['GEOID'].values
        
        tract_exposure = pd.DataFrame({
            'GEOID': geoids,
            value_column: interpolated_values
        })
        
        return tract_exposure
    
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Must be 'spatial_join' or 'idw_interpolation'")


def generate_exposure_from_aermod(
    landing_files: List[Tuple[Union[str, Path], float]],
    takeoff_files: List[Tuple[Union[str, Path], float]],
    tracts_gdf: gpd.GeoDataFrame,
    calibration_file: Union[str, Path],
    aermod_crs: str = 'EPSG:32616',
    aggregation_method: str = 'spatial_join',
    idw_power: int = 2,
    idw_max_distance: Optional[float] = None,
    idw_num_neighbors: Optional[int] = None,
    receptor_match_tolerance: float = 1.0,
    **aggregation_kwargs
) -> pd.DataFrame:
    """
    Generate exposure data from AERMOD files following the expert-defined pipeline.
    
    Pipeline:
    1. Extract annual averages from AERMOD files
    2. Weighted combination of flows for landing and takeoff separately
    3. Convert to percentiles and apply log-linear calibration to obtain UFP
    4. Aggregate UFP values at receptor locations to census tracts
    5. Sum landing and takeoff UFP for total exposure
    
    Args:
        landing_files: List of (file_path, weight) tuples for landing flows
        takeoff_files: List of (file_path, weight) tuples for takeoff flows
        tracts_gdf: GeoDataFrame with census tract geometries (must have GEOID)
        calibration_file: Path to JSON file with calibration coefficients
        aermod_crs: Coordinate reference system for AERMOD data
        aggregation_method: Method for aggregating to tracts ('spatial_join' or 'idw_interpolation')
        idw_power: Power parameter for IDW interpolation
        idw_max_distance: Maximum distance for IDW interpolation (None = no limit)
        idw_num_neighbors: Number of neighbors for IDW interpolation (None = use all)
        receptor_match_tolerance: Tolerance for matching receptors (meters)
        **aggregation_kwargs: Additional parameters passed to aggregate_to_tracts
        
    Returns:
        DataFrame with GEOID and baseline_pollutant_concentration columns
    """
    logger.info("Starting AERMOD exposure generation workflow")
    
    # Step 1: Extract annual averages from landing files
    logger.info(f"Extracting annual averages from {len(landing_files)} landing file(s)")
    landing_gdfs = []
    for file_path, weight in landing_files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Landing file not found: {file_path}")
        
        gdf = extract_annual_average(file_path, aermod_crs=aermod_crs)
        if gdf is None:
            raise ValueError(f"No annual average data found in {file_path}")
        
        landing_gdfs.append(gdf)
        logger.info(f"  Extracted {len(gdf)} receptor points from {file_path.name}")
    
    # Step 2: Extract annual averages from takeoff files
    logger.info(f"Extracting annual averages from {len(takeoff_files)} takeoff file(s)")
    takeoff_gdfs = []
    for file_path, weight in takeoff_files:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Takeoff file not found: {file_path}")
        
        gdf = extract_annual_average(file_path, aermod_crs=aermod_crs)
        if gdf is None:
            raise ValueError(f"No annual average data found in {file_path}")
        
        takeoff_gdfs.append(gdf)
        logger.info(f"  Extracted {len(gdf)} receptor points from {file_path.name}")
    
    # Step 3: Weighted combination of flows
    logger.info("Combining landing flows with weights")
    landing_weights = [w for _, w in landing_files]
    landing_combined = weighted_combine_flows(
        landing_gdfs, 
        landing_weights,
        aermod_crs=aermod_crs,
        tolerance=receptor_match_tolerance
    )
    logger.info(f"  Combined landing: {len(landing_combined)} receptor points")
    
    logger.info("Combining takeoff flows with weights")
    takeoff_weights = [w for _, w in takeoff_files]
    takeoff_combined = weighted_combine_flows(
        takeoff_gdfs,
        takeoff_weights,
        aermod_crs=aermod_crs,
        tolerance=receptor_match_tolerance
    )
    logger.info(f"  Combined takeoff: {len(takeoff_combined)} receptor points")
    
    # Step 4: Apply calibration to convert CO to UFP
    logger.info("Applying calibration to convert CO to UFP")
    landing_ufp, takeoff_ufp = apply_calibration(
        landing_combined,
        takeoff_combined,
        calibration_file
    )
    logger.info(f"  Landing UFP: {len(landing_ufp)} receptor points")
    logger.info(f"  Takeoff UFP: {len(takeoff_ufp)} receptor points")
    
    # Step 5: Aggregate to census tracts
    logger.info(f"Aggregating to census tracts using method: {aggregation_method}")
    
    # Ensure tracts are in AERMOD CRS for spatial operations
    tracts_aermod_crs = tracts_gdf.to_crs(aermod_crs) if tracts_gdf.crs != aermod_crs else tracts_gdf.copy()
    
    # Aggregate landing UFP
    tracts_landing = aggregate_to_tracts(
        landing_ufp,
        'ufp_concentration',
        tracts_aermod_crs,
        method=aggregation_method,
        idw_power=idw_power,
        idw_max_distance=idw_max_distance,
        idw_num_neighbors=idw_num_neighbors,
        **aggregation_kwargs
    )
    tracts_landing = tracts_landing.rename(columns={'ufp_concentration': 'landing_ufp'})
    logger.info(f"  Landing UFP aggregated to {len(tracts_landing)} tracts")
    
    # Aggregate takeoff UFP
    tracts_takeoff = aggregate_to_tracts(
        takeoff_ufp,
        'ufp_concentration',
        tracts_aermod_crs,
        method=aggregation_method,
        idw_power=idw_power,
        idw_max_distance=idw_max_distance,
        idw_num_neighbors=idw_num_neighbors,
        **aggregation_kwargs
    )
    tracts_takeoff = tracts_takeoff.rename(columns={'ufp_concentration': 'takeoff_ufp'})
    logger.info(f"  Takeoff UFP aggregated to {len(tracts_takeoff)} tracts")
    
    # Step 6: Combine landing and takeoff
    tracts_exposure = pd.merge(
        tracts_landing,
        tracts_takeoff,
        on='GEOID',
        how='outer'
    )
    
    # Sum landing and takeoff for total UFP
    tracts_exposure['baseline_pollutant_concentration'] = (
        tracts_exposure['landing_ufp'].fillna(0) + 
        tracts_exposure['takeoff_ufp'].fillna(0)
    )
    
    logger.info(f"Generated exposure data for {len(tracts_exposure)} tracts")
    logger.info(f"  Total UFP range: {tracts_exposure['baseline_pollutant_concentration'].min():.6f} - {tracts_exposure['baseline_pollutant_concentration'].max():.6f}")
    
    # Return only GEOID and baseline_pollutant_concentration
    return tracts_exposure[['GEOID', 'baseline_pollutant_concentration']]
