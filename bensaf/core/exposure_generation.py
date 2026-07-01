"""
AERMOD Exposure Generation Module

This module provides functions to generate exposure data from AERMOD files
following the expert-defined pipeline:
1. Extract annual averages from AERMOD files
2. Weighted combination of flows (e.g., east/west)
3. Apply log-linear calibration: rank percentiles of landing/takeoff CO jointly in one model, then exp
4. Aggregate UFP at receptors to census tracts
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from aermod_parser import AermodFile

logger = logging.getLogger(__name__)


def _extract_aermod_sections(
    file_path: Union[str, Path],
    section_types: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Parse an AERMOD file with the ``aermod_parser`` package and shape the result
    tables to match the legacy ``AermodParser`` output: each table gets x_coord/
    y_coord columns (converted from polar direction/distance using each
    receptor's network origin, or renamed directly from Cartesian x/y).

    Returns:
        (results, network_info) where results maps section_type -> DataFrame,
        and network_info maps network_id -> {'network_type', 'origin_x', 'origin_y'}.
    """
    f = AermodFile.from_path(file_path)
    networks = f.metadata.networks
    network_info = {
        nid: {
            'network_type': net.network_type,
            'origin_x': net.origin_x,
            'origin_y': net.origin_y,
        }
        for nid, net in networks.items()
    }
    origins = {nid: (net.origin_x or 0.0, net.origin_y or 0.0) for nid, net in networks.items()}

    results: Dict[str, pd.DataFrame] = {}
    for section_type in section_types:
        if section_type == 'ANNUAL_AVERAGE':
            df = f.annual_average
        elif section_type.endswith('_HIGHEST'):
            df = f.n_highest(rank=int(section_type[0]))
        elif section_type == 'CONCURRENT_AVERAGE':
            df = f.concurrent
        else:
            continue

        if df.empty:
            results[section_type] = df
            continue

        df = df.copy()
        if 'direction' in df.columns and 'distance' in df.columns:
            origin_x = df['network_id'].map(lambda nid: origins.get(nid, (0.0, 0.0))[0])
            origin_y = df['network_id'].map(lambda nid: origins.get(nid, (0.0, 0.0))[1])
            df['x_coord'] = origin_x + df['distance'] * np.sin(np.radians(df['direction']))
            df['y_coord'] = origin_y + df['distance'] * np.cos(np.radians(df['direction']))
        elif 'x' in df.columns and 'y' in df.columns:
            df = df.rename(columns={'x': 'x_coord', 'y': 'y_coord'})
        results[section_type] = df

    return results, network_info


def extract_annual_average(ado_file_path: Union[str, Path], aermod_crs: str = 'EPSG:32616') -> Optional[gpd.GeoDataFrame]:
    """
    Extract annual average concentration data from an AERMOD .ADO file.

    Args:
        ado_file_path: Path to AERMOD .ADO file
        aermod_crs: Coordinate reference system for AERMOD data

    Returns:
        GeoDataFrame with annual average concentrations, or None if no data found
    """
    results, _ = _extract_aermod_sections(ado_file_path, ['ANNUAL_AVERAGE'])

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
        Tuple of (intercept, coef_landing, coef_takeoff) for use with percentile-ranked predictors
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


def apply_calibration(
    landing_combined: Optional[gpd.GeoDataFrame],
    takeoff_combined: Optional[gpd.GeoDataFrame],
    calibration_file: Union[str, Path],
    receptor_match_tolerance: float = 1.0,
) -> Optional[gpd.GeoDataFrame]:
    """
    Calibrate AERMOD CO to UFP with one log-linear model on **percentile-ranked** concentrations:

        log(UFP) = intercept
                   + coef_landing * rank_pct(landing CO)
                   + coef_takeoff * rank_pct(takeoff CO)
        UFP = exp(log(UFP))

    Percentiles are ``Series.rank(pct=True)`` computed **within** landing and **within** takeoff
    separately. When both flows exist, rows use landing geometry; takeoff percentile is taken from
    the nearest takeoff receptor within ``receptor_match_tolerance`` m (else takeoff term uses 0).

    Landing only: log(UFP) = intercept + coef_landing * rank_pct(landing).
    Takeoff only: log(UFP) = intercept + coef_takeoff * rank_pct(takeoff).

    Returns:
        GeoDataFrame with ``ufp_concentration`` and the reference flow geometry, or None if both
        inputs are None.
    """
    intercept, coef_landing, coef_takeoff = load_calibration_coefficients(calibration_file)

    if landing_combined is None and takeoff_combined is None:
        return None

    if landing_combined is not None and takeoff_combined is not None:
        ref = landing_combined.copy()
        L_pct = ref['concentration'].rank(pct=True).to_numpy(dtype=float)
        takeoff_pct = takeoff_combined['concentration'].rank(pct=True).to_numpy(dtype=float)
        coords_l = np.array([[g.x, g.y] for g in ref.geometry])
        coords_t = np.array([[g.x, g.y] for g in takeoff_combined.geometry])
        tree = cKDTree(coords_t)
        dist, idx = tree.query(coords_l, k=1)
        T_pct = np.where(dist <= receptor_match_tolerance, takeoff_pct[idx], 0.0)
        log_ufp = intercept + coef_landing * L_pct + coef_takeoff * T_pct
        ref['ufp_concentration'] = np.exp(log_ufp)
        return ref

    if landing_combined is not None:
        ref = landing_combined.copy()
        L_pct = ref['concentration'].rank(pct=True).to_numpy(dtype=float)
        ref['ufp_concentration'] = np.exp(intercept + coef_landing * L_pct)
        return ref

    ref = takeoff_combined.copy()
    T_pct = ref['concentration'].rank(pct=True).to_numpy(dtype=float)
    ref['ufp_concentration'] = np.exp(intercept + coef_takeoff * T_pct)
    return ref


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


def _metric_crs_pair(receptor_gdf: gpd.GeoDataFrame, tract_geometries: gpd.GeoSeries):
    """
    If CRS is geographic, project receptors and tract geometries to estimated UTM for
    centroid and distance-based steps (avoids GeoPandas centroid warnings and degree-based IDW).
    """
    crs = receptor_gdf.crs
    if crs is None or not crs.is_geographic:
        return receptor_gdf, tract_geometries
    combined = gpd.GeoSeries(
        pd.concat([receptor_gdf.geometry, tract_geometries], ignore_index=True),
        crs=crs,
    )
    try:
        proj = combined.estimate_utm_crs()
    except (RuntimeError, IndexError, ValueError):
        proj = "EPSG:3857"
    return receptor_gdf.to_crs(proj), tract_geometries.to_crs(proj)


def aggregate_to_tracts(receptor_gdf: gpd.GeoDataFrame,
                       value_column: str,
                       tracts_gdf: gpd.GeoDataFrame,
                       method: str = 'spatial_join',
                       idw_power: int = 2,
                       idw_max_distance: Optional[float] = None,
                       idw_num_neighbors: Optional[int] = None,
                       **extra: Any) -> pd.DataFrame:
    """
    Aggregate receptor point values to census tracts.
    
    For method='spatial_join':
        - Uses spatial join (average) for tracts with direct receptor intersections
        - Uses IDW interpolation for tracts without intersections
        - Uses nearest neighbor as fallback for isolated tracts
    
    For method='idw_interpolation':
        - Uses pure IDW interpolation to all tract centroids

    For method='polar' (Magali / Louie receptor-to-tract rule):
        - Zero receptors inside tract: assign value at the single nearest receptor to the
          tract centroid (Euclidean distance in projected CRS when using _metric_crs_pair).
        - One receptor inside: that receptor's value.
        - More than one inside: mean of those values.
        (Same as spatial_join for tracts with receptors; differs by using nearest neighbor
        instead of IDW when a tract has no receptor inside.)
        Legacy keys polar_center_xy, polar_center_crs, polar_ring_*, polar_extrapolation
        are accepted and ignored for backward compatibility.
    
    Args:
        receptor_gdf: GeoDataFrame with receptor points and values
        value_column: Name of column containing values to aggregate
        tracts_gdf: GeoDataFrame with census tract geometries (must have GEOID)
        method: 'spatial_join', 'idw_interpolation', or 'polar'
        idw_power: Power parameter for IDW (if used)
        idw_max_distance: Maximum distance for IDW (if used)
        idw_num_neighbors: Number of neighbors for IDW (if used)
        **extra: Ignored legacy polar_* keys; any other key raises TypeError.
        
    Returns:
        DataFrame with GEOID and aggregated values
    """
    for _legacy in (
        'polar_center_xy',
        'polar_center_crs',
        'polar_ring_radii',
        'polar_ring_bin_m',
        'polar_snap_tol_m',
        'polar_extrapolation',
    ):
        extra.pop(_legacy, None)
    if extra:
        unknown = ", ".join(sorted(extra.keys()))
        raise TypeError(f"aggregate_to_tracts() got unexpected keyword arguments: {unknown}")
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
            rec_m, twd_m = _metric_crs_pair(receptor_gdf, tracts_without_data.geometry)
            receptor_coords = np.array([[g.x, g.y] for g in rec_m.geometry])
            receptor_values = receptor_gdf[value_column].values

            tract_centroids = twd_m.centroid
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
        tract_geom = tracts_gdf.geometry
        rec_m, tract_m = _metric_crs_pair(receptor_gdf, tract_geom)
        receptor_coords = np.array([[g.x, g.y] for g in rec_m.geometry])
        receptor_values = receptor_gdf[value_column].values

        tract_centroids = tract_m.centroid
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

    elif method == 'polar':
        points_in_tracts = gpd.sjoin(
            receptor_gdf[[value_column, 'geometry']],
            all_tracts,
            how='inner',
            predicate='within'
        )
        tracts_with_data = points_in_tracts.groupby('GEOID')[value_column].mean().reset_index()
        tracts_without_data = all_tracts[~all_tracts['GEOID'].isin(tracts_with_data['GEOID'])]

        if len(tracts_without_data) == 0:
            return tracts_with_data

        rec_m, twd_m = _metric_crs_pair(receptor_gdf, tracts_without_data.geometry)
        receptor_coords = np.array([[g.x, g.y] for g in rec_m.geometry])
        receptor_values = receptor_gdf[value_column].values
        tract_centroids = twd_m.centroid
        target_coords = np.array([[g.x, g.y] for g in tract_centroids])
        tree = cKDTree(receptor_coords)
        _, indices = tree.query(target_coords, k=1)
        nearest_values = receptor_values[indices]

        tracts_interpolated = pd.DataFrame({
            'GEOID': tracts_without_data['GEOID'].values,
            value_column: nearest_values
        })
        return pd.concat([tracts_with_data, tracts_interpolated], ignore_index=True)

    else:
        raise ValueError(
            f"Unknown aggregation method: {method}. "
            f"Must be 'spatial_join', 'idw_interpolation', or 'polar'"
        )


def generate_exposure_from_aermod(
    landing_files: Optional[List[Tuple[Union[str, Path], float]]],
    takeoff_files: Optional[List[Tuple[Union[str, Path], float]]],
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
    3. Apply one log-linear calibration on landing/takeoff **percentile ranks** to obtain UFP
    4. Aggregate UFP to census tracts

    Args:
        landing_files: List of (file_path, weight) tuples for landing flows, or None to skip
        takeoff_files: List of (file_path, weight) tuples for takeoff flows, or None to skip
        tracts_gdf: GeoDataFrame with census tract geometries (must have GEOID)
        calibration_file: Path to JSON file with calibration coefficients
        aermod_crs: Coordinate reference system for AERMOD data
        aggregation_method: 'spatial_join', 'idw_interpolation', or 'polar' (nearest-or-mean rule; see aggregate_to_tracts)
        idw_power: Power parameter for IDW interpolation
        idw_max_distance: Maximum distance for IDW interpolation (None = no limit)
        idw_num_neighbors: Number of neighbors for IDW interpolation (None = use all)
        receptor_match_tolerance: Meters; used for flow combination and for pairing takeoff
            percentile to landing receptors in ``apply_calibration``
        **aggregation_kwargs: Passed to aggregate_to_tracts (legacy polar_* keys are ignored)

    Returns:
        DataFrame with GEOID and ufp columns
    """
    if landing_files is None and takeoff_files is None:
        raise ValueError("At least one of 'landing_files' or 'takeoff_files' must be provided")
    
    logger.info("Starting AERMOD exposure generation workflow")
    
    landing_combined = None
    takeoff_combined = None
    
    # Step 1: Extract annual averages from landing files
    if landing_files is not None:
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
        
        # Step 2: Weighted combination of landing flows
        logger.info("Combining landing flows with weights")
        landing_weights = [w for _, w in landing_files]
        landing_combined = weighted_combine_flows(
            landing_gdfs, 
            landing_weights,
            aermod_crs=aermod_crs,
            tolerance=receptor_match_tolerance
        )
        logger.info(f"  Combined landing: {len(landing_combined)} receptor points")
    
    # Step 3: Extract annual averages from takeoff files
    if takeoff_files is not None:
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
        
        # Step 4: Weighted combination of takeoff flows
        logger.info("Combining takeoff flows with weights")
        takeoff_weights = [w for _, w in takeoff_files]
        takeoff_combined = weighted_combine_flows(
            takeoff_gdfs,
            takeoff_weights,
            aermod_crs=aermod_crs,
            tolerance=receptor_match_tolerance
        )
        logger.info(f"  Combined takeoff: {len(takeoff_combined)} receptor points")
    
    # Step 5: Apply calibration to convert CO to UFP
    logger.info("Applying calibration to convert CO to UFP")
    receptors_ufp = apply_calibration(
        landing_combined,
        takeoff_combined,
        calibration_file,
        receptor_match_tolerance=receptor_match_tolerance,
    )
    if receptors_ufp is None:
        raise ValueError("No exposure data generated from landing or takeoff files")
    logger.info(f"  Calibrated UFP at {len(receptors_ufp)} receptor points")

    # Step 6: Aggregate to census tracts
    logger.info(f"Aggregating to census tracts using method: {aggregation_method}")

    # Ensure tracts are in AERMOD CRS for spatial operations
    tracts_aermod_crs = tracts_gdf.to_crs(aermod_crs) if tracts_gdf.crs != aermod_crs else tracts_gdf.copy()

    tracts_exposure = aggregate_to_tracts(
        receptors_ufp,
        'ufp_concentration',
        tracts_aermod_crs,
        method=aggregation_method,
        idw_power=idw_power,
        idw_max_distance=idw_max_distance,
        idw_num_neighbors=idw_num_neighbors,
        **aggregation_kwargs
    )
    tracts_exposure = tracts_exposure.rename(columns={'ufp_concentration': 'ufp'})
    logger.info(f"  UFP aggregated to {len(tracts_exposure)} tracts")
    
    logger.info(f"Generated exposure data for {len(tracts_exposure)} tracts")
    logger.info(f"  Total UFP range: {tracts_exposure['ufp'].min():.6f} - {tracts_exposure['ufp'].max():.6f}")
    
    # Return only GEOID and pollutant column
    return tracts_exposure[['GEOID', 'ufp']]


def load_exposure_from_aermod(
    aermod_file_path: Union[str, Path, List[Union[str, Path]]],
    tracts_gdf: gpd.GeoDataFrame,
    section_types: Optional[List[str]] = None,
    center_location: Optional[Tuple[float, float]] = None,
    center_crs: Optional[str] = None,
    aermod_crs: Optional[str] = None,
    pollutant_name: str = 'ufp'
) -> pd.DataFrame:
    """
    Load baseline exposure data from one or more AERMOD files.
    
    This function:
    1. Parses AERMOD file(s) to extract concentration data
    2. Handles coordinate transformations for polar coordinates (if center provided)
    3. Creates point geometries from x_coord, y_coord
    4. Intersects points with tracts
    5. Averages concentrations within each tract
    6. If multiple files, sums the annual average values across all files
    
    Args:
        aermod_file_path: Path to AERMOD .OUT or .ADO file, or list of paths.
                         If multiple files, annual averages are summed.
        tracts_gdf: GeoDataFrame with census tract geometries (must have GEOID)
        section_types: Optional list of section types to extract.
                      If None, extracts 'ANNUAL_AVERAGE' only.
                      Options: 'ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST'
        center_location: Optional tuple (x, y) of center point in center_crs.
                       Required for polar coordinate systems to properly convert coordinates.
        center_crs: CRS of the center_location (e.g., 'EPSG:4326' for lat/lon).
                   If None, uses tracts_gdf.crs.
        aermod_crs: CRS of the AERMOD coordinate data (e.g., 'EPSG:32616' for UTM).
                   If None, assumes same as center_crs or tracts_gdf.crs.
        pollutant_name: Name of the pollutant column in output (default: 'ufp')
        
    Returns:
        DataFrame with GEOID as index and pollutant column
    """
    # Normalize to list of paths
    if isinstance(aermod_file_path, (str, Path)):
        file_paths = [aermod_file_path]
    else:
        file_paths = list(aermod_file_path)
    
    if len(file_paths) == 0:
        raise ValueError("At least one AERMOD file path must be provided")
    
    logger.info(f"Loading baseline exposure from {len(file_paths)} AERMOD file(s)")
    
    # Set default CRS values
    if center_crs is None:
        center_crs = tracts_gdf.crs if tracts_gdf.crs else 'EPSG:4326'
    if aermod_crs is None:
        aermod_crs = center_crs
    
    if section_types is None:
        section_types = ['ANNUAL_AVERAGE']
    
    # Process each file and collect tract-level exposures
    all_tract_exposures = []
    
    for file_idx, file_path in enumerate(file_paths):
        logger.info(f"Processing file {file_idx + 1}/{len(file_paths)}: {file_path}")
        
        # Parse AERMOD file using the aermod_parser package
        results, network_info_by_id = _extract_aermod_sections(str(file_path), section_types)

        # Extract DataFrame from results dictionary
        aermod_df = pd.DataFrame()
        for section_type in section_types:
            if section_type in results and isinstance(results[section_type], pd.DataFrame):
                if len(aermod_df) == 0:
                    aermod_df = results[section_type]
                else:
                    aermod_df = pd.concat([aermod_df, results[section_type]], ignore_index=True)
        
        if len(aermod_df) == 0:
            logger.warning(f"No concentration data found in {file_path}, skipping")
            continue
        
        # Check if we have coordinates
        if 'x_coord' not in aermod_df.columns or 'y_coord' not in aermod_df.columns:
            logger.warning(f"Missing coordinates in {file_path}, skipping")
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
                if network_id in network_info_by_id:
                    network_info = network_info_by_id[network_id]
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
        
        # Create point geometries from coordinates
        if center_location is not None:
            points_crs = center_crs
        else:
            points_crs = aermod_crs
            if aermod_crs != tracts_gdf.crs:
                points_aermod = gpd.GeoDataFrame(
                    aermod_df,
                    geometry=gpd.points_from_xy(aermod_df['x_coord'], aermod_df['y_coord']),
                    crs=aermod_crs
                )
                points_tracts_crs = points_aermod.to_crs(tracts_gdf.crs)
                aermod_df['x_coord'] = points_tracts_crs.geometry.x.values
                aermod_df['y_coord'] = points_tracts_crs.geometry.y.values
                points_crs = tracts_gdf.crs
        
        points_gdf = gpd.GeoDataFrame(
            aermod_df,
            geometry=gpd.points_from_xy(aermod_df['x_coord'], aermod_df['y_coord']),
            crs=points_crs
        )
        
        # Transform to tract CRS if needed
        if points_gdf.crs != tracts_gdf.crs:
            logger.info(f"Transforming AERMOD points from {points_gdf.crs} to {tracts_gdf.crs}")
            points_gdf = points_gdf.to_crs(tracts_gdf.crs)
        
        # Spatial join with tracts
        points_in_tracts = gpd.sjoin(
            points_gdf[['concentration', 'geometry']],
            tracts_gdf.reset_index() if 'GEOID' in tracts_gdf.index.names else tracts_gdf,
            how='inner',
            predicate='within'
        )
        
        # Group by GEOID and average concentrations for this file
        tract_exposure = points_in_tracts.groupby('GEOID')['concentration'].mean().reset_index()
        tract_exposure.columns = ['GEOID', pollutant_name]
        all_tract_exposures.append(tract_exposure)
    
    # Combine all tract exposures
    if len(all_tract_exposures) == 0:
        raise ValueError("No concentration data found in any AERMOD file")
    
    # Sum concentrations across all files by GEOID
    combined_exposure = all_tract_exposures[0].set_index('GEOID')
    for tract_exposure in all_tract_exposures[1:]:
        tract_exposure_idx = tract_exposure.set_index('GEOID')
        combined_exposure = combined_exposure.add(
            tract_exposure_idx, 
            fill_value=0.0
        )
    
    # Align with tract geometries (allow missing GEOIDs, will fill with mean)
    tract_geoids = set(tracts_gdf.index if isinstance(tracts_gdf.index, pd.Index) else tracts_gdf['GEOID'].astype(int))
    exposure_geoids = set(combined_exposure.index.astype(int))
    
    # Check for missing GEOIDs
    missing_geoids = tract_geoids - exposure_geoids
    if missing_geoids:
        logger.warning(
            f"{len(missing_geoids)} tracts missing exposure data from AERMOD, "
            f"will fill with mean"
        )
    
    # Reindex to match all tract GEOIDs
    exposure_df = combined_exposure.reindex(tract_geoids)
    
    # Fill missing tracts with mean
    missing = exposure_df[pollutant_name].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} tracts missing exposure data from AERMOD, filling with mean")
        mean_exposure = exposure_df[pollutant_name].mean()
        exposure_df[pollutant_name] = exposure_df[pollutant_name].fillna(mean_exposure)
    
    exposure_df.index.name = 'GEOID'
    exposure_df = exposure_df.reset_index()
    
    logger.info(
        f"Loaded baseline exposure from {len(file_paths)} AERMOD file(s): "
        f"{len(exposure_df)} tracts, summed concentrations"
    )
    
    return exposure_df
