"""
Geographic analysis utilities for SAF health impact assessment.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


def bin_tracts_by_distance(
    tracts_gdf: gpd.GeoDataFrame,
    point_location: Union[Tuple[float, float], gpd.GeoSeries],
    distance_bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None,
    distance_col: str = 'distance_from_point',
    bin_col: str = 'distance_bin',
) -> gpd.GeoDataFrame:
    """
    Bin census tracts based on distance from a specific location.

    Calculates the haversine distance from each tract centroid to a specified
    point and assigns each tract to a distance bin.

    Assumes data is in EPSG:4326 (WGS84).

    Args:
        tracts_gdf: GeoDataFrame with census tract geometries (EPSG:4326)
        point_location: (longitude, latitude) tuple or GeoSeries point
        distance_bins: Bin edges in km. Defaults to [0, 2, 5, 10, 20, 50].
        bin_labels: Labels for each bin (len = len(distance_bins) - 1).
        distance_col: Column name for calculated distances.
        bin_col: Column name for bin assignments.

    Returns:
        GeoDataFrame with added distance and bin columns.
    """
    if not isinstance(tracts_gdf, gpd.GeoDataFrame):
        raise ValueError("tracts_gdf must be a GeoDataFrame")
    if tracts_gdf.empty:
        raise ValueError("tracts_gdf cannot be empty")

    if distance_bins is None:
        distance_bins = [0, 2, 5, 10, 20, 50]

    if len(distance_bins) < 2:
        raise ValueError("distance_bins must have at least 2 values")
    if not all(distance_bins[i] <= distance_bins[i + 1] for i in range(len(distance_bins) - 1)):
        raise ValueError("distance_bins must be in ascending order")

    if bin_labels is None:
        bin_labels = []
        for i in range(len(distance_bins) - 1):
            if i == len(distance_bins) - 2:
                bin_labels.append(f"{distance_bins[i]}+ km")
            else:
                bin_labels.append(f"{distance_bins[i]}-{distance_bins[i + 1]} km")

    if len(bin_labels) != len(distance_bins) - 1:
        raise ValueError(f"bin_labels must have {len(distance_bins) - 1} elements")

    result_gdf = tracts_gdf.copy()

    if isinstance(point_location, tuple):
        from shapely.geometry import Point
        point_geom = Point(point_location[0], point_location[1])
        point_gdf = gpd.GeoDataFrame([1], geometry=[point_geom], crs="EPSG:4326")
        point_location = point_gdf.geometry.iloc[0]

    if result_gdf.crs != "EPSG:4326":
        logger.warning(f"Tract data is in {result_gdf.crs}, reprojecting to EPSG:4326")
        result_gdf = result_gdf.to_crs("EPSG:4326")

    if hasattr(point_location, 'crs') and point_location.crs != "EPSG:4326":
        logger.warning(f"Point location is in {point_location.crs}, reprojecting to EPSG:4326")
        point_location = point_location.to_crs("EPSG:4326")

    centroids = result_gdf.geometry.centroid
    point_lat, point_lon = point_location.y, point_location.x
    centroid_lats = centroids.y.values
    centroid_lons = centroids.x.values

    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    distances = np.array([
        _haversine(point_lat, point_lon, lat, lon)
        for lat, lon in zip(centroid_lats, centroid_lons)
    ])

    result_gdf[distance_col] = distances
    result_gdf[bin_col] = pd.cut(
        distances, bins=distance_bins, labels=bin_labels, include_lowest=True
    )

    logger.info(
        f"Distance statistics: min={distances.min():.2f} km, "
        f"max={distances.max():.2f} km, mean={distances.mean():.2f} km"
    )
    for bin_name, count in result_gdf[bin_col].value_counts().sort_index().items():
        logger.info(f"  {bin_name}: {count} tracts")

    return result_gdf


def analyze_impacts_by_distance(
    tracts_gdf: gpd.GeoDataFrame,
    impact_col: str,
    distance_bin_col: str = 'distance_bin',
    population_col: str = 'population',
    include_rates: bool = True,
) -> pd.DataFrame:
    """
    Summarise health impacts by distance bin.

    Args:
        tracts_gdf: GeoDataFrame with health impact data and distance bins
        impact_col: Column containing health impact values
        distance_bin_col: Column containing distance bin assignments
        population_col: Column containing population counts
        include_rates: Whether to include per-capita rates

    Returns:
        DataFrame with summary statistics per distance bin.
    """
    for col in (distance_bin_col, impact_col, population_col):
        if col not in tracts_gdf.columns:
            raise ValueError(f"Column '{col}' not found in data")

    summary = tracts_gdf.groupby(distance_bin_col).agg(
        {impact_col: ['sum', 'mean', 'std', 'min', 'max'], population_col: 'sum'}
    ).round(4)
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]
    summary = summary.rename(columns={f'{population_col}_sum': 'total_population'})

    if include_rates:
        summary['impact_rate_per_100k'] = (
            summary[f'{impact_col}_sum'] / summary['total_population'] * 100000
        ).round(2)
        summary['impact_rate_per_capita'] = (
            summary[f'{impact_col}_sum'] / summary['total_population']
        ).round(6)

    col_order = [
        'total_population',
        f'{impact_col}_sum', f'{impact_col}_mean',
        f'{impact_col}_std', f'{impact_col}_min', f'{impact_col}_max',
    ]
    if include_rates:
        col_order.extend(['impact_rate_per_100k', 'impact_rate_per_capita'])

    return summary[col_order]


def create_distance_analysis_plots(
    tracts_gdf: gpd.GeoDataFrame,
    impact_col: str,
    distance_bin_col: str = 'distance_bin',
    population_col: str = 'population',
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Dict:
    """
    Create bar and scatter plots for distance-based impact analysis.

    Args:
        tracts_gdf: GeoDataFrame with health impact data and distance bins
        impact_col: Column containing health impact values
        distance_bin_col: Column containing distance bin assignments
        population_col: Column containing population counts
        output_dir: Optional directory to save PNG files
        figsize: Figure size

    Returns:
        Dict of matplotlib Figure objects keyed by plot name.
    """
    import matplotlib.pyplot as plt

    figures = {}
    impact_summary = analyze_impacts_by_distance(
        tracts_gdf, impact_col, distance_bin_col, population_col
    )

    def _bar(col, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=figsize)
        impact_summary[col].plot(kind='bar', ax=ax)
        ax.set_xlabel('Distance from Point Source')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(output_dir) / filename, dpi=300, bbox_inches='tight')
        return fig

    label = impact_col.replace('_', ' ').title()
    figures['total_impacts_by_distance'] = _bar(
        f'{impact_col}_sum', f'Total {label}', f'{label} by Distance',
        'total_impacts_by_distance.png',
    )
    figures['impact_rate_by_distance'] = _bar(
        'impact_rate_per_100k', f'{label} per 100,000 Population',
        f'{label} Rate by Distance', 'impact_rate_by_distance.png',
    )
    figures['population_by_distance'] = _bar(
        'total_population', 'Total Population',
        'Population Distribution by Distance', 'population_by_distance.png',
    )

    distance_numeric = tracts_gdf.get('distance_from_point')
    if distance_numeric is not None:
        fig, ax = plt.subplots(figsize=figsize)
        tract_rates = tracts_gdf[impact_col] / tracts_gdf[population_col] * 100000
        ax.scatter(distance_numeric, tract_rates, alpha=0.6)
        ax.set_xlabel('Distance from Point Source (km)')
        ax.set_ylabel(f'{label} per 100,000 Population')
        ax.set_title(f'{label} Rate vs Distance')
        z = np.polyfit(distance_numeric, tract_rates, 1)
        ax.plot(distance_numeric, np.poly1d(z)(distance_numeric), "r--", alpha=0.8)
        figures['impact_rate_vs_distance'] = fig
        if output_dir:
            fig.savefig(
                Path(output_dir) / 'impact_rate_vs_distance.png', dpi=300, bbox_inches='tight'
            )

    return figures
