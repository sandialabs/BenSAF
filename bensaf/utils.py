import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union, Tuple, Optional, List, Dict
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def create_synthetic_data(tracts_gdf: gpd.GeoDataFrame):
    """
    Create synthetic data for demonstration purposes using existing tract geometries.
    
    Args:
        tracts_gdf: GeoDataFrame with existing tract geometries and optional COUNTYFP column
        
    Returns:
        Tuple of (tracts_gdf, exposure_df, mortality_df)
    """
    logger.info("Creating synthetic data")
    
    # Validate input
    if not isinstance(tracts_gdf, gpd.GeoDataFrame):
        raise ValueError("tracts_gdf must be a GeoDataFrame")
    
    if tracts_gdf.empty:
        raise ValueError("tracts_gdf cannot be empty")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use existing GeoDataFrame as basis
    logger.info(f"Using existing GeoDataFrame with {len(tracts_gdf)} tracts")
    
    # Get tract IDs from the existing GeoDataFrame
    if 'GEOID' in tracts_gdf.columns:
        tract_ids = tracts_gdf['GEOID'].tolist()
    else:
        # Use index if GEOID column doesn't exist
        tract_ids = [f"1400000US{i:08d}" for i in tracts_gdf.index]
    
    # Calculate centroids for distance calculations
    centroids = tracts_gdf.geometry.centroid
    x_coords = centroids.x.values
    y_coords = centroids.y.values
    
    # Use existing geometries
    geometries = tracts_gdf.geometry.tolist()
    
    num_tracts = len(tracts_gdf)
    
    # Calculate distance from airport (use centroid of study area as airport location)
    airport_x, airport_y = np.mean(x_coords), np.mean(y_coords)
    distances = np.sqrt((x_coords - airport_x)**2 + (y_coords - airport_y)**2)
    
    # Create population data (higher near airport)
    base_population = 5000
    population = (base_population * (1 + np.random.normal(0, 0.3, num_tracts))).astype(int)
    
    # Create demographic data
    poc_proportion = 0.3 + 0.4 * np.exp(-distances/5) + np.random.normal(0, 0.1, num_tracts)
    poc_proportion = np.clip(poc_proportion, 0.05, 0.95)
    
    low_income_proportion = 0.2 + 0.3 * np.exp(-distances/6) + np.random.normal(0, 0.1, num_tracts)
    low_income_proportion = np.clip(low_income_proportion, 0.05, 0.9)
    
    # Create pollutant concentration (higher near airport)
    pollutant_concentration = 1000 * np.exp(-distances/3) + np.random.normal(0, 100, num_tracts)
    pollutant_concentration = np.clip(pollutant_concentration, 50, 2000)
    
    # Create county-based natural mortality rates (NMR)
    # Generate county-level mortality rates first, then apply to tracts
    if 'COUNTYFP' in tracts_gdf.columns:
        # Use existing county information
        county_codes = tracts_gdf['COUNTYFP'].unique()
        logger.info(f"Found {len(county_codes)} counties in data")
        
        # Generate county-level mortality rates
        base_mortality_rate = 0.007  # 700 per 100,000 baseline
        county_mortality_rates = {}
        
        for county in county_codes:
            # Create realistic variation between counties
            # Higher rates in more urban/polluted areas, lower in rural areas
            county_factor = np.random.normal(1.0, 0.2)  # ±20% variation
            county_mortality_rate = base_mortality_rate * county_factor
            county_mortality_rate = np.clip(county_mortality_rate, 0.005, 0.012)  # 500-1200 per 100,000
            county_mortality_rates[county] = county_mortality_rate
        
        # Apply county rates to tracts
        mortality_rate = np.array([county_mortality_rates[county] for county in tracts_gdf['COUNTYFP']])
        
        logger.info(f"Generated county-level mortality rates: {list(county_mortality_rates.values())}")
        
    else:
        # Fallback: create synthetic county structure and mortality rates
        logger.info("No COUNTYFP found, creating synthetic county structure")
        
        # Create synthetic counties (assuming ~10 counties for demonstration)
        num_counties = min(10, max(1, num_tracts // 20))  # Roughly 20 tracts per county
        synthetic_counties = [f"{i:03d}" for i in range(num_counties)]
        
        # Assign tracts to counties
        tracts_per_county = num_tracts // num_counties
        county_assignments = []
        for i in range(num_counties):
            if i == num_counties - 1:  # Last county gets remaining tracts
                county_assignments.extend([synthetic_counties[i]] * (num_tracts - len(county_assignments)))
            else:
                county_assignments.extend([synthetic_counties[i]] * tracts_per_county)
        
        # Generate county-level mortality rates
        base_mortality_rate = 0.007  # 700 per 100,000 baseline
        county_mortality_rates = {}
        
        for county in synthetic_counties:
            # Create realistic variation between counties
            county_factor = np.random.normal(1.0, 0.2)  # ±20% variation
            county_mortality_rate = base_mortality_rate * county_factor
            county_mortality_rate = np.clip(county_mortality_rate, 0.005, 0.012)  # 500-1200 per 100,000
            county_mortality_rates[county] = county_mortality_rate
        
        # Apply county rates to tracts
        mortality_rate = np.array([county_mortality_rates[county] for county in county_assignments])
        
        # Add COUNTYFP to the tracts_gdf
        tracts_gdf = tracts_gdf.copy()
        tracts_gdf['COUNTYFP'] = county_assignments
        
        logger.info(f"Created {num_counties} synthetic counties with mortality rates: {list(county_mortality_rates.values())}")
    
    # Update existing GeoDataFrame with new synthetic data
    tracts_gdf = tracts_gdf.copy()
    tracts_gdf['population'] = population
    tracts_gdf['poc_proportion'] = poc_proportion
    tracts_gdf['poc_population'] = population * poc_proportion
    tracts_gdf['nonpoc_population'] = population * (1 - poc_proportion)
    tracts_gdf['low_income_proportion'] = low_income_proportion
    tracts_gdf['low_income_population'] = population * low_income_proportion
    tracts_gdf['not_low_income_population'] = population * (1 - low_income_proportion)
    tracts_gdf['distance_from_airport'] = distances
    
    # Create exposure DataFrame
    exposure_df = pd.DataFrame({
        'GEOID': tract_ids,
        'pollutant_concentration': pollutant_concentration
    })
    
    # Create mortality DataFrame
    mortality_df = pd.DataFrame({
        'GEOID': tract_ids,
        'mortality_rate': mortality_rate
    })
    
    logger.info(f"Created synthetic data with {num_tracts} census tracts")
    return tracts_gdf, exposure_df, mortality_df

def calculate_weighted_ufp(ufp_estimata, weight_col):
    """
    Calculate weighted UFP based on a weight column.
    
    Args:
        ufp_estimata: Series of UFP estimates
        weight_col: Column name for weights
        
    Returns:
        Series of weighted UFP estimates
    """
    return sum(ufp_estimata * weight_col) / sum(weight_col)


def bin_tracts_by_distance(
    tracts_gdf: gpd.GeoDataFrame,
    point_location: Union[Tuple[float, float], gpd.GeoSeries],
    distance_bins: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None,
    distance_col: str = 'distance_from_point',
    bin_col: str = 'distance_bin'
) -> gpd.GeoDataFrame:
    """
    Bin census tracts based on distance from a specific location.
    
    This function calculates the distance from each tract centroid to a specified
    point location and assigns each tract to a distance bin. This is useful for
    analyzing how health impacts, exposures, or demographic characteristics vary
    with distance from point sources like airports.
    
    Assumes all data is in EPSG:4326 (WGS84) coordinate system.
    
    Args:
        tracts_gdf: GeoDataFrame containing census tract data with geometry (EPSG:4326)
        point_location: Location to measure distance from. Can be:
            - Tuple of (longitude, latitude) in WGS84
            - GeoSeries with a single point geometry
        distance_bins: List of distance values defining bin edges (in km).
            If None, uses default bins: [0, 2, 5, 10, 20, 50]
        bin_labels: List of labels for each bin. Must be one fewer than distance_bins.
            If None, generates labels like "0-2 km", "2-5 km", etc.
        distance_col: Name of the column to store calculated distances
        bin_col: Name of the column to store bin assignments
        
    Returns:
        GeoDataFrame with added distance and bin columns
        
    Example:
        ```python
        # Bin tracts by distance from airport
        airport_location = (-122.3088, 47.4502)  # Seattle-Tacoma Airport
        tracts_with_bins = bin_tracts_by_distance(
            tracts_gdf, 
            airport_location,
            distance_bins=[0, 2, 5, 10, 20, 50],
            bin_labels=['0-2 km', '2-5 km', '5-10 km', '10-20 km', '20+ km']
        )
        
        # Analyze health impacts by distance bin
        impacts_by_distance = tracts_with_bins.groupby('distance_bin')['attributable_cases'].sum()
        ```
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not isinstance(tracts_gdf, gpd.GeoDataFrame):
        raise ValueError("tracts_gdf must be a GeoDataFrame")
    
    if tracts_gdf.empty:
        raise ValueError("tracts_gdf cannot be empty")
    
    # Set default distance bins if not provided
    if distance_bins is None:
        distance_bins = [0, 2, 5, 10, 20, 50]
    
    # Validate distance bins
    if len(distance_bins) < 2:
        raise ValueError("distance_bins must have at least 2 values")
    
    if not all(distance_bins[i] <= distance_bins[i+1] for i in range(len(distance_bins)-1)):
        raise ValueError("distance_bins must be in ascending order")
    
    # Generate bin labels if not provided
    if bin_labels is None:
        bin_labels = []
        for i in range(len(distance_bins) - 1):
            if i == len(distance_bins) - 2:  # Last bin
                bin_labels.append(f"{distance_bins[i]}+ km")
            else:
                bin_labels.append(f"{distance_bins[i]}-{distance_bins[i+1]} km")
    
    # Validate bin labels
    if len(bin_labels) != len(distance_bins) - 1:
        raise ValueError(f"bin_labels must have {len(distance_bins) - 1} elements")
    
    # Create a copy to avoid modifying the original
    result_gdf = tracts_gdf.copy()
    
    # Handle point location input
    if isinstance(point_location, tuple):
        # Convert tuple to GeoSeries
        from shapely.geometry import Point
        point_geom = Point(point_location[0], point_location[1])
        point_gdf = gpd.GeoDataFrame([1], geometry=[point_geom], crs="EPSG:4326")
        point_location = point_gdf.geometry.iloc[0]
    
    # Ensure tract data is in EPSG:4326
    if result_gdf.crs != "EPSG:4326":
        logger.warning(f"Tract data is in {result_gdf.crs}, reprojecting to EPSG:4326")
        result_gdf = result_gdf.to_crs("EPSG:4326")
    
    # Ensure point location is in EPSG:4326
    if hasattr(point_location, 'crs') and point_location.crs != "EPSG:4326":
        logger.warning(f"Point location is in {point_location.crs}, reprojecting to EPSG:4326")
        point_location = point_location.to_crs("EPSG:4326")
    
    # Calculate centroids of tracts
    centroids = result_gdf.geometry.centroid
    
    # Calculate distances from point location to each centroid using haversine formula
    # for accurate geographic distance calculations
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points using haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    # Extract coordinates
    point_lat, point_lon = point_location.y, point_location.x
    centroid_lats = centroids.y.values
    centroid_lons = centroids.x.values
    
    # Calculate distances using haversine formula
    distances = np.array([
        haversine_distance(point_lat, point_lon, lat, lon) 
        for lat, lon in zip(centroid_lats, centroid_lons)
    ])
    
    # Add distance column
    result_gdf[distance_col] = distances
    
    # Assign bins based on distances
    bin_assignments = pd.cut(
        distances, 
        bins=distance_bins, 
        labels=bin_labels, 
        include_lowest=True
    )
    
    # Add bin column
    result_gdf[bin_col] = bin_assignments
    
    # Log summary statistics
    logger.info(f"Distance statistics:")
    logger.info(f"  Min distance: {distances.min():.2f} km")
    logger.info(f"  Max distance: {distances.max():.2f} km")
    logger.info(f"  Mean distance: {distances.mean():.2f} km")
    
    bin_counts = result_gdf[bin_col].value_counts().sort_index()
    logger.info(f"Tracts per distance bin:")
    for bin_name, count in bin_counts.items():
        logger.info(f"  {bin_name}: {count} tracts")
    
    return result_gdf


def analyze_impacts_by_distance(
    tracts_gdf: gpd.GeoDataFrame,
    impact_col: str,
    distance_bin_col: str = 'distance_bin',
    population_col: str = 'population',
    include_rates: bool = True
) -> pd.DataFrame:
    """
    Analyze health impacts by distance bin.
    
    This function provides summary statistics for health impacts across different
    distance bins, including total cases, population, and rates.
    
    Args:
        tracts_gdf: GeoDataFrame with health impact data and distance bins
        impact_col: Column name containing health impact values (e.g., 'attributable_cases')
        distance_bin_col: Column name containing distance bin assignments
        population_col: Column name containing population counts
        include_rates: Whether to include per-capita rates in the analysis
        
    Returns:
        DataFrame with summary statistics by distance bin
        
    Example:
        ```python
        # Analyze attributable cases by distance from airport
        impact_summary = analyze_impacts_by_distance(
            tracts_with_bins,
            impact_col='attributable_cases',
            distance_bin_col='distance_bin'
        )
        print(impact_summary)
        ```
    """
    if distance_bin_col not in tracts_gdf.columns:
        raise ValueError(f"Distance bin column '{distance_bin_col}' not found in data")
    
    if impact_col not in tracts_gdf.columns:
        raise ValueError(f"Impact column '{impact_col}' not found in data")
    
    if population_col not in tracts_gdf.columns:
        raise ValueError(f"Population column '{population_col}' not found in data")
    
    # Group by distance bin and calculate summary statistics
    summary = tracts_gdf.groupby(distance_bin_col).agg({
        impact_col: ['sum', 'mean', 'std', 'min', 'max'],
        population_col: 'sum'
    }).round(4)
    
    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
    
    # Rename population column
    summary = summary.rename(columns={f'{population_col}_sum': 'total_population'})
    
    # Calculate rates if requested
    if include_rates:
        summary['impact_rate_per_100k'] = (
            summary[f'{impact_col}_sum'] / summary['total_population'] * 100000
        ).round(2)
        
        summary['impact_rate_per_capita'] = (
            summary[f'{impact_col}_sum'] / summary['total_population']
        ).round(6)
    
    # Reorder columns for better readability
    col_order = [
        'total_population',
        f'{impact_col}_sum',
        f'{impact_col}_mean',
        f'{impact_col}_std',
        f'{impact_col}_min',
        f'{impact_col}_max'
    ]
    
    if include_rates:
        col_order.extend(['impact_rate_per_100k', 'impact_rate_per_capita'])
    
    summary = summary[col_order]
    
    return summary


def create_distance_analysis_plots(
    tracts_gdf: gpd.GeoDataFrame,
    impact_col: str,
    distance_bin_col: str = 'distance_bin',
    population_col: str = 'population',
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, plt.Figure]:
    """
    Create visualization plots for distance-based analysis.
    
    Args:
        tracts_gdf: GeoDataFrame with health impact data and distance bins
        impact_col: Column name containing health impact values
        distance_bin_col: Column name containing distance bin assignments
        population_col: Column name containing population counts
        output_dir: Optional directory to save plots
        figsize: Figure size for plots
        
    Returns:
        Dictionary of matplotlib figures
    """
    import matplotlib.pyplot as plt
    
    figures = {}
    
    # 1. Bar plot of total impacts by distance bin
    fig, ax = plt.subplots(figsize=figsize)
    
    impact_summary = analyze_impacts_by_distance(
        tracts_gdf, impact_col, distance_bin_col, population_col
    )
    
    impact_summary[f'{impact_col}_sum'].plot(kind='bar', ax=ax)
    ax.set_xlabel('Distance from Point Source')
    ax.set_ylabel(f'Total {impact_col.replace("_", " ").title()}')
    ax.set_title(f'{impact_col.replace("_", " ").title()} by Distance')
    ax.tick_params(axis='x', rotation=45)
    
    figures['total_impacts_by_distance'] = fig
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'total_impacts_by_distance.png', dpi=300, bbox_inches='tight')
    
    # 2. Rate plot (impacts per 100,000 population)
    fig, ax = plt.subplots(figsize=figsize)
    
    impact_summary['impact_rate_per_100k'].plot(kind='bar', ax=ax)
    ax.set_xlabel('Distance from Point Source')
    ax.set_ylabel(f'{impact_col.replace("_", " ").title()} per 100,000 Population')
    ax.set_title(f'{impact_col.replace("_", " ").title()} Rate by Distance')
    ax.tick_params(axis='x', rotation=45)
    
    figures['impact_rate_by_distance'] = fig
    
    if output_dir:
        fig.savefig(output_dir / 'impact_rate_by_distance.png', dpi=300, bbox_inches='tight')
    
    # 3. Population distribution by distance
    fig, ax = plt.subplots(figsize=figsize)
    
    impact_summary['total_population'].plot(kind='bar', ax=ax)
    ax.set_xlabel('Distance from Point Source')
    ax.set_ylabel('Total Population')
    ax.set_title('Population Distribution by Distance')
    ax.tick_params(axis='x', rotation=45)
    
    figures['population_by_distance'] = fig
    
    if output_dir:
        fig.savefig(output_dir / 'population_by_distance.png', dpi=300, bbox_inches='tight')
    
    # 4. Scatter plot of distance vs impact rate
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate impact rate for each tract
    tract_rates = tracts_gdf[impact_col] / tracts_gdf[population_col] * 100000
    
    # Get numeric distance values for x-axis
    distance_numeric = tracts_gdf['distance_from_point'] if 'distance_from_point' in tracts_gdf.columns else None
    
    if distance_numeric is not None:
        ax.scatter(distance_numeric, tract_rates, alpha=0.6)
        ax.set_xlabel('Distance from Point Source (km)')
        ax.set_ylabel(f'{impact_col.replace("_", " ").title()} per 100,000 Population')
        ax.set_title(f'{impact_col.replace("_", " ").title()} Rate vs Distance')
        
        # Add trend line
        z = np.polyfit(distance_numeric, tract_rates, 1)
        p = np.poly1d(z)
        ax.plot(distance_numeric, p(distance_numeric), "r--", alpha=0.8)
        
        figures['impact_rate_vs_distance'] = fig
        
        if output_dir:
            fig.savefig(output_dir / 'impact_rate_vs_distance.png', dpi=300, bbox_inches='tight')
    
    return figures