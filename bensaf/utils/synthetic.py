"""
Synthetic data generation utilities.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


def calculate_weighted_ufp(ufp_estimata, weight_col):
    """Calculate weighted UFP based on a weight column."""
    return sum(ufp_estimata * weight_col) / sum(weight_col)


def create_synthetic_data(tracts_gdf: gpd.GeoDataFrame) -> Tuple:
    """
    Create synthetic exposure and mortality data for demonstration purposes.

    Args:
        tracts_gdf: GeoDataFrame with existing tract geometries and optional COUNTYFP column.

    Returns:
        Tuple of (tracts_gdf, exposure_df, mortality_df)
    """
    logger.info("Creating synthetic data")

    if not isinstance(tracts_gdf, gpd.GeoDataFrame):
        raise ValueError("tracts_gdf must be a GeoDataFrame")
    if tracts_gdf.empty:
        raise ValueError("tracts_gdf cannot be empty")

    np.random.seed(42)
    num_tracts = len(tracts_gdf)
    logger.info(f"Using existing GeoDataFrame with {num_tracts} tracts")

    tract_ids = (
        tracts_gdf['GEOID'].tolist()
        if 'GEOID' in tracts_gdf.columns
        else [f"1400000US{i:08d}" for i in tracts_gdf.index]
    )

    centroids = tracts_gdf.geometry.centroid
    x_coords = centroids.x.values
    y_coords = centroids.y.values

    airport_x, airport_y = np.mean(x_coords), np.mean(y_coords)
    distances = np.sqrt((x_coords - airport_x) ** 2 + (y_coords - airport_y) ** 2)

    population = (5000 * (1 + np.random.normal(0, 0.3, num_tracts))).astype(int)

    poc_proportion = np.clip(
        0.3 + 0.4 * np.exp(-distances / 5) + np.random.normal(0, 0.1, num_tracts), 0.05, 0.95
    )
    low_income_proportion = np.clip(
        0.2 + 0.3 * np.exp(-distances / 6) + np.random.normal(0, 0.1, num_tracts), 0.05, 0.9
    )
    pollutant_concentration = np.clip(
        1000 * np.exp(-distances / 3) + np.random.normal(0, 100, num_tracts), 50, 2000
    )

    if 'COUNTYFP' in tracts_gdf.columns:
        county_codes = tracts_gdf['COUNTYFP'].unique()
        base_rate = 0.007
        county_rates = {
            c: np.clip(base_rate * np.random.normal(1.0, 0.2), 0.005, 0.012)
            for c in county_codes
        }
        mortality_rate = np.array([county_rates[c] for c in tracts_gdf['COUNTYFP']])
    else:
        num_counties = min(10, max(1, num_tracts // 20))
        synthetic_counties = [f"{i:03d}" for i in range(num_counties)]
        tracts_per_county = num_tracts // num_counties
        assignments = []
        for i, county in enumerate(synthetic_counties):
            n = num_tracts - len(assignments) if i == num_counties - 1 else tracts_per_county
            assignments.extend([county] * n)
        base_rate = 0.007
        county_rates = {
            c: np.clip(base_rate * np.random.normal(1.0, 0.2), 0.005, 0.012)
            for c in synthetic_counties
        }
        mortality_rate = np.array([county_rates[c] for c in assignments])
        tracts_gdf = tracts_gdf.copy()
        tracts_gdf['COUNTYFP'] = assignments

    tracts_gdf = tracts_gdf.copy()
    tracts_gdf['population'] = population
    tracts_gdf['poc_proportion'] = poc_proportion
    tracts_gdf['poc_population'] = population * poc_proportion
    tracts_gdf['nonpoc_population'] = population * (1 - poc_proportion)
    tracts_gdf['low_income_proportion'] = low_income_proportion
    tracts_gdf['low_income_population'] = population * low_income_proportion
    tracts_gdf['not_low_income_population'] = population * (1 - low_income_proportion)
    tracts_gdf['distance_from_airport'] = distances

    exposure_df = pd.DataFrame(
        {'GEOID': tract_ids, 'pollutant_concentration': pollutant_concentration}
    )
    mortality_df = pd.DataFrame({'GEOID': tract_ids, 'mortality_rate': mortality_rate})

    logger.info(f"Created synthetic data with {num_tracts} census tracts")
    return tracts_gdf, exposure_df, mortality_df
