"""
Census tract data fetching utilities.
"""

import logging
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry import box

logger = logging.getLogger(__name__)

_STATE_FIPS_MAP = {
    'alabama': '01', 'al': '01',
    'alaska': '02', 'ak': '02',
    'arizona': '04', 'az': '04',
    'arkansas': '05', 'ar': '05',
    'california': '06', 'ca': '06',
    'colorado': '08', 'co': '08',
    'connecticut': '09', 'ct': '09',
    'delaware': '10', 'de': '10',
    'florida': '12', 'fl': '12',
    'georgia': '13', 'ga': '13',
    'hawaii': '15', 'hi': '15',
    'idaho': '16', 'id': '16',
    'illinois': '17', 'il': '17',
    'indiana': '18', 'in': '18',
    'iowa': '19', 'ia': '19',
    'kansas': '20', 'ks': '20',
    'kentucky': '21', 'ky': '21',
    'louisiana': '22', 'la': '22',
    'maine': '23', 'me': '23',
    'maryland': '24', 'md': '24',
    'massachusetts': '25', 'ma': '25',
    'michigan': '26', 'mi': '26',
    'minnesota': '27', 'mn': '27',
    'mississippi': '28', 'ms': '28',
    'missouri': '29', 'mo': '29',
    'montana': '30', 'mt': '30',
    'nebraska': '31', 'ne': '31',
    'nevada': '32', 'nv': '32',
    'new hampshire': '33', 'nh': '33',
    'new jersey': '34', 'nj': '34',
    'new mexico': '35', 'nm': '35',
    'new york': '36', 'ny': '36',
    'north carolina': '37', 'nc': '37',
    'north dakota': '38', 'nd': '38',
    'ohio': '39', 'oh': '39',
    'oklahoma': '40', 'ok': '40',
    'oregon': '41', 'or': '41',
    'pennsylvania': '42', 'pa': '42',
    'rhode island': '44', 'ri': '44',
    'south carolina': '45', 'sc': '45',
    'south dakota': '46', 'sd': '46',
    'tennessee': '47', 'tn': '47',
    'texas': '48', 'tx': '48',
    'utah': '49', 'ut': '49',
    'vermont': '50', 'vt': '50',
    'virginia': '51', 'va': '51',
    'washington': '53', 'wa': '53',
    'west virginia': '54', 'wv': '54',
    'wisconsin': '55', 'wi': '55',
    'wyoming': '56', 'wy': '56',
    'district of columbia': '11', 'dc': '11',
}


def fetch_census_tracts(
    state: Union[str, int],
    counties: Optional[Union[str, int, List[Union[str, int]]]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    year: int = 2022,
    cache_dir: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch census tracts for a state, optionally filtered by county or bounding box.

    Downloads TIGER/Line shapefiles from the Census Bureau and returns
    census tracts as a GeoDataFrame. Files are cached locally.

    Args:
        state: State FIPS code (2-digit string or int) or state name/abbreviation.
        counties: Optional county FIPS codes or names (single value or list).
        bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat).
        year: Census year for TIGER/Line data (default: 2022).
        cache_dir: Optional directory for cached downloads.

    Returns:
        GeoDataFrame with census tract geometries. CRS: EPSG:4269 (NAD83).
    """
    if isinstance(state, int):
        state_fips = f"{state:02d}"
    elif isinstance(state, str):
        state_lower = state.lower().strip()
        if state_lower in _STATE_FIPS_MAP:
            state_fips = _STATE_FIPS_MAP[state_lower]
        elif len(state) == 2 and state.isdigit():
            state_fips = state
        else:
            raise ValueError(
                f"Unknown state: {state}. Use FIPS code or state name/abbreviation."
            )
    else:
        raise ValueError("state must be a string or integer")

    county_fips_list = None
    if counties is not None:
        if isinstance(counties, (str, int)):
            county_fips_list = [f"{int(counties):03d}"]
        elif isinstance(counties, list):
            county_fips_list = [f"{int(c):03d}" for c in counties]
        else:
            raise ValueError("counties must be a string, int, or list")

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "bensaf_census_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    zip_file = cache_dir / f"tl_{year}_{state_fips}_tract.zip"

    if not zip_file.exists():
        logger.info(f"Downloading census tracts for state FIPS {state_fips}...")
        ssl_context = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)
        try:
            urllib.request.urlretrieve(url, zip_file)
        except Exception as e:
            raise RuntimeError(f"Failed to download census tracts: {e}")
        logger.info(f"Downloaded to {zip_file}")
    else:
        logger.info(f"Using cached file: {zip_file}")

    extract_dir = cache_dir / f"tl_{year}_{state_fips}_tract"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(extract_dir)

    shp_files = list(extract_dir.glob("*.shp"))
    if not shp_files:
        raise RuntimeError(f"No shapefile found in {extract_dir}")

    gdf = gpd.read_file(shp_files[0])
    logger.info(f"Loaded {len(gdf)} census tracts")

    if county_fips_list:
        if 'COUNTYFP' in gdf.columns:
            gdf = gdf[gdf['COUNTYFP'].isin(county_fips_list)]
            logger.info(f"Filtered to {len(gdf)} tracts in counties {county_fips_list}")
        else:
            logger.warning("COUNTYFP column not found, cannot filter by county")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        gdf = gdf[gdf.geometry.intersects(box(min_lon, min_lat, max_lon, max_lat))]
        logger.info(f"Filtered to {len(gdf)} tracts within bounding box")

    if 'GEOID' not in gdf.columns:
        if 'GEOID20' in gdf.columns:
            gdf['GEOID'] = gdf['GEOID20']
        elif all(c in gdf.columns for c in ('TRACTCE', 'STATEFP', 'COUNTYFP')):
            gdf['GEOID'] = gdf['STATEFP'] + gdf['COUNTYFP'] + gdf['TRACTCE']
        else:
            logger.warning("Could not create GEOID column from available fields")

    if 'GEOID' in gdf.columns:
        gdf['GEOID'] = gdf['GEOID'].astype(str)

    return gdf
