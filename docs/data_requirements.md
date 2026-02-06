# Data Requirements

This document outlines the data requirements for the SAF Toolkit, including data sources, input formats, and internal data structures.

## Census and Demographic Data

### Online Source
- American Community Survey (ACS) 5-Year Estimates
- URL: [Census Bureau Data](https://data.census.gov/)
- Tables: B03002 (Race/Ethnicity) and C17002 (Income)

### Input Format
- CSV files from Census Bureau download
- Required files:
  - `ACSDT5Y2023.B03002-Data.csv`
  - `ACSDT5Y2023.C17002-Data.csv`
- Required columns:
  - `B03002_001E`: Total population
  - `B03002_003E`: White alone
  - `C17002_001E`: Total households
  - `C17002_008E`: Below poverty level
  - `GEOID`: Census tract identifier

### Internal Format
- Pandas DataFrame with columns:
  - `geoid`: Census tract identifier (str)
  - `total_pop`: Total population (int)
  - `white_pop`: White alone population (int)
  - `total_households`: Total households (int)
  - `poverty_households`: Households below poverty (int)

## Geographic Data

### Online Source
- Census Bureau TIGER/Line Shapefiles
- URL: [TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

### Input Format
- Shapefile format (.shp, .shx, .dbf, .prj)
- Required files:
  - `Illinois Shapefile Bounded`
  - `Illinois Shapefile Bounded Circular`
- Required fields:
  - `GEOID`: Census tract identifier
- Coordinate system: EPSG:4269 (NAD83)

### Internal Format
- GeoPandas GeoDataFrame with columns:
  - `geoid`: Census tract identifier (str)
  - `geometry`: Polygon/MultiPolygon geometry
  - `area`: Area in square meters (float)
  - `centroid`: Center point coordinates (Point)

## Air Quality Data

### Online Source
- Local airport emissions data
- Regional air quality monitoring data

### Input Format
- Excel files (.xlsx)
- Required files:
  - `LTO Concentrations.xlsx`
    - Sheet: 'ORD Coord'
    - Columns: Longitude, Latitude, UFP concentrations
  - `illinois.xlsx`
    - Columns: GEOID, COUNTYFP, UFP Estimate

### Internal Format
- Pandas DataFrame with columns:
  - `geoid`: Census tract identifier (str)
  - `county_fips`: County FIPS code (str)
  - `ufp_concentration`: UFP concentration value (float)
  - `longitude`: Longitude coordinate (float)
  - `latitude`: Latitude coordinate (float)

## Health Data

### Online Source
- CDC WONDER Database
- URL: [CDC WONDER](https://wonder.cdc.gov/)

### Input Format
- Tab-delimited text file from CDC WONDER
- Required columns:
  - County FIPS codes
  - Mortality rates per 100,000 population
- Required counties:
  - Cook (031)
  - DuPage (043)
  - Kane (089)
  - Kendall (093)
  - Lake (097)
  - McHenry (111)
  - Will (197)

### Internal Format
- Pandas DataFrame with columns:
  - `county_fips`: County FIPS code (str)
  - `county_name`: County name (str)
  - `mortality_rate`: Rate per 100,000 (float)
  - `year`: Year of data (int)

## Vulnerability Data

### Online Source
- CDC Social Vulnerability Index
- URL: [CDC SVI](https://www.atsdr.cdc.gov/placeandhealth/svi/index.html)

### Input Format
- Excel file (.xlsx)
- Required file: `ORD_Vulnerable_Tracts.dbf.xlsx`
- Required columns:
  - `GEOID`: Census tract identifier
  - `VULEOPCT`: Vulnerability percentile

### Internal Format
- Pandas DataFrame with columns:
  - `geoid`: Census tract identifier (str)
  - `vulnerability_score`: Vulnerability percentile (float, 0-1)
  - `vulnerability_rank`: Rank within study area (int)

## Configuration

### Input Format
- YAML file
- Required settings:
  ```yaml
  study_areas:
    counties:
      - "King"
      - "Pierce"
      - "Snohomish"
    state: "WA"
    year: 2020

  data_paths:
    census_data: "../data_files/census_data"
    acs_data: "../data_files/acs_data"
    mortality_data: "../data_files/cdc_wonder/mortality.txt"
    svi_data: "../data_files/svi"
    health_outcomes: "../data_files/health_outcomes"
    aermod_predictions: "../data_files/predictions.rda"
  ```

### Internal Format
- Python dictionary with nested structure
- All paths converted to Path objects
- Study areas validated against available data

## Data Quality Requirements

1. **Completeness**
   - No missing values in critical fields
   - All required columns present
   - All required files available

2. **Consistency**
   - GEOID formats match across datasets
   - Coordinate systems consistent (EPSG:4269)
   - Date formats consistent

3. **Accuracy**
   - Population counts match Census Bureau totals
   - Rates properly calculated
   - Coordinates accurate

4. **Format**
   - Numeric fields properly formatted
   - No special characters in field names
   - Consistent date formats 