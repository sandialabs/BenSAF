# BenSAF Dash Workflow Summary

## Overview

BenSAF Dash provides a web-based interface for conducting health impact assessments of Sustainable Aviation Fuel (SAF) adoption. The workflow processes air quality exposure data, applies SAF blend scenarios, and calculates health impacts and economic benefits at the census tract level.

## Minimal Data Requirements and Assumptions

### Required Data

1. **Census Tract Geometries**
   - Format: GeoJSON, Shapefile, or GeoPackage
   - Required columns: `GEOID` (unique identifier), `geometry` (polygon geometries)
   - Coordinate Reference System (CRS): Configurable (default: EPSG:4326)

2. **Demographics Data**
   - Format: CSV
   - Required columns: `GEOID` (must match tract geometries), `population`
   - Optional columns: Additional demographic covariates for benefit distribution analysis

3. **Baseline Exposure Data** (Two paths available - see Methodology section)

4. **Incidence/Mortality Data**
   - Format: CSV
   - Required columns: `GEOID` (must match tract geometries), `mortality_rate`
   - Represents baseline mortality rates per census tract

5. **Preterm Birth Data** (Optional)
   - Format: CSV
   - Required columns: `GEOID`, `baseline_preterm_births`
   - Enables preterm birth benefit calculations

### Key Assumptions

- All datasets must share the same `GEOID` identifiers to enable spatial joins
- Census tracts serve as the spatial unit of analysis
- Exposure data represents annual average concentrations
- Health impact functions assume log-linear concentration-response relationships
- SAF blend percentages are limited to 0-50%

## General Methodology

### Major Workflow Steps

1. **Data Loading**
   - Load census tract geometries
   - Load demographics data (population required)
   - Load baseline exposure data (via CSV upload or AERMOD extraction)
   - Load incidence/mortality data
   - Load preterm birth data (optional)

2. **Configuration**
   - Select mortality function(s) from available library
   - Configure SAF blend scenarios (percentages to analyze, e.g., 25%, 50%)
   - Health impact function parameters loaded from mortality function library

3. **Scenario Execution**
   - For each SAF blend scenario:
     a. Calculate pollutant reduction using SAF blend polynomial coefficients
     b. Compute reduced exposure concentrations
     c. Calculate health impacts using concentration-response functions
     d. Estimate economic benefits using value of statistical life and other parameters
     e. Aggregate results to tract-level and population-weighted totals

4. **Results Generation**
   - Tract-level results for each scenario
   - Population-weighted aggregated totals
   - Confidence intervals (mean, lower 95% CI, upper 95% CI)

### Data Input Usage

- **Tract Geometries**: Used for spatial operations, aggregation, and visualization
- **Demographics**: Population used for weighting health impacts and aggregating results
- **Baseline Exposure**: Starting point for calculating exposure reductions under SAF scenarios
- **Incidence Data**: Baseline rates used to calculate attributable cases from exposure changes
- **Preterm Birth Data**: Baseline counts used to calculate preterm birth reductions

## Baseline Exposure: Two Paths

### Path 1: Direct CSV Upload

**Methodology:**
- User uploads pre-processed exposure CSV file
- File must contain `GEOID` column matching census tract identifiers
- Exposure column can be named `pollutant_concentration` or `baseline_pollutant_concentration` (automatically renamed to `ufp`)
- Data is directly loaded into the workflow without transformation

**Assumptions:**
- Exposure values represent annual average concentrations
- Values are already aggregated to census tract level
- Coordinate system matches the configured CRS for tract geometries
- No additional processing or calibration required

### Path 2: Extract from AERMOD Files

**Methodology:**

The AERMOD extraction follows an expert-defined pipeline:

1. **Extract Annual Averages**
   - Parse AERMOD .ADO files using AERMOD parser
   - Extract `ANNUAL_AVERAGE` section data
   - Extract receptor point coordinates (`x_coord`, `y_coord`) and concentration values
   - Handle coordinate transformations (supports polar coordinate systems if center location provided)

2. **Weighted Combination of Flows**
   - Process landing and takeoff flows separately
   - For each flow type, combine multiple AERMOD files using user-specified weights
   - Match receptor points across files using spatial tolerance (default: 1.0 meters)
   - Weighted average concentrations at matched receptor locations
   - If weights not specified, equal weights assumed

3. **Convert to Percentiles and Apply Log-Linear Calibration**
   - Convert CO concentrations to percentiles using rank-based method (separately for landing and takeoff)
   - Apply log-linear calibration model to convert CO to UFP (ultrafine particles):
     - Landing: `log(UFP) = intercept + coef_landing * landing_percentile`
     - Takeoff: `log(UFP) = intercept + coef_takeoff * takeoff_percentile`
   - Exponentiate to obtain UFP concentrations
   - Calibration coefficients loaded from `data/aermod_calibration_coefficients.json`

4. **Aggregate to Census Tracts**
   - Convert receptor point geometries to GeoDataFrame in AERMOD CRS (default: EPSG:32616)
   - Aggregate UFP values to census tracts using spatial join method:
     - Points within tract polygons are averaged
     - If multiple files processed, sum annual averages across files
   - Alternative: IDW (Inverse Distance Weighting) interpolation available but not default

5. **Combine Landing and Takeoff**
   - If both landing and takeoff flows provided, sum UFP concentrations
   - Final output: DataFrame with `GEOID` and `ufp` columns

**AERMOD Parsing Assumptions:**

- AERMOD files contain annual average concentration data in standard format
- Receptor coordinates are valid and within reasonable bounds (< 1e10)
- Coordinate reference system defaults to EPSG:32616 (UTM Zone 16N) but configurable
- Calibration coefficients are derived from SEA-TAC calibration model
- Percentile-based calibration assumes rank-based percentile computation
- Spatial aggregation assumes uniform distribution within tracts (spatial join averages points)
- Multiple AERMOD files represent different wind directions or operational conditions
- Weights represent relative importance or frequency of different conditions

**Calibration File Requirements:**
- JSON format with keys: `intercept`, `coef_landing`, `coef_takeoff`
- Coefficients derived from SEA-TAC R model calibration
- Default location: `data/aermod_calibration_coefficients.json`

## Outputs

### Scenario-Level Results

For each SAF blend scenario, the workflow produces:

1. **Exposure Changes**
   - `reduced_concentration`: Pollutant concentration after SAF adoption (tract-level Series)
   - `delta_concentration`: Reduction in concentration (baseline - reduced)

2. **Health Impacts** (by endpoint)
   - `relative_risk_mean`, `relative_risk_lower`, `relative_risk_upper`: Relative risk estimates
   - `attributable_fraction_mean`, `attributable_fraction_lower`, `attributable_fraction_upper`: Attributable fractions
   - `attributable_cases_mean`, `attributable_cases_lower`, `attributable_cases_upper`: Attributable cases per tract
   - `total_attributable_cases`: Population-weighted total cases (aggregated)

3. **Economic Benefits**
   - `mortality_value_mean`, `mortality_value_lower`, `mortality_value_upper`: Economic value of mortality reductions
   - `preterm_birth_value_mean`, `preterm_birth_value_lower`, `preterm_birth_value_upper`: Economic value of preterm birth reductions (if applicable)
   - All values in configured currency units

### Aggregated Results

- Population-weighted totals across all census tracts
- Confidence intervals (mean, lower 95% CI, upper 95% CI)
- Summary statistics by scenario

### Visualization Outputs

- Summary cards with key metrics per scenario
- Bar charts comparing scenarios
- Interactive maps showing spatial distribution
- Detailed results tables with tract-level data

## Limitations

### Data Limitations

- **Spatial Resolution**: Analysis limited to census tract level; sub-tract variation not captured
- **Temporal Resolution**: Assumes annual average exposure; temporal patterns not considered
- **Pollutant Scope**: Primary focus on ultrafine particles (UFP); other pollutants require separate analysis
- **Geographic Scope**: Analysis limited to census tracts with available data; gaps in coverage not interpolated

### Methodology Limitations

- **Exposure Aggregation**: Spatial join method assumes uniform distribution within tracts; may not capture gradients
- **Calibration Assumptions**: AERMOD calibration assumes percentile-based log-linear relationship; may not hold for all conditions
- **Health Functions**: Log-linear concentration-response functions may not capture threshold effects or non-linearities
- **Economic Valuation**: Uses standard value of statistical life; may not reflect local preferences or equity considerations

### AERMOD Processing Limitations

- **Receptor Matching**: Tolerance-based matching (1.0 m default) may miss or incorrectly match receptors
- **Coordinate Systems**: Requires accurate CRS specification; polar coordinate handling requires center location
- **File Format**: Assumes standard AERMOD .ADO format; variations may not be supported
- **Calibration Transferability**: Calibration coefficients derived from specific study; applicability to other airports/regions uncertain

### Workflow Limitations

- **Single Pollutant**: Each analysis handles one pollutant at a time
- **Scenario Independence**: Scenarios analyzed independently; cumulative effects not modeled
- **Static Demographics**: Uses fixed population/demographics; does not account for future changes
- **No Uncertainty Propagation**: Some uncertainty sources not fully propagated through calculations

### Technical Limitations

- **File Size**: Large AERMOD files or many tracts may impact performance
- **Memory**: All data held in memory; very large study areas may exceed available memory
- **Coordinate Transformations**: Complex CRS transformations may introduce small errors
- **Web Interface**: Single-user interface; concurrent analyses not supported

## Additional Notes

- The dashboard is a thin frontend; all computation performed by `bensaf.Workflow` core
- Configuration files (economic parameters, calibration coefficients) must be present in `data/` directory
- Mortality function library loaded from configuration; functions can be selected for analysis
- Results can be exported for further analysis or reporting outside the dashboard
