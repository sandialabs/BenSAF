# BenSAF Dashboard: Input Data Requirements

## Required Inputs

### 1. Case Study Boundary
**Format:** GeoJSON, Shapefile, or GeoPackage

| Field | Type | Description |
|---|---|---|
| `GEOID` | string | Census tract identifier |
| `geometry` | polygon | Tract boundary |

Defines the spatial frame used for all joins, maps, and aggregations. Keep attribute columns minimal.

---

### 2. Demographics
**Format:** CSV, one row per census tract

| Field | Type | Required | Description |
|---|---|---|---|
| `GEOID` | string | yes | Census tract identifier |
| `population` | integer | yes | Total tract population |
| additional covariates | numeric | no | Age strata, income, etc. used by health pipelines |

Used for population weighting and demographic context in health impact calculations.

---

### 3. Baseline Pollutant Exposure
**Format:** CSV or AERMOD `.ADO` files — choose one source.

#### Option A — CSV upload
One row per census tract.

| Field | Type | Description |
|---|---|---|
| `GEOID` | string | Census tract identifier |
| `ufp` / `pollutant_concentration` / `baseline_pollutant_concentration` | numeric | Baseline UFP concentration (pt/cm³) |

#### Option B — AERMOD `.ADO` files
Upload one or more landing-phase files and/or one or more takeoff-phase files. Each file is assigned a numeric weight (defaults to equal split across files within each phase). A CRS string for the AERMOD receptor grid must be provided (e.g. `EPSG:32610`). Exposure values are derived using the bundled calibration model.

> Only the phases uploaded are included in the blend. Landing-only or takeoff-only submissions are valid.

---

### 4. Baseline Mortality Rate
**Format:** CSV, one row per census tract

| Field | Type | Description |
|---|---|---|
| `GEOID` | string | Census tract identifier |
| `mortality_rate` | numeric | Baseline all-cause mortality rate |

Drives attributable mortality cases under each SAF scenario using the Bouma et al. concentration–response function.

---

## Optional Input

### 5. Per-capita Expenditure
**Format:** CSV, one row per census tract

| Field | Type | Required | Description |
|---|---|---|---|
| `GEOID` | string | yes | Census tract identifier |
| `per_capita_expenditure` | numeric | yes | Also accepted as `per_capita_consumption` or `income` (legacy) |
| `life_years_gained` | numeric | no | Supplemental economic valuation input |

Enables monetized mortality benefits in results. If omitted, only case counts are reported.

---

## Shared Reference Files

These files are bundled with the repository and are not uploaded by the user.

| File | Description |
|---|---|
| `data/aermod_calibration_coefficients.json` | Linear model (intercept + landing/takeoff coefficients) converting raw AERMOD concentrations to calibrated UFP values. Sourced from a SEA-TAC calibration. |
| `data/saf_blend_parameters.json` | Quadratic polynomial mapping SAF blend % to pollutant reduction %. Default: Reduction = −0.0152·SAF + 0.00009·SAF². |
| `data/mortality_functions.json` | Concentration–response parameters (Bouma et al.) used by the health pipeline. |

---

## Analysis Configuration

SAF blend scenarios are configured in-app on the Configuration tab — no file upload required.

| Parameter | Type | Range | Description |
|---|---|---|---|
| SAF blend % | integer | 0–50 | One or more blend percentages to compare across scenarios |
