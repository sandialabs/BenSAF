# BenSAF

A toolkit for health-based assessments of sustainable aviation fuels.

## Overview

BenSAF provides a generalized framework for:

1. Processing geospatial data for airport-adjacent communities
2. Estimating health impacts of sustainable aviation fuel blend scenarios
3. Analyzing impacts across different demographic groups
4. Generating visualizations and reports

## Installation

1. Create and activate a virtual environment:
```bash
venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package in development (editable) mode:
```bash
pip install -e .
```

## Documentation

The documentation is built using Sphinx. To build the documentation:

1. Make sure you have installed the development dependencies:
```bash
uv pip install -e ".[dev]"
```

2. Build the documentation:
```bash
cd docs
make html
```

The built documentation will be available in `docs/build/html/`.

## Project Structure

```
bensaf/                   # Main BenSAF Python package
├── __init__.py
├── workflow.py          # Core workflow orchestration
├── health_impacts.py    # Health impact calculation functions
├── utils.py             # Utility functions and data processing
├── graphics.py          # Visualization and plotting utilities
```

## Usage

### Basic Usage

```python
from bensaf.workflow import Workflow
import geopandas as gpd
import pandas as pd

# Load your data
tracts_gdf = gpd.read_file("census_tracts.gpkg")
exposure_df = pd.read_csv("exposure_data.csv")
mortality_df = pd.read_csv("mortality_data.csv")

# Initialize workflow
config = {
    'control_scenarios': [5, 25, 50],  # Emission reduction percentages
    'demographic_columns': ['race', 'income_level']
}
workflow = Workflow(config)

# Load data
workflow.load_tract_data(tracts_gdf)
workflow.load_exposure_data(exposure_df)
workflow.load_mortality_data(mortality_df)

# Load health impact function (Bouma et al.)
workflow.load_health_impact_function(
    mean_rr=1.012,
    lower_rr=1.010,
    upper_rr=1.015,
    unit_increase=2723  # pt/cm3
)

# Run analysis
results = workflow.run_complete_analysis("results")
```

### Using Configuration Files

You can also use YAML configuration files:

```python
import yaml
from pathlib import Path
from bensaf.workflow import Workflow

# Load configuration
with open("bensaf_workflow_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize workflow with config
workflow = Workflow(config)

# Continue with data loading and analysis...
```

### Example Scripts

The `examples/` directory contains example scripts demonstrating the workflow:

- `workflow_example.py`: Complete workflow with synthetic data


## Data Requirements

The workflow requires the following data:

1. **Census Tract Data** (GeoDataFrame):
   - GEOID: Census tract identifier
   - geometry: Tract geometry
   - population: Total population
   - Optional demographic columns

2. **Exposure Data** (DataFrame or GeoDataFrame):
   - GEOID: Census tract identifier
   - pollutant_concentration: Baseline pollutant concentration

3. **Mortality Data** (DataFrame):
   - GEOID: Census tract identifier
   - mortality_rate: Baseline mortality rate (deaths per person per year)

## License

TBD
