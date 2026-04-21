# BenSAF Dash - Web Dashboard

A web-based interface for the BenSAF (Benefits of Sustainable Aviation Fuels) health impact assessment toolkit.

## Overview

BenSAF Dash provides a Plotly Dash web application that allows users to interact with the BenSAF core analysis toolkit through a web browser. This interface is ideal for:

- Collaborative analysis sessions
- Remote access to BenSAF functionality
- Users who prefer web-based interfaces
- Quick prototyping and exploration
- Sharing results with stakeholders

## Architecture

BenSAF Dash follows a **thin frontend** architecture:

- **No analysis logic**: All calculations performed by `bensaf.workflow.Workflow`
- **No data transformations**: All processing done by `bensaf` core functions
- **Pure UI layer**: Only handles user input, file uploads, and visualization

The dashboard simply:
1. Accepts file uploads and configuration from users
2. Passes data to `bensaf.Workflow`
3. Receives results from `bensaf.Workflow`
4. Displays results using Plotly charts

## Installation

Install BenSAF Dash dependencies:

```bash
uv pip install dash dash-bootstrap-components plotly
```

Or install from the project root with extras:

```bash
uv pip install -e ".[dash]"
```

## Usage

### Starting the Dashboard

From the command line:

```bash
python -m bensaf_dash.app
```

Or programmatically:

```python
from bensaf_dash.app import app
app.run_server(debug=True)
```

The dashboard will be available at `http://localhost:8050`

### Using the Dashboard

#### 1. Data Upload Tab

Upload your analysis data or load example data:

**Option A: Upload Your Own Data**
- **Census Tract Data**: GeoJSON, Shapefile, or GeoPackage with tract geometries and demographics
  - Required columns: `GEOID`, `geometry`, `population`
- **Exposure Data**: CSV file with pollutant concentrations by tract
  - Required columns: `GEOID`, `pollutant_concentration`
- **Mortality Data**: CSV file with baseline mortality rates
  - Required columns: `GEOID`, `mortality_rate`

**Option B: Load Example Case Study**
- Choose a case study in the dropdown, then **Load Selected Case Study** (inputs under `data/case-studies/`)
- Loads tract geometry, demographics, exposure, and mortality inputs for that study
- Useful for testing and learning the workflow

#### 2. Configuration Tab

Set analysis parameters:

- **Health Impact Function**: Configure relative risk parameters
  - Mean RR, Lower/Upper 95% CI, Unit Increase
  - Defaults from Bouma et al. (2024) provided
- **Control Scenarios**: Define SAF blend percentages to analyze
  - E.g., 25%, 50%, 75% SAF

#### 3. Analysis Tab

Run the analysis:

- Click "Run Complete Analysis" button
- View progress and status messages
- Analysis runs via `bensaf.Workflow` in the backend

#### 4. Results Tab

View and explore results:

- **Summary Cards**: Key metrics for each scenario
- **Bar Chart**: Comparison of health impacts across scenarios
- **Map**: Spatial distribution of health impacts
- **Table**: Detailed numerical results with confidence intervals

## Project Structure

```
bensaf_dash/
├── __init__.py                 # Package initialization
├── app.py                      # Main Dash application entry point
├── layouts/
│   ├── __init__.py
│   └── main_layout.py         # UI layout components
├── callbacks/
│   ├── __init__.py
│   └── workflow_callbacks.py  # Callback functions connecting UI to bensaf
└── README.md                   # This file
```

## Component Responsibilities

### `app.py`
- Initializes Dash application
- Sets up Bootstrap styling
- Registers callbacks
- Provides main() entry point

### `layouts/main_layout.py`
- Defines UI structure
- Creates tabs for workflow steps
- Provides upload widgets, input forms, and visualization containers
- No logic - pure UI

### `callbacks/workflow_callbacks.py`
- Handles file uploads and parsing
- Instantiates `bensaf.Workflow`
- Calls `bensaf` methods with user inputs
- Converts `bensaf` results to Plotly visualizations
- All analysis delegated to `bensaf` core

## Key Design Decisions

### 1. Stateful Workflow
The dashboard maintains a global `workflow_instance` that persists across callbacks. This allows:
- Sequential loading of data files
- Progressive configuration
- Single analysis execution on complete dataset

### 2. Data Storage
Uses Dash `dcc.Store` components to track:
- Workflow state (data loaded, config set)
- Analysis results (for visualization)

### 3. File Handling
Uploaded files are:
- Base64 decoded
- Written to temporary files
- Loaded by `bensaf` (geopandas, pandas)
- Temporary files cleaned up

### 4. Visualization Translation
Results from `bensaf.Workflow` (DataFrames, dicts) are converted to:
- Plotly bar charts
- Plotly maps (scattermapbox)
- Bootstrap tables

## Dependencies

- `dash`: Web application framework
- `dash-bootstrap-components`: Bootstrap UI components
- `plotly`: Interactive visualizations
- `pandas`: Data manipulation (inherited from bensaf)
- `geopandas`: Spatial data (inherited from bensaf)
- `bensaf`: Core analysis toolkit

## Deployment

For production deployment:

### Option 1: Gunicorn (Linux/Mac)

```bash
gunicorn bensaf_dash.app:server -b 0.0.0.0:8050
```

### Option 2: Waitress (Windows)

```bash
pip install waitress
waitress-serve --port=8050 bensaf_dash.app:server
```

### Option 3: Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -e ".[dash]"

EXPOSE 8050

CMD ["python", "-m", "bensaf_dash.app"]
```

Build and run:

```bash
docker build -t bensaf-dash .
docker run -p 8050:8050 bensaf-dash
```

## Future Enhancements

Potential improvements:

- **Multi-user support**: Separate workflow instances per session
- **Data persistence**: Save/load analysis sessions
- **Export functionality**: Download results as CSV/Excel
- **Advanced visualizations**: 3D plots, animated scenarios
- **Real-time progress**: Websocket updates during analysis
- **Comparison mode**: Side-by-side scenario comparison
- **Parameter sweeps**: Batch analysis over parameter ranges

## Troubleshooting

### Port already in use
Change the port in `app.py`:
```python
app.run_server(debug=True, port=8051)
```

### Large file uploads
Increase upload size limit in `app.py`:
```python
app.config.suppress_callback_exceptions = True
app.server.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Memory issues with large datasets
Consider implementing:
- Chunked data processing
- File streaming instead of in-memory storage
- Database backend for large results

## Contributing

To add new features:

1. **New visualizations**: Add to `workflow_callbacks.py`, create new callback
2. **New configuration options**: Update `main_layout.py` and config callback
3. **New data inputs**: Add upload component and parsing callback

Remember: Keep all analysis logic in `bensaf` core, not in the dashboard.

## License

Same as parent BenSAF project (MIT)

