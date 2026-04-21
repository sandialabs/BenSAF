"""
Workflow callbacks for BenSAF Dash application

These callbacks handle user interactions and connect the UI to the bensaf.Workflow class.
"""

import base64
import io
import json
from pathlib import Path
import tempfile

import matplotlib
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import callback, Input, Output, State, html, dcc, ALL, ctx as dash_ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash

from bensaf.model.data_model import AnalysisResults
from bensaf.model.workflow import Workflow
from bensaf.utils.params import load_mortality_functions

PLOT_PRIMARY = "#18BC9C"
PLOT_SECONDARY = "#3498DB"


def _mpl_plotly_colorscale(cmap_name: str, n: int = 9):
    cmap = matplotlib.colormaps[cmap_name]
    return [[i / max(1, n - 1), mcolors.to_hex(cmap(i / max(1, n - 1)))] for i in range(n)]


def _map_annotation_layout(title, text, x=0.5, y=0.5):
    return go.Figure().update_layout(
        title=title,
        height=600,
        template="plotly_white",
        annotations=[
            {
                "text": text,
                "xref": "paper",
                "yref": "paper",
                "x": x,
                "y": y,
                "showarrow": False,
                "font": {"size": 16},
            }
        ],
        uirevision="constant",
    )


def _inputs_core_merged_gdf(workflow):
    """Tract geometry plus core input layers only (no scenario columns)."""
    return AnalysisResults(workflow.inputs).get_merged_data(core_only=True)

workflow_instance = None
cached_geojson = None
cached_center = None
_mortality_functions_cache = None


def _mortality_functions():
    global _mortality_functions_cache
    if _mortality_functions_cache is None:
        _mortality_functions_cache = load_mortality_functions()
    return _mortality_functions_cache


def _mortality_function_rows():
    return [
        {"id": fid, "title": data["title"]}
        for fid, data in sorted(_mortality_functions().items())
    ]

def _numeric_saf_scenarios(scenarios, default=(25, 50)):
    """Valid integers 0–50 for workflow/plots; None or invalid entries are dropped."""
    if not scenarios:
        return list(default)
    out = []
    for s in scenarios:
        if s is None:
            continue
        try:
            n = int(round(float(s)))
        except (TypeError, ValueError):
            continue
        out.append(max(0, min(50, n)))
    return out if out else list(default)


def load_saf_blend_parameters():
    """Load SAF blend parameters from JSON file."""
    project_root = Path(__file__).parent.parent.parent
    saf_params_file = project_root / 'data' / 'saf_blend_parameters.json'
    
    if not saf_params_file.exists():
        return [0.0, 1.0, 0.0]
    
    with open(saf_params_file, 'r') as f:
        data = json.load(f)
    
    return data.get('polynomial_coefficients', [0.0, 1.0, 0.0])

def load_case_studies():
    """Load and validate case studies from JSON metadata file."""
    project_root = Path(__file__).parent.parent.parent
    case_studies_file = project_root / 'data' / 'case-studies' / 'case_studies.json'
    
    if not case_studies_file.exists():
        raise FileNotFoundError(f"Case studies metadata not found at {case_studies_file}")
    
    with open(case_studies_file, 'r') as f:
        data = json.load(f)
    
    if 'case_studies' not in data:
        raise ValueError("Invalid case_studies.json: missing 'case_studies' key")
    
    return data['case_studies']

def get_case_study_by_id(case_study_id):
    """Get a specific case study by ID."""
    case_studies = load_case_studies()
    for case_study in case_studies:
        if case_study.get('id') == case_study_id:
            return case_study
    raise ValueError(f"Case study '{case_study_id}' not found")

def resolve_case_study_paths(case_study):
    """Resolve all file paths for a case study relative to project root."""
    project_root = Path(__file__).parent.parent.parent
    case_studies_root = project_root / 'data' / 'case-studies'
    case_study_dir = case_studies_root / case_study['folder']
    
    files = case_study['files']
    resolved = {}
    
    # Resolve basic files
    resolved['tracts_geometries'] = case_study_dir / files['tracts_geometries']
    resolved['demographics'] = case_study_dir / files['demographics']
    resolved['mortality'] = case_study_dir / files['mortality']
    
    # Optional tract-level mortality economic inputs (GEOID, per_capita_consumption, ...)
    if files.get('mortality_economic'):
        resolved['mortality_economic'] = case_study_dir / files['mortality_economic']

    # Optional CSV exposure file
    if 'exposure_csv' in files:
        resolved['exposure_csv'] = case_study_dir / files['exposure_csv']
    
    # Optional AERMOD files with weights
    if 'aermod' in files:
        resolved['aermod'] = {}
        resolved['aermod']['landing'] = [
            (case_study_dir / item['file'], item['weight'])
            for item in files['aermod']['landing']
        ]
        resolved['aermod']['takeoff'] = [
            (case_study_dir / item['file'], item['weight'])
            for item in files['aermod']['takeoff']
        ]
    
    # Resolve calibration file
    if case_study.get('calibration_file'):
        resolved['calibration'] = case_study_dir / case_study['calibration_file']
    else:
        # Use shared calibration file
        resolved['calibration'] = project_root / 'data' / 'aermod_calibration_coefficients.json'
    
    return resolved


def _aermod_weight_table_ui(filenames, id_type):
    """Table mapping each file name to a weight input. id_type: 'landing-weight' or 'takeoff-weight'."""
    n = len(filenames)
    if n == 0:
        return html.Small("No files uploaded.", className="text-muted")
    default_w = round(1.0 / n, 6)
    header = html.Thead(html.Tr([html.Th("File"), html.Th("Weight", style={"width": "140px"})]))
    rows = []
    for i, fn in enumerate(filenames):
        rows.append(
            html.Tr(
                [
                    html.Td(fn, className="align-middle text-break small"),
                    html.Td(
                        dbc.Input(
                            id={"type": id_type, "index": i},
                            type="number",
                            value=default_w,
                            step=0.01,
                            min=0,
                            className="form-control form-control-sm",
                        )
                    ),
                ]
            )
        )
    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        striped=True,
        size="sm",
        className="mb-0",
        responsive=True,
    )


def register_callbacks(app):
    
    @app.callback(
        [Output('data-config', 'data'),
         Output('aermod-crs-config-status', 'children')],
        Input('input-aermod-crs', 'value'),
        State('data-config', 'data'),
        prevent_initial_call=False,
    )
    def update_aermod_data_config(aermod_crs, current_config):
        """Keep AERMOD grid CRS in the data store (used when generating exposure from .ADO files)."""
        config = dict(current_config) if current_config else {}
        text = (aermod_crs or "").strip()
        config['aermod_crs'] = text if text else 'EPSG:4326'
        hint = html.Small(f"Using {config['aermod_crs']} for AERMOD x/y coordinates.", className="text-muted")
        return config, hint
    
    @app.callback(
        Output('case-study-dropdown', 'options'),
        Input('case-studies-init', 'data'),
        prevent_initial_call=False
    )
    def populate_case_study_dropdown(init_data):
        """Populate dropdown with available case studies."""
        try:
            case_studies = load_case_studies()
            options = [
                {
                    'label': f"{cs['name']}{' - ' + cs['description'] if cs.get('description') else ''}",
                    'value': cs['id']
                }
                for cs in case_studies
            ]
            return options
        except Exception as e:
            return [{'label': f'Error loading case studies: {str(e)}', 'value': None}]
    
    @app.callback(
        Output('btn-load-example', 'disabled'),
        Input('case-study-dropdown', 'value'),
        prevent_initial_call=False
    )
    def enable_load_button(selected_case_study):
        """Enable load button when a case study is selected."""
        return selected_case_study is None
    
    @app.callback(
        [Output('example-load-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('btn-load-example', 'n_clicks'),
        [State('case-study-dropdown', 'value'),
         State('workflow-state', 'data'),
         State('data-config', 'data')],
        prevent_initial_call=True
    )
    def load_example_data(n_clicks, selected_case_study_id, state, data_config):
        if n_clicks is None or selected_case_study_id is None:
            return "", state
        
        try:
            state = dict(state) if state else {}

            # Get case study metadata
            case_study = get_case_study_by_id(selected_case_study_id)
            resolved_paths = resolve_case_study_paths(case_study)
            
            # Validate core files exist
            if not resolved_paths['tracts_geometries'].exists():
                raise FileNotFoundError(f"Tract geometries not found at {resolved_paths['tracts_geometries']}")
            if not resolved_paths['demographics'].exists():
                raise FileNotFoundError(f"Demographics data not found at {resolved_paths['demographics']}")
            if not resolved_paths['mortality'].exists():
                raise FileNotFoundError(f"Mortality data not found at {resolved_paths['mortality']}")
            
            exposure_source = case_study.get('exposure_source', 'aermod' if 'aermod' in resolved_paths else 'csv')
            
            if exposure_source == 'aermod':
                if 'aermod' not in resolved_paths:
                    raise ValueError("Case study is configured for AERMOD exposure but no AERMOD files are defined")
                if not resolved_paths['calibration'].exists():
                    raise FileNotFoundError(f"Calibration coefficients file not found at {resolved_paths['calibration']}")
                for file_path, _ in resolved_paths['aermod']['landing']:
                    if not file_path.exists():
                        raise FileNotFoundError(f"Landing AERMOD file not found at {file_path}")
                for file_path, _ in resolved_paths['aermod']['takeoff']:
                    if not file_path.exists():
                        raise FileNotFoundError(f"Takeoff AERMOD file not found at {file_path}")
            else:
                if 'exposure_csv' not in resolved_paths or not resolved_paths['exposure_csv'].exists():
                    raise FileNotFoundError("Exposure CSV file not found for this case study")
            
            # Load data files
            tracts_gdf = gpd.read_file(resolved_paths['tracts_geometries'])
            demographics_df = pd.read_csv(resolved_paths['demographics'])
            mortality_df = pd.read_csv(resolved_paths['mortality'])
            mortality_economic_df = None
            if resolved_paths.get('mortality_economic'):
                me_path = resolved_paths['mortality_economic']
                if me_path.exists():
                    mortality_economic_df = pd.read_csv(me_path)

            global workflow_instance, cached_geojson, cached_center
            cached_geojson = None
            cached_center = None
            
            if workflow_instance is None:
                workflow_instance = Workflow()
            
            # Use load_inputs for cleaner API
            if exposure_source == 'aermod':
                aermod_crs = case_study.get('aermod_crs')
                if not aermod_crs and data_config:
                    aermod_crs = data_config.get('aermod_crs')
                if not aermod_crs:
                    aermod_crs = 'EPSG:4326'
                exposure_data = {
                    'landing_files': resolved_paths['aermod']['landing'],
                    'takeoff_files': resolved_paths['aermod']['takeoff'],
                    'calibration_file': resolved_paths['calibration'],
                    'aggregation_method': case_study.get('aggregation_method', 'spatial_join'),
                    'aermod_crs': aermod_crs,
                }
                agg_kw = case_study.get('aggregation_kwargs')
                if agg_kw:
                    exposure_data['aggregation_kwargs'] = agg_kw
                exposure_source_str = 'aermod_workflow'
            else:
                exposure_df = pd.read_csv(resolved_paths['exposure_csv'])
                if 'pollutant_concentration' in exposure_df.columns:
                    exposure_df = exposure_df.rename(columns={'pollutant_concentration': 'ufp'})
                elif 'baseline_pollutant_concentration' in exposure_df.columns:
                    exposure_df = exposure_df.rename(columns={'baseline_pollutant_concentration': 'ufp'})
                exposure_data = exposure_df
                exposure_source_str = 'csv'
            
            workflow_instance.load_inputs(
                tracts_gdf=tracts_gdf,
                demographics_df=demographics_df,
                exposure_source=exposure_source_str,
                exposure_data=exposure_data,
                incidence_df=mortality_df,
                mortality_economic_df=mortality_economic_df,
                pollutant_name='ufp'
            )
            
            n_exposure = len(workflow_instance.inputs.baseline_exposure)
            
            state['tracts_loaded'] = True
            state['n_tracts'] = len(tracts_gdf)
            state['demographics_loaded'] = True
            state['n_demographics'] = len(demographics_df)
            state['exposure_loaded'] = True
            state['n_exposure'] = n_exposure
            state['exposure_source'] = exposure_source
            state['mortality_loaded'] = True
            state['n_mortality'] = len(mortality_df)
            if mortality_economic_df is not None:
                state['mortality_economic_loaded'] = True
                state['n_mortality_economic'] = len(mortality_economic_df)
            else:
                state['mortality_economic_loaded'] = False
                state['n_mortality_economic'] = 0

            if exposure_source == 'aermod':
                exposure_msg = f"Generated {n_exposure} exposure records from AERMOD files"
            else:
                exposure_msg = f"Loaded {n_exposure} exposure records from CSV"
            
            alert_children = [
                html.H5(f"{case_study['name']} Loaded Successfully!", className="alert-heading"),
                html.P(f"Loaded {len(tracts_gdf)} census tract geometries"),
                html.P(f"Loaded {len(demographics_df)} demographic records"),
                html.P(exposure_msg),
                html.P(f"Loaded {len(mortality_df)} mortality records"),
            ]
            if mortality_economic_df is not None:
                alert_children.append(
                    html.P(
                        f"Loaded {len(mortality_economic_df)} mortality economic tract records",
                        className="mb-0",
                    )
                )
            else:
                alert_children[-1] = html.P(
                    f"Loaded {len(mortality_df)} mortality records",
                    className="mb-0",
                )

            status = dbc.Alert(alert_children, color="success", className="mt-2")
            
            return status, state
            
        except FileNotFoundError as e:
            status = dbc.Alert(
                [
                    html.H5("Example Data Not Found", className="alert-heading"),
                    html.P(str(e)),
                    html.P("Please ensure all required files exist for this case study.", className="mb-0")
                ],
                color="warning",
                className="mt-2"
            )
            return status, state
        except Exception as e:
            status = dbc.Alert(
                [
                    html.H5("Error Loading Example Data", className="alert-heading"),
                    html.P(f"Error: {str(e)}")
                ],
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-tracts-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-tracts', 'contents'),
        [State('upload-tracts', 'filename'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_tract_data(contents, filename, state):
        if contents is None:
            return "", state
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            
            tracts_gdf = gpd.read_file(tmp_path)
            
            Path(tmp_path).unlink()
            
            global workflow_instance, cached_geojson, cached_center
            cached_geojson = None
            cached_center = None
            
            if workflow_instance is None:
                workflow_instance = Workflow()
            
            # Load tract geometries (should only have GEOID and geometry); CRS comes from file metadata
            workflow_instance.inputs.load_tract_geometries(tracts_gdf)
            
            state['tracts_loaded'] = True
            state['n_tracts'] = len(tracts_gdf)
            
            status = dbc.Alert(
                f"Successfully loaded {len(tracts_gdf)} census tracts (CRS: {workflow_instance.inputs.crs})",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error loading tract data: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-demographics-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-demographics', 'contents'),
        [State('upload-demographics', 'filename'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_demographics_data(contents, filename, state):
        if contents is None:
            return "", state
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            demographics_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            global workflow_instance
            if workflow_instance is None:
                workflow_instance = Workflow()
            
            workflow_instance.inputs.load_demographics(demographics_df)
            
            state['demographics_loaded'] = True
            state['n_demographics'] = len(demographics_df)
            
            status = dbc.Alert(
                f"Successfully loaded demographics data with {len(demographics_df)} records",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error loading demographics data: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-exposure-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-exposure', 'contents'),
        [State('upload-exposure', 'filename'),
         State('exposure-source-radio', 'value'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_exposure_data(contents, filename, exposure_source, state):
        if contents is None:
            return "", state
        
        # Only process if CSV mode is selected
        if exposure_source != 'csv':
            return "", state
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            exposure_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            global workflow_instance
            if workflow_instance is None:
                workflow_instance = Workflow()
            
            # Rename column if needed
            if 'pollutant_concentration' in exposure_df.columns:
                exposure_df = exposure_df.rename(columns={'pollutant_concentration': 'ufp'})
            elif 'baseline_pollutant_concentration' in exposure_df.columns:
                exposure_df = exposure_df.rename(columns={'baseline_pollutant_concentration': 'ufp'})
            
            if workflow_instance.inputs.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")
            
            workflow_instance.inputs.load_baseline_exposure(exposure_df, pollutant_columns=['ufp'])
            
            state['exposure_loaded'] = True
            state['n_exposure'] = len(exposure_df)
            state['exposure_source'] = 'csv'
            
            status = dbc.Alert(
                f"Successfully loaded exposure data with {len(exposure_df)} records",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error loading exposure data: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-mortality-incidence-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-mortality-incidence', 'contents'),
        [State('upload-mortality-incidence', 'filename'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_mortality_incidence_data(contents, filename, state):
        if contents is None:
            return "", state
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            incidence_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            global workflow_instance
            if workflow_instance is None:
                workflow_instance = Workflow()
            
            if workflow_instance.inputs.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")
            
            workflow_instance.inputs.load_incidence_data(incidence_df, endpoint_columns=['mortality_rate'])
            
            state['mortality_loaded'] = True
            state['n_mortality'] = len(incidence_df)
            
            status = dbc.Alert(
                f"Successfully loaded incidence data with {len(incidence_df)} records",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error loading incidence data: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-mortality-economic-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-mortality-economic', 'contents'),
        [State('upload-mortality-economic', 'filename'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_mortality_economic_data(contents, filename, state):
        if contents is None:
            return "", state

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            economic_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            global workflow_instance
            if workflow_instance is None:
                workflow_instance = Workflow()

            if workflow_instance.inputs.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")

            workflow_instance.inputs.load_mortality_economic_tract_data(economic_df)

            state = dict(state) if state else {}
            state['mortality_economic_loaded'] = True
            state['n_mortality_economic'] = len(economic_df)

            status = dbc.Alert(
                f"Successfully loaded mortality economic tract data ({len(economic_df)} records)",
                color="success",
                className="mt-2",
            )
            return status, state

        except Exception as e:
            status = dbc.Alert(
                f"Error loading mortality economic data: {str(e)}",
                color="danger",
                className="mt-2",
            )
            return status, state

    @app.callback(
        Output('mortality-pipeline-status', 'children'),
        Input('workflow-state', 'data')
    )
    def update_mortality_pipeline_status(state):
        """Update status badge for mortality pipeline."""
        if not state:
            return dbc.Badge("Not Ready", color="secondary", className="ms-2")
        
        has_data = state.get('mortality_loaded', False)
        has_demographics = state.get('demographics_loaded', False)
        
        if has_data and has_demographics:
            return dbc.Badge("Ready", color="success", className="ms-2")
        elif has_data or has_demographics:
            return dbc.Badge("Partial", color="warning", className="ms-2")
        else:
            return dbc.Badge("Not Ready", color="secondary", className="ms-2")
    
    @app.callback(
        Output('preterm-birth-pipeline-status', 'children'),
        Input('workflow-state', 'data')
    )
    def update_preterm_birth_pipeline_status(state):
        """Update status badge for preterm birth pipeline."""
        if not state:
            return dbc.Badge("Not Ready", color="secondary", className="ms-2")
        
        has_data = state.get('ptb_data_loaded', False)
        has_demographics = state.get('demographics_loaded', False)
        
        if has_data and has_demographics:
            return dbc.Badge("Ready", color="success", className="ms-2")
        elif has_data or has_demographics:
            return dbc.Badge("Partial", color="warning", className="ms-2")
        else:
            return dbc.Badge("Not Ready", color="secondary", className="ms-2")
    
    @app.callback(
        Output('mortality-function-dropdown', 'options'),
        Output('mortality-function-dropdown', 'value'),
        Input('workflow-state', 'data'),
        prevent_initial_call=False
    )
    def load_mortality_functions(state):
        functions = _mortality_function_rows()
        options = [
            {'label': func['title'], 'value': func['id']}
            for func in functions
        ]
        
        default_value = 0 if functions else None
        
        return options, default_value
    
    @app.callback(
        Output('mortality-function-details', 'children'),
        Input('mortality-function-dropdown', 'value'),
        prevent_initial_call=False
    )
    def update_function_details(function_id):
        if function_id is None:
            return html.Div("Select a mortality function to view details", className="text-muted")
        
        function_data = _mortality_functions().get(function_id)
        
        if function_data is None:
            return html.Div("Function not found", className="text-danger")
        
        return dbc.Card([
            dbc.CardBody([
                html.H6(function_data['title'], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Mean Relative Risk", className="fw-bold"),
                        html.P(f"{function_data['mean_rr']:.4f}", className="mb-2"),
                    ], md=6),
                    dbc.Col([
                        html.Label("Unit Increase (pt/cm³)", className="fw-bold"),
                        html.P(f"{function_data['unit_increase']:.1f}", className="mb-2"),
                    ], md=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Lower 95% CI", className="fw-bold"),
                        html.P(f"{function_data['lower_rr']:.4f}", className="mb-2"),
                    ], md=6),
                    dbc.Col([
                        html.Label("Upper 95% CI", className="fw-bold"),
                        html.P(f"{function_data['upper_rr']:.4f}", className="mb-2"),
                    ], md=6),
                ]),
            ])
        ], className="bg-light")
    
    @app.callback(
        Output('mortality-function-checkboxes', 'children'),
        Input('selected-mortality-functions-store', 'data'),
        prevent_initial_call=False
    )
    def create_mortality_function_checkboxes(selected):
        functions = _mortality_function_rows()

        if not functions:
            return html.Div("No mortality functions available", className="text-muted")

        selected_set = set(selected or [])
        checkboxes = []
        for func in functions:
            checkbox_id = {'type': 'mortality-function-checkbox', 'index': func['id']}
            fid = func['id']
            checkboxes.append(
                dbc.Checklist(
                    options=[{'label': func['title'], 'value': fid}],
                    value=[fid] if fid in selected_set else [],
                    id=checkbox_id,
                    className="mb-2"
                )
            )

        return html.Div(checkboxes)
    
    @app.callback(
        Output('saf-scenarios-list', 'children'),
        Input('saf-scenarios-store', 'data'),
        prevent_initial_call=False
    )
    def update_saf_scenarios_list(scenarios):
        if not scenarios:
            return html.Div("No scenarios defined", className="text-muted")
        
        items = []
        for i, scenario in enumerate(scenarios):
            items.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id={'type': 'saf-scenario-input', 'index': i},
                            type='number',
                            value=scenario,
                            min=0,
                            step=1,
                            debounce=True,
                            className="mb-2"
                        ),
                    ], md=10),
                    dbc.Col([
                        dbc.Button(
                            "×",
                            id={'type': 'saf-scenario-remove', 'index': i},
                            color="danger",
                            size="sm",
                            className="w-100 mb-2"
                        ),
                    ], md=2),
                ], className="mb-2")
            )
        
        return html.Div(items)
    
    @app.callback(
        [Output('saf-scenarios-store', 'data'),
         Output('saf-scenarios-status', 'children')],
        Input('btn-add-scenario', 'n_clicks'),
        Input({'type': 'saf-scenario-remove', 'index': ALL}, 'n_clicks'),
        Input({'type': 'saf-scenario-input', 'index': ALL}, 'value'),
        State('saf-scenarios-store', 'data'),
        prevent_initial_call=False
    )
    def manage_saf_scenarios(add_clicks, remove_clicks, input_values, current_scenarios):
        if not dash_ctx.triggered:
            return current_scenarios or [25, 50], ""
        
        trigger_id = dash_ctx.triggered[0]['prop_id']
        
        if 'btn-add-scenario' in trigger_id:
            new_scenarios = current_scenarios.copy() if current_scenarios else [25, 50]
            new_scenarios.append(0)
            return new_scenarios, html.Small("Scenario added", className="text-success")
        
        if 'saf-scenario-remove' in trigger_id:
            trigger_data = json.loads(trigger_id.split('.')[0])
            index = trigger_data['index']
            new_scenarios = current_scenarios.copy() if current_scenarios else [25, 50]
            if 0 <= index < len(new_scenarios):
                new_scenarios.pop(index)
            return new_scenarios, html.Small("Scenario removed", className="text-success")
        
        if 'saf-scenario-input' in trigger_id:
            if not input_values:
                return current_scenarios or [25, 50], ""
            out = []
            clamped = False
            for v in input_values:
                if v is None or v == '':
                    out.append(None)
                    continue
                try:
                    num = float(v)
                except (TypeError, ValueError):
                    out.append(None)
                    continue
                if num < 0 or num > 50:
                    clamped = True
                out.append(int(round(max(0, min(50, num)))))
            msg = (
                html.Small(
                    "Scenarios must be between 0 and 50 (out-of-range values were adjusted).",
                    className="text-warning",
                )
                if clamped
                else ""
            )
            return out, msg
        
        return current_scenarios or [25, 50], ""
    
    @app.callback(
        Output('selected-mortality-functions-store', 'data'),
        Input({'type': 'mortality-function-checkbox', 'index': ALL}, 'value'),
        prevent_initial_call=False
    )
    def collect_selected_mortality_functions(checkbox_values):
        selected = []
        if checkbox_values:
            for values in checkbox_values:
                if values and isinstance(values, list):
                    selected.extend(values)
                elif values:
                    selected.append(values)
        return list(set(selected)) if selected else []
    
    @app.callback(
        [Output('config-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('selected-mortality-functions-store', 'data'),
        Input('saf-scenarios-store', 'data'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def update_config(selected_functions, scenarios, state):
        state['config_set'] = True
        state['config_explicitly_set'] = True
        state['scenarios'] = scenarios if scenarios else [25, 50]
        state['selected_mortality_functions'] = selected_functions if selected_functions else []
        
        status = dbc.Alert(
            "Configuration updated successfully",
            color="success",
            className="mt-2"
        )
        
        return status, state
    
    @app.callback(
        [Output('upload-ptb-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-ptb-data', 'contents'),
        [State('upload-ptb-data', 'filename'),
         State('workflow-state', 'data')],
        prevent_initial_call=True
    )
    def upload_preterm_birth_data(contents, filename, state):
        if contents is None:
            return "", state

        try:
            global workflow_instance

            if workflow_instance is None:
                msg = dbc.Alert("Workflow not initialized. Please load data first.", color="warning")
                return msg, state

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            if filename.endswith('.csv'):
                ptb_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            else:
                msg = dbc.Alert("Unsupported file format. Please upload a CSV file.", color="danger")
                return msg, state

            workflow_instance.inputs.load_preterm_birth_data(ptb_df)

            state['ptb_data_loaded'] = True

            status = html.Span(f"✓ Loaded {len(ptb_df)} records from {filename}", className="text-success")
            return status, state

        except Exception as e:
            status = html.Span(f"✗ Error loading file: {str(e)}", className="text-danger")
            return status, state
    
    @app.callback(
        Output('btn-run-analysis', 'disabled'),
        Input('workflow-state', 'data')
    )
    def enable_analysis_button(state):
        if not state:
            return True
        
        # Core required data: tracts, demographics, exposure
        # Mortality and preterm birth are optional (pipeline-specific)
        all_loaded = (
            state.get('tracts_loaded', False) and
            state.get('demographics_loaded', False) and
            state.get('exposure_loaded', False) and
            state.get('config_set', False) and
            state.get('config_explicitly_set', False) and
            state.get('config_tab_visited', False)
        )
        
        return not all_loaded
    
    @app.callback(
        [Output('analysis-status', 'children'),
         Output('analysis-results', 'data')],
        Input('btn-run-analysis', 'n_clicks'),
        State('workflow-state', 'data'),
        State('saf-scenarios-store', 'data'),
        prevent_initial_call=True
    )
    def run_analysis(n_clicks, state, scenarios_store):
        import traceback
        
        if n_clicks is None:
            return "", {}
        
        try:
            global workflow_instance
            
            if workflow_instance is None:
                raise ValueError("Workflow not initialized. Please load data first.")
            
            raw_scenarios = scenarios_store if scenarios_store else state.get('scenarios', [25, 50])
            scenarios = _numeric_saf_scenarios(raw_scenarios)
            
            # Update config with scenarios
            workflow_instance.config.saf_scenarios = scenarios
            
            analysis_results = workflow_instance.run_scenarios(scenarios=scenarios, pollutant_name='ufp')

            pop_series = None
            dc = workflow_instance.inputs.demographics_core
            if dc is not None and 'population' in dc.columns:
                pop_series = dc['population']

            results = {}
            for scenario in scenarios:
                scenario_id = int(scenario)
                sr = analysis_results.scenarios.get(scenario_id)
                if sr is None:
                    continue

                scenario_agg = sr.get_aggregated_results(population=pop_series)

                if 'mortality' in scenario_agg:
                    mortality_agg = scenario_agg['mortality']
                    tac = mortality_agg['total_attributable_cases']
                    total_cases = tac.mean
                    lower_cases = tac.lower
                    upper_cases = tac.upper
                else:
                    total_cases = 0.0
                    lower_cases = 0.0
                    upper_cases = 0.0

                pollutant_reduction = sr.pollutant_reduction

                result_dict = {
                    'scenario': scenario,
                    'saf_percentage': scenario,
                    'pollutant_reduction': float(pollutant_reduction),
                    'total_cases': float(total_cases),
                    'lower_cases': float(lower_cases),
                    'upper_cases': float(upper_cases),
                }

                if 'economic_benefits' in scenario_agg:
                    econ_agg = scenario_agg['economic_benefits']
                    if 'mortality_economic_value' in econ_agg:
                        ev = econ_agg['mortality_economic_value']
                        result_dict['mortality_economic_value'] = float(ev.mean)
                        result_dict['mortality_economic_value_lower'] = float(ev.lower)
                        result_dict['mortality_economic_value_upper'] = float(ev.upper)

                    if 'preterm_birth_economic_value' in econ_agg:
                        ev = econ_agg['preterm_birth_economic_value']
                        result_dict['ptb_economic_value'] = float(ev.mean)
                        result_dict['ptb_economic_value_lower'] = float(ev.lower)
                        result_dict['ptb_economic_value_upper'] = float(ev.upper)

                results[str(scenario)] = result_dict
            
            status = html.Div([
                html.Span("✓ Analysis Complete! ", className="text-success fw-bold"),
                html.Span(f"{len(scenarios)} scenarios analyzed. ", className="text-muted"),
                html.A("View Results", href="#", className="text-primary", style={"textDecoration": "underline"})
            ])
            
            return status, results
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Analysis error traceback:\n{error_traceback}")
            
            status = html.Div([
                html.Span("✗ Analysis Failed: ", className="text-danger fw-bold"),
                html.Span(str(e), className="text-muted"),
                html.Br(),
                html.Small(f"See console for full traceback", className="text-muted")
            ])
            return status, {}
    
    @app.callback(
        Output('results-bar-chart', 'figure'),
        Input('analysis-results', 'data')
    )
    def update_bar_chart(results):
        empty = dict(
            template="plotly_white",
            height=400,
            annotations=[
                {
                    "text": "No analysis results available. Please run analysis first.",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14, "color": "gray"},
                }
            ],
            uirevision="constant",
        )
        if not results:
            return go.Figure().update_layout(
                title="Health impacts by SAF scenario",
                xaxis_title="SAF scenario",
                yaxis_title="Attributable cases avoided",
                **empty,
            )

        scenarios = []
        mean_cases = []
        lower_cases = []
        upper_cases = []

        for scenario_key in sorted(results.keys(), key=lambda x: int(x)):
            scenario_results = results[scenario_key]
            scenarios.append(f"{scenario_results['scenario']}% SAF")
            mean_cases.append(scenario_results["total_cases"])
            lower_cases.append(scenario_results["lower_cases"])
            upper_cases.append(scenario_results["upper_cases"])

        has_economic = any("total_economic_benefits" in results[k] for k in results.keys())
        has_mort_econ = any("mortality_economic_value" in results[k] for k in results.keys())
        has_ptb_econ = any("ptb_economic_value" in results[k] for k in results.keys())

        err_cases = dict(
            type="data",
            symmetric=False,
            array=[u - m for u, m in zip(upper_cases, mean_cases)],
            arrayminus=[m - l for m, l in zip(mean_cases, lower_cases)],
        )

        if has_economic or has_mort_econ or has_ptb_econ:
            econ_values = []
            econ_lower = []
            econ_upper = []

            for scenario_key in sorted(results.keys(), key=lambda x: int(x)):
                scenario_results = results[scenario_key]
                if "total_economic_benefits" in scenario_results:
                    econ_values.append(scenario_results["total_economic_benefits"] / 1e6)
                    econ_lower.append(scenario_results.get("total_economic_benefits_lower", 0) / 1e6)
                    econ_upper.append(scenario_results.get("total_economic_benefits_upper", 0) / 1e6)
                elif "mortality_economic_value" in scenario_results:
                    econ_values.append(scenario_results["mortality_economic_value"] / 1e6)
                    econ_lower.append(scenario_results.get("mortality_economic_value_lower", 0) / 1e6)
                    econ_upper.append(scenario_results.get("mortality_economic_value_upper", 0) / 1e6)
                else:
                    econ_values.append(0)
                    econ_lower.append(0)
                    econ_upper.append(0)

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.14,
                subplot_titles=(
                    "Attributable cases avoided",
                    "Economic benefits ($ millions, mean and 95% CI)",
                ),
                row_heights=[0.52, 0.48],
            )
            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=mean_cases,
                    name="Cases",
                    error_y=err_cases,
                    marker_color=PLOT_PRIMARY,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=econ_values,
                    name="Economic ($M)",
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[u - m for u, m in zip(econ_upper, econ_values)],
                        arrayminus=[m - l for m, l in zip(econ_values, econ_lower)],
                    ),
                    marker_color=PLOT_SECONDARY,
                ),
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text="SAF scenario", row=2, col=1)
            fig.update_yaxes(title_text="Cases", row=1, col=1)
            fig.update_yaxes(title_text="$M", row=2, col=1)
            fig.update_layout(
                title="Health impacts and economic benefits by SAF scenario",
                template="plotly_white",
                height=520,
                uirevision="constant",
                showlegend=False,
                margin=dict(t=64),
            )
            return fig

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=scenarios,
                y=mean_cases,
                name="Cases",
                error_y=err_cases,
                marker_color=PLOT_PRIMARY,
            )
        )
        fig.update_layout(
            title="Health impacts by SAF scenario",
            xaxis_title="SAF scenario",
            yaxis_title="Attributable cases avoided",
            template="plotly_white",
            height=400,
            uirevision="constant",
        )
        return fig
    
    @app.callback(
        [Output('results-scenario-dropdown', 'options'),
         Output('results-scenario-dropdown', 'value')],
        Input('analysis-results', 'data')
    )
    def update_scenario_dropdown(results):
        if not results:
            return [], None
        
        options = []
        for scenario_key in sorted(results.keys(), key=lambda x: int(x)):
            scenario_results = results[scenario_key]
            scenario_num = scenario_results['scenario']
            options.append({
                'label': f'{scenario_num}% SAF Blend',
                'value': str(scenario_num)
            })
        
        default_value = options[0]['value'] if options else None
        return options, default_value
    
    @app.callback(
        [Output('results-map-dropdown', 'options'),
         Output('results-map-dropdown', 'disabled'),
         Output('results-map-dropdown', 'value')],
        Input('results-scenario-dropdown', 'value'),
        Input('analysis-results', 'data'),
        State('results-map-dropdown', 'value')
    )
    def update_variable_dropdown(selected_scenario, results, current_value):
        if not selected_scenario or not results:
            return [], True, None
        
        global workflow_instance
        
        options = [
            {'label': 'Attributable Cases Avoided', 'value': 'attributable_cases'},
            {'label': 'Attributable Fraction', 'value': 'attributable_fraction'},
            {'label': 'Relative Risk', 'value': 'relative_risk'},
            {'label': 'Delta Concentration', 'value': 'delta_concentration'},
            {'label': 'Reduced Concentration', 'value': 'reduced_concentration'},
        ]
        
        # Add economic benefit options if available
        if workflow_instance is not None and workflow_instance.results:
            scenarios_map = workflow_instance.results.scenarios
            scenario_id = int(selected_scenario)
            if scenario_id in scenarios_map:
                scenario_result = scenarios_map[scenario_id]
                econ_names = {b.name for b in scenario_result.economic_benefits}
                if 'mortality_economic_value' in econ_names:
                    options.append({'label': 'Mortality Economic Value ($)', 'value': 'mortality_economic_value'})
                if 'preterm_birth_economic_value' in econ_names:
                    options.append({'label': 'Preterm Birth Economic Value ($)', 'value': 'preterm_birth_economic_value'})
                if 'preterm_birth_reduction' in econ_names:
                    options.append({'label': 'Preterm Birth Reduction', 'value': 'preterm_birth_reduction'})
        
        if current_value is None:
            return options, False, options[0]['value']
        
        option_values = [opt['value'] for opt in options]
        if current_value in option_values:
            return options, False, current_value
        
        return options, False, options[0]['value']
    
    @app.callback(
        Output('results-map', 'figure'),
        Input('workflow-state', 'data'),
        Input('analysis-results', 'data'),
        Input('results-scenario-dropdown', 'value'),
        Input('results-map-dropdown', 'value')
    )
    def update_map(workflow_state, results, selected_scenario, selected_variable):
        global workflow_instance, cached_geojson, cached_center
        
        if workflow_instance is None or workflow_instance.inputs.tract_geometries is None:
            cached_geojson = None
            cached_center = None
            return _map_annotation_layout(
                "No data loaded",
                "Please load inputs from the Data tab.",
            )

        if not results or not workflow_instance.results or not workflow_instance.results.scenarios:
            return _map_annotation_layout(
                "No analysis results available",
                "Run analysis using Execute Analysis in the header.",
            )

        if not selected_scenario or not selected_variable:
            return _map_annotation_layout(
                "Select scenario and variable",
                "Choose a SAF scenario and a result variable from the dropdowns above.",
            )
        
        scenario_num = int(selected_scenario)
        scenario_id = scenario_num
        
        scenarios_map = workflow_instance.results.scenarios
        if scenario_id not in scenarios_map:
            return go.Figure().update_layout(
                title=f"Scenario {scenario_num}% not found",
                height=600,
                template="plotly_white",
                uirevision="constant",
            )
        
        gdf = workflow_instance.results.get_merged_data()
        scenario_result = scenarios_map[scenario_id]
        scenario_df = scenario_result.to_dataframe()
        scenario_name = scenario_result.spec.scenario_label
        
        # Map variable names to column names in scenario outputs
        # Health impacts are now nested under endpoint names (e.g., 'mortality_attributable_cases_mean')
        variable_mapping = {
            'attributable_cases': ('mortality_attributable_cases_mean', 'Attributable Cases Avoided'),
            'attributable_fraction': ('mortality_attributable_fraction_mean', 'Attributable Fraction'),
            'relative_risk': ('mortality_relative_risk_mean', 'Relative Risk'),
            'delta_concentration': ('delta_concentration', 'Delta Concentration (pt/cm³)'),
            'reduced_concentration': ('reduced_concentration', 'Reduced Concentration (pt/cm³)'),
            'mortality_economic_value': ('mortality_economic_value_mean', 'Mortality Economic Value ($)'),
            'preterm_birth_economic_value': ('preterm_birth_economic_value_mean', 'Preterm Birth Economic Value ($)'),
            'preterm_birth_reduction': ('preterm_birth_reduction_mean', 'Preterm Birth Reduction')
        }
        
        if selected_variable not in variable_mapping:
            selected_variable = 'attributable_cases'
        
        column_name, colorbar_title = variable_mapping[selected_variable]
        
        # Check if column exists in merged data (which has prefixed scenario columns)
        prefixed_column = f"{scenario_name}_{column_name}"
        if prefixed_column in gdf.columns:
            z_column = prefixed_column
        elif column_name in scenario_df.columns:
            # Merge scenario data
            gdf = gdf.join(scenario_df[[column_name]], how='left')
            z_column = column_name
        else:
            return go.Figure().update_layout(
                title=f"Variable '{selected_variable}' not available for scenario {scenario_num}%",
                height=600,
                template="plotly_white",
                uirevision="constant",
            )
        
        if cached_geojson is None or cached_center is None:
            gdf_simplified = gdf.copy()
            gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(tolerance=0.001, preserve_topology=True)
            
            gdf_json = json.loads(gdf_simplified.to_json())
            
            if not gdf_json.get('features'):
                return go.Figure().update_layout(
                    title="No features to display",
                    height=600,
                    template="plotly_white",
                    uirevision="constant",
                )
            
            # Project to a projected CRS for centroid calculation to avoid geographic CRS warning
            if gdf_simplified.crs and gdf_simplified.crs.is_geographic:
                # Use Web Mercator for centroid calculation
                gdf_projected = gdf_simplified.to_crs('EPSG:3857')
                centroids = gdf_projected.geometry.centroid
                # Calculate mean in projected CRS, then convert back to geographic
                mean_x = centroids.x.mean()
                mean_y = centroids.y.mean()
                mean_point = gpd.GeoDataFrame([1], geometry=gpd.points_from_xy([mean_x], [mean_y]), crs='EPSG:3857')
                mean_point_geo = mean_point.to_crs(gdf_simplified.crs)
                center_lat = mean_point_geo.geometry.iloc[0].y
                center_lon = mean_point_geo.geometry.iloc[0].x
            else:
                centroids = gdf_simplified.geometry.centroid
                center_lat = centroids.y.mean()
                center_lon = centroids.x.mean()
            
            cached_geojson = gdf_json
            cached_center = (center_lat, center_lon)
        else:
            gdf_json = cached_geojson
            center_lat, center_lon = cached_center
        
        for feature in gdf_json['features']:
            feature['id'] = str(feature['properties'].get('GEOID', feature.get('id', '')))
        
        z_values = gdf[z_column].fillna(0).tolist()
        locations = [str(f['id']) for f in gdf_json['features']]
        
        variable_labels = {
            'attributable_cases': 'Attributable Cases Avoided',
            'attributable_fraction': 'Attributable Fraction',
            'relative_risk': 'Relative Risk',
            'delta_concentration': 'Delta Concentration',
            'reduced_concentration': 'Reduced Concentration',
            'mortality_economic_value': 'Mortality Economic Value',
            'preterm_birth_economic_value': 'Preterm Birth Economic Value',
            'preterm_birth_reduction': 'Preterm Birth Reduction'
        }
        var_label = variable_labels.get(selected_variable, selected_variable)
        title_text = f'{var_label} by Census Tract ({scenario_num}% SAF)'
        
        if "economic" in selected_variable.lower():
            colorscale = _mpl_plotly_colorscale("Blues")
        elif "reduction" in selected_variable.lower() or "ptb" in selected_variable.lower():
            colorscale = _mpl_plotly_colorscale("YlOrRd")
        else:
            colorscale = _mpl_plotly_colorscale("YlGnBu")
        
        customdata = [[gid, z] for gid, z in zip(locations, z_values)]
        hovertemplate = (
            f"<b>{var_label}</b><br>"
            "GEOID: %{customdata[0]}<br>"
            "Value: %{customdata[1]:,.4f}<extra></extra>"
        )
        choropleth_kw = dict(
            geojson=gdf_json,
            locations=locations,
            z=z_values,
            colorscale=colorscale,
            marker_opacity=0.68,
            marker_line_width=0.5,
            marker_line_color="white",
            colorbar=dict(title=colorbar_title),
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
        if selected_variable == 'reduced_concentration':
            choropleth_kw['zmin'] = 0
            choropleth_kw['zmax'] = 1500

        fig = go.Figure(go.Choroplethmapbox(**choropleth_kw))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=9
            ),
            title=title_text,
            height=600,
            margin={"r":0,"t":40,"l":0,"b":0},
            template="plotly_white",
            uirevision=f"{selected_scenario}_{selected_variable}"
        )
        
        return fig
    
    @app.callback(
        Output('results-summary-table', 'children'),
        Input('results-scenario-dropdown', 'value'),
        Input('analysis-results', 'data')
    )
    def update_results_summary_table(selected_scenario, results):
        if not selected_scenario or not results:
            return html.Small(
                "Run analysis, then pick a SAF scenario above to see key metrics.",
                className="text-muted",
            )
        
        global workflow_instance
        
        if workflow_instance is None or not workflow_instance.results or not workflow_instance.results.scenarios:
            return html.Small("Run analysis to populate scenario metrics.", className="text-muted")

        scenario_num = int(selected_scenario)
        scenario_id = scenario_num
        scenarios_map = workflow_instance.results.scenarios

        if scenario_id not in scenarios_map:
            return html.Small("Selected scenario is not available.", className="text-muted")
        
        scenario_result = scenarios_map[scenario_id]
        
        summary_data = []
        
        # Get pollutant reduction from scenario result
        pollutant_reduction = scenario_result.pollutant_reduction
        summary_data.append({
            'Metric': 'Pollutant Reduction',
            'Value': f"{pollutant_reduction:.2f}%"
        })
        
        if str(scenario_num) in results:
            scenario_results = results[str(scenario_num)]
            summary_data.append({
                'Metric': 'Total Attributable Cases Avoided',
                'Value': f"{scenario_results['total_cases']:.2f}"
            })
            summary_data.append({
                'Metric': 'Lower 95% CI',
                'Value': f"{scenario_results['lower_cases']:.2f}"
            })
            summary_data.append({
                'Metric': 'Upper 95% CI',
                'Value': f"{scenario_results['upper_cases']:.2f}"
            })
        
        if not summary_data:
            return html.Small("No summary rows for this scenario.", className="text-muted")

        rows = []
        for item in summary_data:
            rows.append(html.Tr([
                html.Td(item['Metric'], className="fw-bold"),
                html.Td(item['Value'])
            ]))
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Metric"),
                html.Th("Value")
            ])),
            html.Tbody(rows)
        ], bordered=True, hover=True, responsive=True, striped=True, size='sm', className="mb-0")
        
        return table
    
    @app.callback(
        Output('results-table', 'children'),
        Input('analysis-results', 'data')
    )
    def update_results_table(results):
        if not results:
            return ""
        
        rows = []
        for scenario_key in sorted(results.keys(), key=lambda x: int(x)):
            scenario_results = results[scenario_key]
            rows.append(html.Tr([
                html.Td(f"{scenario_results['scenario']}%"),
                html.Td(f"{scenario_results['total_cases']:.2f}"),
                html.Td(f"{scenario_results['lower_cases']:.2f}"),
                html.Td(f"{scenario_results['upper_cases']:.2f}"),
            ]))
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("SAF Blend"),
                html.Th("Mean Cases"),
                html.Th("Lower 95% CI"),
                html.Th("Upper 95% CI"),
            ])),
            html.Tbody(rows)
        ], bordered=True, hover=True, responsive=True, striped=True, size='sm', className="mb-0")
        
        return table

    @app.callback(
        Output('download-results-summary-csv', 'data'),
        Input('btn-export-results-summary-csv', 'n_clicks'),
        State('analysis-results', 'data'),
        prevent_initial_call=True,
    )
    def export_results_summary_csv(n_clicks, results):
        if not n_clicks or not results:
            raise PreventUpdate
        rows = [results[k] for k in sorted(results.keys(), key=lambda x: int(x))]
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "bensaf_scenario_summary.csv", index=False)

    @app.callback(
        Output('download-results-tract-csv', 'data'),
        Input('btn-export-results-tract-csv', 'n_clicks'),
        State('analysis-results', 'data'),
        prevent_initial_call=True,
    )
    def export_results_tract_csv(n_clicks, results):
        global workflow_instance
        if not n_clicks or not results:
            raise PreventUpdate
        if workflow_instance is None or not workflow_instance.results:
            raise PreventUpdate
        gdf = workflow_instance.results.get_merged_data()
        df = gdf.drop(columns=["geometry"], errors="ignore").reset_index()
        return dcc.send_data_frame(df.to_csv, "bensaf_tract_level_results.csv", index=False)
    
    @app.callback(
        [Output('data-viewer-dropdown', 'options'),
         Output('data-viewer-dropdown', 'value')],
        Input('workflow-state', 'data')
    )
    def update_data_viewer_dropdown(workflow_state):
        global workflow_instance
        
        if workflow_instance is None or workflow_instance.inputs.tract_geometries is None:
            return [], None
        
        gdf = _inputs_core_merged_gdf(workflow_instance)
        available_cols = [col for col in gdf.columns if col not in ['geometry']]
        
        options = [
            {'label': col.replace('_', ' ').title(), 'value': col}
            for col in available_cols
        ]
        
        # Set default value to first available column (prefer population or baseline_pollutant_concentration)
        default_value = None
        for preferred in ['population', 'baseline_pollutant_concentration', 'mortality_rate']:
            if preferred in available_cols:
                default_value = preferred
                break
        
        if default_value is None and available_cols:
            default_value = available_cols[0]
        
        return options, default_value
    
    @app.callback(
        Output('data-viewer-map', 'figure'),
        Input('data-viewer-dropdown', 'value'),
        Input('workflow-state', 'data')
    )
    def update_data_viewer_map(selected_variable, workflow_state):
        global workflow_instance, cached_geojson, cached_center
        
        if workflow_instance is None or workflow_instance.inputs.tract_geometries is None:
            return _map_annotation_layout(
                "No data loaded",
                "Please load inputs from the Data tab.",
            )

        if selected_variable is None:
            return go.Figure().update_layout(
                title="Please select a variable",
                height=600,
                template="plotly_white",
                uirevision="constant",
            )
        
        gdf = _inputs_core_merged_gdf(workflow_instance)
        
        if selected_variable not in gdf.columns:
            available_cols = [col for col in gdf.columns if col not in ['geometry']]
            return go.Figure().update_layout(
                title=f"Variable '{selected_variable}' not available",
                height=600,
                template="plotly_white",
                annotations=[
                    {
                        "text": f'Available columns: {", ".join(available_cols[:5])}...',
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "showarrow": False,
                        "font": {"size": 14},
                    }
                ],
                uirevision="constant",
            )
        
        # Create a readable label for the colorbar
        colorbar_title = selected_variable.replace('_', ' ').title()
        
        # Add units for specific variables
        if 'baseline_pollutant_concentration' in selected_variable or 'concentration' in selected_variable.lower():
            colorbar_title += ' (pt/cm³)'
        elif 'mortality_rate' in selected_variable or 'rate' in selected_variable.lower():
            colorbar_title += ' (rate)'
        elif 'distance' in selected_variable.lower():
            colorbar_title += ' (km)'
        
        if cached_geojson is None or cached_center is None:
            gdf_simplified = gdf.copy()
            gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(tolerance=0.001, preserve_topology=True)
            
            gdf_json = json.loads(gdf_simplified.to_json())
            
            # Project to a projected CRS for centroid calculation to avoid geographic CRS warning
            if gdf_simplified.crs and gdf_simplified.crs.is_geographic:
                # Use Web Mercator for centroid calculation
                gdf_projected = gdf_simplified.to_crs('EPSG:3857')
                centroids = gdf_projected.geometry.centroid
                # Calculate mean in projected CRS, then convert back to geographic
                mean_x = centroids.x.mean()
                mean_y = centroids.y.mean()
                mean_point = gpd.GeoDataFrame([1], geometry=gpd.points_from_xy([mean_x], [mean_y]), crs='EPSG:3857')
                mean_point_geo = mean_point.to_crs(gdf_simplified.crs)
                center_lat = mean_point_geo.geometry.iloc[0].y
                center_lon = mean_point_geo.geometry.iloc[0].x
            else:
                centroids = gdf_simplified.geometry.centroid
                center_lat = centroids.y.mean()
                center_lon = centroids.x.mean()
            
            cached_geojson = gdf_json
            cached_center = (center_lat, center_lon)
        else:
            gdf_json = cached_geojson
            center_lat, center_lon = cached_center
        
        for feature in gdf_json['features']:
            feature['id'] = str(feature['properties'].get('GEOID', feature.get('id', '')))
        
        z_values = gdf[selected_variable].fillna(0).tolist()
        locations = [str(f['id']) for f in gdf_json['features']]
        customdata = [[gid, z] for gid, z in zip(locations, z_values)]
        hovertemplate = (
            f"<b>{colorbar_title}</b><br>"
            "GEOID: %{customdata[0]}<br>"
            "Value: %{customdata[1]:,.4f}<extra></extra>"
        )

        choropleth_kw = dict(
            geojson=gdf_json,
            locations=locations,
            z=z_values,
            colorscale=_mpl_plotly_colorscale("YlGnBu"),
            marker_opacity=0.68,
            marker_line_width=0.5,
            marker_line_color="white",
            colorbar=dict(title=colorbar_title),
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
        if selected_variable in ('baseline_pollutant_concentration', 'ufp'):
            choropleth_kw['zmin'] = 0
            choropleth_kw['zmax'] = 1500

        fig = go.Figure(go.Choroplethmapbox(**choropleth_kw))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=9
            ),
            title=f'{colorbar_title} by Census Tract',
            height=600,
            margin={"r":0,"t":40,"l":0,"b":0},
            template="plotly_white",
            uirevision=selected_variable
        )
        
        return fig
    
    @app.callback(
        Output('saf-reduction-curve', 'figure'),
        Input('saf-scenarios-store', 'data')
    )
    def update_saf_curve(scenarios):
        import numpy as np
        
        coeffs = load_saf_blend_parameters()
        
        saf_range = np.linspace(0, 100, 101)
        
        # Calculate reductions (negative values, clamped to [-100, 0])
        # Multiply by 100 to convert from decimal to percentage
        reductions = np.array([
            max(-100.0, min(0.0, sum(coeff * (saf ** i) for i, coeff in enumerate(coeffs)) * 100))
            for saf in saf_range
        ])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=saf_range,
            y=reductions,
            mode='lines',
            name='Polynomial fit',
            line=dict(color=PLOT_PRIMARY, width=3),
        ))
        
        scenarios_list = _numeric_saf_scenarios(scenarios)
        scenario_reductions = [
            max(-100.0, min(0.0, sum(coeff * (s ** i) for i, coeff in enumerate(coeffs)) * 100))
            for s in scenarios_list
        ]
        
        if scenarios_list:
            fig.add_trace(go.Scatter(
                x=scenarios_list,
                y=scenario_reductions,
                mode='markers',
                name='Analysis scenarios',
                marker=dict(color=PLOT_SECONDARY, size=11, symbol='circle', line=dict(width=1, color='white')),
            ))
        
        equation_text = f"Reduction = {coeffs[0]:.3f} + {coeffs[1]:.3f}·SAF + {coeffs[2]:.5f}·SAF²"
        
        fig.update_layout(
            title='SAF Blend to Pollutant Reduction',
            xaxis_title='SAF Blend Percentage (%)',
            yaxis_title='Pollutant Reduction (%)',
            template='plotly_white',
            height=400,
            uirevision='constant',
            annotations=[
                {
                    "text": equation_text,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.02,
                    "y": 0.98,
                    "showarrow": False,
                    "font": {"size": 11, "color": "#5a6570"},
                    "align": "left",
                    "bgcolor": "rgba(255, 255, 255, 0.94)",
                    "bordercolor": "rgba(0, 0, 0, 0.06)",
                    "borderwidth": 1,
                }
            ],
        )
        
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[-100, 0])
        
        return fig
    
    @app.callback(
        Output('data-viewer-table', 'children'),
        Input('workflow-state', 'data')
    )
    def update_data_viewer_table(workflow_state):
        global workflow_instance
        
        if workflow_instance is None or workflow_instance.inputs.tract_geometries is None:
            return dbc.Alert(
                "No data loaded. Please load data from the Data tab.",
                color="info"
            )
        
        gdf = _inputs_core_merged_gdf(workflow_instance)
        
        # Reset index to show GEOID as a column if it's the index
        if gdf.index.name == 'GEOID' or isinstance(gdf.index, pd.Index):
            gdf = gdf.reset_index()
        
        display_cols = [col for col in gdf.columns if col != 'geometry']
        
        df_display = gdf[display_cols].head(100)
        
        header_row = html.Tr([html.Th(col.replace('_', ' ').title()) for col in display_cols])
        
        table_rows = []
        for idx, row in df_display.iterrows():
            cells = []
            for col in display_cols:
                val = row[col]
                if pd.isna(val):
                    cells.append(html.Td("—", className="text-muted"))
                elif isinstance(val, (int, float)):
                    if 'proportion' in col.lower() or 'rate' in col.lower() or 'fraction' in col.lower():
                        cells.append(html.Td(f"{val:.4f}"))
                    elif isinstance(val, int) and 'geoid' in col.lower():
                        cells.append(html.Td(str(val)))
                    else:
                        cells.append(html.Td(f"{val:.2f}"))
                else:
                    cells.append(html.Td(str(val)))
            table_rows.append(html.Tr(cells))
        
        table = dbc.Table([
            html.Thead(header_row),
            html.Tbody(table_rows)
        ], bordered=True, hover=True, responsive=True, striped=True, size='sm', className="mb-0")
        
        return html.Div([
            html.Small(
                f"Showing {len(df_display)} of {len(gdf)} total census tracts",
                className="text-muted d-block mb-2"
            ),
            table
        ])
    
    @app.callback(
        [Output('exposure-csv-upload', 'style'),
         Output('exposure-aermod-upload', 'style')],
        Input('exposure-source-radio', 'value')
    )
    def toggle_exposure_source(source):
        if source == 'csv':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    @app.callback(
        [Output('upload-landing-aermod-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True),
         Output('landing-aermod-weights-table', 'children')],
        Input('upload-landing-aermod', 'contents'),
        State('upload-landing-aermod', 'filename'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def upload_landing_aermod(contents_list, filename_list, state):
        if contents_list is None or len(contents_list) == 0:
            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['landing'] = []
            return (
                "",
                state,
                html.Small("Upload landing .ADO files to assign a weight to each file.", className="text-muted"),
            )

        try:
            if not isinstance(contents_list, list):
                contents_list = [contents_list]
            if not isinstance(filename_list, list):
                filename_list = [filename_list]

            landing_files = []
            for contents, filename in zip(contents_list, filename_list):
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    tmp.write(decoded)
                    tmp_path = tmp.name

                landing_files.append((tmp_path, filename))

            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['landing'] = landing_files

            names = [fn for _, fn in landing_files]
            status = dbc.Alert(
                f"Uploaded {len(landing_files)} landing file(s). Set weights in the table, then generate exposure.",
                color="success",
                className="mt-2 py-2",
            )

            return status, state, _aermod_weight_table_ui(names, "landing-weight")

        except Exception as e:
            status = dbc.Alert(
                f"Error uploading landing AERMOD files: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state, html.Small("Upload failed; try again.", className="text-danger small")
    
    @app.callback(
        [Output('upload-takeoff-aermod-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True),
         Output('takeoff-aermod-weights-table', 'children')],
        Input('upload-takeoff-aermod', 'contents'),
        State('upload-takeoff-aermod', 'filename'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def upload_takeoff_aermod(contents_list, filename_list, state):
        if contents_list is None or len(contents_list) == 0:
            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['takeoff'] = []
            return (
                "",
                state,
                html.Small("Upload takeoff .ADO files to assign a weight to each file.", className="text-muted"),
            )

        try:
            if not isinstance(contents_list, list):
                contents_list = [contents_list]
            if not isinstance(filename_list, list):
                filename_list = [filename_list]

            takeoff_files = []
            for contents, filename in zip(contents_list, filename_list):
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    tmp.write(decoded)
                    tmp_path = tmp.name

                takeoff_files.append((tmp_path, filename))

            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['takeoff'] = takeoff_files

            names = [fn for _, fn in takeoff_files]
            status = dbc.Alert(
                f"Uploaded {len(takeoff_files)} takeoff file(s). Set weights in the table, then generate exposure.",
                color="success",
                className="mt-2 py-2",
            )

            return status, state, _aermod_weight_table_ui(names, "takeoff-weight")

        except Exception as e:
            status = dbc.Alert(
                f"Error uploading takeoff AERMOD files: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state, html.Small("Upload failed; try again.", className="text-danger small")
    
    @app.callback(
        [Output('generate-exposure-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('btn-generate-exposure', 'n_clicks'),
        [
            State('workflow-state', 'data'),
            State({'type': 'landing-weight', 'index': ALL}, 'value'),
            State({'type': 'takeoff-weight', 'index': ALL}, 'value'),
            State('data-config', 'data'),
        ],
        prevent_initial_call=True
    )
    def generate_exposure_from_aermod(n_clicks, state, landing_weight_values, takeoff_weight_values, data_config):
        if n_clicks is None:
            return "", state

        def weights_from_inputs(raw_values, num_files):
            if num_files == 0:
                return []
            if raw_values is None or len(raw_values) != num_files:
                return [1.0 / num_files] * num_files
            out = []
            for v in raw_values:
                if v is None or v == '':
                    out.append(1.0 / num_files)
                else:
                    out.append(float(v))
            return out

        try:
            global workflow_instance

            if workflow_instance is None:
                workflow_instance = Workflow()

            if workflow_instance.inputs.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")

            if 'aermod_files' not in state:
                raise ValueError("Please upload AERMOD files first")

            aermod_files = state['aermod_files']
            landing_files = aermod_files.get('landing') or []
            takeoff_files = aermod_files.get('takeoff') or []

            if len(landing_files) == 0 and len(takeoff_files) == 0:
                raise ValueError("Upload at least one landing or takeoff AERMOD file")

            project_root = Path(__file__).parent.parent.parent
            calibration_file = project_root / 'data' / 'aermod_calibration_coefficients.json'

            if not calibration_file.exists():
                raise FileNotFoundError(f"Default calibration file not found at {calibration_file}")

            landing_weights = weights_from_inputs(landing_weight_values, len(landing_files))
            takeoff_weights = weights_from_inputs(takeoff_weight_values, len(takeoff_files))

            landing_file_tuples = (
                [(path, weight) for (path, _), weight in zip(landing_files, landing_weights)]
                if landing_files
                else None
            )
            takeoff_file_tuples = (
                [(path, weight) for (path, _), weight in zip(takeoff_files, takeoff_weights)]
                if takeoff_files
                else None
            )
            
            # Generate exposure using load_inputs method
            if workflow_instance.inputs.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")
            
            from bensaf.core.exposure_generation import generate_exposure_from_aermod
            
            aermod_crs = data_config.get('aermod_crs', 'EPSG:4326') if data_config else 'EPSG:4326'
            exposure_df = generate_exposure_from_aermod(
                landing_files=landing_file_tuples,
                takeoff_files=takeoff_file_tuples,
                tracts_gdf=workflow_instance.inputs.tract_geometries.reset_index(),
                calibration_file=calibration_file,
                aermod_crs=aermod_crs,
                aggregation_method='spatial_join'
            )
            # Rename 'ufp' column if needed
            if 'ufp' in exposure_df.columns:
                exposure_df = exposure_df.rename(columns={'ufp': 'ufp'})
            
            workflow_instance.inputs.load_baseline_exposure(exposure_df, pollutant_columns=['ufp'])
            
            state['exposure_loaded'] = True
            state['n_exposure'] = len(workflow_instance.inputs.baseline_exposure)
            state['exposure_source'] = 'aermod'
            
            status = dbc.Alert(
                [
                    html.H5("Exposure Generated Successfully!", className="alert-heading"),
                    html.P(f"Generated exposure data for {state['n_exposure']} census tracts from AERMOD files."),
                    html.P("You can now proceed with the analysis.", className="mb-0")
                ],
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                [
                    html.H5("Error Generating Exposure", className="alert-heading"),
                    html.P(f"Error: {str(e)}")
                ],
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        Output('workflow-state', 'data', allow_duplicate=True),
        Input('tabs', 'active_tab'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def track_config_tab_visit(active_tab, state):
        """Track when Configuration tab is visited"""
        if active_tab == 'tab-configuration':
            if 'config_tab_visited' not in state:
                state['config_tab_visited'] = True
        return state
    
    @app.callback(
        [Output('step-data-icon', 'children'),
         Output('step-config-icon', 'children'),
         Output('step-analysis-icon', 'children'),
         Output('step-results-icon', 'children')],
        Input('workflow-state', 'data'),
        Input('analysis-results', 'data')
    )
    def update_workflow_progress(workflow_state, analysis_results):
        """Update workflow progress indicator icons"""
        if not workflow_state:
            workflow_state = {}
        
        data_loaded = (
            workflow_state.get('tracts_loaded', False) and
            workflow_state.get('exposure_loaded', False) and
            workflow_state.get('mortality_loaded', False)
        )
        # Only show config as done if it was explicitly set AND the tab has been visited
        config_set = (
            workflow_state.get('config_set', False) and 
            workflow_state.get('config_explicitly_set', False) and
            workflow_state.get('config_tab_visited', False)
        )
        analysis_complete = bool(analysis_results)
        
        def _step_chip(done):
            if done:
                return html.Div(
                    className="bensaf-step-circle bg-success text-white border-0 d-inline-flex align-items-center justify-content-center",
                    children=[html.I(className="bi bi-check-lg")],
                )
            return html.Div(
                className="bensaf-step-circle bg-light text-muted border d-inline-flex align-items-center justify-content-center",
                children=[html.I(className="bi bi-circle")],
            )

        data_icon = _step_chip(data_loaded)
        config_icon = _step_chip(config_set)
        analysis_icon = _step_chip(analysis_complete)
        results_icon = _step_chip(analysis_complete)
        
        return data_icon, config_icon, analysis_icon, results_icon
    
    @app.callback(
        Output('results-summary-cards', 'children'),
        Input('analysis-results', 'data')
    )
    def update_results_summary_cards(analysis_results):
        """Update summary cards in results tab"""
        if not analysis_results:
            return dbc.Alert("No analysis results available. Please run the analysis first.", color="info")
        
        cards = []
        for scenario_key in sorted(analysis_results.keys(), key=lambda x: int(x)):
            scenario_results = analysis_results[scenario_key]
            scenario = scenario_results['scenario']
            total_cases = scenario_results['total_cases']
            lower_cases = scenario_results['lower_cases']
            upper_cases = scenario_results['upper_cases']
            
            card_body = [
                html.H3(f"{total_cases:.2f}", className="text-center text-primary mb-2"),
                html.P("Attributable Cases Avoided", className="text-center text-muted mb-2"),
                html.P(
                    f"95% CI: [{lower_cases:.2f}, {upper_cases:.2f}]",
                    className="text-center small text-muted mb-3"
                )
            ]
            
            # Add economic benefits if available
            if 'total_economic_benefits' in scenario_results:
                econ_value = scenario_results['total_economic_benefits'] / 1e6  # Convert to millions
                econ_lower = scenario_results.get('total_economic_benefits_lower', 0) / 1e6
                econ_upper = scenario_results.get('total_economic_benefits_upper', 0) / 1e6
                card_body.extend([
                    html.Hr(className="my-2"),
                    html.H4(f"${econ_value:.2f}M", className="text-center text-success mb-2"),
                    html.P("Total Economic Benefits", className="text-center text-muted mb-2"),
                    html.P(
                        f"95% CI: [${econ_lower:.2f}M, ${econ_upper:.2f}M]",
                        className="text-center small text-muted mb-0"
                    )
                ])
            elif 'mortality_economic_value' in scenario_results:
                mort_value = scenario_results['mortality_economic_value'] / 1e6
                mort_lower = scenario_results.get('mortality_economic_value_lower', 0) / 1e6
                mort_upper = scenario_results.get('mortality_economic_value_upper', 0) / 1e6
                card_body.extend([
                    html.Hr(className="my-2"),
                    html.H4(f"${mort_value:.2f}M", className="text-center text-success mb-2"),
                    html.P("Mortality Economic Benefits", className="text-center text-muted mb-2"),
                    html.P(
                        f"95% CI: [${mort_lower:.2f}M, ${mort_upper:.2f}M]",
                        className="text-center small text-muted mb-0"
                    )
                ])
            
            cards.append(
                dbc.Col([
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H5(f"{scenario}% SAF Blend", className="mb-0"),
                                className="bg-light",
                            ),
                            dbc.CardBody(card_body),
                        ],
                        className="h-100 shadow-sm border-start border-primary border-4",
                    )
                ], md=4, className="mb-3")
            )
        
        return dbc.Row(cards)
    
    @app.callback(
        Output('analysis-prerequisites-checklist', 'children'),
        Input('workflow-state', 'data')
    )
    def update_analysis_prerequisites(workflow_state):
        """Update prerequisites checklist in analysis tab"""
        if not workflow_state:
            return html.Div([
                html.P("No workflow state available", className="text-muted")
            ])
        
        tracts_loaded = workflow_state.get('tracts_loaded', False)
        exposure_loaded = workflow_state.get('exposure_loaded', False)
        mortality_loaded = workflow_state.get('mortality_loaded', False)
        config_set = (
            workflow_state.get('config_set', False) and 
            workflow_state.get('config_explicitly_set', False) and
            workflow_state.get('config_tab_visited', False)
        )
        
        all_ready = tracts_loaded and exposure_loaded and mortality_loaded and config_set
        
        items = [
            ("Census Tract Geometries", tracts_loaded),
            ("Exposure Data", exposure_loaded),
            ("Mortality Data", mortality_loaded),
            ("Configuration Set", config_set),
        ]
        
        checklist_items = []
        for item_name, item_status in items:
            icon = (
                html.I(className="bi bi-check-circle-fill text-success me-2")
                if item_status
                else html.I(className="bi bi-x-circle-fill text-danger me-2")
            )
            checklist_items.append(
                html.Div([
                    icon,
                    html.Span(item_name)
                ], className="mb-2")
            )
        
        if all_ready:
            checklist_items.append(
                html.Div([
                    html.Strong("All prerequisites met! Ready to run analysis.", className="text-success")
                ], className="mt-3")
            )
        else:
            checklist_items.append(
                html.Div([
                    html.Strong("Please complete all prerequisites before running analysis.", className="text-warning")
                ], className="mt-3")
            )
        
        return html.Div(checklist_items)
    
    
    @app.callback(
        Output('health-impact-function-plot', 'figure'),
        Input('mortality-function-dropdown', 'value')
    )
    def update_health_impact_function_plot(function_id):
        if function_id is None:
            return go.Figure().update_layout(
                title='Health Impact Function',
                xaxis_title='Pollutant Concentration Increase (pt/cm³)',
                yaxis_title='Relative Risk',
                template='plotly_white',
                height=400,
                annotations=[{
                    'text': 'Select a mortality function to view the health impact function',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }],
                uirevision='constant'
            )
        
        function_data = _mortality_functions().get(function_id)
        if function_data is None:
            return go.Figure().update_layout(
                title='Health Impact Function',
                xaxis_title='Pollutant Concentration Increase (pt/cm³)',
                yaxis_title='Relative Risk',
                template='plotly_white',
                height=400,
                annotations=[{
                    'text': 'Function not found',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }],
                uirevision='constant'
            )
        
        mean_rr = function_data['mean_rr']
        lower_rr = function_data['lower_rr']
        upper_rr = function_data['upper_rr']
        unit_increase = function_data['unit_increase']
        import numpy as np
        
        if mean_rr is None or lower_rr is None or upper_rr is None or unit_increase is None:
            return go.Figure().update_layout(
                title='Health Impact Function',
                xaxis_title='Pollutant Concentration Increase (pt/cm³)',
                yaxis_title='Relative Risk',
                template='plotly_white',
                height=400,
                annotations=[{
                    'text': 'Enter parameters to see the health impact function',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }],
                uirevision='constant'
            )
        
        # Calculate concentration range
        max_conc = unit_increase * 3
        conc_range = np.linspace(0, max_conc, 101)
        
        # Calculate relative risk for each concentration
        mean_log = np.log(mean_rr)
        lower_log = np.log(lower_rr)
        upper_log = np.log(upper_rr)
        z = 1.96
        
        se_log = ((upper_log - mean_log) + (mean_log - lower_log)) / (2 * z)
        mean_log_one_unit = mean_log / unit_increase
        se_log_one_unit = se_log / unit_increase
        
        mean_rr_values = np.exp(mean_log_one_unit * conc_range)
        lower_rr_values = np.exp((mean_log_one_unit - 1.96 * se_log_one_unit) * conc_range)
        upper_rr_values = np.exp((mean_log_one_unit + 1.96 * se_log_one_unit) * conc_range)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=conc_range,
            y=mean_rr_values,
            mode='lines',
            name='Mean',
            line=dict(color=PLOT_PRIMARY, width=3),
        ))
        
        fig.add_trace(go.Scatter(
            x=conc_range,
            y=lower_rr_values,
            mode='lines',
            name='Lower 95% CI',
            line=dict(color=PLOT_PRIMARY, width=1, dash='dash'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=conc_range,
            y=upper_rr_values,
            mode='lines',
            name='Upper 95% CI',
            line=dict(color=PLOT_PRIMARY, width=1, dash='dash'),
            fill='tonexty',
            fillcolor="rgba(24, 188, 156, 0.15)",
            showlegend=True
        ))
        
        fig.add_vline(
            x=unit_increase,
            line_dash="dot",
            line_color=PLOT_SECONDARY,
            annotation_text=f"Unit Increase: {unit_increase} pt/cm³",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Health Impact Function',
            xaxis_title='Pollutant Concentration Increase (pt/cm³)',
            yaxis_title='Relative Risk',
            template='plotly_white',
            height=400,
            uirevision='constant',
            hovermode='x unified'
        )
        
        return fig
    
