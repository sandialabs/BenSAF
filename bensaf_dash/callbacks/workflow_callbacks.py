"""
Workflow callbacks for BenSAF Dash application

These callbacks handle user interactions and connect the UI to the bensaf.Workflow class.
"""

import base64
import io
import json
from pathlib import Path
import tempfile

import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from dash import callback, Input, Output, State, html
import dash_bootstrap_components as dbc

from bensaf.workflow import Workflow
from bensaf.mortality_functions import MortalityFunctionLibrary

workflow_instance = None
cached_geojson = None
cached_center = None
mortality_library = None

def register_callbacks(app):
    
    @app.callback(
        [Output('example-load-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('btn-load-example', 'n_clicks'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def load_example_data(n_clicks, state):
        if n_clicks is None:
            return "", state
        
        try:
            from pathlib import Path
            
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / 'data' / 'ord'
            aermod_dir = project_root / 'data' / 'aermod-examples'
            
            tracts_geometries_file = data_dir / 'tracts_geometries.geojson'
            demographics_file = data_dir / 'demographics_df.csv'
            mortality_file = data_dir / 'mortality_df.csv'
            
            # AERMOD files for exposure generation
            landing_eastflow_file = aermod_dir / 'landing_eastflow.ADO'
            landing_westflow_file = aermod_dir / 'landing_westflow.ADO'
            takeoff_eastflow_file = aermod_dir / 'takeoff_eastflow.ADO'
            takeoff_westflow_file = aermod_dir / 'westflow_takeoff.ADO'
            calibration_file = project_root / 'data' / 'aermod_calibration_coefficients.json'
            
            if not tracts_geometries_file.exists():
                raise FileNotFoundError(f"Tract geometries not found at {tracts_geometries_file}")
            if not demographics_file.exists():
                raise FileNotFoundError(f"Demographics data not found at {demographics_file}")
            if not mortality_file.exists():
                raise FileNotFoundError(f"Mortality data not found at {mortality_file}")
            if not landing_eastflow_file.exists():
                raise FileNotFoundError(f"Landing eastflow AERMOD file not found at {landing_eastflow_file}")
            if not landing_westflow_file.exists():
                raise FileNotFoundError(f"Landing westflow AERMOD file not found at {landing_westflow_file}")
            if not takeoff_eastflow_file.exists():
                raise FileNotFoundError(f"Takeoff eastflow AERMOD file not found at {takeoff_eastflow_file}")
            if not takeoff_westflow_file.exists():
                raise FileNotFoundError(f"Takeoff westflow AERMOD file not found at {takeoff_westflow_file}")
            if not calibration_file.exists():
                raise FileNotFoundError(f"Calibration coefficients file not found at {calibration_file}")
            
            tracts_gdf = gpd.read_file(tracts_geometries_file)
            demographics_df = pd.read_csv(demographics_file)
            mortality_df = pd.read_csv(mortality_file)
            
            global workflow_instance, cached_geojson, cached_center
            cached_geojson = None
            cached_center = None
            
            if workflow_instance is None:
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            # Load tract geometries
            workflow_instance.data.load_tract_geometries(tracts_gdf)
            
            # Load demographics
            workflow_instance.data.load_demographics(demographics_df)
            
            # Generate exposure data from AERMOD files
            # Using weights: East = 1/3, West = 2/3 (as in the notebook)
            landing_files = [
                (landing_eastflow_file, 1/3),
                (landing_westflow_file, 2/3)
            ]
            takeoff_files = [
                (takeoff_eastflow_file, 1/3),
                (takeoff_westflow_file, 2/3)
            ]
            
            workflow_instance.data.load_baseline_exposure_from_aermod_workflow(
                landing_files=landing_files,
                takeoff_files=takeoff_files,
                calibration_file=calibration_file,
                aggregation_method='spatial_join'
            )
            
            # Load mortality data
            workflow_instance.data.load_mortality_data(mortality_df)
            
            state['tracts_loaded'] = True
            state['n_tracts'] = len(tracts_gdf)
            state['demographics_loaded'] = True
            state['n_demographics'] = len(demographics_df)
            state['exposure_loaded'] = True
            state['n_exposure'] = len(workflow_instance.data.baseline_exposure)
            state['exposure_source'] = 'aermod'
            state['mortality_loaded'] = True
            state['n_mortality'] = len(mortality_df)
            
            status = dbc.Alert(
                [
                    html.H5("ORD Example Data Loaded Successfully!", className="alert-heading"),
                    html.P(f"Loaded {len(tracts_gdf)} census tract geometries"),
                    html.P(f"Loaded {len(demographics_df)} demographic records"),
                    html.P(f"Generated {state['n_exposure']} exposure records from AERMOD files"),
                    html.P(f"Loaded {len(mortality_df)} mortality records", className="mb-0")
                ],
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except FileNotFoundError as e:
            status = dbc.Alert(
                [
                    html.H5("Example Data Not Found", className="alert-heading"),
                    html.P(str(e)),
                    html.P("Please ensure the ORD example data exists in data/ord/", className="mb-0")
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
        State('upload-tracts', 'filename'),
        State('workflow-state', 'data'),
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
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            # Load tract geometries (should only have GEOID and geometry)
            workflow_instance.data.load_tract_geometries(tracts_gdf)
            
            state['tracts_loaded'] = True
            state['n_tracts'] = len(tracts_gdf)
            
            status = dbc.Alert(
                f"Successfully loaded {len(tracts_gdf)} census tracts",
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
        State('upload-demographics', 'filename'),
        State('workflow-state', 'data'),
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
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            workflow_instance.data.load_demographics(demographics_df)
            
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
        State('upload-exposure', 'filename'),
        State('exposure-source-radio', 'value'),
        State('workflow-state', 'data'),
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
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            # Rename column if needed
            if 'pollutant_concentration' in exposure_df.columns:
                exposure_df = exposure_df.rename(columns={'pollutant_concentration': 'baseline_pollutant_concentration'})
            workflow_instance.data.load_baseline_exposure_data(exposure_df)
            
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
        [Output('upload-mortality-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-mortality', 'contents'),
        State('upload-mortality', 'filename'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def upload_mortality_data(contents, filename, state):
        if contents is None:
            return "", state
        
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            mortality_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
            global workflow_instance
            if workflow_instance is None:
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            workflow_instance.data.load_mortality_data(mortality_df)
            
            state['mortality_loaded'] = True
            state['n_mortality'] = len(mortality_df)
            
            status = dbc.Alert(
                f"Successfully loaded mortality data with {len(mortality_df)} records",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error loading mortality data: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        Output('data-summary', 'children'),
        Input('workflow-state', 'data')
    )
    def update_data_summary(state):
        if not state:
            return ""
        
        tracts_status = "✓" if state.get('tracts_loaded') else "✗"
        demographics_status = "✓" if state.get('demographics_loaded') else "✗"
        exposure_status = "✓" if state.get('exposure_loaded') else "✗"
        mortality_status = "✓" if state.get('mortality_loaded') else "✗"
        
        return dbc.Card([
            dbc.CardHeader(html.H5("Data Loading Status")),
            dbc.CardBody([
                html.P([
                    html.Strong(f"{tracts_status} Census Tract Geometries: "),
                    f"{state.get('n_tracts', 0)} records loaded" if state.get('tracts_loaded') else "Not loaded"
                ]),
                html.P([
                    html.Strong(f"{demographics_status} Demographics Data: "),
                    f"{state.get('n_demographics', 0)} records loaded" if state.get('demographics_loaded') else "Not loaded"
                ]),
                html.P([
                    html.Strong(f"{exposure_status} Exposure Data: "),
                    f"{state.get('n_exposure', 0)} records loaded" if state.get('exposure_loaded') else "Not loaded"
                ]),
                html.P([
                    html.Strong(f"{mortality_status} Mortality Data: "),
                    f"{state.get('n_mortality', 0)} records loaded" if state.get('mortality_loaded') else "Not loaded"
                ], className="mb-0"),
            ])
        ], className="mt-3")
    
    @app.callback(
        Output('mortality-function-dropdown', 'options'),
        Output('mortality-function-dropdown', 'value'),
        Input('workflow-state', 'data'),
        prevent_initial_call=False
    )
    def load_mortality_functions(state):
        global mortality_library
        
        if mortality_library is None:
            mortality_library = MortalityFunctionLibrary()
        
        functions = mortality_library.list_functions()
        options = [
            {'label': func['title'], 'value': func['id']}
            for func in functions
        ]
        
        default_value = 0 if functions else None
        
        return options, default_value
    
    @app.callback(
        [Output('input-mean-rr', 'value'),
         Output('input-lower-rr', 'value'),
         Output('input-upper-rr', 'value'),
         Output('input-unit-increase', 'value')],
        Input('mortality-function-dropdown', 'value'),
        prevent_initial_call=False
    )
    def update_function_inputs(function_id):
        global mortality_library
        
        if mortality_library is None:
            mortality_library = MortalityFunctionLibrary()
        
        if function_id is None:
            function_id = 0
        
        function_data = mortality_library.get_function(function_id)
        
        if function_data is None:
            return 1.012, 1.010, 1.015, 2723
        
        return (
            function_data['mean_rr'],
            function_data['lower_rr'],
            function_data['upper_rr'],
            function_data['unit_increase']
        )
    
    @app.callback(
        [Output('config-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('input-mean-rr', 'value'),
        Input('input-lower-rr', 'value'),
        Input('input-upper-rr', 'value'),
        Input('input-unit-increase', 'value'),
        Input('poly-coeff-0', 'value'),
        Input('poly-coeff-1', 'value'),
        Input('poly-coeff-2', 'value'),
        Input('scenario-1', 'value'),
        Input('scenario-2', 'value'),
        Input('scenario-3', 'value'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def update_config(mean_rr, lower_rr, upper_rr, unit_increase,
                     poly_coeff_0, poly_coeff_1, poly_coeff_2,
                     scenario_1, scenario_2, scenario_3, state):
        
        global workflow_instance
        
        if workflow_instance is not None:
            if mean_rr is not None and lower_rr is not None and upper_rr is not None and unit_increase is not None:
                workflow_instance.set_health_impact_function(
                    mean_rr=mean_rr,
                    lower_rr=lower_rr,
                    upper_rr=upper_rr,
                    unit_increase=unit_increase
                )
            
            # Update SAF polynomial coefficients in config
            workflow_instance.config.saf_polynomial_coeffs = [
                poly_coeff_0 if poly_coeff_0 is not None else 0.0,
                poly_coeff_1 if poly_coeff_1 is not None else 1.0,
                poly_coeff_2 if poly_coeff_2 is not None else 0.0
            ]
        
        state['config_set'] = True
        state['config_explicitly_set'] = True
        state['scenarios'] = [s for s in [scenario_1, scenario_2, scenario_3] if s is not None]
        state['poly_coeffs'] = [poly_coeff_0, poly_coeff_1, poly_coeff_2]
        
        status = dbc.Alert(
            "Configuration updated successfully",
            color="success",
            className="mt-2"
        )
        
        return status, state
    
    @app.callback(
        Output('btn-run-analysis', 'disabled'),
        Input('workflow-state', 'data')
    )
    def enable_analysis_button(state):
        if not state:
            return True
        
        all_loaded = (
            state.get('tracts_loaded', False) and
            state.get('exposure_loaded', False) and
            state.get('mortality_loaded', False) and
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
        prevent_initial_call=True
    )
    def run_analysis(n_clicks, state):
        if n_clicks is None:
            return "", {}
        
        try:
            global workflow_instance
            
            if workflow_instance is None:
                raise ValueError("Workflow not initialized. Please load data first.")
            
            scenarios = state.get('scenarios', [25, 50, 75])
            
            # Update config with scenarios
            workflow_instance.config.saf_scenarios = scenarios
            
            # Apply control scenarios and calculate health impacts
            workflow_instance.apply_control_scenarios(scenarios, use_saf_polynomial=True)
            workflow_instance.calculate_health_impacts()
            
            # Get aggregated results
            aggregated_results = workflow_instance.aggregated_results
            
            results = {}
            for scenario in scenarios:
                scenario_name = f"{scenario}% SAF Usage"
                
                if scenario_name in aggregated_results:
                    scenario_results = aggregated_results[scenario_name]
                    total_cases = scenario_results['total_attributable_cases']['mean']
                    lower_cases = scenario_results['total_attributable_cases']['lower']
                    upper_cases = scenario_results['total_attributable_cases']['upper']
                    
                    # Get pollutant reduction from scenario outputs
                    if scenario_name in workflow_instance.data.scenario_outputs:
                        scenario_df = workflow_instance.data.scenario_outputs[scenario_name]
                        pollutant_reduction = scenario_df['pollutant_reduction'].iloc[0] if 'pollutant_reduction' in scenario_df.columns else scenario
                    else:
                        pollutant_reduction = scenario
                    
                    results[str(scenario)] = {
                        'scenario': scenario,
                        'saf_percentage': scenario,
                        'pollutant_reduction': float(pollutant_reduction),
                        'total_cases': float(total_cases),
                        'lower_cases': float(lower_cases),
                        'upper_cases': float(upper_cases)
                    }
            
            status = html.Div([
                html.Span("✓ Analysis Complete! ", className="text-success fw-bold"),
                html.Span(f"{len(scenarios)} scenarios analyzed. ", className="text-muted"),
                html.A("View Results", href="#", className="text-primary", style={"textDecoration": "underline"})
            ])
            
            return status, results
            
        except Exception as e:
            status = html.Div([
                html.Span("✗ Analysis Failed: ", className="text-danger fw-bold"),
                html.Span(str(e), className="text-muted")
            ])
            return status, {}
    
    @app.callback(
        Output('results-summary', 'children'),
        Input('analysis-results', 'data')
    )
    def update_results_summary(results):
        if not results:
            return dbc.Alert(
                "No analysis results available. Please run the analysis first.",
                color="info"
            )
        
        cards = []
        for scenario_key, scenario_results in results.items():
            scenario = scenario_results['scenario']
            total_cases = scenario_results['total_cases']
            lower_cases = scenario_results['lower_cases']
            upper_cases = scenario_results['upper_cases']
            
            card = dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5(f"{scenario}% SAF Blend")),
                    dbc.CardBody([
                        html.H3(f"{total_cases:.2f}", className="text-center text-primary"),
                        html.P("Attributable Cases Avoided", className="text-center text-muted"),
                        html.P(
                            f"95% CI: [{lower_cases:.2f}, {upper_cases:.2f}]",
                            className="text-center small"
                        )
                    ])
                ])
            ], md=4, className="mb-3")
            cards.append(card)
        
        return dbc.Row(cards)
    
    @app.callback(
        Output('results-bar-chart', 'figure'),
        Input('analysis-results', 'data')
    )
    def update_bar_chart(results):
        if not results:
            return go.Figure().update_layout(
                title='Health Impacts by SAF Scenario',
                xaxis_title='SAF Blend Percentage',
                yaxis_title='Attributable Cases Avoided',
                template='plotly_white',
                height=500,
                annotations=[{
                    'text': 'No analysis results available. Please run analysis first.',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14, 'color': 'gray'}
                }],
                uirevision='constant'
            )
        
        scenarios = []
        mean_cases = []
        lower_cases = []
        upper_cases = []
        
        for scenario_key in sorted(results.keys(), key=lambda x: int(x)):
            scenario_results = results[scenario_key]
            scenarios.append(f"{scenario_results['scenario']}% SAF")
            mean_cases.append(scenario_results['total_cases'])
            lower_cases.append(scenario_results['lower_cases'])
            upper_cases.append(scenario_results['upper_cases'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scenarios,
            y=mean_cases,
            name='Mean',
            error_y=dict(
                type='data',
                symmetric=False,
                array=[u - m for u, m in zip(upper_cases, mean_cases)],
                arrayminus=[m - l for m, l in zip(mean_cases, lower_cases)]
            ),
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.update_layout(
            title='Health Impacts by SAF Scenario',
            xaxis_title='SAF Blend Percentage',
            yaxis_title='Attributable Cases Avoided',
            template='plotly_white',
            height=400,
            uirevision='constant'
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
        
        options = [
            {'label': 'Attributable Cases Avoided', 'value': 'attributable_cases'},
            {'label': 'Attributable Fraction', 'value': 'attributable_fraction'},
            {'label': 'Relative Risk', 'value': 'relative_risk'},
            {'label': 'Delta Concentration', 'value': 'delta_concentration'},
            {'label': 'Reduced Concentration', 'value': 'reduced_concentration'},
        ]
        
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
        
        if workflow_instance is None or workflow_instance.data.tract_geometries is None:
            cached_geojson = None
            cached_center = None
            return go.Figure().update_layout(
                title="No data loaded",
                height=600,
                annotations=[{
                    'text': 'Please load data in the Setup tab',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                uirevision='constant'
            )
        
        if not results or not workflow_instance.data.scenario_outputs:
            return go.Figure().update_layout(
                title="No analysis results available",
                height=600,
                annotations=[{
                    'text': 'Please run analysis in the Setup tab',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                uirevision='constant'
            )
        
        if not selected_scenario or not selected_variable:
            return go.Figure().update_layout(
                title="Please select scenario and variable",
                height=600,
                annotations=[{
                    'text': 'Select a scenario and variable from the dropdowns above',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                uirevision='constant'
            )
        
        scenario_num = int(selected_scenario)
        scenario_name = f"{scenario_num}% SAF Usage"
        
        if scenario_name not in workflow_instance.data.scenario_outputs:
            return go.Figure().update_layout(
                title=f"Scenario {scenario_num}% not found",
                height=600,
                uirevision='constant'
            )
        
        # Get merged data with all scenario outputs
        gdf = workflow_instance.data.get_merged_data()
        scenario_df = workflow_instance.data.scenario_outputs[scenario_name]
        
        # Map variable names to column names in scenario outputs
        variable_mapping = {
            'attributable_cases': ('attributable_cases_mean', 'Attributable Cases Avoided'),
            'attributable_fraction': ('attributable_fraction_mean', 'Attributable Fraction'),
            'relative_risk': ('relative_risk_mean', 'Relative Risk'),
            'delta_concentration': ('delta_concentration', 'Delta Concentration (pt/cm³)'),
            'reduced_concentration': ('reduced_concentration', 'Reduced Concentration (pt/cm³)')
        }
        
        if selected_variable not in variable_mapping:
            selected_variable = 'attributable_cases'
        
        column_name, colorbar_title = variable_mapping[selected_variable]
        
        # Check if column exists in scenario outputs or merged data
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
                uirevision='constant'
            )
        
        if cached_geojson is None or cached_center is None:
            gdf_simplified = gdf.copy()
            gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(tolerance=0.001, preserve_topology=True)
            
            gdf_json = json.loads(gdf_simplified.to_json())
            
            if not gdf_json.get('features'):
                return go.Figure().update_layout(
                    title="No features to display",
                    height=600,
                    uirevision='constant'
                )
            
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
            'reduced_concentration': 'Reduced Concentration'
        }
        var_label = variable_labels.get(selected_variable, selected_variable)
        title_text = f'{var_label} by Census Tract ({scenario_num}% SAF)'
        
        fig = go.Figure(go.Choroplethmapbox(
            geojson=gdf_json,
            locations=locations,
            z=z_values,
            colorscale="Viridis",
            marker_opacity=0.5,
            marker_line_width=0.5,
            marker_line_color='white',
            colorbar=dict(title=colorbar_title),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=9
            ),
            title=title_text,
            height=600,
            margin={"r":0,"t":40,"l":0,"b":0},
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
            return ""
        
        global workflow_instance
        
        if workflow_instance is None or not workflow_instance.data.scenario_outputs:
            return ""
        
        scenario_num = int(selected_scenario)
        scenario_name = f"{scenario_num}% SAF Usage"
        
        if scenario_name not in workflow_instance.data.scenario_outputs:
            return ""
        
        scenario_df = workflow_instance.data.scenario_outputs[scenario_name]
        
        summary_data = []
        
        if 'pollutant_reduction' in scenario_df.columns:
            pollutant_reduction = scenario_df['pollutant_reduction'].iloc[0]
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
            return ""
        
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
        [Output('data-viewer-dropdown', 'options'),
         Output('data-viewer-dropdown', 'value')],
        Input('workflow-state', 'data')
    )
    def update_data_viewer_dropdown(workflow_state):
        global workflow_instance
        
        if workflow_instance is None or workflow_instance.data.tract_geometries is None:
            return [], None
        
        gdf = workflow_instance.data.get_merged_data()
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
        
        if workflow_instance is None or workflow_instance.data.tract_geometries is None:
            return go.Figure().update_layout(
                title="No data loaded",
                height=600,
                annotations=[{
                    'text': 'Please load data in the Setup tab',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                uirevision='constant'
            )
        
        if selected_variable is None:
            return go.Figure().update_layout(
                title="Please select a variable",
                height=600,
                uirevision='constant'
            )
        
        gdf = workflow_instance.data.get_merged_data()
        
        if selected_variable not in gdf.columns:
            available_cols = [col for col in gdf.columns if col not in ['geometry']]
            return go.Figure().update_layout(
                title=f"Variable '{selected_variable}' not available",
                height=600,
                annotations=[{
                    'text': f'Available columns: {", ".join(available_cols[:5])}...',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 14}
                }],
                uirevision='constant'
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
        
        fig = go.Figure(go.Choroplethmapbox(
            geojson=gdf_json,
            locations=locations,
            z=z_values,
            colorscale="Viridis",
            marker_opacity=0.5,
            marker_line_width=0.5,
            marker_line_color='white',
            colorbar=dict(title=colorbar_title),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=9
            ),
            title=f'{colorbar_title} by Census Tract',
            height=600,
            margin={"r":0,"t":40,"l":0,"b":0},
            uirevision=selected_variable
        )
        
        return fig
    
    @app.callback(
        Output('saf-reduction-curve', 'figure'),
        Input('poly-coeff-0', 'value'),
        Input('poly-coeff-1', 'value'),
        Input('poly-coeff-2', 'value'),
        Input('scenario-1', 'value'),
        Input('scenario-2', 'value'),
        Input('scenario-3', 'value')
    )
    def update_saf_curve(poly_coeff_0, poly_coeff_1, poly_coeff_2,
                        scenario_1, scenario_2, scenario_3):
        import numpy as np
        
        coeffs = [
            poly_coeff_0 if poly_coeff_0 is not None else 0.0,
            poly_coeff_1 if poly_coeff_1 is not None else 1.0,
            poly_coeff_2 if poly_coeff_2 is not None else 0.0
        ]
        
        saf_range = np.linspace(0, 100, 101)
        
        reductions = np.array([
            max(0.0, min(100.0, sum(coeff * (saf ** i) for i, coeff in enumerate(coeffs))))
            for saf in saf_range
        ])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=saf_range,
            y=reductions,
            mode='lines',
            name='Polynomial Fit',
            line=dict(color='blue', width=3)
        ))
        
        scenarios = [s for s in [scenario_1, scenario_2, scenario_3] if s is not None]
        scenario_reductions = [
            max(0.0, min(100.0, sum(coeff * (s ** i) for i, coeff in enumerate(coeffs))))
            for s in scenarios
        ]
        
        fig.add_trace(go.Scatter(
            x=scenarios,
            y=scenario_reductions,
            mode='markers',
            name='Analysis Scenarios',
            marker=dict(color='red', size=12, symbol='circle')
        ))
        
        equation_text = f"Reduction = {coeffs[0]:.3f} + {coeffs[1]:.3f}·SAF + {coeffs[2]:.5f}·SAF²"
        
        fig.update_layout(
            title='SAF Blend to Pollutant Reduction',
            xaxis_title='SAF Blend Percentage (%)',
            yaxis_title='Pollutant Reduction (%)',
            template='plotly_white',
            height=400,
            uirevision='constant',
            annotations=[{
                'text': equation_text,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.02,
                'y': 0.98,
                'showarrow': False,
                'font': {'size': 12},
                'align': 'left',
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': 'black',
                'borderwidth': 1
            }]
        )
        
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
        
        return fig
    
    @app.callback(
        Output('data-viewer-table', 'children'),
        Input('workflow-state', 'data')
    )
    def update_data_viewer_table(workflow_state):
        global workflow_instance
        
        if workflow_instance is None or workflow_instance.data.tract_geometries is None:
            return dbc.Alert(
                "No data loaded. Please load data in the Setup tab.",
                color="info"
            )
        
        gdf = workflow_instance.data.get_merged_data()
        
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
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-landing-aermod', 'contents'),
        State('upload-landing-aermod', 'filename'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def upload_landing_aermod(contents_list, filename_list, state):
        if contents_list is None or len(contents_list) == 0:
            return "", state
        
        try:
            if not isinstance(contents_list, list):
                contents_list = [contents_list]
            if not isinstance(filename_list, list):
                filename_list = [filename_list]
            
            landing_files = []
            for contents, filename in zip(contents_list, filename_list):
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    tmp.write(decoded)
                    tmp_path = tmp.name
                
                landing_files.append((tmp_path, filename))
            
            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['landing'] = landing_files
            
            status = dbc.Alert(
                f"Successfully uploaded {len(landing_files)} landing AERMOD file(s)",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error uploading landing AERMOD files: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('upload-takeoff-aermod-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('upload-takeoff-aermod', 'contents'),
        State('upload-takeoff-aermod', 'filename'),
        State('workflow-state', 'data'),
        prevent_initial_call=True
    )
    def upload_takeoff_aermod(contents_list, filename_list, state):
        if contents_list is None or len(contents_list) == 0:
            return "", state
        
        try:
            if not isinstance(contents_list, list):
                contents_list = [contents_list]
            if not isinstance(filename_list, list):
                filename_list = [filename_list]
            
            takeoff_files = []
            for contents, filename in zip(contents_list, filename_list):
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
                    tmp.write(decoded)
                    tmp_path = tmp.name
                
                takeoff_files.append((tmp_path, filename))
            
            if 'aermod_files' not in state:
                state['aermod_files'] = {}
            state['aermod_files']['takeoff'] = takeoff_files
            
            status = dbc.Alert(
                f"Successfully uploaded {len(takeoff_files)} takeoff AERMOD file(s)",
                color="success",
                className="mt-2"
            )
            
            return status, state
            
        except Exception as e:
            status = dbc.Alert(
                f"Error uploading takeoff AERMOD files: {str(e)}",
                color="danger",
                className="mt-2"
            )
            return status, state
    
    @app.callback(
        [Output('generate-exposure-status', 'children'),
         Output('workflow-state', 'data', allow_duplicate=True)],
        Input('btn-generate-exposure', 'n_clicks'),
        State('workflow-state', 'data'),
        State('landing-weights-input', 'value'),
        State('takeoff-weights-input', 'value'),
        prevent_initial_call=True
    )
    def generate_exposure_from_aermod(n_clicks, state, landing_weights_str, takeoff_weights_str):
        if n_clicks is None:
            return "", state
        
        try:
            global workflow_instance
            
            if workflow_instance is None:
                config = {
                    'saf_polynomial_coeffs': [0.0, 1.0, 0.0]
                }
                workflow_instance = Workflow(config)
            
            if workflow_instance.data.tract_geometries is None:
                raise ValueError("Tract geometries must be loaded first")
            
            # Get AERMOD files from state
            if 'aermod_files' not in state:
                raise ValueError("Please upload AERMOD files first")
            
            aermod_files = state['aermod_files']
            
            if 'landing' not in aermod_files or len(aermod_files['landing']) == 0:
                raise ValueError("Please upload at least one landing AERMOD file")
            
            if 'takeoff' not in aermod_files or len(aermod_files['takeoff']) == 0:
                raise ValueError("Please upload at least one takeoff AERMOD file")
            
            # Use default calibration file
            project_root = Path(__file__).parent.parent.parent
            calibration_file = project_root / 'data' / 'aermod_calibration_coefficients.json'
            
            if not calibration_file.exists():
                raise FileNotFoundError(f"Default calibration file not found at {calibration_file}")
            
            # Parse weights
            def parse_weights(weights_str, num_files):
                if not weights_str or weights_str.strip() == '':
                    # Default: equal weights
                    return [1.0 / num_files] * num_files
                
                try:
                    weights = [float(w.strip()) for w in weights_str.split(',')]
                    if len(weights) != num_files:
                        raise ValueError(f"Number of weights ({len(weights)}) must match number of files ({num_files})")
                    return weights
                except ValueError as e:
                    raise ValueError(f"Invalid weights format: {str(e)}")
            
            landing_files = aermod_files['landing']
            takeoff_files = aermod_files['takeoff']
            
            landing_weights = parse_weights(landing_weights_str, len(landing_files))
            takeoff_weights = parse_weights(takeoff_weights_str, len(takeoff_files))
            
            # Create file + weight tuples
            landing_file_tuples = [(path, weight) for (path, _), weight in zip(landing_files, landing_weights)]
            takeoff_file_tuples = [(path, weight) for (path, _), weight in zip(takeoff_files, takeoff_weights)]
            
            # Generate exposure using spatial_join method
            workflow_instance.data.load_baseline_exposure_from_aermod_workflow(
                landing_files=landing_file_tuples,
                takeoff_files=takeoff_file_tuples,
                calibration_file=calibration_file,
                aggregation_method='spatial_join'
            )
            
            state['exposure_loaded'] = True
            state['n_exposure'] = len(workflow_instance.data.baseline_exposure)
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
        
        data_icon = html.Span("✓", className="text-success", style={"fontSize": "20px", "fontWeight": "bold"}) if data_loaded else html.Span("○", className="text-muted", style={"fontSize": "20px"})
        config_icon = html.Span("✓", className="text-success", style={"fontSize": "20px", "fontWeight": "bold"}) if config_set else html.Span("○", className="text-muted", style={"fontSize": "20px"})
        analysis_icon = html.Span("✓", className="text-success", style={"fontSize": "20px", "fontWeight": "bold"}) if analysis_complete else html.Span("○", className="text-muted", style={"fontSize": "20px"})
        results_icon = html.Span("✓", className="text-success", style={"fontSize": "20px", "fontWeight": "bold"}) if analysis_complete else html.Span("○", className="text-muted", style={"fontSize": "20px"})
        
        return data_icon, config_icon, analysis_icon, results_icon
    
    @app.callback(
        Output('header-data-status', 'children'),
        Input('workflow-state', 'data')
    )
    def update_header_data_status(workflow_state):
        """Update compact data status in header"""
        if not workflow_state:
            return html.Span("No data loaded", className="text-muted")
        
        tracts_loaded = workflow_state.get('tracts_loaded', False)
        exposure_loaded = workflow_state.get('exposure_loaded', False)
        mortality_loaded = workflow_state.get('mortality_loaded', False)
        config_set = (
            workflow_state.get('config_set', False) and 
            workflow_state.get('config_explicitly_set', False) and
            workflow_state.get('config_tab_visited', False)
        )
        
        status_items = []
        status_items.append(html.Span("Tracts: ", className="fw-bold"))
        status_items.append(html.Span("✓" if tracts_loaded else "✗", className="text-success" if tracts_loaded else "text-danger"))
        status_items.append(html.Span(" | ", className="mx-1"))
        status_items.append(html.Span("Exposure: ", className="fw-bold"))
        status_items.append(html.Span("✓" if exposure_loaded else "✗", className="text-success" if exposure_loaded else "text-danger"))
        status_items.append(html.Span(" | ", className="mx-1"))
        status_items.append(html.Span("Mortality: ", className="fw-bold"))
        status_items.append(html.Span("✓" if mortality_loaded else "✗", className="text-success" if mortality_loaded else "text-danger"))
        status_items.append(html.Span(" | ", className="mx-1"))
        status_items.append(html.Span("Config: ", className="fw-bold"))
        status_items.append(html.Span("✓" if config_set else "✗", className="text-success" if config_set else "text-danger"))
        
        return html.Div(status_items)
    
    @app.callback(
        Output('overview-data-status', 'children'),
        Input('workflow-state', 'data')
    )
    def update_overview_data_status(workflow_state):
        """Update data status cards in overview tab"""
        if not workflow_state:
            return dbc.Alert("No data loaded yet", color="info")
        
        cards = []
        
        tracts_status = workflow_state.get('tracts_loaded', False)
        tracts_count = workflow_state.get('n_tracts', 0)
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Census Tracts", className="card-title"),
                    html.P(f"{'✓ Loaded' if tracts_status else '✗ Not loaded'}", className="mb-1"),
                    html.Small(f"{tracts_count} records" if tracts_status else "No data", className="text-muted")
                ])
            ], color="success" if tracts_status else "light", className="mb-2")
        )
        
        demographics_status = workflow_state.get('demographics_loaded', False)
        demographics_count = workflow_state.get('n_demographics', 0)
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Demographics", className="card-title"),
                    html.P(f"{'✓ Loaded' if demographics_status else '✗ Not loaded'}", className="mb-1"),
                    html.Small(f"{demographics_count} records" if demographics_status else "No data", className="text-muted")
                ])
            ], color="success" if demographics_status else "light", className="mb-2")
        )
        
        exposure_status = workflow_state.get('exposure_loaded', False)
        exposure_count = workflow_state.get('n_exposure', 0)
        exposure_source = workflow_state.get('exposure_source', 'unknown')
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Exposure Data", className="card-title"),
                    html.P(f"{'✓ Loaded' if exposure_status else '✗ Not loaded'}", className="mb-1"),
                    html.Small(f"{exposure_count} records ({exposure_source})" if exposure_status else "No data", className="text-muted")
                ])
            ], color="success" if exposure_status else "light", className="mb-2")
        )
        
        mortality_status = workflow_state.get('mortality_loaded', False)
        mortality_count = workflow_state.get('n_mortality', 0)
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Mortality Data", className="card-title"),
                    html.P(f"{'✓ Loaded' if mortality_status else '✗ Not loaded'}", className="mb-1"),
                    html.Small(f"{mortality_count} records" if mortality_status else "No data", className="text-muted")
                ])
            ], color="success" if mortality_status else "light", className="mb-2")
        )
        
        return html.Div(cards)
    
    @app.callback(
        Output('overview-config-status', 'children'),
        Input('workflow-state', 'data')
    )
    def update_overview_config_status(workflow_state):
        """Update configuration status in overview tab"""
        if not workflow_state:
            return dbc.Alert("Configuration not set", color="info")
        
        config_set = workflow_state.get('config_set', False)
        scenarios = workflow_state.get('scenarios', [])
        
        if config_set:
            return dbc.Card([
                dbc.CardBody([
                    html.H6("Configuration", className="card-title"),
                    html.P("✓ Configured", className="mb-1 text-success"),
                    html.Small(f"Scenarios: {', '.join([f'{s}%' for s in scenarios])}", className="text-muted")
                ])
            ], color="success")
        else:
            return dbc.Card([
                dbc.CardBody([
                    html.H6("Configuration", className="card-title"),
                    html.P("✗ Not configured", className="mb-1 text-muted"),
                    html.Small("Go to Configuration tab to set parameters", className="text-muted")
                ])
            ], color="light")
    
    @app.callback(
        Output('overview-results-summary', 'children'),
        Input('analysis-results', 'data')
    )
    def update_overview_results_summary(analysis_results):
        """Update results summary in overview tab"""
        if not analysis_results:
            return dbc.Alert("No analysis results available. Run analysis to see results.", color="info")
        
        cards = []
        for scenario_key in sorted(analysis_results.keys(), key=lambda x: int(x)):
            scenario_results = analysis_results[scenario_key]
            scenario = scenario_results['scenario']
            total_cases = scenario_results['total_cases']
            lower_cases = scenario_results['lower_cases']
            upper_cases = scenario_results['upper_cases']
            
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6(f"{scenario}% SAF Blend", className="mb-0")),
                        dbc.CardBody([
                            html.H4(f"{total_cases:.2f}", className="text-center text-primary mb-2"),
                            html.P("Cases Avoided", className="text-center text-muted mb-1 small"),
                            html.P(
                                f"95% CI: [{lower_cases:.2f}, {upper_cases:.2f}]",
                                className="text-center small text-muted mb-0"
                            )
                        ])
                    ])
                ], md=4, className="mb-3")
            )
        
        return dbc.Row(cards)
    
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
            
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H5(f"{scenario}% SAF Blend", className="mb-0")),
                        dbc.CardBody([
                            html.H3(f"{total_cases:.2f}", className="text-center text-primary mb-2"),
                            html.P("Attributable Cases Avoided", className="text-center text-muted mb-2"),
                            html.P(
                                f"95% CI: [{lower_cases:.2f}, {upper_cases:.2f}]",
                                className="text-center small text-muted mb-0"
                            )
                        ])
                    ], className="h-100")
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
            icon = html.Span("✓", className="text-success me-2", style={"fontWeight": "bold"}) if item_status else html.Span("✗", className="text-danger me-2", style={"fontWeight": "bold"})
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
        Input('input-mean-rr', 'value'),
        Input('input-lower-rr', 'value'),
        Input('input-upper-rr', 'value'),
        Input('input-unit-increase', 'value')
    )
    def update_health_impact_function_plot(mean_rr, lower_rr, upper_rr, unit_increase):
        """Create health impact function visualization"""
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
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=conc_range,
            y=lower_rr_values,
            mode='lines',
            name='Lower 95% CI',
            line=dict(color='blue', width=1, dash='dash'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=conc_range,
            y=upper_rr_values,
            mode='lines',
            name='Upper 95% CI',
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.1)',
            showlegend=True
        ))
        
        # Add vertical line at unit_increase
        fig.add_vline(
            x=unit_increase,
            line_dash="dot",
            line_color="red",
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
    
