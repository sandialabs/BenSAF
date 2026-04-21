"""
Main layout for BenSAF Dash application
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span("BenSAF", className="fw-bold text-primary fs-2"),
                                    html.Span(" Dashboard", className="text-body fs-2"),
                                ],
                                className="d-flex align-items-baseline flex-wrap",
                            ),
                            html.P(
                                "Benefits of Sustainable Aviation Fuels · health impact analysis",
                                className="text-muted mb-0 mt-1 small",
                            ),
                        ],
                    ),
                ],
                className="mb-3 pb-3 border-bottom",
            ),
            create_header(),
            dbc.Tabs(
                [
                    dbc.Tab(label="Data", tab_id="tab-data", children=[create_data_tab()]),
                    dbc.Tab(
                        label="Configuration",
                        tab_id="tab-configuration",
                        children=[create_configuration_tab()],
                    ),
                    dbc.Tab(label="Results", tab_id="tab-results", children=[create_results_tab()]),
                ],
                id="tabs",
                active_tab="tab-data",
                className="mb-3 bensaf-tabs",
            ),
            dcc.Store(id="workflow-state", data={}),
            dcc.Store(id="analysis-results", data={}),
            dcc.Store(id="case-studies-init", data={"init": True}),
            dcc.Store(id="data-config", data={"aermod_crs": "EPSG:4326"}),
        ],
        fluid=True,
        className="py-3",
    )


def create_header():
    """Header: workflow steps, readiness checklist, run controls."""
    step_wrap = "text-center px-1"
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H6("Workflow", className="mb-3 text-uppercase small text-muted"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Div(id="step-data-icon", className="mb-1 d-flex justify-content-center"),
                                                    html.P("Data", className=f"{step_wrap} small text-muted mb-0"),
                                                ],
                                                xs=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Div(id="step-config-icon", className="mb-1 d-flex justify-content-center"),
                                                    html.P("Config", className=f"{step_wrap} small text-muted mb-0"),
                                                ],
                                                xs=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Div(id="step-analysis-icon", className="mb-1 d-flex justify-content-center"),
                                                    html.P("Analysis", className=f"{step_wrap} small text-muted mb-0"),
                                                ],
                                                xs=3,
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Div(id="step-results-icon", className="mb-1 d-flex justify-content-center"),
                                                    html.P("Results", className=f"{step_wrap} small text-muted mb-0"),
                                                ],
                                                xs=3,
                                            ),
                                        ],
                                        className="g-0",
                                    ),
                                ],
                                xs=12,
                                lg=4,
                                className="mb-3 mb-lg-0",
                            ),
                            dbc.Col(
                                [
                                    html.H6("Readiness", className="mb-2 text-uppercase small text-muted"),
                                    html.Div(
                                        id="analysis-prerequisites-checklist",
                                        className="small p-2 rounded bg-light border",
                                    ),
                                ],
                                xs=12,
                                lg=5,
                                className="mb-3 mb-lg-0",
                            ),
                            dbc.Col(
                                [
                                    html.H6("Run", className="mb-2 text-uppercase small text-muted"),
                                    dbc.Button(
                                        "Execute Analysis",
                                        id="btn-run-analysis",
                                        color="primary",
                                        size="sm",
                                        className="w-100 mb-2",
                                        disabled=True,
                                    ),
                                    html.Div(id="analysis-status", className="small text-muted"),
                                    dcc.Loading(
                                        id="loading-analysis",
                                        type="circle",
                                        children=html.Div(id="loading-output"),
                                    ),
                                ],
                                xs=12,
                                lg=3,
                            ),
                        ],
                        className="g-3 align-items-start",
                    ),
                ],
            ),
        ],
        className="mb-4 shadow-sm border-0",
    )


def create_data_tab():
    """Data tab: file upload vs example case study, accordion inputs, data explorer."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Data", className="mt-4 mb-3"),
                html.P(
                    "Choose one of two ways to configure the analysis case study: upload your own inputs below, or load a prepared example case study.",
                    className="text-muted",
                ),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("File Upload", className="mb-3"),
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Label("Census Tract Geometries (GeoJSON, Shapefile, or GeoPackage)", className="fw-bold"),
                        html.Small("Must contain GEOID and geometry columns only", className="text-muted d-block mb-2"),
                        dcc.Upload(
                            id='upload-tracts',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Tract Geometries File')
                            ]),
                            className="bensaf-upload-zone",
                            multiple=False
                        ),
                        html.Div(id='upload-tracts-status', className="text-muted mt-2"),
                    ], title="1. Census Tract Geometries", item_id="item-tracts"),
                    
                    dbc.AccordionItem([
                        html.Label("Demographics Data (CSV)", className="fw-bold"),
                        html.Small("Must contain GEOID and demographic columns (population, etc.)", className="text-muted d-block mb-2"),
                        dcc.Upload(
                            id='upload-demographics',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Demographics File')
                            ]),
                            className="bensaf-upload-zone",
                            multiple=False
                        ),
                        html.Div(id='upload-demographics-status', className="text-muted mt-2"),
                    ], title="2. Demographics Data", item_id="item-demographics"),
                    
                    dbc.AccordionItem([
                        html.Label("Exposure Data Source", className="fw-bold mb-2"),
                        dcc.RadioItems(
                            id='exposure-source-radio',
                            options=[
                                {'label': 'Upload Exposure CSV', 'value': 'csv'},
                                {'label': 'Generate from AERMOD Files', 'value': 'aermod'}
                            ],
                            value='csv',
                            className="mb-3"
                        ),
                        
                        html.Div(id='exposure-csv-upload', children=[
                            html.Label("Exposure Data (CSV)", className="fw-bold"),
                            dcc.Upload(
                                id='upload-exposure',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Exposure Data File')
                                ]),
                                className="bensaf-upload-zone",
                                multiple=False
                            ),
                            html.Div(id='upload-exposure-status', className="text-muted mt-2"),
                        ]),
                        
                        html.Div(id='exposure-aermod-upload', style={'display': 'none'}, children=[
                            html.Label("AERMOD grid CRS", className="fw-bold"),
                            html.Small(
                                "Coordinate system for x/y values in the .ADO receptor grid (often a projected CRS in meters; set to match your run).",
                                className="text-muted d-block mb-2",
                            ),
                            dbc.Input(
                                id='input-aermod-crs',
                                type='text',
                                placeholder='EPSG:4326',
                                value='EPSG:4326',
                                className="mb-1",
                            ),
                            html.Div(id='aermod-crs-config-status', className="text-muted small mb-3"),
                            html.P(
                                "Provide landing .ADO files, takeoff .ADO files, or both. Each uploaded file gets its own weight in the tables below (defaults split weight equally). Only the phases you upload are used in the blend.",
                                className="text-muted small mb-3"
                            ),
                            html.Label("Landing AERMOD files", className="fw-bold mt-2"),
                            html.Small("Optional. One or more .ADO files with annual-average landing concentrations.", className="text-muted d-block mb-1"),
                            dcc.Upload(
                                id='upload-landing-aermod',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Landing AERMOD Files')
                                ]),
                                className="bensaf-upload-zone",
                                multiple=True
                            ),
                            html.Div(id='upload-landing-aermod-status', className="text-muted mb-2"),
                            html.Label("Weight per landing file", className="fw-bold small"),
                            html.Div(
                                id='landing-aermod-weights-table',
                                className="mb-4",
                                children=html.Small(
                                    "Upload landing .ADO files to assign a weight to each file.",
                                    className="text-muted",
                                ),
                            ),
                            
                            html.Label("Takeoff AERMOD files", className="fw-bold mt-2"),
                            html.Small("Optional. One or more .ADO files with annual-average takeoff concentrations.", className="text-muted d-block mb-1"),
                            dcc.Upload(
                                id='upload-takeoff-aermod',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Takeoff AERMOD Files')
                                ]),
                                className="bensaf-upload-zone",
                                multiple=True
                            ),
                            html.Div(id='upload-takeoff-aermod-status', className="text-muted mb-2"),
                            html.Label("Weight per takeoff file", className="fw-bold small"),
                            html.Div(
                                id='takeoff-aermod-weights-table',
                                className="mb-3",
                                children=html.Small(
                                    "Upload takeoff .ADO files to assign a weight to each file.",
                                    className="text-muted",
                                ),
                            ),
                            
                            html.Small("Using default calibration coefficients from data/aermod_calibration_coefficients.json", className="text-muted d-block mb-3"),
                            
                            dbc.Button(
                                "Generate Exposure from AERMOD",
                                id='btn-generate-exposure',
                                color="primary",
                                className="mt-3 w-100"
                            ),
                            html.Div(id='generate-exposure-status', className="mt-2"),
                        ]),
                    ], title="3. Exposure Data", item_id="item-exposure"),
                    
                    dbc.AccordionItem([
                        html.P(
                            "Select and configure health benefit pipelines. Each pipeline has specific data requirements.",
                            className="text-muted mb-3",
                        ),
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.H5("Mortality Pipeline", className="mb-0 d-inline"),
                                    html.Span(id='mortality-pipeline-status', className="ms-2"),
                                ])
                            ], className="bg-light"),
                            dbc.CardBody([
                                html.P(
                                    "Computes mortality health impacts and economic benefits from pollutant reduction.",
                                    className="text-muted mb-3",
                                ),
                                html.H6("Required Data", className="fw-bold mt-3 mb-2"),
                                html.Ul([
                                    html.Li("Incidence data with 'mortality_rate' column (CSV)"),
                                    html.Li("Demographics data with 'population' column (already uploaded)"),
                                ], className="mb-3"),
                                html.H6("Configuration", className="fw-bold mt-3 mb-2"),
                                html.Ul([
                                    html.Li("Mortality function: Selected in Configuration tab"),
                                    html.Li(
                                        "Mortality economics (optional): CSV with GEOID and "
                                        "per_capita_consumption; optional life_years_gained. "
                                        "Included when you load the ORD example; or upload below."
                                    ),
                                ], className="mb-3"),
                                html.Label("Upload Incidence Data (CSV)", className="fw-bold"),
                                html.Small(
                                    "Must contain GEOID and mortality_rate columns",
                                    className="text-muted d-block mb-2",
                                ),
                                dcc.Upload(
                                    id='upload-mortality-incidence',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Incidence Data File'),
                                    ]),
                                    className="bensaf-upload-zone",
                                    multiple=False,
                                ),
                                html.Div(id='upload-mortality-incidence-status', className="text-muted mt-2"),
                                html.Label(
                                    "Upload mortality economic tract data (CSV, optional)",
                                    className="fw-bold mt-3",
                                ),
                                html.Small(
                                    "GEOID, per_capita_consumption; optional life_years_gained",
                                    className="text-muted d-block mb-2",
                                ),
                                dcc.Upload(
                                    id='upload-mortality-economic',
                                    children=html.Div([
                                        "Drag and Drop or ",
                                        html.A("Select economic tract CSV"),
                                    ]),
                                    className="bensaf-upload-zone",
                                    multiple=False,
                                ),
                                html.Div(
                                    id="upload-mortality-economic-status",
                                    className="text-muted mt-2",
                                ),
                            ]),
                        ], className="mb-3 shadow-sm border-0"),
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.H5("Preterm Birth Pipeline", className="mb-0 d-inline"),
                                    html.Span(id='preterm-birth-pipeline-status', className="ms-2"),
                                ])
                            ], className="bg-light"),
                            dbc.CardBody([
                                html.P(
                                    "Computes reduction in preterm births and economic benefits from UFP reduction.",
                                    className="text-muted mb-3",
                                ),
                                html.H6("Required Data", className="fw-bold mt-3 mb-2"),
                                html.Ul([
                                    html.Li("Preterm birth data with 'baseline_preterm_births' column (CSV)"),
                                    html.Li("Demographics data with 'population' column (already uploaded)"),
                                ], className="mb-3"),
                                html.H6("Configuration", className="fw-bold mt-3 mb-2"),
                                html.Ul([
                                    html.Li(
                                        "Odds ratio and monetary value per PtB: set on AnalysisInputs "
                                        "via the API (e.g. set_preterm_birth_economic_parameters)."
                                    ),
                                ], className="mb-3"),
                                html.Label("Upload Preterm Birth Data (CSV)", className="fw-bold"),
                                html.Small(
                                    "Must contain GEOID and baseline_preterm_births columns",
                                    className="text-muted d-block mb-2",
                                ),
                                dcc.Upload(
                                    id='upload-ptb-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Preterm Birth Data File'),
                                    ]),
                                    className="bensaf-upload-zone",
                                    multiple=False,
                                ),
                                html.Div(id='upload-ptb-status', className="text-muted mt-2"),
                            ]),
                        ], className="mb-0 shadow-sm border-0"),
                    ], title="4. Health Pipelines", item_id="item-health-pipelines"),
                ], start_collapsed=False),
            ], md=6),
            
            dbc.Col([
                html.H4("Load Example Case Study", className="mb-3"),
                dbc.Card(
                    [
                    dbc.CardBody([
                        html.P(
                            "Pick a packaged example and load all inputs at once (same analysis case study as a full file upload).",
                            className="text-muted mb-3",
                        ),
                        dcc.Dropdown(
                            id='case-study-dropdown',
                            options=[],
                            value=None,
                            clearable=False,
                            placeholder="Select a case study...",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "Load Selected Case Study",
                            id='btn-load-example',
                            color="secondary",
                            size="lg",
                            className="w-100",
                            disabled=True
                        ),
                        html.Div(id='example-load-status', className="mt-3")
                    ]),
                    ],
                    className="shadow-sm border-0 h-100",
                ),
            ], md=6),
        ]),
        
        html.Hr(className="my-4"),
        
        dbc.Row([
            dbc.Col([
                html.H4("Data Explorer", className="mt-4 mb-3"),
                html.Label("Select Variable to Display", className="fw-bold"),
                dcc.Dropdown(
                    id='data-viewer-dropdown',
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="Select a variable...",
                    className="mb-3"
                ),
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("Map", className="mb-2"),
                dcc.Loading(
                    id="loading-data-viewer-map",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id='data-viewer-map',
                            config={'displayModeBar': True, 'scrollZoom': True},
                            style={'height': '600px'}
                        )
                    ]
                ),
            ], md=7),
            
            dbc.Col([
                html.H5("Data table", className="mb-2"),
                html.P("First 100 records", className="text-muted small mb-2"),
                dcc.Loading(
                    id="loading-data-table",
                    type="circle",
                    children=[
                        html.Div(
                            id='data-viewer-table',
                            style={
                                'height': '580px',
                                'overflowY': 'auto',
                                'overflowX': 'auto'
                            }
                        )
                    ]
                ),
            ], md=5)
        ]),
        
    ], className="pt-1 pb-4")

def create_configuration_tab():
    """Create configuration tab with health impact function and SAF scenarios"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Configuration", className="mt-4 mb-3"),
                html.P("Configure health impact function parameters and SAF blend scenarios.", className="text-muted"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='health-impact-function-plot',
                    config={'displayModeBar': True},
                    style={'height': '400px'}
                ),
            ], md=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Health Impact Function", className="mb-0"), className="bg-light"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Mortality Function to View", className="fw-bold"),
                                dcc.Dropdown(
                                    id='mortality-function-dropdown',
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    className="mb-3"
                                ),
                            ], md=12),
                        ]),
                        
                        html.Div(id='mortality-function-details', className="mb-3"),
                        
                        html.Hr(className="my-3"),
                        
                        html.Label("Select Functions to Compute in Analysis", className="fw-bold mb-2"),
                        html.Div(id='mortality-function-checkboxes', className="mb-3"),
                        
                        html.Div(id='config-status', className="mt-3")
                    ])
                ], className="mb-4 shadow-sm border-0"),
            ], md=7)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("SAF Blend Scenarios", className="mb-0"), className="bg-light"),
                    dbc.CardBody([
                        html.P("Define the SAF blend percentages to analyze (0-50%)", className="text-muted small mb-3"),
                        html.Div(id='saf-scenarios-list', className="mb-3"),
                        dbc.Button(
                            "Add Scenario",
                            id='btn-add-scenario',
                            color="primary",
                            outline=True,
                            size="sm",
                            className="mb-2"
                        ),
                        html.Div(id='saf-scenarios-status', className="mt-2")
                    ])
                ], className="mb-4 shadow-sm border-0"),
            ], md=6),
            dbc.Col([
                dcc.Graph(
                    id='saf-reduction-curve',
                    config={'displayModeBar': True},
                    style={'height': '500px'}
                ),
            ], md=6)
        ]),
        
        dcc.Store(id='saf-scenarios-store', data=[25, 50]),
        dcc.Store(id='selected-mortality-functions-store', data=[])
        
    ], className="pt-1 pb-4")



def create_results_tab():
    """Create enhanced results tab with summary cards and comparison charts"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Analysis Results", className="mt-4 mb-3"),
                html.P("View and explore your analysis results", className="text-muted"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='results-summary-cards', className="mb-4")
            ], md=12)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Selected scenario — key metrics", className="mb-0"),
                            className="bg-light",
                        ),
                        dbc.CardBody(html.Div(id="results-summary-table")),
                    ],
                    className="mb-4 shadow-sm border-start border-primary border-4",
                ),
            ], md=12),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Scenario Comparison", className="mb-0"), className="bg-light"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-bar-chart",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='results-bar-chart',
                                    config={'displayModeBar': True},
                                    style={'height': '520px'}
                                )
                            ]
                        )
                    ])
                ], className="mb-4 shadow-sm border-0")
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Geographic Visualization", className="mt-4 mb-3"),
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select SAF Scenario", className="fw-bold"),
                dcc.Dropdown(
                    id='results-scenario-dropdown',
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="Select scenario...",
                    className="mb-3"
                ),
            ], md=6),
            dbc.Col([
                html.Label("Select Result Variable", className="fw-bold"),
                dcc.Dropdown(
                    id='results-map-dropdown',
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="Select variable...",
                    disabled=True,
                    className="mb-3"
                ),
            ], md=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H5("Map", className="mb-2"),
                dcc.Loading(
                    id="loading-map",
                    type="circle",
                    children=[
                        dcc.Graph(
                            id='results-map',
                            config={'displayModeBar': True, 'scrollZoom': True},
                            style={'height': '600px'}
                        )
                    ]
                ),
            ], md=7),
            
            dbc.Col([
                html.H5("Detailed results table", className="mb-2"),
                dcc.Loading(
                    id="loading-results-table",
                    type="circle",
                    children=[
                        html.Div(
                            id='results-table',
                            style={
                                'height': '580px',
                                'overflowY': 'auto',
                                'overflowX': 'auto'
                            }
                        )
                    ]
                ),
            ], md=5)
        ]),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Export", className="mb-0"), className="bg-light"),
                    dbc.CardBody([
                        html.P(
                            "Scenario summary matches the detailed results table. "
                            "Tract-level CSV includes inputs and all scenario outputs (wide columns).",
                            className="mb-3 text-body fw-semibold small",
                        ),
                        dbc.ButtonGroup([
                            dbc.Button(
                                "Summary table (CSV)",
                                id="btn-export-results-summary-csv",
                                color="primary",
                                size="sm",
                            ),
                            dbc.Button(
                                "Tract-level data (CSV)",
                                id="btn-export-results-tract-csv",
                                color="primary",
                                size="sm",
                            ),
                        ]),
                        dcc.Download(id="download-results-summary-csv"),
                        dcc.Download(id="download-results-tract-csv"),
                    ]),
                ], className="mt-2 mb-4 shadow-sm border-0"),
            ], md=12),
        ]),
        
    ], className="pt-1 pb-4")

