"""
Main layout for BenSAF Dash application
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("BenSAF Dashboard", className="text-center mb-2"),
                html.P(
                    "Benefits of Sustainable Aviation Fuels - Health Impact Analysis",
                    className="text-center text-muted mb-4"
                ),
            ])
        ]),
        
        create_header(),
        
        dbc.Tabs([
            dbc.Tab(label="Data", tab_id="tab-data", children=[
                create_data_tab()
            ]),
            dbc.Tab(label="Configuration", tab_id="tab-configuration", children=[
                create_configuration_tab()
            ]),
            dbc.Tab(label="Results", tab_id="tab-results", children=[
                create_results_tab()
            ]),
        ], id="tabs", active_tab="tab-data"),
        
        dcc.Store(id='workflow-state', data={}),
        dcc.Store(id='analysis-results', data={}),
        dcc.Store(id='case-studies-init', data={'init': True}),
        dcc.Store(id='data-config', data={'crs': 'EPSG:4326', 'aermod_crs': 'EPSG:32616', 'airport_coordinates': None}),
        
    ], fluid=True, className="py-4")

def create_header():
    """Create header with workflow progress, status, and analysis controls"""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Workflow Progress", className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div(id='step-data-icon', className="text-center mb-1"),
                                html.P("Data", className="text-center mb-0", style={"fontSize": "12px"})
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.Div(id='step-config-icon', className="text-center mb-1"),
                                html.P("Config", className="text-center mb-0", style={"fontSize": "12px"})
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.Div(id='step-analysis-icon', className="text-center mb-1"),
                                html.P("Analysis", className="text-center mb-0", style={"fontSize": "12px"})
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.Div(id='step-results-icon', className="text-center mb-1"),
                                html.P("Results", className="text-center mb-0", style={"fontSize": "12px"})
                            ])
                        ], md=3),
                    ])
                ], md=4),
                dbc.Col([
                    html.H6("Status", className="mb-2"),
                    html.Div(id='header-data-status', style={"fontSize": "12px"}),
                ], md=3),
                dbc.Col([
                    html.H6("Run Analysis", className="mb-2"),
                    dbc.Button(
                        "Execute Analysis",
                        id='btn-run-analysis',
                        color="primary",
                        size="sm",
                        className="w-100 mb-2",
                        disabled=True
                    ),
                    html.Div(id='analysis-status', style={"fontSize": "11px"}),
                    dcc.Loading(
                        id="loading-analysis",
                        type="default",
                        children=html.Div(id="loading-output")
                    ),
                ], md=5)
            ])
        ])
    ], className="mb-3")


def create_data_tab():
    """Create data upload tab with collapsible sections"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Data Upload", className="mt-4 mb-3"),
                html.P("Configure your study area and upload data files or load example datasets.", className="text-muted"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Study Area Configuration", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Airport Coordinates", className="fw-bold"),
                                html.Small("Optional: Latitude and longitude of the airport center", className="text-muted d-block mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Input(
                                            id='input-airport-lat',
                                            type='number',
                                            placeholder='Latitude (e.g., 41.9786)',
                                            step=0.0001,
                                            className="mb-2"
                                        ),
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Input(
                                            id='input-airport-lon',
                                            type='number',
                                            placeholder='Longitude (e.g., -87.9048)',
                                            step=0.0001,
                                            className="mb-2"
                                        ),
                                    ], md=6),
                                ]),
                            ], md=4),
                            dbc.Col([
                                html.Label("Coordinate Reference System (CRS)", className="fw-bold"),
                                html.Small("CRS for input data (e.g., EPSG:4326 for WGS84)", className="text-muted d-block mb-2"),
                                dbc.Input(
                                    id='input-crs',
                                    type='text',
                                    placeholder='EPSG:4326',
                                    value='EPSG:4326',
                                    className="mb-2"
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("AERMOD CRS", className="fw-bold"),
                                html.Small("CRS for AERMOD files (e.g., EPSG:32616 for UTM Zone 16N)", className="text-muted d-block mb-2"),
                                dbc.Input(
                                    id='input-aermod-crs',
                                    type='text',
                                    placeholder='EPSG:32616',
                                    value='EPSG:32616',
                                    className="mb-2"
                                ),
                            ], md=4),
                        ]),
                        html.Div(id='config-data-status', className="mt-2", style={"fontSize": "12px"}),
                    ])
                ], className="mb-4"),
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Data Upload", className="mb-3"),
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
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
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
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
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
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px 0'
                                },
                                multiple=False
                            ),
                            html.Div(id='upload-exposure-status', className="text-muted mt-2"),
                        ]),
                        
                        html.Div(id='exposure-aermod-upload', style={'display': 'none'}, children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Landing AERMOD Files", className="fw-bold mt-3"),
                                    html.Small("Upload one or more AERMOD .ADO files for landing flows", className="text-muted d-block mb-1"),
                                    dcc.Upload(
                                        id='upload-landing-aermod',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Landing AERMOD Files')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px 0'
                                        },
                                        multiple=True
                                    ),
                                    html.Div(id='upload-landing-aermod-status', className="text-muted mb-3"),
                                ], md=6),
                                dbc.Col([
                                    html.Label("Landing File Weights", className="fw-bold mt-3"),
                                    html.Small("Enter weights for each landing file (comma-separated, e.g., 0.33, 0.67)", className="text-muted d-block mb-1"),
                                    dbc.Input(
                                        id='landing-weights-input',
                                        type='text',
                                        placeholder='0.33, 0.67',
                                        className="mb-3"
                                    ),
                                ], md=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Takeoff AERMOD Files", className="fw-bold mt-3"),
                                    html.Small("Upload one or more AERMOD .ADO files for takeoff flows", className="text-muted d-block mb-1"),
                                    dcc.Upload(
                                        id='upload-takeoff-aermod',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Takeoff AERMOD Files')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px 0'
                                        },
                                        multiple=True
                                    ),
                                    html.Div(id='upload-takeoff-aermod-status', className="text-muted mb-3"),
                                ], md=6),
                                dbc.Col([
                                    html.Label("Takeoff File Weights", className="fw-bold mt-3"),
                                    html.Small("Enter weights for each takeoff file (comma-separated, e.g., 0.33, 0.67)", className="text-muted d-block mb-1"),
                                    dbc.Input(
                                        id='takeoff-weights-input',
                                        type='text',
                                        placeholder='0.33, 0.67',
                                        className="mb-3"
                                    ),
                                ], md=6)
                            ]),
                            
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
                ], start_collapsed=False),
            ], md=6),
            
            dbc.Col([
                html.H4("Load Example Data", className="mb-3"),
                dbc.Card([
                    dbc.CardBody([
                        html.P("Select a case study to load example data", className="text-muted mb-3"),
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
                    ])
                ]),
            ], md=6),
        ]),
        
        html.Hr(className="my-4"),
        
        dbc.Row([
            dbc.Col([
                html.H4("Health Pipelines", className="mt-4 mb-3"),
                html.P("Select and configure health benefit pipelines. Each pipeline has specific data requirements.", className="text-muted mb-3"),
            ], md=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Mortality Pipeline", className="mb-0 d-inline"),
                            html.Span(id='mortality-pipeline-status', className="ms-2")
                        ])
                    ]),
                    dbc.CardBody([
                        html.P("Computes mortality health impacts and economic benefits from pollutant reduction.", className="text-muted mb-3"),
                        
                        html.H6("Required Data", className="fw-bold mt-3 mb-2"),
                        html.Ul([
                            html.Li("Incidence data with 'mortality_rate' column (CSV)"),
                            html.Li("Demographics data with 'population' column (already uploaded)"),
                        ], className="mb-3"),
                        
                        html.H6("Configuration", className="fw-bold mt-3 mb-2"),
                        html.Ul([
                            html.Li("Mortality function: Selected in Configuration tab"),
                            html.Li("Economic parameters: Configured in data/economic_parameters.json"),
                        ], className="mb-3"),
                        
                        html.Label("Upload Incidence Data (CSV)", className="fw-bold"),
                        html.Small("Must contain GEOID and mortality_rate columns", className="text-muted d-block mb-2"),
                        dcc.Upload(
                            id='upload-mortality-incidence',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Incidence Data File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-mortality-incidence-status', className="text-muted mt-2"),
                    ])
                ], className="mb-3"),
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Preterm Birth Pipeline", className="mb-0 d-inline"),
                            html.Span(id='preterm-birth-pipeline-status', className="ms-2")
                        ])
                    ]),
                    dbc.CardBody([
                        html.P("Computes reduction in preterm births and economic benefits from UFP reduction.", className="text-muted mb-3"),
                        
                        html.H6("Required Data", className="fw-bold mt-3 mb-2"),
                        html.Ul([
                            html.Li("Preterm birth data with 'baseline_preterm_births' column (CSV)"),
                            html.Li("Demographics data with 'population' column (already uploaded)"),
                        ], className="mb-3"),
                        
                        html.H6("Configuration", className="fw-bold mt-3 mb-2"),
                        html.Ul([
                            html.Li("Odds ratio: Configured in data/economic_parameters.json"),
                            html.Li("Monetary value per PtB: Configured in data/economic_parameters.json"),
                        ], className="mb-3"),
                        
                        html.Label("Upload Preterm Birth Data (CSV)", className="fw-bold"),
                        html.Small("Must contain GEOID and baseline_preterm_births columns", className="text-muted d-block mb-2"),
                        dcc.Upload(
                            id='upload-ptb-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Preterm Birth Data File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-ptb-status', className="text-muted mt-2"),
                    ])
                ], className="mb-3"),
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
                html.H5("Data Table", className="mb-2 mt-3"),
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
        
    ], fluid=True, className="py-4")

def create_configuration_tab():
    """Create configuration tab with health impact function and SAF scenarios"""
    return dbc.Container([
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
                    dbc.CardHeader(html.H5("Health Impact Function", className="mb-0")),
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
                ], className="mb-4"),
            ], md=7)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("SAF Blend Scenarios", className="mb-0")),
                    dbc.CardBody([
                        html.P("Define the SAF blend percentages to analyze (0-50%)", className="text-muted small mb-3"),
                        html.Div(id='saf-scenarios-list', className="mb-3"),
                        dbc.Button(
                            "Add Scenario",
                            id='btn-add-scenario',
                            color="secondary",
                            size="sm",
                            className="mb-2"
                        ),
                        html.Div(id='saf-scenarios-status', className="mt-2")
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader(html.H5("Preterm Birth Data (Optional)", className="mb-0")),
                    dbc.CardBody([
                        html.P("Upload preterm birth data to enable preterm birth benefit calculations. Economic parameters are configured in data/economic_parameters.json", className="text-muted small mb-3"),
                        html.Label("Upload Preterm Birth Data", className="fw-bold"),
                        dcc.Upload(
                            id='upload-ptb-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select PtB Data File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '40px',
                                'lineHeight': '40px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'fontSize': '12px'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-ptb-status', className="text-muted mt-1", style={"fontSize": "11px"}),
                    ])
                ]),
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
        
    ], fluid=True, className="py-4")



def create_results_tab():
    """Create enhanced results tab with summary cards and comparison charts"""
    return dbc.Container([
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
                dbc.Card([
                    dbc.CardHeader(html.H5("Scenario Comparison", className="mb-0")),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-bar-chart",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id='results-bar-chart',
                                    config={'displayModeBar': True},
                                    style={'height': '400px'}
                                )
                            ]
                        )
                    ])
                ], className="mb-4")
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
                html.H5("Detailed Results Table", className="mb-3"),
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
        
    ], fluid=True, className="py-4")

