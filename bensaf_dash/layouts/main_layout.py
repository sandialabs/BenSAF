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
                html.P("Upload your data files or load the ORD example dataset.", className="text-muted"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
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
                    
                    dbc.AccordionItem([
                        html.Label("Mortality Data (CSV)", className="fw-bold"),
                        html.Small("Must contain GEOID and mortality_rate columns", className="text-muted d-block mb-2"),
                        dcc.Upload(
                            id='upload-mortality',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Mortality Data File')
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
                        html.Div(id='upload-mortality-status', className="text-muted mt-2"),
                    ], title="4. Mortality Data", item_id="item-mortality"),
                ], start_collapsed=False, className="mb-4"),
                
                html.Hr(className="my-4"),
                
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Load Example Data", className="text-center mb-3"),
                        html.P("Quick start with the ORD case study dataset", className="text-center text-muted mb-3"),
                        dbc.Button(
                            "Load ORD Example Data",
                            id='btn-load-example',
                            color="secondary",
                            size="lg",
                            className="w-100"
                        ),
                        html.Div(id='example-load-status', className="mt-3")
                    ])
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
                
            ], md=12)
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
                                html.Label("Select Mortality Function", className="fw-bold"),
                                dcc.Dropdown(
                                    id='mortality-function-dropdown',
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    className="mb-3"
                                ),
                            ], md=12),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Mean Relative Risk", className="fw-bold"),
                                dbc.Input(
                                    id='input-mean-rr',
                                    type='number',
                                    value=1.012,
                                    step=0.001,
                                    className="mb-3"
                                ),
                                html.Small("Mean relative risk per unit increase", className="text-muted"),
                            ], md=6),
                            dbc.Col([
                                html.Label("Unit Increase (pt/cm³)", className="fw-bold"),
                                dbc.Input(
                                    id='input-unit-increase',
                                    type='number',
                                    value=2723,
                                    step=1,
                                    className="mb-3"
                                ),
                                html.Small("Pollutant concentration unit increase", className="text-muted"),
                            ], md=6),
                        ]),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Lower 95% CI", className="fw-bold"),
                                dbc.Input(
                                    id='input-lower-rr',
                                    type='number',
                                    value=1.010,
                                    step=0.001,
                                    className="mb-3"
                                ),
                            ], md=6),
                            dbc.Col([
                                html.Label("Upper 95% CI", className="fw-bold"),
                                dbc.Input(
                                    id='input-upper-rr',
                                    type='number',
                                    value=1.015,
                                    step=0.001,
                                    className="mb-3"
                                ),
                            ], md=6)
                        ]),
                        
                        html.Div(id='config-status', className="mt-3")
                    ])
                ], className="mb-4"),
            ], md=7)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("SAF to Pollutant Reduction", className="mb-0")),
                    dbc.CardBody([
                        html.P("Polynomial coefficients: Reduction = a₀ + a₁·SAF + a₂·SAF²", className="text-muted small mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("a₀ (constant)", className="fw-bold"),
                                dbc.Input(
                                    id='poly-coeff-0',
                                    type='number',
                                    value=0.0,
                                    step=0.01,
                                    className="mb-3"
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("a₁ (linear)", className="fw-bold"),
                                dbc.Input(
                                    id='poly-coeff-1',
                                    type='number',
                                    value=1.0,
                                    step=0.01,
                                    className="mb-3"
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("a₂ (quadratic)", className="fw-bold"),
                                dbc.Input(
                                    id='poly-coeff-2',
                                    type='number',
                                    value=0.0,
                                    step=0.001,
                                    className="mb-3"
                                ),
                            ], md=4),
                        ]),
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader(html.H5("SAF Blend Scenarios", className="mb-0")),
                    dbc.CardBody([
                        html.P("Define the SAF blend percentages to analyze", className="text-muted small mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Scenario 1 (%)", className="fw-bold"),
                                dbc.Input(
                                    id='scenario-1',
                                    type='number',
                                    value=25,
                                    min=0,
                                    max=100,
                                    step=1,
                                    className="mb-3"
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Scenario 2 (%)", className="fw-bold"),
                                dbc.Input(
                                    id='scenario-2',
                                    type='number',
                                    value=50,
                                    min=0,
                                    max=100,
                                    step=1,
                                    className="mb-3"
                                ),
                            ], md=4),
                            dbc.Col([
                                html.Label("Scenario 3 (%)", className="fw-bold"),
                                dbc.Input(
                                    id='scenario-3',
                                    type='number',
                                    value=75,
                                    min=0,
                                    max=100,
                                    step=1,
                                    className="mb-3"
                                ),
                            ], md=4),
                        ]),
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

