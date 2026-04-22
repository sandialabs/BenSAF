"""
Main layout for BenSAF Dash application
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_sidebar_nav():
    """Workflow navigation only; readiness is shown under Execute Analysis (prerequisites)."""
    nav_btn = "bensaf-nav-link-btn text-start px-2 py-2 w-100 rounded-2 border-0 shadow-none mb-1"

    return html.Div(
        [
            html.H6("Workflow", className="text-uppercase small text-muted mb-2"),
            dbc.Button(
                "Data",
                id="nav-tab-data",
                n_clicks=0,
                color="link",
                className=nav_btn,
            ),
            dbc.Button(
                "Explore",
                id="nav-tab-explore",
                n_clicks=0,
                color="link",
                className=nav_btn,
            ),
            dbc.Button(
                "Configuration",
                id="nav-tab-configuration",
                n_clicks=0,
                color="link",
                className=nav_btn,
            ),
            dbc.Button(
                "Results",
                id="nav-tab-results",
                n_clicks=0,
                color="link",
                className=nav_btn,
            ),
        ],
        className="bensaf-sidebar-nav mb-3",
    )


def create_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Button(
                                        html.Img(
                                            src="/assets/BenSAF.png",
                                            alt="BenSAF",
                                            className="bensaf-logo-img",
                                        ),
                                        id="nav-logo-overview",
                                        n_clicks=0,
                                        type="button",
                                        title="Overview",
                                        className="bensaf-logo-btn p-0 border-0 bg-transparent w-100 text-start mb-2",
                                    ),
                                ],
                                className="bensaf-sidebar-brand",
                            ),
                            create_sidebar_nav(),
                            create_header(),
                        ],
                        xs=12,
                        lg=3,
                        xl=3,
                        className="bensaf-sidebar pb-3 pb-lg-0 pe-lg-3",
                    ),
                    dbc.Col(
                        [
                            html.Div(id="panel-overview", children=[create_overview_tab()]),
                            html.Div(
                                id="panel-data",
                                children=[create_data_tab()],
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="panel-explore",
                                children=[create_explore_tab()],
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="panel-configuration",
                                children=[create_configuration_tab()],
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="panel-results",
                                children=[create_results_tab()],
                                style={"display": "none"},
                            ),
                        ],
                        xs=12,
                        lg=9,
                        xl=9,
                        className="ps-lg-3 bensaf-main-panels",
                    ),
                ],
                className="align-items-lg-start g-lg-4",
            ),
            dcc.Store(id="active-tab", data="tab-overview"),
            dcc.Store(id="workflow-state", data={}),
            dcc.Store(id="analysis-results", data={}),
            dcc.Store(id="case_studies-init", data={"init": True}),
            dcc.Store(id="data-config", data={"aermod_crs": "EPSG:4326"}),
        ],
        fluid=False,
        className="py-3 bensaf-app-root",
    )


def create_overview_tab():
    """Landing tab: short stakeholder overview and how BenSAF relates to AERMOD, NAS/TRB, and BenMAP."""
    stages = [
        (
            "Load and preview",
            "Bring in your airport-area case study or use the example, then preview inputs on a map or table.",
        ),
        (
            "Set scenarios",
            "Choose SAF blend levels to compare; the sidebar shows when everything is ready to run.",
        ),
        (
            "See results",
            "Run the analysis, then review maps and exports for briefings or reports.",
        ),
    ]
    items = [
        dbc.ListGroupItem(
            [
                html.H5(title, className="mb-1 fs-6"),
                html.P(text, className="mb-0 text-muted small"),
            ],
            className="py-2 px-3",
        )
        for title, text in stages
    ]

    method_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H6("AERMOD", className="mb-0 small")),
                        dbc.CardBody(
                            html.P(
                                "Baseline air pollution: concentrations communities experience before SAF-driven changes, "
                                "often supported by dispersion modeling.",
                                className="small text-muted mb-0",
                            ),
                            className="py-2",
                        ),
                    ],
                    className="h-100 shadow-sm border-0",
                ),
                md=4,
                className="mb-2",
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H6("NAS / TRB", className="mb-0 small"),
                        ),
                        dbc.CardBody(
                            html.P(
                                "Pollutant reduction from SAF blends follow Excel-based methods released by the "
                                "National Academies and Transportation Research Board (NAS/TRB).",
                                className="small text-muted mb-0",
                            ),
                            className="py-2",
                        ),
                    ],
                    className="h-100 shadow-sm border-0",
                ),
                md=4,
                className="mb-2",
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H6("BenMAP", className="mb-0 small")),
                        dbc.CardBody(
                            html.P(
                                "Health and economic benefits use a BenMAP-like procedure: changes in concentration drive "
                                "health impacts, which can optionally be converted to economic impact when the necessary "
                                "economic inputs are provided.",
                                className="small text-muted mb-0",
                            ),
                            className="py-2",
                        ),
                    ],
                    className="h-100 shadow-sm border-0",
                ),
                md=4,
                className="mb-2",
            ),
        ],
        className="g-2 mt-0",
    )

    return html.Div(
        [
            html.H3("Overview", className="mt-2 mb-2 fs-4"),
            html.P(
                "BenSAF is an analytical and visualization tool for estimating the health and economic benefits that "
                "could result from the use of sustainable aviation fuel (SAF). Click the logo to return here; use the "
                "sidebar for Data, Explore, Configuration, and Results.",
                className="text-muted small mb-2",
            ),
            html.H5("Using this dashboard", className="fs-6 mb-1 text-body"),
            dbc.ListGroup(items, flush=True, className="border rounded shadow-sm mb-2"),
            html.H5("Analytical backbone", className="fs-6 mb-1 text-body"),
            html.P(
                "BenSAF follows an integrated multi-step procedure established in literature, which includes methods "
                "from AERMOD, NAS/TRB, and BenMAP.",
                className="text-muted small mb-2",
            ),
            method_cards,
        ],
        className="pt-0 pb-2",
    )


def create_header():
    """Run control first, then prerequisites (sole readiness checklist for Execute Analysis)."""
    inner = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "Execute Analysis",
                            id="btn-run-analysis",
                            color="primary",
                            size="md",
                            className="w-100",
                            disabled=True,
                        ),
                        html.Div(id="analysis-status", className="small text-muted mt-2"),
                        dcc.Loading(
                            id="loading-analysis",
                            type="circle",
                            children=html.Div(id="loading-output"),
                        ),
                    ],
                    width=12,
                ),
            ],
            className="g-2 align-items-start",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6("Prerequisites", className="mb-2 mt-3 text-uppercase small text-muted"),
                        html.Div(
                            id="analysis-prerequisites-checklist",
                            className="small p-2 rounded bg-light border",
                        ),
                    ],
                    width=12,
                    className="mb-0",
                ),
            ],
            className="g-2 align-items-start",
        ),
    ]
    return dbc.Card(
        [
            dbc.CardBody(inner, className="p-3"),
        ],
        className="mb-0 shadow-sm border-0 bg-light",
    )


def create_explore_tab():
    """Map and table preview of merged inputs (requires data loaded on the Data tab)."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Explore", className="mt-4 mb-3"),
                            html.P(
                                "Preview merged tract data on a map or in a table. Load inputs on the Data tab "
                                "(uploads or example case study) first.",
                                className="text-muted mb-3",
                            ),
                            html.Label("Select Variable to Display", className="fw-bold"),
                            dcc.Dropdown(
                                id="data-viewer-dropdown",
                                options=[],
                                value=None,
                                clearable=False,
                                placeholder="Select a variable...",
                                className="mb-3",
                            ),
                            dbc.Tabs(
                                [
                                    dbc.Tab(
                                        label="Map",
                                        tab_id="data-explorer-map",
                                        children=[
                                            dcc.Loading(
                                                id="loading-data-viewer-map",
                                                type="circle",
                                                children=[
                                                    dcc.Graph(
                                                        id="data-viewer-map",
                                                        config={"displayModeBar": True, "scrollZoom": True},
                                                        style={"height": "380px"},
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                    dbc.Tab(
                                        label="Data table",
                                        tab_id="data-explorer-table",
                                        children=[
                                            html.P("First 100 records", className="text-muted small mb-2"),
                                            dcc.Loading(
                                                id="loading-data-table",
                                                type="circle",
                                                children=[
                                                    html.Div(
                                                        id="data-viewer-table",
                                                        style={
                                                            "height": "420px",
                                                            "overflowY": "auto",
                                                            "overflowX": "auto",
                                                        },
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                                id="data-explorer-tabs",
                                active_tab="data-explorer-map",
                                className="mb-0",
                            ),
                        ],
                        md=12,
                    )
                ]
            ),
        ],
        className="pt-1 pb-4",
    )


def create_data_tab():
    """Data tab: file upload vs example case study, accordion inputs."""
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
                        html.P([html.Strong("Data type: "), "Vector parcel boundaries (GeoJSON, Shapefile, or GeoPackage)."], className="mb-1 small"),
                        html.P([html.Strong("Required fields: "), "GEOID and geometry; keep attribute columns minimal."], className="mb-1 small"),
                        html.P(
                            [html.Strong("Role in analysis: "), "Defines tract polygons for spatial joins, mapping, and alignment of all tabular inputs."],
                            className="mb-3 small text-muted",
                        ),
                        dcc.Upload(
                            id='upload-tracts',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select file')
                            ]),
                            className="bensaf-upload-zone",
                            multiple=False
                        ),
                        html.Div(id='upload-tracts-status', className="text-muted mt-2"),
                    ], title="Case Study Boundary", item_id="item-tracts"),
                    
                    dbc.AccordionItem([
                        html.P([html.Strong("Data type: "), "Tabular (CSV), one row per census tract."], className="mb-1 small"),
                        html.P([html.Strong("Required fields: "), "GEOID; include population and other covariates used by the health pipelines."], className="mb-1 small"),
                        html.P(
                            [html.Strong("Role in analysis: "), "Population weighting and demographic context for incidence and impact calculations."],
                            className="mb-3 small text-muted",
                        ),
                        dcc.Upload(
                            id='upload-demographics',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select file')
                            ]),
                            className="bensaf-upload-zone",
                            multiple=False
                        ),
                        html.Div(id='upload-demographics-status', className="text-muted mt-2"),
                    ], title="Demographics", item_id="item-demographics"),
                    
                    dbc.AccordionItem([
                        html.P([html.Strong("Data type: "), "Gridded or tabular baseline pollutant concentrations per tract; upload CSV or build from AERMOD .ADO outputs."], className="mb-1 small"),
                        html.P(
                            [html.Strong("Required fields: "), "CSV: GEOID and a concentration column (e.g. ufp / baseline_pollutant_concentration). AERMOD: weighted .ADO files and CRS as described below."],
                            className="mb-1 small",
                        ),
                        html.P(
                            [html.Strong("Role in analysis: "), "Baseline exposure merged to tracts; combined with SAF blend reduction to estimate concentration changes under each scenario."],
                            className="mb-3 small text-muted",
                        ),
                        html.Label("Source", className="fw-bold mb-2 small"),
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
                            dcc.Upload(
                                id='upload-exposure',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select file')
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
                    ], title="Baseline Pollutant Exposure", item_id="item-exposure"),
                    
                    dbc.AccordionItem([
                        html.P([html.Strong("Data type: "), "Tabular (CSV), one row per census tract."], className="mb-1 small"),
                        html.P(
                            [html.Strong("Required fields: "), "GEOID and mortality_rate."],
                            className="mb-1 small",
                        ),
                        html.P(
                            [html.Strong("Role in analysis: "), "Baseline incidence drives attributable mortality cases in the health pipeline."],
                            className="mb-3 small text-muted",
                        ),
                        dbc.Card([
                            dbc.CardHeader([
                                html.Div([
                                    html.H5("Mortality", className="mb-0 d-inline"),
                                    html.Span(id='mortality-pipeline-status', className="ms-2"),
                                ])
                            ], className="bg-light"),
                            dbc.CardBody([
                                html.P(
                                    [
                                        html.Strong("Analysis note: "),
                                        "Mortality impacts use the Bouma et al. concentration–response parameters by default.",
                                    ],
                                    className="small text-muted mb-3",
                                ),
                                dcc.Upload(
                                    id='upload-mortality-incidence',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select incidence CSV'),
                                    ]),
                                    className="bensaf-upload-zone",
                                    multiple=False,
                                ),
                                html.Div(id='upload-mortality-incidence-status', className="text-muted mt-2"),
                            ]),
                        ], className="mb-0 shadow-sm border-0"),
                    ], title="Baseline Mortality Rate", item_id="item-health-pipelines"),
                    dbc.AccordionItem([
                        html.P([html.Strong("Data type: "), "Tabular (CSV), one row per census tract."], className="mb-1 small"),
                        html.P(
                            [html.Strong("Required fields: "), "GEOID and per_capita_expenditure. Optional: life_years_gained."],
                            className="mb-1 small",
                        ),
                        html.P(
                            [html.Strong("Role in analysis: "), "Tract-level expenditure supports monetized mortality benefits when you run scenarios."],
                            className="mb-3 small text-muted",
                        ),
                        dcc.Upload(
                            id='upload-mortality-economic',
                            children=html.Div([
                                "Drag and Drop or ",
                                html.A("Select per-capita expenditure CSV"),
                            ]),
                            className="bensaf-upload-zone",
                            multiple=False,
                        ),
                        html.Div(
                            id="upload-mortality-economic-status",
                            className="text-muted mt-2",
                        ),
                    ], title="Per-capita expenditure (optional)", item_id="item-per-capita-expenditure"),
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                "After inputs are loaded, open the ",
                                html.Strong("Explore"),
                                " tab to map or tabulate merged variables.",
                            ],
                            className="text-muted small mt-3 mb-0",
                        ),
                    ],
                    md=12,
                ),
            ]
        ),
    ], className="pt-1 pb-4")

def create_configuration_tab():
    """SAF blend scenarios; mortality uses Bouma et al. by default."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Configuration", className="mt-4 mb-3"),
                html.P(
                    "Define SAF blend scenarios to compare. Mortality impacts use the Bouma et al. model by default.",
                    className="text-muted",
                ),
            ])
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
                        html.Div(id='saf-scenarios-status', className="mt-2"),
                        html.Div(id='config-status', className="mt-2"),
                    ])
                ], className="mb-4 shadow-sm border-0"),
            ], md=6),
            dbc.Col([
                dcc.Graph(
                    id='saf-reduction-curve',
                    config={'displayModeBar': True},
                    style={'height': '400px'}
                ),
            ], md=6)
        ]),
        dcc.Store(id='saf-scenarios-store', data=[]),
    ], className="pt-1 pb-4")



def create_results_tab():
    """Tract maps, table, and export."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Results", className="mt-2 mb-1 fs-4"),
                html.P(
                    "Map results by census tract and scenario, or export tables for briefings and reports.",
                    className="text-muted small mb-2",
                ),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H5("Maps by census tract", className="mt-2 mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("SAF scenario", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="results-scenario-dropdown",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    placeholder="Scenario…",
                                    className="mb-2",
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Variable", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="results-map-dropdown",
                                    options=[],
                                    value=None,
                                    clearable=False,
                                    placeholder="Variable…",
                                    disabled=True,
                                    className="mb-2",
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="g-2 align-items-end",
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            label="Map",
                            tab_id="results-geo-map",
                            children=[
                                dcc.Loading(
                                    id="loading-map",
                                    type="circle",
                                    children=[
                                        dcc.Graph(
                                            id="results-map",
                                            config={"displayModeBar": True, "scrollZoom": True},
                                            style={"height": "380px"},
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dbc.Tab(
                            label="Table",
                            tab_id="results-geo-table",
                            children=[
                                dcc.Loading(
                                    id="loading-results-table",
                                    type="circle",
                                    children=[
                                        html.Div(
                                            id="results-table",
                                            style={
                                                "height": "320px",
                                                "overflowY": "auto",
                                                "overflowX": "auto",
                                            },
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                    id="results-geo-tabs",
                    active_tab="results-geo-map",
                    className="mb-0",
                ),
            ], md=12),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H6("Export", className="mb-0 text-uppercase small text-muted"),
                        className="py-2 bg-light border-0",
                    ),
                    dbc.CardBody(
                        className="py-2",
                        children=[
                            html.P(
                                "Summary CSV matches the scenario table; tract CSV adds geometry-linked columns.",
                                className="small text-muted mb-2 mb-md-0",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Summary (CSV)",
                                        id="btn-export-results-summary-csv",
                                        color="primary",
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "Tracts (CSV)",
                                        id="btn-export-results-tract-csv",
                                        color="primary",
                                        size="sm",
                                    ),
                                ],
                                className="mt-1",
                            ),
                            dcc.Download(id="download-results-summary-csv"),
                            dcc.Download(id="download-results-tract-csv"),
                        ],
                    ),
                ], className="mb-2 shadow-sm border-0"),
            ], md=12),
        ]),
    ], className="pt-1 pb-2")

