"""
BenSAF Dash Application

Main entry point for the Plotly Dash web interface to BenSAF.
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

from bensaf_dash.layouts.main_layout import create_layout
from bensaf_dash.callbacks import workflow_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.title = "BenSAF - Benefits of Sustainable Aviation Fuels"

app.layout = create_layout()

workflow_callbacks.register_callbacks(app)

server = app.server

def main():
    app.run(debug=True, host='0.0.0.0', port=8052)

if __name__ == '__main__':
    main()


