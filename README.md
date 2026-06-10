# BenSAF

![BenSAF logo](BenSAF.png)

A toolkit for health-based assessments of sustainable aviation fuels.

## Overview

BenSAF provides a generalized framework for:

1. Processing geospatial data for airport-adjacent communities
2. Estimating health impacts of sustainable aviation fuel blend scenarios
3. Analyzing impacts across different demographic groups
4. Generating visualizations and reports

## Installation

BenSAF requires **Python 3.11 or higher**.

1. Clone the repository:
```bash
git clone <repository-url>
cd BenSAF
```

2. Create and activate a virtual environment. We recommend using conda (miniconda) or uv to manage the virtual environment:

   **Using miniconda:**
```bash
conda create -n bensaf python=3.11
conda activate bensaf
```

   **Using uv:**
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package in development (editable) mode:
```bash
pip install -e .
```

## Project Structure

```
bensaf/                   # Core analysis toolkit
├── __init__.py
├── workflow.py          # Core workflow orchestration
├── health_impacts.py    # Health impact calculation functions
├── utils.py             # Utility functions and data processing
├── graphics.py          # Visualization and plotting utilities

bensaf_gui/              # Desktop GUI application (PyQt/PySide6)
├── model/               # MVC Model layer
├── view/                # MVC View layer
├── controller/          # MVC Controller layer

bensaf_dash/             # Web-based dashboard (Plotly Dash)
├── app.py               # Main Dash application
├── layouts/             # UI layouts
├── callbacks/           # Interactive callbacks
```

## User Interfaces

BenSAF provides multiple interfaces to suit different workflows:

### 1. Web Dashboard (Dash)

Launch the web-based interface:

```bash
python -m bensaf_dash.app
```

Then open your browser to `http://localhost:8050`

Features:
- Web browser-based interface
- File upload for data
- Interactive parameter configuration
- Real-time visualization
- No installation required for end users (when deployed)

Included in the base install (`pip install -e .`). See `bensaf_dash/README.md` for detailed documentation.

### 2. Desktop GUI (PyQt/PySide6)

Launch the native desktop application:

```bash
python -m bensaf_gui.main
```

Features:
- Native desktop experience
- Offline capability
- Rich file dialogs
- Embedded matplotlib visualizations

Install GUI dependencies:
```bash
pip install -e ".[gui]"
```

See `bensaf_gui/README.md` for detailed documentation.

### 3. Python API

Use BenSAF programmatically in scripts or notebooks:

```python
from bensaf.workflow import Workflow

workflow = Workflow()
# ... continue with analysis
```

All interfaces use the same core `bensaf` package, ensuring consistent results.

## License

BSD 3-Clause License

## Sandia Funding Statement

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525.
