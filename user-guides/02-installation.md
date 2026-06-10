# Installation

## Requirements

BenSAF requires Python 3.11 or higher.

## Dependencies

Core dependencies include:

- pandas >= 2.0.0
- numpy >= 1.24.0
- geopandas >= 0.13.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- contextily >= 1.5.0

Optional dependencies:

- **GUI**: PySide6 >= 6.5.0
- **Dash**: dash >= 2.14.0, dash-bootstrap-components >= 1.5.0, plotly >= 5.18.0

## Installation Methods

### From Source

```bash
git clone <repository-url>
cd BenSAF
pip install -e .
```

### Development Installation

For development, install with optional dependencies:

```bash
pip install -e ".[gui,dash]"
```

## Verification

After installation, verify the installation:

```python
import bensaf
print(bensaf.__version__)
```
