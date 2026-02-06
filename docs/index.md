# SAF Toolkit Documentation

Welcome to the SAF Toolkit documentation! This toolkit provides tools and utilities for working with various data sources and APIs.

## Getting Started

To get started with the SAF Toolkit, you'll need to:

1. Install the package
2. Configure your environment
   - See [API Keys Setup Guide](../api_keys.md) for required API keys
3. Start using the tools

## Installation

```bash
pip install saf-toolkit
```

## Quick Start

Here's a quick example of how to use the toolkit:

```python
from saf_toolkit.data import fetch_mortality_data

# Fetch mortality data for Washington state
df = fetch_mortality_data(
    output_path="data/mortality.csv",
    year_range="2018-2022",
    state="Washington"
)
```

## Features

- CDC WONDER API integration
- Data fetching and processing utilities
- Configuration management
- More features coming soon!

## Documentation

- [API Reference](api.md) - Detailed API documentation
- [Data Requirements](data_requirements.md) - Data sources, formats, and requirements
- [Contributing Guide](contributing.md) - How to contribute to the project

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for more information.

## License

This project is licensed under the Revised BSD License.