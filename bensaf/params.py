"""
Parameter loading for SAF health impact assessment.

All functions accept an optional path argument so the file location can be
injected rather than resolved from __file__ at call time.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from bensaf.mortality_functions import MortalityFunctionLibrary

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data'


def load_saf_blend_parameters(path: Optional[Path] = None) -> List[float]:
    """
    Load SAF blend polynomial coefficients from JSON.

    Returns:
        List of polynomial coefficients [a0, a1, a2, ...] such that
        reduction = a0 + a1*SAF + a2*SAF^2 + ... (negative decimal, e.g. -0.3 = 30% reduction)
    """
    if path is None:
        path = _DEFAULT_DATA_DIR / 'saf_blend_parameters.json'

    default_coeffs = [0.0, -0.0152, 0.00009]

    if not path.exists():
        logger.warning(f"SAF blend parameters file not found at {path}, using default coefficients")
        return default_coeffs

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        coeffs = data.get('polynomial_coefficients', default_coeffs)
        logger.info(f"Loaded SAF blend parameters: {coeffs}")
        return coeffs
    except Exception as e:
        logger.error(f"Error loading SAF blend parameters: {e}, using defaults")
        return default_coeffs


def load_economic_parameters(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load economic parameters from JSON.

    Returns:
        Dict with keys: per_capita_consumption, life_years_gained,
        preterm_birth_odds_ratio, monetary_value_per_ptb
    """
    if path is None:
        path = _DEFAULT_DATA_DIR / 'economic_parameters.json'

    defaults: Dict[str, Any] = {
        'per_capita_consumption': None,
        'life_years_gained': 10.0,
        'preterm_birth_odds_ratio': None,
        'monetary_value_per_ptb': None,
    }

    if not path.exists():
        logger.warning(f"Economic parameters file not found at {path}, using defaults")
        return defaults

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        params = {**defaults, **data}
        logger.info(f"Loaded economic parameters from {path}")
        return params
    except Exception as e:
        logger.warning(f"Error loading economic parameters: {e}, using defaults")
        return defaults


def load_mortality_function_config(
    function_id: Optional[int] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a single mortality function configuration from the library.

    Args:
        function_id: ID of the function to load. If None, uses the first available.
        path: Path to the mortality functions JSON file. If None, uses the default.

    Returns:
        Dict with keys: title, mean_rr, lower_rr, upper_rr, unit_increase
    """
    library = MortalityFunctionLibrary(json_path=path)

    if function_id is None:
        functions = library.list_functions()
        if not functions:
            raise ValueError("No mortality functions available")
        function_id = functions[0]['id']

    function_data = library.get_function(function_id)
    if function_data is None:
        raise ValueError(f"Mortality function {function_id} not found")

    return function_data
