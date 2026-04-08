"""
Utility subpackage for SAF health impact assessment.

Submodules:
    bensaf.utils.params     — load_saf_blend_parameters, load_economic_parameters, etc.
    bensaf.utils.synthetic  — create_synthetic_data, calculate_weighted_ufp
    bensaf.utils.geo        — bin_tracts_by_distance, analyze_impacts_by_distance,
                              create_distance_analysis_plots
    bensaf.utils.census     — fetch_census_tracts
"""

from bensaf.utils.synthetic import calculate_weighted_ufp, create_synthetic_data
from bensaf.utils.geo import (
    analyze_impacts_by_distance,
    bin_tracts_by_distance,
    create_distance_analysis_plots,
)
from bensaf.utils.census import fetch_census_tracts

__all__ = [
    'calculate_weighted_ufp',
    'create_synthetic_data',
    'bin_tracts_by_distance',
    'analyze_impacts_by_distance',
    'create_distance_analysis_plots',
    'fetch_census_tracts',
]
