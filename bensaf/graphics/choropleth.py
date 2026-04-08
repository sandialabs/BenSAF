"""
Graphics utilities for creating maps and visualizations.

This module provides functions for creating standardized maps and visualizations
for health impact analysis, with support for different geographic regions and datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional, Union, Tuple, List
import geopandas as gpd


def plot_background_map(ax: plt.Axes, gdf: gpd.GeoDataFrame = None,
                       edgecolor: str = 'black', linewidth: float = 0.2,
                       facecolor: str = 'white', add_basemap: bool = True,
                       basemap_style: str = 'osm', alpha: float = 0.6,
                       show_boundaries: bool = True) -> None:
    """
    Plot background basemap with optional geographic boundaries.
    
    Args:
        ax: Matplotlib axes object
        gdf: Optional GeoDataFrame containing geographic boundaries to overlay
        edgecolor: Color of boundary lines
        linewidth: Width of boundary lines
        facecolor: Fill color of boundaries
        add_basemap: Whether to add a web-based basemap
        basemap_style: Style of basemap ('osm', 'cartodb-positron', 'cartodb-darkmatter', 'stamen-terrain', 'stamen-toner')
        alpha: Transparency of the basemap
        show_boundaries: Whether to show the GeoDataFrame boundaries
    """
    # Add web-based basemap if requested
    if add_basemap:
        try:
            import contextily as ctx
            
            # Define basemap source mapping
            basemap_sources = {
                'osm': ctx.providers.OpenStreetMap.Mapnik,
                'cartodb-positron': ctx.providers.CartoDB.Positron,
                'cartodb-darkmatter': ctx.providers.CartoDB.DarkMatter,
            }
            
            source = basemap_sources.get(basemap_style, ctx.providers.OpenStreetMap.Mapnik)

            ctx.add_basemap(ax, crs=gdf.crs, source=source, alpha=alpha)
            
        except ImportError:
            print("Warning: contextily not available. Install with 'pip install contextily' for basemap support")
        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")
    return ax

def plot_choropleth_map(ax: plt.Axes, gdf: gpd.GeoDataFrame, column: Union[str, pd.Series],
                       vmin: Optional[float] = None, vmax: Optional[float] = None,
                       cmap: str = 'viridis', alpha: float = 0.75,
                       linewidth: float = 0.2, legend: bool = False) -> None:
    """
    Plot choropleth map with variable coloring.
    
    Args:
        ax: Matplotlib axes object
        gdf: GeoDataFrame containing tract geometries and data
        column: Column name (str) or pandas Series with matching index to use for coloring
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap to use
        alpha: Transparency level
        linewidth: Width of tract boundary lines
        legend: Whether to show legend
    """
    # If column is a string, use it directly
    gdf.plot(ax=ax, column=column, alpha=alpha, legend=legend, 
            cmap=cmap, linewidth=linewidth, vmin=vmin, vmax=vmax)


def add_colorbar(ax: plt.Axes, vmin: float, vmax: float, label: str,
                cmap: str = 'viridis', orientation: str = 'horizontal', 
                fraction: float = 0.05, pad: float = 0.05, alpha: float = 0.75, 
                fontsize: int = 12) -> None:
    """
    Add colorbar to map.
    
    Args:
        ax: Matplotlib axes object
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        label: Label for colorbar
        cmap: Colormap to use (should match the main plot)
        orientation: Orientation of colorbar ('horizontal' or 'vertical')
        fraction: Fraction of axes to use for colorbar
        pad: Padding around colorbar
        alpha: Transparency level
        fontsize: Font size for label and ticks
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, orientation=orientation, fraction=fraction, 
                             pad=pad, alpha=alpha)
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)


def standardize_map_appearance(ax: plt.Axes, title: str, fontsize: int = 15) -> None:
    """
    Apply standard formatting to map.
    
    Args:
        ax: Matplotlib axes object
        title: Title for the map
        fontsize: Font size for title
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fontsize)
    # Note: tight_layout should be called on the figure, not here
    # This function now only handles axes formatting


def plot_point_icon(ax: plt.Axes, longitudes: Union[float, List[float]], 
                   latitudes: Union[float, List[float]], 
                   icon_path: Optional[Union[str, Path]] = None, 
                   marker: str = 'o', markersize: int = 100, color: str = 'red',
                   zoom: float = 0.04, zorder: int = 1) -> None:
    """
    Plot icon or marker at specified locations.
    
    Args:
        ax: Matplotlib axes object
        longitudes: Longitude(s) for icon placement
        latitudes: Latitude(s) for icon placement
        icon_path: Path to icon image file. If None, uses matplotlib marker
        marker: Matplotlib marker symbol to use when icon_path is None
        markersize: Size of marker when icon_path is None
        color: Color of marker when icon_path is None
        zoom: Zoom factor for icon size (only used when icon_path is provided)
        zorder: Z-order for layering
    """
    # Convert single values to lists for consistent processing
    if isinstance(longitudes, (int, float)):
        longitudes = [longitudes]
    if isinstance(latitudes, (int, float)):
        latitudes = [latitudes]
    
    if icon_path is None:
        # Use matplotlib marker
        ax.scatter(longitudes, latitudes, marker=marker, s=markersize, 
                  color=color, zorder=zorder, edgecolors='black', linewidth=1)
    else:
        # Use custom icon image
        try:
            # Load image
            from matplotlib.image import imread
            image = imread(icon_path)
            
            # Handle both RGB and grayscale images
            if len(image.shape) == 3:  # RGB image
                img_height, img_width, _ = image.shape
            else:  # Grayscale image
                img_height, img_width = image.shape
            
            aspect_ratio = img_width / img_height
            
            # Plot icon at each location
            for lon, lat in zip(longitudes, latitudes):
                ax.imshow(image, 
                         extent=(lon - zoom*aspect_ratio*0.9, lon + zoom*aspect_ratio*0.9,
                                lat - zoom*0.7, lat + zoom*0.7), 
                         zorder=zorder)
        except Exception as e:
            # Fallback to marker if image loading fails
            print(f"Warning: Could not load icon from {icon_path}. Using marker instead. Error: {e}")
            ax.scatter(longitudes, latitudes, marker=marker, s=markersize, 
                      color=color, zorder=zorder, edgecolors='black', linewidth=1)


def plot_excluded_tracts(ax: plt.Axes, gdf: gpd.GeoDataFrame, 
                        tract_ids: List[str], exclude_ids: Optional[List[str]] = None,
                        color: str = 'white') -> None:
    """
    Plot tracts that should be excluded from analysis (e.g., zero population tracts).
    
    Args:
        ax: Matplotlib axes object
        gdf: GeoDataFrame containing tract geometries
        tract_ids: List of tract IDs to potentially exclude
        exclude_ids: List of tract IDs to exclude from exclusion (e.g., airport tracts)
        color: Color to use for excluded tracts
    """
    if exclude_ids is None:
        exclude_ids = []
    
    for tract_id in tract_ids:
        if tract_id not in exclude_ids:
            tract_data = gdf[gdf['GEOID'] == tract_id]
            if not tract_data.empty:
                tract_data.plot(ax=ax, color=color)


def create_choropleth_map(gdf: gpd.GeoDataFrame, column: str, title: str,
                         vmin: Optional[float] = None, vmax: Optional[float] = None,
                         cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 8),
                         show_background: bool = True, show_boundaries: bool = True,
                         point_locations: Optional[List[Tuple[float, float]]] = None,
                         icon_path: Optional[Union[str, Path]] = None,
                         excluded_tracts: Optional[List[str]] = None,
                         exclude_from_exclusion: Optional[List[str]] = None,
                         ax: Optional[plt.Axes] = None) -> Union[plt.Figure, plt.Axes]:
    """
    Create a complete choropleth map with optional features.
    
    Args:
        gdf: GeoDataFrame containing tract geometries and data
        column: Column name to use for coloring
        title: Title for the map
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        cmap: Colormap to use
        figsize: Figure size
        show_background: Whether to show background tract boundaries
        show_boundaries: Whether to show tract boundaries on choropleth
        point_locations: List of (lon, lat) tuples for point icons
        icon_path: Path to icon image file
        excluded_tracts: List of tract IDs to exclude
        exclude_from_exclusion: List of tract IDs to exclude from exclusion
        ax: Optional axes to plot on. If None, creates new figure and axes
        
    Returns:
        Matplotlib figure object if ax is None, otherwise the axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot background if requested
    if show_background:
        plot_background_map(ax, gdf)
    
    # Plot choropleth
    plot_choropleth_map(ax, gdf, column, vmin=vmin, vmax=vmax, cmap=cmap,
                       legend=False, linewidth=0.2 if show_boundaries else 0)
    
    # Add point icons if provided
    if point_locations:
        # Handle single location or list of locations
        if isinstance(point_locations[0], (int, float)):
            # Single location as tuple
            longitudes, latitudes = [point_locations[0]], [point_locations[1]]
        else:
            # List of locations
            longitudes, latitudes = zip(*point_locations)
        plot_point_icon(ax, longitudes, latitudes, icon_path)
    
    # Plot excluded tracts if provided
    if excluded_tracts:
        plot_excluded_tracts(ax, gdf, excluded_tracts, exclude_from_exclusion)
    
    # Add colorbar if vmin/vmax are provided
    if vmin is not None and vmax is not None:
        if isinstance(column, str):
            label = column
        elif isinstance(column, pd.Series):
            label = column.name
        else:
            label = None
        add_colorbar(ax, vmin, vmax, label, cmap=cmap)
    
    # Standardize appearance
    standardize_map_appearance(ax, title)
    
    return ax

def save_map(fig: plt.Figure, output_path: Union[str, Path], 
             dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """
    Save map to file.
    
    Args:
        fig: Matplotlib figure object
        output_path: Path to save the figure
        dpi: Resolution for saved image
        bbox_inches: Bounding box setting for saved image
    """
    fig.savefig(str(output_path), dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)