"""
Plotting utilities and settings for axon morphometry figures.

Provides consistent styling across all plots by loading settings from
config/plot_settings.yaml.
"""

import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def get_config_path() -> Path:
    """Get path to plot_settings.yaml config file."""
    # Look relative to this file's location
    module_dir = Path(__file__).parent
    repo_root = module_dir.parent
    return repo_root / "config" / "plot_settings.yaml"


def load_plot_settings(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load plot settings from YAML config file.

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        Dictionary of plot settings
    """
    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(f"Plot settings file not found: {config_path}")

    with open(config_path, 'r') as f:
        settings = yaml.safe_load(f)

    return settings


class PlotSettings:
    """
    Container for plot settings with convenient attribute access.

    Usage:
        settings = PlotSettings()
        color = settings.colors['sham']
        dpi = settings.figure['dpi']
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._settings = load_plot_settings(config_path)

    @property
    def colors(self) -> Dict[str, str]:
        """Color palette for groups and indicators."""
        return self._settings.get('colors', {})

    @property
    def markers(self) -> Dict[str, str]:
        """Marker styles for populations."""
        return self._settings.get('markers', {})

    @property
    def figure(self) -> Dict[str, Any]:
        """Figure settings (dpi, figsize, etc.)."""
        return self._settings.get('figure', {})

    @property
    def fonts(self) -> Dict[str, Any]:
        """Font settings (sizes, weight)."""
        return self._settings.get('fonts', {})

    @property
    def grid(self) -> Dict[str, Any]:
        """Grid settings."""
        return self._settings.get('grid', {})

    @property
    def frame(self) -> Dict[str, Any]:
        """Frame/box settings."""
        return self._settings.get('frame', {'box': True, 'linewidth': 1.0})

    @property
    def error_bars(self) -> Dict[str, Any]:
        """Error bar settings."""
        return self._settings.get('error_bars', {})

    @property
    def histogram(self) -> Dict[str, Any]:
        """Histogram settings."""
        return self._settings.get('histogram', {})

    @property
    def line(self) -> Dict[str, Any]:
        """Line plot settings."""
        return self._settings.get('line', {})

    @property
    def panel_labels(self) -> Dict[str, Any]:
        """Panel label settings (a, b, c, ...)."""
        return self._settings.get('panel_labels', {
            'enabled': True,
            'fontsize': 16,
            'fontweight': 'bold',
            'position': [-0.12, 1.05],
            'va': 'bottom',
            'ha': 'left'
        })

    def get_group_color(self, group: str) -> str:
        """Get color for a group (sham/tbi), case-insensitive."""
        return self.colors.get(group.lower(), self.colors.get('unknown', '#7f7f7f'))

    def get_marker(self, population: Optional[str]) -> str:
        """Get marker for a population (cc/cg), case-insensitive."""
        if population is None:
            return self.markers.get('default', 's')
        return self.markers.get(population.lower(), self.markers.get('default', 's'))


# Default instance for easy import
_default_settings: Optional[PlotSettings] = None


def get_plot_settings() -> PlotSettings:
    """Get the default PlotSettings instance (lazy loaded)."""
    global _default_settings
    if _default_settings is None:
        _default_settings = PlotSettings()
    return _default_settings


def style_axis(
    ax,
    settings: Optional[PlotSettings] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Apply consistent font styling to an axis.

    Args:
        ax: Matplotlib axis
        settings: Optional PlotSettings instance. If None, uses default.
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
    """
    if settings is None:
        settings = get_plot_settings()

    fonts = settings.fonts
    grid_settings = settings.grid
    frame_settings = settings.frame

    # Apply tick label size
    ax.tick_params(labelsize=fonts.get('tick_size', 12))

    # Apply axis labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fonts.get('label_size', 14))
    elif ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=fonts.get('label_size', 14))

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fonts.get('label_size', 14))
    elif ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=fonts.get('label_size', 14))

    # Apply grid (disabled by default)
    if grid_settings.get('enabled', False):
        ax.grid(True, alpha=grid_settings.get('alpha', 0.3))
    else:
        ax.grid(False)

    # Apply frame/box settings
    if frame_settings.get('box', True):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(frame_settings.get('linewidth', 1.0))


def add_panel_labels(
    axes: Union[List, "np.ndarray"],
    labels: Optional[List[str]] = None,
    settings: Optional[PlotSettings] = None
) -> None:
    """
    Add panel labels (a, b, c, ...) to a list of axes.

    Uses settings from plot_settings.yaml for consistent styling.

    Args:
        axes: List of matplotlib axes, or 2D array from plt.subplots()
        labels: Optional custom labels. If None, uses lowercase letters (a, b, c, ...)
        settings: Optional PlotSettings instance. If None, uses default.

    Example:
        fig, axes = plt.subplots(2, 3)
        add_panel_labels(axes)  # Adds 'a' through 'f'
    """
    import numpy as np

    if settings is None:
        settings = get_plot_settings()

    panel_settings = settings.panel_labels
    if not panel_settings.get('enabled', True):
        return

    # Flatten axes array if needed
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flat
    else:
        axes_flat = axes

    # Generate default labels if not provided
    if labels is None:
        labels = list(string.ascii_lowercase[:len(list(axes_flat))])
        # Reset iterator
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flat
        else:
            axes_flat = axes

    pos = panel_settings.get('position', [-0.12, 1.05])

    for ax, label in zip(axes_flat, labels):
        ax.text(
            pos[0], pos[1], label,
            transform=ax.transAxes,
            fontsize=panel_settings.get('fontsize', 16),
            fontweight=panel_settings.get('fontweight', 'bold'),
            va=panel_settings.get('va', 'bottom'),
            ha=panel_settings.get('ha', 'left')
        )
