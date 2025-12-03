"""
Plotting utilities and settings for axon morphometry figures.

Provides consistent styling across all plots by loading settings from
config/plot_settings.yaml.
"""

from pathlib import Path
from typing import Any, Dict, Optional

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
