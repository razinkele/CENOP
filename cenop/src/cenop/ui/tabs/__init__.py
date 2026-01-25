"""
UI Tabs Module

Contains all tab UI definitions.
"""

from .dashboard import dashboard_tab
from .settings import settings_tab
from .population import population_tab
from .disturbance import disturbance_tab
from .export import export_tab

__all__ = [
    "dashboard_tab",
    "settings_tab", 
    "population_tab",
    "disturbance_tab",
    "export_tab"
]
