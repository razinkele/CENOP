"""
UI Tabs Module

Contains all tab UI definitions.
"""

from ui.tabs.dashboard import dashboard_tab
from ui.tabs.settings import settings_tab
from ui.tabs.population import population_tab
from ui.tabs.disturbance import disturbance_tab
from ui.tabs.export import export_tab

__all__ = [
    "dashboard_tab",
    "settings_tab", 
    "population_tab",
    "disturbance_tab",
    "export_tab"
]
