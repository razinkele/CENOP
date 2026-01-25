"""
Simple test Shiny app to verify rendering works.
"""

from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map, basemaps
import logging

# Add CENOP core imports
from cenop import Simulation, SimulationParameters
from cenop.landscape import CellData, create_homogeneous_landscape
from cenop.parameters import SimulationConstants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Sidebar"),
        ui.output_text("sidebar_text"),
    ),
    ui.h1("Test App"),
    ui.output_text("test_output"),
    ui.output_ui("test_ui"),
    output_widget("test_map"),
)

def server(input, output, session):
    logger.info("SERVER INITIALIZED")
    
    # Reactive value
    my_value = reactive.Value("Hello Reactive!")
    
    @render.text
    def sidebar_text():
        logger.info("RENDER sidebar_text called!")
        return "Sidebar working"
    
    @render.text
    def test_output():
        logger.info("RENDER test_output called!")
        return "Hello, this is a test text output"
    
    @render.ui
    def test_ui():
        logger.info("RENDER test_ui called!")
        return ui.p("This is a UI output", style="color: green;")
    
    @render_widget
    def test_map():
        logger.info("RENDER test_map called!")
        return Map(center=(55.5, 4.0), zoom=6, basemap=basemaps.Esri.OceanBasemap)

app = App(app_ui, server)
