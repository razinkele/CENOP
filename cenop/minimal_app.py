"""
Minimal test of the app structure
"""

from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map, basemaps
import logging

from cenop import Simulation, SimulationParameters
from cenop.landscape import CellData, create_homogeneous_landscape
from cenop.parameters import SimulationConstants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MINIMAL")

# Simplified UI - just sidebar with outputs
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.output_text("test1"),
        ui.output_ui("test2"),
        ui.input_action_button("run", "Run", class_="btn-primary"),
        width=250
    ),
    # Main area
    ui.h1("Minimal Test"),
    ui.output_text("test3"),
    output_widget("test_map"),
    title="Test",
    fillable=True
)

def server(input, output, session):
    logger.info("=== SERVER INITIALIZED ===")
    
    # Reactive values
    counter = reactive.Value(0)
    
    @reactive.effect
    @reactive.event(input.run)
    def run_handler():
        logger.info("RUN clicked")
        counter.set(counter() + 1)
    
    @render.text
    def test1():
        logger.info("RENDER test1 called!")
        return f"Test 1: Counter = {counter()}"
    
    @render.ui
    def test2():
        logger.info("RENDER test2 called!")
        return ui.p(f"Test 2: {counter()}", style="color:green")
    
    @render.text
    def test3():
        logger.info("RENDER test3 called!")
        return f"Test 3: {counter()}"
    
    @render_widget
    def test_map():
        logger.info("RENDER test_map called!")
        return Map(center=(55.5, 4.0), zoom=6, basemap=basemaps.Esri.OceanBasemap)
    
    logger.info("All renders defined")

app = App(app_ui, server)
