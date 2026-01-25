"""
CENOP - CETacean Noise-Population Model

Shiny Application Entry Point

Main entry point for the Python Shiny web interface.
Visualization design based on DEPONS (Disturbance Effects on the 
Harbour Porpoise Population in the North Sea).

This is a minimal entry point that imports the modular UI and server components.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CENOP")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
# Add src directory to path to ensure cenop package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from shiny import App

# Import modular components
from cenop.ui.layout import app_ui
from cenop.server.main import server

# Static files directory for logo and icon
APP_DIR = Path(__file__).parent / "static"

# Create the Shiny app with static file serving
app = App(app_ui, server, static_assets=APP_DIR)


if __name__ == "__main__":
    from shiny import run_app
    run_app(app)
