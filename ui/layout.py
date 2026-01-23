"""
Main UI Layout for CENOP Shiny App
CENOP - CETacean Noise-Population Model
"""

from shiny import ui
import shinyswatch

from ui.sidebar import create_sidebar
from ui.tabs.dashboard import dashboard_tab
from ui.tabs.settings import settings_tab
from ui.tabs.population import population_tab
from ui.tabs.disturbance import disturbance_tab
from ui.tabs.export import export_tab


# Custom CSS for styling
CUSTOM_CSS = """
/* Progress bar styling */
.progress { height: 20px; }
.progress-bar { transition: width 0.3s ease-in-out; }

/* Card improvements */
.card { box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.card-header { font-weight: 600; background-color: #f8f9fa; }

/* Value box styling */
.value-box { min-height: 100px; }

/* Sidebar styling */
.sidebar { background-color: #f8f9fa; }

/* Error display */
.shiny-output-error { color: #dc3545; }
.shiny-output-error:before { content: '⚠ '; }

/* Help button styling */
.help-btn {
    background: transparent;
    border: none;
    color: #6c757d;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 5px 10px;
    margin-left: auto;
}
.help-btn:hover {
    color: #2c3e50;
}

/* Help modal content styling */
.help-content h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 25px; }
.help-content h3 { color: #34495e; margin-top: 20px; }
.help-content h4 { color: #7f8c8d; margin-top: 15px; }
.help-content ul { margin-left: 20px; }
.help-content li { margin-bottom: 5px; }
.help-content code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
.help-content .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
.help-content .param-table th, .help-content .param-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.help-content .param-table th { background: #3498db; color: white; }
.help-content .param-table tr:nth-child(even) { background: #f9f9f9; }
.help-content .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 15px 0; }
.help-content .tip { background: #d4edda; border-left: 4px solid #28a745; padding: 10px 15px; margin: 15px 0; }
"""


def create_help_modal():
    """Create the help modal with comprehensive documentation."""
    return ui.modal(
        ui.div(
            ui.HTML("""
<div class="help-content">
    <h2>📖 CENOP User Manual</h2>
    <p><strong>CENOP</strong> (CETacean Noise-Population Model) is a Python translation of the DEPONS 3.0 model 
    for simulating how harbour porpoise population dynamics are affected by disturbances from offshore 
    wind farm construction and ship noise.</p>
    
    <h2>🚀 Quick Start</h2>
    <ol>
        <li><strong>Set Initial Population</strong> - Enter the number of porpoises (default: 1000)</li>
        <li><strong>Set Simulation Years</strong> - How long to run (1-100 years)</li>
        <li><strong>Select Landscape</strong> - Choose the geographical area</li>
        <li><strong>Click "Load Landscape"</strong> - Load bathymetry and food distribution</li>
        <li><strong>Select Wind Turbines</strong> (optional) - Choose a turbine scenario</li>
        <li><strong>Click "Load Turbines"</strong> - Display turbines and noise contours</li>
        <li><strong>Click "Run Simulation"</strong> - Start the simulation</li>
        <li><strong>Adjust Speed</strong> - Use the slider to speed up or slow down</li>
    </ol>
    
    <h2>🗺️ Landscapes</h2>
    <table class="param-table">
        <tr><th>Landscape</th><th>Description</th><th>Available Turbine Scenarios</th></tr>
        <tr><td>Homogeneous</td><td>Uniform test landscape (400×400 cells)</td><td>None</td></tr>
        <tr><td>NorthSea</td><td>North Sea with real bathymetry (400×400 @ 400m)</td><td>Scenarios 1-3 (80-240 turbines)</td></tr>
        <tr><td>UserDefined</td><td>DEPONS default landscape data files</td><td>User-defined</td></tr>
    </table>
    <div class="note">
        <strong>Note:</strong> Other DEPONS landscapes (Kattegat, InnerDanishWaters, DanTysk, Gemini) 
        require separate data files not included in this distribution.
    </div>
    
    <h2>🌬️ Wind Turbine Scenarios</h2>
    <p>Turbine scenarios define the location and construction timing of offshore wind farms. 
    Each turbine generates pile-driving noise during construction that deters porpoises.</p>
    <div class="note">
        <strong>Note:</strong> The noise overlay (red shading) shows areas where received sound levels 
        exceed the deterrence threshold (158 dB). Porpoises avoid these areas during pile-driving.
    </div>
    
    <h2>📊 Dashboard Visualizations</h2>
    <h3>Map Layers (Toggle On/Off)</h3>
    <ul>
        <li><strong>Porpoises</strong> (blue dots) - Current positions of simulated animals</li>
        <li><strong>Depth</strong> - Bathymetry from EMODnet (toggle in Layers panel)</li>
        <li><strong>Turbines</strong> (orange dots) - Wind turbine locations</li>
        <li><strong>Noise</strong> (red shading) - Sound levels above deterrence threshold</li>
        <li><strong>Foraging</strong> (green shading) - Food availability / patch distribution</li>
    </ul>
    
    <h3>Charts</h3>
    <ul>
        <li><strong>Population Size</strong> - Total porpoises and lactating females with calves over time</li>
        <li><strong>Life and Death</strong> - Daily births and deaths</li>
        <li><strong>Energy Balance</strong> - Average food eaten vs energy expended</li>
    </ul>
    
    <h2>⚙️ Model Settings</h2>
    <p>All parameters have tooltip icons (ⓘ) - hover for detailed descriptions.</p>
    
    <h3>Basic Tab</h3>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Random Seed</td><td>0 (auto)</td><td>Seed for reproducibility (0 = random each run)</td></tr>
        <tr><td>Tracked Porpoises</td><td>1</td><td>Number of individuals to track in detail</td></tr>
        <tr><td>Ship Traffic</td><td>Off</td><td>Enable/disable vessel noise disturbance</td></tr>
        <tr><td>Bycatch Probability</td><td>0.0</td><td>Annual probability of fishing net mortality</td></tr>
    </table>
    
    <h3>Movement Tab (CRW Parameters)</h3>
    <p>Correlated Random Walk parameters controlling fine-scale movement:</p>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>k (Inertia)</td><td>0.001</td><td>Directional persistence - higher = straighter paths</td></tr>
        <tr><td>a0</td><td>0.35</td><td>Step length autocorrelation</td></tr>
        <tr><td>a1</td><td>0.0005</td><td>Water depth effect on step length</td></tr>
        <tr><td>a2</td><td>-0.02</td><td>Salinity effect on step length</td></tr>
        <tr><td>b0</td><td>-0.024</td><td>Turning angle autocorrelation</td></tr>
        <tr><td>b1</td><td>-0.008</td><td>Water depth effect on turning</td></tr>
        <tr><td>b2</td><td>0.93</td><td>Salinity effect on turning</td></tr>
        <tr><td>b3</td><td>-14.0</td><td>Intercept for turning angle</td></tr>
    </table>
    
    <h3>Dispersal Tab</h3>
    <p>Controls large-scale movement when porpoises have declining energy:</p>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Dispersal Type</td><td>PSM-Type2</td><td>PSM-Type2 (memory-based with heading dampening), Undirected, or Off</td></tr>
        <tr><td>tDisp</td><td>3 days</td><td>Consecutive days of declining energy to trigger dispersal</td></tr>
        <tr><td>PSM_log</td><td>0.6</td><td>Logistic increase rate for spatial memory</td></tr>
        <tr><td>PSM_dist</td><td>N(300;100)</td><td>Preferred dispersal distance: mean 300km, std 100km</td></tr>
        <tr><td>PSM_tol</td><td>5 km</td><td>Tolerance for reaching dispersal target</td></tr>
        <tr><td>PSM_angle</td><td>20°</td><td>Maximum heading change per step during dispersal</td></tr>
    </table>
    
    <h3>Energy Tab</h3>
    <p>Memory decay rates and food dynamics:</p>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>rS (Satiation)</td><td>0.04</td><td>Decay rate for satiation memory - how fast porpoises forget satisfaction</td></tr>
        <tr><td>rR (Reference)</td><td>0.04</td><td>Decay rate for reference memory - how fast they forget food locations</td></tr>
        <tr><td>rU (Replenishment)</td><td>0.1</td><td>Rate at which depleted food patches recover</td></tr>
    </table>
    
    <h2>📈 Population Tab</h2>
    <p>Detailed population analytics:</p>
    <ul>
        <li><strong>Age Distribution</strong> - Histogram of porpoise ages (0-30 years)</li>
        <li><strong>Energy Distribution</strong> - Histogram of energy levels (0-20 units)</li>
        <li><strong>Landscape Energy</strong> - Total food availability over time</li>
        <li><strong>Average Movement</strong> - Daily movement distances</li>
        <li><strong>Vital Statistics</strong> - Summary table of population metrics</li>
    </ul>
    
    <h2>🔊 Disturbance Tab</h2>
    <p>Noise impact analysis:</p>
    <ul>
        <li><strong>Dispersal Plot</strong> - Number of porpoises in dispersal states over time</li>
        <li><strong>Deterrence Events</strong> - Count of currently deterred porpoises</li>
        <li><strong>Noise Exposure Map</strong> - Spatial visualization of noise impacts</li>
    </ul>
    
    <h2>💾 Export Tab</h2>
    <p>Export simulation results:</p>
    <ul>
        <li><strong>Download Results CSV</strong> - Population time series with tick, population, births, deaths, energy</li>
    </ul>
    <div class="tip">
        <strong>Tip:</strong> For DEPONS-compatible outputs (Population.txt, PorpoiseStatistics.txt, Mortality.txt, 
        Dispersal.txt, Energy.txt), use the batch runner API programmatically via Python.
    </div>
    
    <h2>⚡ Simulation Speed</h2>
    <p>The speed slider controls how fast the simulation runs:</p>
    <ul>
        <li><strong>1%</strong> - Slowest (0.3 seconds per simulated day) - good for watching individual movements</li>
        <li><strong>50%</strong> - Medium (~0.075 seconds per day)</li>
        <li><strong>100%</strong> - Maximum speed (no delay) - for long runs</li>
    </ul>
    
    <h2>🔬 Scientific Background</h2>
    <p>CENOP is based on the DEPONS 3.0 model (Nabe-Nielsen et al., 2018). Key features:</p>
    <ul>
        <li><strong>Agent-based</strong> - Each porpoise is an individual with its own state</li>
        <li><strong>Spatially explicit</strong> - 400m × 400m grid cells</li>
        <li><strong>30-minute time steps</strong> - 48 ticks per day, 17,280 ticks per year</li>
        <li><strong>Mechanistic</strong> - Population dynamics emerge from individual behavior</li>
        <li><strong>Energy-based mortality</strong> - Survival depends on energy reserves</li>
        <li><strong>Persistent Spatial Memory</strong> - Porpoises remember good foraging areas</li>
        <li><strong>Deterrence response</strong> - Avoidance of noise above 158 dB threshold</li>
    </ul>
    
    <h3>Energy Model</h3>
    <p>The Dynamic Energy Budget model includes:</p>
    <ul>
        <li><strong>Seasonal scaling</strong> - Higher metabolism in warm months (May-Sep: ×1.3)</li>
        <li><strong>Lactation cost</strong> - Nursing females use ×1.4 more energy</li>
        <li><strong>Starvation mortality</strong> - Increases when energy drops below threshold</li>
    </ul>
    
    <div class="tip">
        <strong>Tip:</strong> For detailed model documentation, see the DEPONS 3.0 TRACE document 
        (Grimm et al., 2014) included with this software.
    </div>
    
    <h2>📚 References</h2>
    <ul>
        <li>Nabe-Nielsen J., et al. (2018). Predicting the impacts of anthropogenic disturbances on marine populations. <em>Conservation Letters</em>.</li>
        <li>Hin V., et al. (2019). A bioenergetics model for harbour porpoise. <em>Ecological Modelling</em>.</li>
        <li>Grimm V., et al. (2014). TRACE documentation standard. <em>Ecological Modelling</em>.</li>
        <li>DEPONS Project: <a href="http://www.depons.dk" target="_blank">www.depons.dk</a></li>
    </ul>
    
    <h2>📧 Contact</h2>
    <p>For questions and support, contact the AI4WIND project team.</p>
    
    <p class="text-muted small mt-4">CENOP Version 1.0 • Python Shiny Implementation • 2024-2026</p>
</div>
            """),
            style="max-height: 70vh; overflow-y: auto; padding: 20px;"
        ),
        title="CENOP Help",
        size="xl",
        easy_close=True,
        footer=ui.modal_button("Close", class_="btn-primary")
    )


def create_app_ui():
    """Create the main application UI."""
    # Create title with logo (80% larger: 42px * 1.8 = ~76px)
    title_with_logo = ui.div(
        ui.img(src="CENOP_logo.png", height="76px", style="vertical-align: middle;"),
        style="display: inline-flex; align-items: center;"
    )
    
    return ui.page_navbar(
        dashboard_tab(),
        settings_tab(),
        population_tab(),
        disturbance_tab(),
        export_tab(),
        # Add help button to the navbar
        ui.nav_spacer(),
        ui.nav_control(
            ui.input_action_link("help_btn", "❓ Help", class_="nav-link")
        ),
        sidebar=create_sidebar(),
        title=title_with_logo,
        theme=shinyswatch.theme.flatly,
        header=ui.tags.style(CUSTOM_CSS),
        fillable=True
    )


# Export the UI for use in app.py
app_ui = create_app_ui()
