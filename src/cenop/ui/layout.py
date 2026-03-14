"""
Main UI Layout for CENOP Shiny App
CENOP - CETacean Noise-Population Model
"""

from shiny import ui
import shinyswatch
from shiny_deckgl import head_includes

from .sidebar import create_sidebar
from .tabs.dashboard import dashboard_tab
from .tabs.settings import settings_tab
from .tabs.population import population_tab
from .tabs.disturbance import disturbance_tab
from .tabs.landscape_editor import landscape_editor_tab
from .tabs.export import export_tab


# Custom CSS for styling
CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════
   CENOP-JASMINE "Hybrid Depth" Marine Theme
   Light coastal shell + dark ocean data zones
   ═══════════════════════════════════════════════════ */

/* CSS Custom Properties */
:root {
    /* Light Zone — Navbar, Sidebar */
    --coastal-bg: #f0f4f8;
    --coastal-sidebar: #e8eff5;
    --coastal-text: #1a3a5c;
    --coastal-border: #d0dbe6;

    /* Dark Zone — Cards, Charts, Map */
    --ocean-bg: #0f1923;
    --ocean-header: #142330;
    --ocean-text: #c8dce8;
    --ocean-muted: #8899aa;
    --ocean-grid: rgba(255,255,255,0.08);
    --ocean-row-alt: #1a2a3a;

    /* Accent Colors — Bioluminescence */
    --accent-teal: #0d7377;
    --accent-cyan: #00b4d8;
    --accent-cyan-light: #90e0ef;
    --accent-coral: #ff6b6b;
    --accent-coral-btn: #d4544e;
    --accent-green: #48c78e;
    --accent-amber: #f0a040;
    --accent-slate: #5a6f80;
}

/* ─── Navbar ─── */
.navbar {
    min-height: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    background: var(--coastal-bg) !important;
    border-bottom: 2px solid var(--accent-teal) !important;
}
.navbar > .container-fluid { min-height: 0 !important; }
.navbar-brand {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-right: 0 !important;
}
.navbar .nav-link {
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
    color: var(--coastal-text) !important;
    font-weight: 500;
    transition: color 0.2s, border-color 0.2s;
    border-bottom: 2px solid transparent;
}
.navbar .nav-link:hover {
    color: var(--accent-teal) !important;
}
.navbar .nav-link.active,
.navbar .nav-item.active .nav-link {
    color: var(--accent-teal) !important;
    border-bottom: 2px solid var(--accent-teal);
}

/* ─── Sidebar ─── */
.sidebar {
    background-color: var(--coastal-sidebar) !important;
    border-right: 1px solid var(--coastal-border);
}
.sidebar h5 {
    color: var(--coastal-text);
    font-weight: 600;
}
.sidebar label {
    color: var(--coastal-text);
    font-weight: 500;
    font-size: 0.85rem;
}
.sidebar .text-muted {
    color: var(--accent-slate) !important;
}
.sidebar hr {
    border-color: var(--coastal-border);
    opacity: 0.6;
}

/* Sidebar tooltip icons: teal instead of Bootstrap blue */
.sidebar span[style*="color: #0d6efd"] {
    color: var(--accent-teal) !important;
}

/* Sidebar buttons */
.sidebar .btn-primary {
    background: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
    border-radius: 6px;
    font-weight: 500;
    transition: background 0.2s, transform 0.1s;
}
.sidebar .btn-primary:hover {
    background: #0b6264 !important;
    transform: translateY(-1px);
}
.sidebar .btn-danger {
    background: var(--accent-coral-btn) !important;
    border-color: var(--accent-coral-btn) !important;
    border-radius: 6px;
}
.sidebar .btn-secondary {
    background: var(--accent-slate) !important;
    border-color: var(--accent-slate) !important;
    border-radius: 6px;
}
.sidebar .btn-outline-secondary {
    color: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
    border-radius: 6px;
}
.sidebar .btn-outline-secondary:hover {
    background: var(--accent-teal) !important;
    color: white !important;
}

/* Slider: teal track */
.sidebar .irs--shiny .irs-bar {
    background: var(--accent-teal) !important;
    border-color: var(--accent-teal) !important;
}
.sidebar .irs--shiny .irs-handle {
    border-color: var(--accent-teal) !important;
}
.sidebar .irs--shiny .irs-single {
    background: var(--accent-teal) !important;
}

/* Progress bar: teal gradient */
.progress { height: 20px; border-radius: 6px; background: var(--coastal-border); }
.progress-bar {
    transition: width 0.3s ease-in-out;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-cyan)) !important;
}

/* ─── Dark Data Cards ─── */
.card.ocean-card {
    background: var(--ocean-bg) !important;
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--accent-teal);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.card.ocean-card .card-header {
    background: var(--ocean-header) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.9rem;
}
.card.ocean-card .card-body {
    color: var(--ocean-text);
}
.card.ocean-card .text-muted {
    color: var(--ocean-muted) !important;
}

/* Default card overrides (for non-ocean cards in Settings) */
.card {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-radius: 8px;
    border: 1px solid var(--coastal-border);
}
.card-header {
    font-weight: 600;
    background-color: var(--coastal-bg) !important;
    color: var(--coastal-text);
    font-size: 0.9rem;
}

/* ─── Dark Tables ─── */
.ocean-card table {
    color: var(--ocean-text) !important;
}
.ocean-card table thead th {
    background: var(--accent-teal) !important;
    color: white !important;
    border-color: rgba(255,255,255,0.1) !important;
}
.ocean-card table tbody tr:nth-child(even) {
    background: var(--ocean-row-alt) !important;
}
.ocean-card table tbody tr:nth-child(odd) {
    background: var(--ocean-bg) !important;
}
.ocean-card table td, .ocean-card table th {
    border-color: rgba(255,255,255,0.06) !important;
}

/* ─── Stat Chips (Dashboard top row) ─── */
.stat-chip {
    border-radius: 6px !important;
    font-weight: 500;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.stat-pop { background: var(--accent-teal) !important; }
.stat-year { background: var(--accent-cyan) !important; color: #000 !important; }
.stat-birth { background: var(--accent-green) !important; }
.stat-death { background: var(--accent-amber) !important; }

/* ─── Collapsible Chart Panel ─── */
.chart-toggle-bar {
    background: rgba(15, 25, 35, 0.92);
    color: var(--accent-cyan);
    text-align: center;
    padding: 6px 0;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    border-radius: 0 0 8px 8px;
    user-select: none;
    transition: background 0.2s;
}
.chart-toggle-bar:hover {
    background: rgba(15, 25, 35, 1);
}
.chart-panel {
    transition: max-height 0.4s ease, opacity 0.3s ease;
    overflow: hidden;
    max-height: 250px;
    opacity: 1;
}
.chart-panel.collapsed {
    max-height: 0;
    opacity: 0;
}

/* ─── Error Display ─── */
.shiny-output-error { color: var(--accent-coral); }
.shiny-output-error:before { content: '⚠ '; }

/* ─── Help Button ─── */
.help-btn {
    background: transparent;
    border: none;
    color: var(--accent-slate);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 5px 10px;
    margin-left: auto;
}
.help-btn:hover { color: var(--accent-teal); }

/* ─── Help Modal ─── */
.help-content h2 { color: var(--coastal-text); border-bottom: 2px solid var(--accent-teal); padding-bottom: 10px; margin-top: 25px; }
.help-content h3 { color: #34495e; margin-top: 20px; }
.help-content h4 { color: #7f8c8d; margin-top: 15px; }
.help-content ul { margin-left: 20px; }
.help-content li { margin-bottom: 5px; }
.help-content code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
.help-content .param-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
.help-content .param-table th, .help-content .param-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.help-content .param-table th { background: var(--accent-teal); color: white; }
.help-content .param-table tr:nth-child(even) { background: #f9f9f9; }
.help-content .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 15px 0; }
.help-content .tip { background: #d4edda; border-left: 4px solid var(--accent-green); padding: 10px 15px; margin: 15px 0; }

/* ─── Settings Tab ─── */
.navset-card-tab .nav-link {
    color: var(--coastal-text) !important;
}
.navset-card-tab .nav-link.active {
    color: var(--accent-teal) !important;
    border-bottom-color: var(--accent-teal) !important;
}

/* ─── Accordion (sidebar social comms) ─── */
.accordion-button {
    background: var(--coastal-sidebar) !important;
    color: var(--coastal-text) !important;
    font-size: 0.85rem;
}
.accordion-button:not(.collapsed) {
    background: var(--coastal-bg) !important;
    color: var(--accent-teal) !important;
}

/* ─── Data Frame / Tables in light zone ─── */
.shiny-data-grid {
    border-radius: 6px;
    overflow: hidden;
}

/* Value box styling */
.value-box { min-height: 100px; }
"""


def create_help_modal():
    """Create the help modal with comprehensive documentation."""
    return ui.modal(
        ui.div(
            ui.HTML("""
<div class="help-content">
    <h2>CENOP-JASMINE User Manual</h2>
    <p><strong>CENOP-JASMINE</strong> (CETacean Noise-Population Model with JASMINE Extensions) is a Python translation
    of the DEPONS 3.2 model for simulating how harbour porpoise population dynamics are affected by disturbances
    from offshore wind farm construction and ship noise.</p>
    <p>The <strong>JASMINE</strong> (Just Another Simulation Model In Nature Environments) extension adds research-grade
    features including physics-based movement, dynamic energy budgets, and learned avoidance behaviors.</p>
    <p>The simulation engine uses a <strong>Structure-of-Arrays (SoA)</strong> architecture with
    <strong>Numba JIT-compiled kernels</strong> for hot-path operations (boundary reflection, CRW movement,
    food consumption, energy costs, social vector accumulation), achieving sub-millisecond per-kernel
    performance for populations of 500+ agents.</p>

    <h2>Simulation Modes</h2>
    <table class="param-table">
        <tr><th>Mode</th><th>Description</th><th>Use Case</th></tr>
        <tr><td><strong>DEPONS</strong></td><td>Regulatory-compatible empirical models aligned with DEPONS 3.2 algorithms</td><td>Environmental impact assessments</td></tr>
        <tr><td><strong>JASMINE</strong></td><td>Physics-based movement, Dynamic Energy Budget (DEB), learned avoidance</td><td>Research and hypothesis testing</td></tr>
    </table>
    <div class="tip">
        <strong>Tip:</strong> Select simulation mode from the "Simulation Mode" dropdown in the sidebar.
    </div>

    <h2>Quick Start</h2>
    <ol>
        <li><strong>Select Simulation Mode</strong> - Choose DEPONS (Regulatory) or JASMINE (Research)</li>
        <li><strong>Set Initial Population</strong> - Enter the number of porpoises (default: 1000)</li>
        <li><strong>Set Simulation Years</strong> - How long to run (1-100 years)</li>
        <li><strong>Select Landscape</strong> - Choose the geographical area</li>
        <li><strong>Click "Load Landscape"</strong> - Load bathymetry and food distribution</li>
        <li><strong>Select Wind Turbines</strong> (optional) - Choose a turbine scenario</li>
        <li><strong>Click "Load Turbines"</strong> - Display turbines and noise contours</li>
        <li><strong>Click "Run Simulation"</strong> - Start the simulation</li>
        <li><strong>Adjust Speed</strong> - Use the slider to speed up or slow down</li>
    </ol>

    <h2>Landscapes</h2>
    <table class="param-table">
        <tr><th>Landscape</th><th>Description</th><th>Available Turbine Scenarios</th></tr>
        <tr><td>Homogeneous</td><td>Uniform test landscape (400x400 cells)</td><td>None</td></tr>
        <tr><td>Lithuania</td><td>Central Baltic / Lithuanian waters</td><td>Curonian Nord scenarios</td></tr>
        <tr><td>CentralBaltic</td><td>Central Baltic Sea (Baltic Proper, 51–55°N)</td><td>None</td></tr>
        <tr><td>Kattegat</td><td>Kattegat / Inner Danish Waters (600x1000 @ 400m)</td><td>None</td></tr>
        <tr><td>NorthSea</td><td>North Sea with real bathymetry (2088x2175 @ 400m)</td><td>Scenarios 1-3</td></tr>
    </table>

    <h2>Landscape Layers</h2>
    <p>The <strong>Landscape</strong> tab provides a spatial viewer for all environmental data layers loaded from
    ASCII grid files (<code>.asc</code>). Each cell in the grid represents a 400m &times; 400m area.
    This cell size is <strong>mandatory</strong> &mdash; DEPONS requires all landscape grids to use exactly 400&nbsp;m
    resolution (<code>REQUIRED_CELL_SIZE = 400</code> in DEPONS). The <code>cellsize</code> header in each
    ASC file is validated at load time; grids with a different resolution will be rejected.
    Movement distances, coordinate conversions, and the persistent spatial memory system all assume
    400&nbsp;m cells.</p>
    <p>Layers can be inspected individually with summary statistics (min, max, mean, coverage).
    Each layer is loaded from files in the landscape data directory (e.g., <code>data/Kattegat/</code>).
    Some layers actively drive the simulation; others are loaded for visualisation and analysis only.</p>

    <table class="param-table">
        <tr><th>Layer</th><th>File(s)</th><th>Role in Simulation</th><th>Time&nbsp;Varying</th></tr>
        <tr><td>Bathymetry</td><td><code>bathy.asc</code></td><td>Active &mdash; movement, land masking</td><td>No</td></tr>
        <tr><td>Salinity</td><td><code>salinity01.asc</code>&ndash;<code>salinity12.asc</code></td><td>Active &mdash; movement</td><td>Monthly</td></tr>
        <tr><td>Food Probability</td><td><code>patches.asc</code></td><td>Active &mdash; food system</td><td>No</td></tr>
        <tr><td>Prey (MaxEnt)</td><td><code>prey01.asc</code>&ndash;<code>prey12.asc</code></td><td>Active &mdash; food carrying capacity</td><td>Monthly</td></tr>
        <tr><td>Distance to Coast</td><td><code>disttocoast.asc</code></td><td>Visualisation only</td><td>No</td></tr>
        <tr><td>Sediment Type</td><td><code>sediment.asc</code></td><td>Visualisation only *</td><td>No</td></tr>
        <tr><td>Blocks</td><td><code>blocks.asc</code></td><td>Visualisation only</td><td>No</td></tr>
    </table>
    <p class="small text-muted">* In DEPONS, sediment feeds the Weston flux ship-noise propagation model. CENOP currently uses a simpler &alpha;/&beta; spreading-loss model that does not require sediment data.</p>

    <h3>Bathymetry (Depth) &mdash; <span style="color: var(--accent-green);">Active</span></h3>
    <p>Water depth in metres below sea level, sourced from EMODnet or equivalent hydrographic surveys.
    This is one of the most important layers &mdash; it directly drives three simulation mechanisms:</p>
    <ul>
        <li><strong>Land masking</strong> &mdash; cells with depth &lt; <code>min_depth</code> (default 1&nbsp;m) are treated as land.
            Porpoises cannot enter them; the model tries turning at 40&deg;, 70&deg;, and 120&deg; in both
            directions, picking the deeper option. If no angle succeeds, it backtracks to the previous
            position or moves to the deepest neighbouring cell.</li>
        <li><strong>CRW movement</strong> &mdash; depth modulates both step length and turning angle every tick through the
            coefficients <code>a1</code> (step&ndash;depth) and <code>b1</code> (angle&ndash;depth). Porpoises take
            shorter, more tortuous steps in shallow water.</li>
        <li><strong>Dispersal depth gate</strong> &mdash; during PSM dispersal, a stricter minimum depth
            (<code>min_depth_dispersal</code>, default 4&nbsp;m) applies, keeping dispersing porpoises in deeper waters.</li>
    </ul>
    <p><strong>File:</strong> <code>bathy.asc</code> &nbsp;|&nbsp; <strong>Units:</strong> metres below sea level &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> viridis (yellow&nbsp;=&nbsp;shallow, purple&nbsp;=&nbsp;deep) &nbsp;|&nbsp;
    <strong>NODATA:</strong> <code>-9999</code> (land)</p>

    <h3>Salinity &mdash; <span style="color: var(--accent-green);">Active</span></h3>
    <p>Monthly sea-surface salinity fields (12 layers, one per month), typically derived from
    oceanographic models or satellite observations. Salinity varies seasonally due to river
    discharge, precipitation, and ocean circulation.</p>
    <p><strong>Use in simulation:</strong> salinity modulates CRW movement every tick through
    coefficients <code>a2</code> (step&ndash;salinity) and <code>b2</code> (angle&ndash;salinity).
    In the Kattegat calibration (<code>b2&nbsp;=&nbsp;0.93</code>), high salinity strongly increases
    turning angle variability. Salinity gradients (e.g., the Baltic&ndash;North Sea transition)
    act as natural habitat boundaries that porpoises tend not to cross.</p>
    <p><strong>Files:</strong> <code>salinity01.asc</code> through <code>salinity12.asc</code>
    (one per calendar month) &nbsp;|&nbsp; <strong>Units:</strong> PSU (Practical Salinity Units) &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> blue gradient &nbsp;|&nbsp;
    <strong>Monthly:</strong> use the month slider to view seasonal variation</p>

    <h3>Food Probability &mdash; <span style="color: var(--accent-green);">Active</span></h3>
    <p>A static spatial layer defining <em>where</em> food can exist and the <em>carrying capacity</em>
    of each cell. Cells with value &gt;&nbsp;0 are food patches; cells with value&nbsp;0 are permanently
    barren.</p>
    <p><strong>Use in simulation:</strong></p>
    <ul>
        <li><strong>Food initialisation</strong> &mdash; at simulation start, if MaxEnt data is available,
            each cell's food level is set to <code>maxU &times; maxEnt / meanMaxEnt</code>;
            otherwise it falls back to the food probability value.</li>
        <li><strong>Foraging</strong> &mdash; porpoises consume food from their current cell each tick, reducing
            the local food level. The amount eaten depends on the porpoise's hunger (energy deficit).</li>
        <li><strong>Replenishment</strong> &mdash; depleted cells regenerate food once per day using logistic
            growth: <code>F += rU &times; F &times; (1 &minus; F/K)</code>, where
            <code>rU</code> (default 0.1) is the per-step growth rate and <code>K</code> is the carrying
            capacity derived from MaxEnt (or food probability as fallback). If the first iteration's delta
            exceeds a threshold, 47 more iterations are applied (48 total per day, matching DEPONS&nbsp;3.2).</li>
    </ul>
    <p><strong>File:</strong> <code>patches.asc</code> &nbsp;|&nbsp; <strong>Units:</strong> probability / relative capacity (0&ndash;1) &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> green gradient &nbsp;|&nbsp;
    <strong>NODATA:</strong> <code>-9999</code> (land)</p>

    <h3>Prey (MaxEnt) &mdash; <span style="color: var(--accent-green);">Active</span></h3>
    <p>Monthly predictions of relative prey density from <strong>Maximum Entropy</strong> (MaxEnt) species
    distribution models. These are generated externally using satellite tracking data combined with
    environmental covariates (depth, distance to coast, sediment type, sea surface temperature,
    chlorophyll concentration).</p>
    <p><strong>Use in simulation:</strong> MaxEnt values set the monthly carrying capacity of each food
    patch, matching DEPONS&nbsp;3.2 behaviour. Food grows logistically towards
    <code>K = maxU &times; maxEnt / meanMaxEntInQuarter</code>, so cells with high MaxEnt hold more food.
    The 12 monthly layers shift prey distribution spatially across the year, capturing seasonal
    productivity cycles. At initialisation, MaxEnt also determines each cell's starting food level.
    When MaxEnt data is not available for a landscape, the food probability layer is used as fallback
    carrying capacity.</p>
    <p><strong>Files:</strong> <code>prey01.asc</code> through <code>prey12.asc</code>
    (one per calendar month; also accepts DEPONS long-form naming: <code>prey0000_01.asc</code>) &nbsp;|&nbsp;
    <strong>Units:</strong> relative habitat suitability (0&ndash;1) &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> green gradient &nbsp;|&nbsp;
    <strong>Monthly:</strong> use the month slider to compare seasonal prey distribution</p>

    <h3>Distance to Coast &mdash; <span style="color: var(--accent-amber);">Visualisation only</span></h3>
    <p>Euclidean distance from each cell to the nearest coastline.</p>
    <p><strong>Role in DEPONS:</strong> distance to coast is used as an environmental covariate in the
    external MaxEnt prey distribution models that generate the Prey layers. It is not directly
    referenced by the DEPONS movement or foraging algorithms.</p>
    <p><strong>Current status in CENOP:</strong> loaded and displayed in the Landscape viewer for
    spatial context and habitat characterisation (e.g., comparing porpoise density across distance
    bands). Not referenced by any simulation logic.</p>
    <p><strong>File:</strong> <code>disttocoast.asc</code> &nbsp;|&nbsp; <strong>Units:</strong> kilometres &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> yellow&ndash;red gradient &nbsp;|&nbsp;
    <strong>NODATA:</strong> <code>-9999</code> (land)</p>

    <h3>Sediment Type &mdash; <span style="color: var(--accent-amber);">Visualisation only</span></h3>
    <p>Seabed grain size on the phi (&phi;) scale &mdash; a logarithmic classification of sediment particle
    diameter. Negative values indicate coarse material (rock, gravel), values near zero are sand,
    and positive values are silt or clay.</p>
    <p><strong>Role in DEPONS:</strong> sediment feeds into the <em>Weston flux</em> acoustic propagation model
    (<code>WestonFlux.java</code>) that calculates transmission loss for ship noise. The grain size
    determines sound speed ratio, density ratio, and attenuation coefficient of the seabed, which
    together control how far ship noise propagates through the water column. Sediment is also a key
    covariate in the external MaxEnt prey models (sandeels prefer coarse sand substrates).</p>
    <p><strong>Current status in CENOP:</strong> loaded and displayed in the Landscape viewer.
    CENOP uses a simpler &alpha;/&beta; spreading-loss formula for sound propagation that does not
    require sediment data. Not referenced by any simulation logic.</p>
    <p><strong>File:</strong> <code>sediment.asc</code> &nbsp;|&nbsp; <strong>Units:</strong> phi (&phi;) scale &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> categorical &nbsp;|&nbsp;
    <strong>NODATA:</strong> <code>-9999</code> (land)</p>

    <h3>Blocks &mdash; <span style="color: var(--accent-amber);">Visualisation only</span></h3>
    <p>A spatial classification layer that divides the landscape into numbered reporting regions
    (e.g., ICES statistical rectangles or management areas). Each cell is assigned an integer
    block ID.</p>
    <p><strong>Role in DEPONS:</strong> blocks serve as spatial output regions. The <code>Block</code> agent
    in DEPONS counts how many porpoises occupy each block at each time step, enabling density maps
    and comparisons with field survey data by region.</p>
    <p><strong>Current status in CENOP:</strong> loaded and displayed in the Landscape viewer for
    spatial reference. Block-based porpoise counting is not yet implemented in the CENOP output
    pipeline. Not referenced by any simulation logic.</p>
    <p><strong>File:</strong> <code>blocks.asc</code> &nbsp;|&nbsp; <strong>Units:</strong> integer region IDs &nbsp;|&nbsp;
    <strong>Colour scheme:</strong> categorical &nbsp;|&nbsp;
    <strong>NODATA:</strong> <code>-9999</code> (land/unclassified)</p>

    <div class="tip">
        <strong>Tip:</strong> In the Landscape tab, select a layer from the dropdown and click "Load Layer"
        to visualise it on the map. For monthly layers (Salinity, Prey), use the month slider to compare
        seasonal patterns. Hover over cells for tooltip values.
    </div>

    <h2>Wind Turbine Scenarios</h2>
    <p>Turbine scenarios define the location and construction timing of offshore wind farms.
    Each turbine generates pile-driving noise during construction that deters porpoises.</p>
    <div class="note">
        <strong>Note:</strong> The noise overlay (red shading) shows areas where received sound levels
        exceed the deterrence threshold (152 dB). Porpoises avoid these areas during pile-driving.
    </div>

    <h2>Dashboard Visualizations</h2>
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

    <h2>Model Settings</h2>
    <p>All parameters have tooltip icons - hover for detailed descriptions.</p>

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
        <tr><td>b0</td><td>-0.024</td><td>Turning angle autocorrelation</td></tr>
    </table>

    <h3>Dispersal Tab</h3>
    <p>Controls large-scale movement when porpoises have declining energy:</p>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Dispersal Type</td><td>PSM-Type2</td><td>Memory-based with heading dampening</td></tr>
        <tr><td>tDisp</td><td>3 days</td><td>Consecutive days of declining energy to trigger dispersal</td></tr>
        <tr><td>PSM_dist</td><td>N(300;100)</td><td>Preferred dispersal distance: mean 300km, std 100km</td></tr>
    </table>

    <h3>Energy Tab</h3>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>rS (Satiation)</td><td>0.03</td><td>Decay rate for satiation memory</td></tr>
        <tr><td>rR (Reference)</td><td>0.03</td><td>Decay rate for reference memory</td></tr>
        <tr><td>rU (Replenishment)</td><td>0.1</td><td>Rate at which depleted food patches recover</td></tr>
    </table>

    <h2>JASMINE Mode Features</h2>
    <p>When JASMINE mode is selected, the following advanced features are enabled:</p>

    <h3>Behavioral State Machine</h3>
    <p>Five behavioral states with configurable transitions:</p>
    <table class="param-table">
        <tr><th>State</th><th>Description</th><th>Movement</th></tr>
        <tr><td>FORAGING</td><td>Searching for/consuming food</td><td>DEPONS CRW</td></tr>
        <tr><td>TRAVELING</td><td>Directed movement between areas</td><td>Physics-based</td></tr>
        <tr><td>RESTING</td><td>Low activity energy recovery</td><td>Physics-based</td></tr>
        <tr><td>DISPERSING</td><td>Memory-driven dispersal to new areas</td><td>PSM-based</td></tr>
        <tr><td>DISTURBED</td><td>Response to disturbance events</td><td>Avoidance</td></tr>
    </table>

    <h3>Dynamic Energy Budget (DEB)</h3>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Body Mass</td><td>50.0 kg</td><td>Adult porpoise body mass</td></tr>
        <tr><td>BMR Scale</td><td>1.0</td><td>Basal metabolic rate multiplier</td></tr>
        <tr><td>Activity Cost</td><td>2.0</td><td>Activity cost multiplier</td></tr>
        <tr><td>Thermal Model</td><td>On</td><td>Temperature-dependent metabolism</td></tr>
        <tr><td>Disturbance Cost</td><td>1.5</td><td>Energy cost during disturbance</td></tr>
    </table>

    <h3>Disturbance Memory</h3>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Memory Decay Rate</td><td>0.001</td><td>Per-tick memory decay</td></tr>
        <tr><td>Avoidance Radius</td><td>20 cells</td><td>Influence radius for avoidance</td></tr>
        <tr><td>Habituation</td><td>On</td><td>Enable habituation to repeated disturbance</td></tr>
        <tr><td>Habituation Rate</td><td>0.05</td><td>Rate of habituation per exposure</td></tr>
    </table>

    <h3>Physics-Based Movement</h3>
    <table class="param-table">
        <tr><th>Parameter</th><th>Default</th><th>Description</th></tr>
        <tr><td>Drag Coefficient</td><td>0.01</td><td>Hydrodynamic drag</td></tr>
        <tr><td>Max Thrust</td><td>100.0 N</td><td>Maximum propulsive thrust</td></tr>
        <tr><td>Current Weight</td><td>0.5</td><td>Ocean current influence (0-1)</td></tr>
    </table>

    <h2>Population Tab</h2>
    <ul>
        <li><strong>Age Distribution</strong> - Histogram of porpoise ages (0-30 years)</li>
        <li><strong>Energy Distribution</strong> - Histogram of energy levels</li>
        <li><strong>Landscape Energy</strong> - Total food availability over time</li>
        <li><strong>Average Movement</strong> - Daily movement distances</li>
        <li><strong>Vital Statistics</strong> - Summary table of population metrics</li>
    </ul>

    <h2>Disturbance Tab</h2>
    <ul>
        <li><strong>Dispersal Plot</strong> - Number of porpoises in dispersal states over time</li>
        <li><strong>Deterrence Events</strong> - Count of currently deterred porpoises</li>
        <li><strong>Noise Exposure Map</strong> - Spatial visualization of noise impacts</li>
    </ul>

    <h2>Export Tab</h2>
    <ul>
        <li><strong>Download Results CSV</strong> - Population time series with tick, population, births, deaths, energy</li>
    </ul>

    <h2>Simulation Speed</h2>
    <ul>
        <li><strong>1%</strong> - Slowest (0.3 seconds per simulated day) - good for watching individual movements</li>
        <li><strong>50%</strong> - Medium (~0.075 seconds per day)</li>
        <li><strong>100%</strong> - Maximum speed (no delay) - for long runs</li>
    </ul>

    <h2>Scientific Background</h2>
    <p>CENOP-JASMINE is based on the DEPONS 3.2 model (Nabe-Nielsen et al., 2018) with JASMINE extensions. Key features:</p>
    <ul>
        <li><strong>Agent-based</strong> - Each porpoise is an individual with its own state</li>
        <li><strong>Spatially explicit</strong> - 400m x 400m grid cells</li>
        <li><strong>30-minute time steps</strong> - 48 ticks per day, 17,280 ticks per year</li>
        <li><strong>Dual-mode</strong> - DEPONS for regulatory, JASMINE for research</li>
        <li><strong>Energy-based mortality</strong> - Survival depends on energy reserves</li>
        <li><strong>Persistent Spatial Memory</strong> - Porpoises remember good foraging areas</li>
        <li><strong>Reference Memory</strong> - Vectorized attraction to previously visited food-rich areas</li>
        <li><strong>Learned Avoidance</strong> (JASMINE) - Porpoises remember disturbance zones</li>
        <li><strong>Habituation</strong> (JASMINE) - Reduced response to repeated exposure</li>
    </ul>

    <h2>Performance</h2>
    <p>CENOP uses Numba JIT-compiled kernels for simulation hot paths, achieving sub-millisecond
    performance per kernel call for populations of 500 agents. Three independent kernels
    (boundary reflection, turn position, BMR cost) run in parallel using <code>prange</code>.</p>

    <h2>Model Validation</h2>
    <ul>
        <li><strong>DEPONS mode</strong> - Algorithmically aligned with DEPONS 3.2 (pregnancy FSM, dispersal, deterrence, reference memory, CRW rejection sampling)</li>
        <li><strong>JASMINE mode</strong> - Research-grade, designed for exploring advanced behavioral hypotheses</li>
    </ul>

    <h2>References</h2>
    <ul>
        <li>Nabe-Nielsen J., Sibly R.M., Tougaard J., Teilmann J., Sveegaard S. (2014). Effects of noise and by-catch on a Danish harbour porpoise population. <em>Ecological Modelling</em>, 272, 242–251. <a href="https://doi.org/10.1016/j.ecolmodel.2013.09.025" target="_blank">doi:10.1016/j.ecolmodel.2013.09.025</a></li>
        <li>Nabe-Nielsen J., van Beest F.M., Grimm V., Sibly R.M., Teilmann J., Thompson P.M. (2018). Predicting the impacts of anthropogenic disturbances on marine populations. <em>Conservation Letters</em>, 11(5), e12563. <a href="https://doi.org/10.1111/conl.12563" target="_blank">doi:10.1111/conl.12563</a></li>
        <li>Nabe-Nielsen J., Harwood J. (2016). Comparison of the iPCoD and DEPONS models for modelling population consequences of noise on harbour porpoises. <em>Scientific Report from DCE</em>, No. 186.</li>
        <li>van Beest F.M., Nabe-Nielsen J., Carstensen J., Teilmann J., Sveegaard S. (2015). Disturbance Effects on the Harbour Porpoise Population in the North Sea (DEPONS): Status report on model development. <em>Scientific Report from DCE</em>, No. 140.</li>
        <li>Hin V., Harwood J., de Roos A.M. (2019). Bio-energetic modeling of medium-sized cetaceans shows that physiological structure is key to determining the cumulative effects of disturbance. <em>Ecological Modelling</em>, 394, 82–93. <a href="https://doi.org/10.1016/j.ecolmodel.2018.12.019" target="_blank">doi:10.1016/j.ecolmodel.2018.12.019</a></li>
        <li>Tougaard J., Wright A.J., Madsen P.T. (2015). Cetacean noise criteria revisited in the light of proposed exposure limits for harbour porpoises. <em>Marine Pollution Bulletin</em>, 90(1–2), 196–208. <a href="https://doi.org/10.1016/j.marpolbul.2014.10.051" target="_blank">doi:10.1016/j.marpolbul.2014.10.051</a></li>
        <li>Kooijman S.A.L.M. (2010). <em>Dynamic Energy Budget theory for metabolic organisation</em>. 3rd ed. Cambridge University Press.</li>
        <li>Grimm V., Railsback S.F. (2005). <em>Individual-based Modeling and Ecology</em>. Princeton University Press.</li>
        <li>DEPONS Project: <a href="http://www.depons.dk" target="_blank">www.depons.dk</a></li>
    </ul>

    <h2>Contact</h2>
    <p>For questions and support, contact the arturas.razinkovas-baziukas@ku.lt.</p>

    <p class="text-muted small mt-4">CENOP-JASMINE Version 2.0 | Python Shiny Implementation | 2024-2026</p>
</div>
            """),
            style="max-height: 70vh; overflow-y: auto; padding: 20px;"
        ),
        title="CENOP-JASMINE Help",
        size="xl",
        easy_close=True,
        footer=ui.modal_button("Close", class_="btn-primary")
    )


def create_app_ui():
    """Create the main application UI."""
    # Create title with logo
    title_with_logo = ui.div(
        ui.img(src="CENOP_logo.png", height="35px", style="vertical-align: middle;"),
        style="display: inline-flex; align-items: center; margin-right: 1rem;"
    )
    
    # Dynamic legend: server sends entries via cenop_legend_update,
    # JS renders checkboxes and toggles layer visibility client-side.
    LEGEND_CSS = """
    .cenop-legend {
        position: absolute; bottom: 10px; right: 10px; z-index: 2;
        background: rgba(255,255,255,0.95); border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18); font-family: sans-serif;
        font-size: 12px; min-width: 150px; pointer-events: auto;
        overflow: hidden;
    }
    .cenop-legend-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 10px; cursor: pointer; user-select: none;
        font-weight: 600; font-size: 12px; color: #333;
        border: none; background: none; width: 100%; text-align: left;
    }
    .cenop-legend-header:hover { background: rgba(0,0,0,0.04); }
    .cenop-legend-body { padding: 2px 0; }
    .cenop-legend-body.collapsed { display: none; }
    .cenop-legend-row {
        display: flex; align-items: center; gap: 6px;
        padding: 3px 10px; cursor: pointer;
    }
    .cenop-legend-row:hover { background: rgba(0,0,0,0.04); }
    .cenop-legend-cb { margin: 0; cursor: pointer; }
    .cenop-legend-swatch {
        width: 14px; height: 14px; flex-shrink: 0; border-radius: 2px;
    }
    .cenop-legend-swatch.circle { border-radius: 50%; }
    .cenop-legend-label { color: #333; white-space: nowrap; }
    """

    LEGEND_JS = """
    (function() {
        var COMPANIONS = {
            'depth-bitmap': ['depth-tooltip'],
            'foraging-bitmap': ['foraging-tooltip'],
            'turbine-poles': ['turbine-blades']
        };
        var userOverrides = {};

        function findMapInstance() {
            var instances = window.__deckgl_instances || {};
            for (var id in instances) return {id: id, inst: instances[id]};
            return null;
        }

        function applyLayerVisibility(mapId, inst) {
            if (!inst || !inst.overlay) return;
            var visMap = {};
            (inst.lastLayers || []).forEach(function(lp) {
                visMap[lp.id] = lp.visible !== false;
            });
            var deckLayers = inst.overlay.props.layers || [];
            var newLayers = deckLayers.map(function(layer) {
                if (layer.id in visMap) {
                    return layer.clone({visible: visMap[layer.id]});
                }
                return layer;
            });
            inst.overlay.setProps({layers: newLayers});
            inst.map.triggerRepaint();
        }

        function setLayerVisible(inst, layerId, visible) {
            if (!inst || !inst.lastLayers) return;
            inst.lastLayers = inst.lastLayers.map(function(lp) {
                if (lp.id !== layerId) return lp;
                return Object.assign({}, lp, {visible: visible});
            });
        }

        function toggleBasemap(inst, visible) {
            if (!inst || !inst.map) return;
            var style = inst.map.getStyle();
            if (!style || !style.layers) return;
            style.layers.forEach(function(layer) {
                inst.map.setLayoutProperty(
                    layer.id, 'visibility', visible ? 'visible' : 'none'
                );
            });
        }

        function renderLegend(container, entries) {
            container.textContent = '';

            var header = document.createElement('button');
            header.className = 'cenop-legend-header';
            var titleSpan = document.createElement('span');
            titleSpan.textContent = 'Layers';
            var arrowSpan = document.createElement('span');
            arrowSpan.className = 'cenop-legend-arrow';
            arrowSpan.textContent = '\\u25BC';
            header.appendChild(titleSpan);
            header.appendChild(arrowSpan);

            var body = document.createElement('div');
            body.className = 'cenop-legend-body';

            header.addEventListener('click', function() {
                body.classList.toggle('collapsed');
                arrowSpan.textContent = body.classList.contains('collapsed')
                    ? '\\u25B6' : '\\u25BC';
            });
            container.appendChild(header);

            entries.forEach(function(entry) {
                var row = document.createElement('label');
                row.className = 'cenop-legend-row';

                var cb = document.createElement('input');
                cb.type = 'checkbox';
                cb.className = 'cenop-legend-cb';
                var checked = (entry.id in userOverrides)
                    ? userOverrides[entry.id] : entry.checked !== false;
                cb.checked = checked;

                cb.addEventListener('change', function() {
                    var vis = this.checked;
                    userOverrides[entry.id] = vis;
                    var m = findMapInstance();
                    if (!m) return;
                    if (entry.id === 'basemap') {
                        toggleBasemap(m.inst, vis);
                        return;
                    }
                    setLayerVisible(m.inst, entry.id, vis);
                    (COMPANIONS[entry.id] || []).forEach(function(cid) {
                        setLayerVisible(m.inst, cid, vis);
                    });
                    applyLayerVisibility(m.id, m.inst);
                });

                var swatch = document.createElement('span');
                swatch.className = 'cenop-legend-swatch'
                    + (entry.shape === 'circle' ? ' circle' : '');
                if (entry.colors && entry.colors.length) {
                    var stops = entry.colors.map(function(c) {
                        return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
                    });
                    swatch.style.background =
                        'linear-gradient(90deg,' + stops.join(',') + ')';
                } else if (entry.color) {
                    swatch.style.backgroundColor = entry.color;
                }

                var lbl = document.createElement('span');
                lbl.className = 'cenop-legend-label';
                lbl.textContent = entry.label;

                row.appendChild(cb);
                row.appendChild(swatch);
                row.appendChild(lbl);
                body.appendChild(row);
            });
            container.appendChild(body);

            // Re-apply user overrides after server update
            var m = findMapInstance();
            if (m) {
                var needRebuild = false;
                entries.forEach(function(entry) {
                    if (entry.id in userOverrides) {
                        var vis = userOverrides[entry.id];
                        if (entry.id === 'basemap') {
                            toggleBasemap(m.inst, vis);
                        } else {
                            setLayerVisible(m.inst, entry.id, vis);
                            (COMPANIONS[entry.id] || []).forEach(function(cid) {
                                setLayerVisible(m.inst, cid, vis);
                            });
                            needRebuild = true;
                        }
                    }
                });
                if (needRebuild) applyLayerVisibility(m.id, m.inst);
            }
        }

        Shiny.addCustomMessageHandler('cenop_legend_update', function(payload) {
            var entries = payload.entries || [];
            var mapEl = document.getElementById('sim_map');
            if (!mapEl) return;
            var container = mapEl.querySelector('.cenop-legend');
            if (!container) {
                container = document.createElement('div');
                container.className = 'cenop-legend';
                mapEl.appendChild(container);
            }
            renderLegend(container, entries);
        });
    })();
    """

    return ui.page_navbar(
        head_includes(),
        dashboard_tab(),
        settings_tab(),
        population_tab(),
        disturbance_tab(),
        landscape_editor_tab(),
        export_tab(),
        # Add help button to the navbar
        ui.nav_spacer(),
        ui.nav_control(
            ui.input_action_link("help_btn", "❓ Help", class_="nav-link")
        ),
        sidebar=create_sidebar(),
        title=title_with_logo,
        theme=shinyswatch.theme.flatly,
        header=ui.TagList(
            ui.tags.style(CUSTOM_CSS),
            ui.tags.style(LEGEND_CSS),
            ui.tags.script(LEGEND_JS),
        ),
        fillable=True
    )


# Export the UI for use in app.py
app_ui = create_app_ui()
