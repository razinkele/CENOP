"""
Export Tab UI
"""

from shiny import ui


def export_tab():
    """Create the Export and About tab."""
    return ui.nav_panel(
        "Export",
        ui.layout_columns(
            ui.card(
                ui.card_header("📊 Export Simulation Data"),
                ui.p("Download simulation results in CSV format."),
                ui.download_button("download_data", "📥 Download Results CSV", class_="btn-success btn-lg mb-3"),
                ui.tags.hr(),
                ui.h6("📁 Exported Data Includes:"),
                ui.tags.ul(
                    ui.tags.li("Tick count and simulation time"),
                    ui.tags.li("Population size over time"),
                    ui.tags.li("Birth and death counts"),
                    ui.tags.li("Energy levels"),
                    ui.tags.li("Year and day markers"),
                ),
                ui.div(
                    ui.p("For DEPONS-compatible outputs (Population.txt, PorpoiseStatistics.txt, etc.), "
                         "use the Python API:", class_="text-muted small"),
                    ui.tags.pre(
                        "from cenop.core.output_writer import OutputWriter\n"
                        "writer = OutputWriter(config)\n"
                        "writer.record_tick(sim)",
                        class_="bg-light p-2 small"
                    ),
                    class_="mt-3"
                ),
            ),
            ui.card(
                ui.card_header("ℹ️ About CENOP"),
                ui.div(
                    ui.img(src="CENOP_logo.png", height="60px", style="margin-bottom: 10px;"),
                    style="text-align: center;"
                ),
                ui.h5("CENOP - CETacean Noise-Population Model", class_="text-center"),
                ui.p("Python Shiny implementation of DEPONS 3.0", class_="text-center text-muted"),
                ui.tags.hr(),
                ui.h6("🔬 Key Features:"),
                ui.tags.ul(
                    ui.tags.li("Agent-based simulation of harbour porpoises"),
                    ui.tags.li("Vectorized NumPy operations for performance"),
                    ui.tags.li("Wind turbine pile-driving noise deterrence"),
                    ui.tags.li("Ship traffic disturbance"),
                    ui.tags.li("Persistent Spatial Memory (PSM) for dispersal"),
                    ui.tags.li("Dynamic Energy Budget model"),
                    ui.tags.li("DEPONS-compatible data outputs"),
                ),
                ui.tags.hr(),
                ui.p("Version 1.0", class_="small text-muted text-center"),
                ui.p("AI4WIND Project • 2024-2026", class_="small text-muted text-center"),
            ),
            col_widths=[6, 6]
        )
    )
