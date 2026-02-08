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
            col_widths=[12]
        )
    )
