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
                ui.card_header("Export Simulation Data"),
                ui.p("Download simulation results in CSV format.", style="color: var(--ocean-text, #c8dce8);"),
                ui.download_button("download_data", "Download Results CSV", class_="btn-lg mb-3",
                                   style="background: var(--accent-teal, #0d7377); border-color: var(--accent-teal, #0d7377); color: white; border-radius: 6px;"),
                ui.tags.hr(style="border-color: rgba(255,255,255,0.1);"),
                ui.h6("Exported Data Includes:", style="color: var(--accent-cyan, #00b4d8);"),
                ui.tags.ul(
                    ui.tags.li("Tick count and simulation time"),
                    ui.tags.li("Population size over time"),
                    ui.tags.li("Birth and death counts"),
                    ui.tags.li("Energy levels"),
                    ui.tags.li("Year and day markers"),
                    style="color: var(--ocean-text, #c8dce8);"
                ),
                ui.div(
                    ui.p("For DEPONS-compatible outputs (Population.txt, PorpoiseStatistics.txt, etc.), "
                         "use the Python API:", style="color: var(--ocean-muted, #8899aa); font-size: 0.85rem;"),
                    ui.tags.pre(
                        "from cenop.core.output_writer import OutputWriter\n"
                        "writer = OutputWriter(config)\n"
                        "writer.record_tick(sim)",
                        style="background: var(--ocean-row-alt, #1a2a3a); color: var(--accent-cyan, #00b4d8); padding: 10px; border-radius: 6px; font-size: 0.85rem;"
                    ),
                    class_="mt-3"
                ),
                class_="ocean-card"
            ),
            col_widths=[12]
        )
    )
