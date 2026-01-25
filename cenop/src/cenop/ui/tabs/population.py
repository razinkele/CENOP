"""
Population Tab UI
"""

from shiny import ui


def population_tab():
    """Create the Population analytics tab."""
    return ui.nav_panel(
        "Population",
        ui.layout_columns(
            ui.card(
                ui.card_header("Porpoise Age Distribution (0-30 years)"),
                ui.output_ui("age_histogram"),
                height="320px"
            ),
            ui.card(
                ui.card_header("Energy Level Distribution (0-20)"),
                ui.output_ui("energy_histogram"),
                height="320px"
            ),
            col_widths=[6, 6]
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Landscape Energy Level"),
                ui.output_ui("landscape_energy_plot"),
                height="320px"
            ),
            ui.card(
                ui.card_header("Average Porpoise Movement"),
                ui.output_ui("movement_plot"),
                height="320px"
            ),
            col_widths=[6, 6]
        ),
        ui.card(
            ui.card_header("Vital Statistics"),
            ui.output_data_frame("vital_stats_table"),
            height="250px"
        )
    )
