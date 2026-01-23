"""
Disturbance Tab UI
"""

from shiny import ui


def disturbance_tab():
    """Create the Disturbance monitoring tab."""
    return ui.nav_panel(
        "Disturbance",
        ui.layout_columns(
            ui.card(
                ui.card_header("Porpoise Dispersal"),
                ui.output_ui("dispersal_plot"),
                height="350px"
            ),
            ui.card(
                ui.card_header("Deterrence Events"),
                ui.output_ui("deterrence_plot"),
                height="350px"
            ),
            col_widths=[6, 6]
        ),
        ui.card(
            ui.card_header("Noise Exposure Map"),
            ui.output_ui("noise_map"),
            height="350px"
        )
    )
