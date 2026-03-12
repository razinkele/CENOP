"""
About Tab UI
"""

from shiny import ui


def about_tab():
    """Create the About tab."""
    return ui.nav_panel(
        "About",
        ui.layout_columns(
            ui.card(
                ui.card_header("ℹ️ About CENOP & JASMINE"),
                ui.div(
                    ui.img(src="CENOP_logo.png", height="120px", style="margin-bottom: 20px;"),
                    style="text-align: center;"
                ),
                ui.h3("CENOP - CETacean Noise-Population Model", class_="text-center"),
                ui.h5("Integrated with JASMINE Physics Engine", class_="text-center text-primary"),
                ui.p("A high-performance Python simulation platform for marine mammal population dynamics.", 
                     class_="text-center lead text-muted"),
                
                ui.tags.hr(),
                
                ui.div(
                    ui.h5("🔬 Simulation Frameworks"),
                    ui.layout_columns(
                        ui.div(
                            ui.h6("DEPONS Mode (Standard)"),
                             ui.p("Regulatory-grade simulation using empirically calibrated Correlated Random Walk (CRW) behavior."),
                             ui.p("Designed for Environmental Impact Assessments (EIA) aligned with DEPONS 3.2 algorithms."),
                             class_="p-3 border rounded bg-light"
                        ),
                        ui.div(
                             ui.h6("JASMINE Mode (Physics)"),
                             ui.p("Next-generation movement model using symplectic integration and environmental advection forces."),
                             ui.p("Enables fine-scale physics-based movement and variable time-stepping."),
                             class_="p-3 border rounded bg-light"
                        ),
                        col_widths=[6, 6]
                    ),
                    class_="mb-4"
                ),

                ui.h5("⚡ Technical Architecture"),
                ui.tags.ul(
                    ui.tags.li(ui.strong("Structure-of-Arrays (SoA):"), " Vectorized NumPy implementation optimized for simulating 10,000+ agents."),
                    ui.tags.li(ui.strong("Numba JIT Kernels:"), " Hot-path operations (CRW movement, boundary reflection, food consumption, energy costs, social vectors) compiled to machine code with prange parallelism."),
                    ui.tags.li(ui.strong("Flexible Timestepping:"), " Supports standard 30-min steps or sub-second physics-driven updates."),
                    ui.tags.li(ui.strong("Noise Modeling:"), " Comprehensive integration of wind turbine pile-driving and ship noise layers."),
                    ui.tags.li(ui.strong("Energetics:"), " Full Dynamic Energy Budget (DEB) model tracking foraging, lactation, and survival."),
                ),

                ui.tags.hr(),
                ui.p("Version 2.0 (JASMINE Integration)", class_="small text-muted text-center"),
                ui.p("Algorithmically aligned with DEPONS 3.2", class_="small text-muted text-center"),
                ui.p("2024-2026", class_="small text-muted text-center"),
            ),
            col_widths=[8]
        )
    )
