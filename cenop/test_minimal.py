"""Minimal test app to verify Shiny renders work."""
from shiny import App, render, ui, reactive

app_ui = ui.page_fluid(
    ui.h2("Minimal Test App"),
    ui.input_action_button("click", "Click me"),
    ui.hr(),
    ui.h4("Static text works if you see this"),
    ui.value_box(
        "Counter",
        ui.output_text("counter"),
        theme="primary"
    ),
    ui.output_text("status"),
    ui.card(
        ui.card_header("Test Card"),
        ui.output_ui("test_ui")
    )
)

def server(input, output, session):
    count = reactive.value(0)
    
    @reactive.effect
    @reactive.event(input.click)
    def increment():
        count.set(count() + 1)
    
    @render.text
    def counter():
        print(f"RENDER counter: {count()}", flush=True)
        return str(count())
    
    @render.text
    def status():
        print(f"RENDER status: clicked {count()} times", flush=True)
        return f"Button clicked {count()} times"
    
    @render.ui
    def test_ui():
        print(f"RENDER test_ui called", flush=True)
        return ui.p(f"This is rendered UI, count = {count()}")

app = App(app_ui, server)
