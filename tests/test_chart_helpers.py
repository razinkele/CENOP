"""Charts draw with plotly.js loaded once in the page head, not one copy per chart."""

import re

import pandas as pd
from htmltools import HTMLDocument

from cenop.server.renderers import create_histogram_chart
from cenop.server.renderers.chart_helpers import create_time_series_chart
from cenop.ui.layout import app_ui

PLOTLY_SCRIPT = re.compile(r"<script[^>]*src=\"https://cdn\.plot\.ly/plotly-[\d.]+\.min\.js\"")


def _charts():
    df = pd.DataFrame({"day": [1, 2, 3], "a": [1.0, 2.0, 3.0]})
    yield str(create_time_series_chart(df, "day", ["a"], ["blue"], ["A"], "t"))
    yield str(create_histogram_chart([1.0, 2.0, 2.5], "t", "x", "y", (0.0, 3.0)))


def test_charts_do_not_load_plotly_themselves():
    for html in _charts():
        assert "Plotly.newPlot" in html
        assert not re.search(r"<script[^>]*\bsrc=", html)


def test_page_head_loads_plotly_once_with_integrity():
    html = HTMLDocument(app_ui(None)).render()["html"]
    head = html.split("</head>", 1)[0]
    tags = PLOTLY_SCRIPT.findall(head)
    assert len(tags) == 1
    assert 'integrity="sha256-' in head[head.index(tags[0]) :].split(">", 1)[0]
    assert len(PLOTLY_SCRIPT.findall(html)) == 1


def _figure(n_traces):
    import plotly.graph_objects as go

    from cenop.server.renderers.chart_helpers import _apply_depons_layout

    fig = go.Figure([go.Scatter(x=[1, 2], y=[1, 2], name=f"s{i}") for i in range(n_traces)])
    return _apply_depons_layout(fig, "Title", 300)


def test_legend_sits_above_the_plot_area_not_on_the_lines():
    layout = _figure(2).layout
    assert layout.legend.orientation == "h"
    assert layout.legend.yanchor == "bottom" and layout.legend.y >= 1.0  # above the plot area
    assert layout.margin.t >= 60  # room for the title and the legend row
    assert layout.title.yref == "container" and layout.title.yanchor == "top"


def test_single_line_charts_keep_the_compact_top_margin():
    assert _figure(1).layout.margin.t == 40
