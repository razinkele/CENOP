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
