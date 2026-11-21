"""plotly.js, loaded once in the page head for every chart.

The charts (cenop.server.renderers.chart_helpers) are built with include_plotlyjs=False.
When each chart carried its own <script src=...plotly...>, Shiny inserted the chart and ran
its drawing script before that download had finished ("Plotly is not defined" in the
browser console), and Plotly hashed its 4.6 MB bundle on every render. This tag loads the
same CDN file, with the same integrity hash, that Plotly would embed.
"""

from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

from shiny import ui


@lru_cache(maxsize=1)
def _cdn_script() -> tuple[str, str]:
    """(URL, integrity hash) of the plotly.js bundled with the installed plotly package."""
    from plotly.offline import get_plotlyjs, get_plotlyjs_version

    digest = hashlib.sha256(get_plotlyjs().encode("utf-8")).digest()
    url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    return url, "sha256-" + base64.b64encode(digest).decode("ascii")


def plotly_head():
    """Head content that loads plotly.js before any chart is drawn."""
    url, integrity = _cdn_script()
    return ui.head_content(
        # Plotly's own config script, which must run before the library loads
        ui.tags.script("window.PlotlyConfig = {MathJaxConfig: 'local'};"),
        ui.tags.script(src=url, integrity=integrity, crossorigin="anonymous", charset="utf-8"),
    )
