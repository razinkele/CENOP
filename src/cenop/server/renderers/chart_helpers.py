"""
Chart Helper Functions for CENOP

Provides standardized Plotly chart creation with DEPONS-style formatting.
Eliminates duplicate chart configuration code across renderers.
"""

import pandas as pd
import plotly.graph_objects as go
from shiny import ui

# Marine "Hybrid Depth" color scheme
DEPONS_COLORS = {
    "primary": "#00b4d8",  # Cyan — population, primary data
    "secondary": "#ff6b6b",  # Coral — deaths, noise, danger
    "success": "#48c78e",  # Sea green — births, food
    "warning": "#f0a040",  # Amber — turbines, energy
    "background": "rgba(0,0,0,0)",  # Transparent (dark card shows through)
    "grid": "rgba(255,255,255,0.08)",
    "text": "#c8dce8",  # Light ocean text
    "muted": "#8899aa",  # Muted axis labels
    "cyan_light": "#90e0ef",  # Light cyan — secondary data
}


def _apply_depons_layout(
    fig: go.Figure,
    title: str,
    height: int,
    x_title: str = "",
    y_title: str = "",
    show_legend: bool = True,
) -> go.Figure:
    """
    Apply standard DEPONS styling to a Plotly figure.

    Args:
        fig: Plotly figure to style
        title: Chart title
        height: Chart height in pixels
        x_title: X-axis label
        y_title: Y-axis label
        show_legend: Whether to show legend

    Returns:
        Styled figure
    """
    legend_config = (
        dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(color=DEPONS_COLORS["text"], size=11),
            bgcolor="rgba(0,0,0,0)",
        )
        if show_legend
        else {}
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=DEPONS_COLORS["text"], size=13)),
        xaxis_title=dict(text=x_title, font=dict(color=DEPONS_COLORS["muted"])),
        yaxis_title=dict(text=y_title, font=dict(color=DEPONS_COLORS["muted"])),
        height=height,
        legend=legend_config,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor=DEPONS_COLORS["background"],
        paper_bgcolor=DEPONS_COLORS["background"],
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=DEPONS_COLORS["grid"],
        tickfont=dict(color=DEPONS_COLORS["muted"]),
        linecolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEPONS_COLORS["grid"],
        tickfont=dict(color=DEPONS_COLORS["muted"]),
        linecolor="rgba(255,255,255,0.1)",
    )

    return fig


def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    colors: list[str],
    names: list[str],
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 180,
) -> ui.HTML:
    """
    Create a standardized time series chart.

    Args:
        df: DataFrame with the data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis traces
        colors: List of colors for each trace
        names: List of legend names for each trace
        title: Chart title
        x_title: X-axis label
        y_title: Y-axis label
        height: Chart height in pixels

    Returns:
        Shiny HTML element containing the chart
    """
    fig = go.Figure()

    for col, color, name in zip(y_cols, colors, names):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col], y=df[col], mode="lines", name=name, line=dict(color=color, width=2)
                )
            )

    _apply_depons_layout(fig, title, height, x_title, y_title)

    return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


def create_histogram_chart(
    data: list[float],
    title: str,
    x_title: str,
    y_title: str,
    x_range: tuple[float, float],
    nbins: int = 30,
    color: str = "red",
    height: int = 300,
) -> ui.HTML:
    """
    Create a standardized histogram chart.

    Args:
        data: List of values to histogram
        title: Chart title
        x_title: X-axis label
        y_title: Y-axis label
        x_range: Tuple of (min, max) for x-axis
        nbins: Number of bins
        color: Bar color
        height: Chart height in pixels

    Returns:
        Shiny HTML element containing the chart
    """
    fig = go.Figure()

    bin_size = (x_range[1] - x_range[0]) / nbins

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=nbins,
            xbins=dict(start=x_range[0], end=x_range[1], size=bin_size),
            marker_color=color,
            name=x_title,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=DEPONS_COLORS["text"], size=13)),
        xaxis_title=dict(text=x_title, font=dict(color=DEPONS_COLORS["muted"])),
        yaxis_title=dict(text=y_title, font=dict(color=DEPONS_COLORS["muted"])),
        height=height,
        xaxis=dict(range=list(x_range)),
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor=DEPONS_COLORS["background"],
        paper_bgcolor=DEPONS_COLORS["background"],
        bargap=0.1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=DEPONS_COLORS["grid"],
        tickfont=dict(color=DEPONS_COLORS["muted"]),
        linecolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=DEPONS_COLORS["grid"],
        tickfont=dict(color=DEPONS_COLORS["muted"]),
        linecolor="rgba(255,255,255,0.1)",
    )

    return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


def create_svg_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    colors: list[str],
    names: list[str],
    title: str = "",
    height: int = 150,
) -> "ui.HTML":
    """
    Lightweight SVG time series chart — no JS dependencies, instant rendering.

    Designed for dashboard sparklines where Plotly's ~3MB library and heavy
    DOM operations cause unnecessary performance overhead.
    """
    import numpy as np

    if df is None or len(df) < 2:
        return ui.HTML(
            f'<div style="height:{height}px;display:flex;align-items:center;'
            f'justify-content:center;color:#8899aa;font-size:12px;">'
            f"No data yet. Run simulation to see results.</div>"
        )

    # Downsample to max 300 points
    if len(df) > 300:
        step = max(1, len(df) // 300)
        df = df.iloc[::step].copy()

    # Dimensions
    W, H = 500, height
    ml, mr, mt, mb = 50, 14, 20, 26
    cw, ch = W - ml - mr, H - mt - mb

    x_vals = df[x_col].values.astype(float)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    x_range = x_max - x_min or 1.0

    # Global y range
    all_y = []
    for col in y_cols:
        if col in df.columns:
            v = df[col].dropna().values
            if len(v):
                all_y.extend(v.tolist())
    if not all_y:
        return ui.HTML(
            f'<div style="height:{height}px;display:flex;align-items:center;'
            f'justify-content:center;color:#8899aa;font-size:12px;">No data</div>'
        )

    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.08 or 1.0
    y_min -= y_pad
    y_max += y_pad
    y_range = y_max - y_min

    p = [
        f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%;height:100%;font-family:system-ui,sans-serif;'
        f'background:transparent;border-radius:4px;">'
    ]

    # Title
    p.append(
        f'<text x="{ml}" y="14" font-size="11" font-weight="600" fill="#c8dce8">{title}</text>'
    )

    # Grid + Y labels
    for i in range(5):
        frac = i / 4
        yv = y_min + y_range * frac
        yp = mt + ch - ch * frac
        p.append(
            f'<line x1="{ml}" y1="{yp:.0f}" x2="{ml+cw}" y2="{yp:.0f}" '
            f'stroke="#1a2a3a" stroke-width="0.5"/>'
        )
        p.append(
            f'<text x="{ml-4}" y="{yp+3:.0f}" text-anchor="end" '
            f'font-size="9" fill="#8899aa">{yv:.0f}</text>'
        )

    # Data lines
    for col, color, name in zip(y_cols, colors, names):
        if col not in df.columns:
            continue
        y_data = df[col].values
        pts = []
        for xv, yv in zip(x_vals, y_data):
            if np.isnan(yv):
                continue
            px = ml + ((float(xv) - x_min) / x_range) * cw
            py = mt + ch - ((float(yv) - y_min) / y_range) * ch
            pts.append(f"{px:.1f},{py:.1f}")
        if pts:
            p.append(
                f'<polyline points="{" ".join(pts)}" '
                f'fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
            )
            # Current value label at right edge
            last_y = df[col].dropna().iloc[-1] if len(df[col].dropna()) else None
            if last_y is not None:
                ly = mt + ch - ((float(last_y) - y_min) / y_range) * ch
                p.append(f'<circle cx="{ml+cw}" cy="{ly:.1f}" r="2.5" fill="{color}"/>')

    # Legend
    lx = ml
    for color, name in zip(colors, names):
        p.append(
            f'<line x1="{lx}" y1="{H-9}" x2="{lx+14}" y2="{H-9}" '
            f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        p.append(f'<text x="{lx+18}" y="{H-6}" font-size="9" fill="#8899aa">{name}</text>')
        lx += len(name) * 5.5 + 30

    # Axes
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ch}" stroke="rgba(255,255,255,0.1)"/>')
    p.append(
        f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="rgba(255,255,255,0.1)"/>'
    )

    p.append("</svg>")
    return ui.HTML("".join(p))


def no_data_placeholder(message: str = "No data yet. Run simulation to see results.") -> ui.Tag:
    """
    Create a standardized placeholder for empty charts.

    Args:
        message: Message to display

    Returns:
        Shiny paragraph element
    """
    return ui.p(message, class_="text-muted text-center mt-5")
