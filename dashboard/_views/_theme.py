## Shared visual theme utilities -- IBM Plex + red/white/blue percentile scale.

from __future__ import annotations

import plotly.graph_objects as go

BLUE = "#1d4ed8"
RED  = "#dc2626"


def pct_bg(pct: float) -> str:
    t = max(0.0, min(100.0, float(pct))) / 100.0
    if t < 0.5:
        s = t * 2
        r = round(214 + (255 - 214) * s)
        g = round(40  + (255 - 40)  * s)
        b = round(40  + (255 - 40)  * s)
    else:
        s = (t - 0.5) * 2
        r = round(255 + (29  - 255) * s)
        g = round(255 + (100 - 255) * s)
        b = round(255 + (220 - 255) * s)
    return f"rgb({r},{g},{b})"


def pct_fg(pct: float) -> str:
    return "#fff" if (pct < 18 or pct > 82) else "#222"


def pct_badge_html(pct: float, label: str) -> str:
    bg, fg = pct_bg(pct), pct_fg(pct)
    return (
        f'<div style="text-align:center;display:inline-block;margin:0 12px 0 0">'
        f'<div style="background:{bg};color:{fg};border-radius:4px;padding:5px 14px;'
        f'font-family:\'IBM Plex Mono\',monospace;font-size:15px;font-weight:600;'
        f'min-width:58px;text-align:center">'
        f'{round(pct)}<span style="font-size:10px;font-weight:400">th</span></div>'
        f'<div style="font-size:11px;color:#888;margin-top:5px;font-weight:500;'
        f'white-space:nowrap">{label}</div>'
        f'</div>'
    )


def pct_rank(series, higher_is_better: bool = True):
    """Return 0-100 percentile rank for each value in a pandas Series."""
    ranked = series.rank(pct=True) * 100
    return ranked if higher_is_better else 100 - ranked


def apply_chart_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    axis_common = dict(
        gridcolor="#f0f0f0",
        linecolor="#e5e5e5",
        tickfont=dict(family="IBM Plex Mono", size=11, color="#999"),
        title_font=dict(family="IBM Plex Sans", size=12, color="#777"),
    )
    layout: dict = dict(
        font=dict(family="IBM Plex Sans", size=12, color="#444"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=axis_common,
        yaxis=axis_common,
        margin=dict(l=60, r=24, t=28, b=48),
        legend=dict(font=dict(family="IBM Plex Sans", size=11)),
        hovermode="x unified",
    )
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


def add_league_band(
    fig: go.Figure,
    buckets: list[float],
    p25: list[float],
    med: list[float],
    p75: list[float],
) -> go.Figure:
    fig.add_trace(go.Scatter(
        x=buckets + buckets[::-1],
        y=p75 + p25[::-1],
        fill="toself",
        fillcolor="rgba(226,234,246,0.7)",
        line=dict(width=0),
        showlegend=True,
        name="League P25–P75",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=buckets,
        y=med,
        mode="lines",
        line=dict(color="#aabbc0", width=1.5, dash="dash"),
        showlegend=True,
        name="League median",
        hoverinfo="skip",
    ))
    return fig
