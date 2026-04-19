## Overview page -- league-wide metrics, distribution charts, leaderboards.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.clickable_table import clickable_player_table
from dashboard._views._theme import (
    apply_chart_theme, pct_bg, pct_fg, pct_rank, BLUE, RED,
)


def _fmt(v, d=4, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"{abs(v):.{d}f}"
    if sign:
        return ("+" if v >= 0 else "-") + s
    return s


def _fmtbe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Never"
    return f"{int(v)}s" if v <= 200 else ">200s"


def _decay_histogram(rapm: pd.DataFrame) -> go.Figure:
    vals = rapm["rapm_decay"].dropna()
    vmin, vmax = float(vals.min()), float(vals.max())
    n_bins = 16
    step   = (vmax - vmin) / n_bins
    edges  = [vmin + i * step for i in range(n_bins + 1)]
    counts = [int(((vals >= edges[i]) & (vals < edges[i + 1])).sum()) for i in range(n_bins)]
    counts[-1] += int((vals == vmax).sum())
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    pcts    = [float((vals < c).mean() * 100) for c in centers]

    fig = go.Figure()
    for c, cnt, pct in zip(centers, counts, pcts):
        fig.add_trace(go.Bar(
            x=[c], y=[cnt],
            width=step * 0.9,
            marker_color=pct_bg(pct),
            marker_line_width=0,
            showlegend=False,
            hovertemplate=f"Decay: {c:.5f}<br>Players: {cnt}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#c00", line_width=1.5,
                  annotation_text="0", annotation_font=dict(color="#c00", size=10))
    fig.update_layout(xaxis_title="Decay coefficient", yaxis_title="Players", barmode="overlay", bargap=0)
    apply_chart_theme(fig, height=220)
    fig.update_layout(margin=dict(l=44, r=16, t=16, b=48))
    return fig


def _rapm_scatter(rapm: pd.DataFrame) -> go.Figure:
    flagged = rapm[rapm["overuse_flag"]]
    normal  = rapm[~rapm["overuse_flag"]]

    fig = go.Figure()
    for df, color, name in [(normal, BLUE, "Normal"), (flagged, RED, "Flagged")]:
        fig.add_trace(go.Scatter(
            x=df["rapm_decay"],
            y=df["rapm_base"],
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.75, line=dict(color="#fff", width=1)),
            name=name,
            text=df["player_name"],
            hovertemplate="<b>%{text}</b><br>Base RAPM: %{y:.4f}<br>Decay: %{x:.6f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#eee", line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color="#ddd", line_width=1)
    fig.update_layout(
        xaxis_title="Decay coefficient",
        yaxis_title="Base RAPM",
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
    )
    apply_chart_theme(fig, height=220)
    fig.update_layout(margin=dict(l=52, r=16, t=24, b=48))
    return fig


def render(season: str, rapm: pd.DataFrame | None):
    st.title("League Overview")
    st.caption(f"Shift-age decay analysis across all skaters · Season {season[:4]}–{season[4:]} · 5v5")

    if rapm is None:
        st.warning(f"No RAPM data for {season}.")
        st.code(
            f"python -m src.ingestion.pipeline --season {season}\n"
            f"Rscript r/rapm/rapm_model.R --season {season}"
        )
        return

    overuse_n  = int(rapm["overuse_flag"].sum())
    avg_decay  = float(rapm["rapm_decay"].mean())
    be_vals    = rapm.apply(
        lambda r: -r["rapm_base"] / r["rapm_decay"]
        if r["rapm_decay"] < -0.00002 and r["rapm_base"] > 0 else np.nan,
        axis=1,
    )
    median_be  = float(be_vals.dropna().median()) if be_vals.notna().any() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players in sample", len(rapm))
    c2.metric("Overuse flags", overuse_n, help="decay < −0.00035 and min 50 min TOI")
    c3.metric("Avg decay coef", f"{avg_decay:.6f}", help="Negative = worsens with shift age")
    c4.metric("Median break-even", _fmtbe(median_be) if not np.isnan(median_be) else "N/A")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    col_hist, col_scatter = st.columns(2)

    with col_hist:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Distribution of Decay Coefficients</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:4px">'
            'Red bars = fastest decay. Blue = holds up well with shift age.</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_decay_histogram(rapm), use_container_width=True, config={"displayModeBar": False})

    with col_scatter:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Base RAPM vs. Decay Rate</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:4px">'
            'Red = overuse-flagged. Ideal: upper-left (good RAPM, low decay).</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_rapm_scatter(rapm), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

    # percentile ranks for cell coloring
    rapm = rapm.copy()
    rapm["pct_rapm"]  = pct_rank(rapm["rapm_base"],  higher_is_better=True)
    rapm["pct_decay"] = pct_rank(rapm["rapm_decay"],  higher_is_better=True)
    rapm["pct_toi"]   = pct_rank(rapm["toi_5v5"],    higher_is_better=True)
    rapm["break_even"] = be_vals

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Most Overused Players</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
            'Fastest decay coefficient. Click a name to view their profile.</div>',
            unsafe_allow_html=True,
        )
        worst_src = rapm.sort_values("rapm_decay").head(12).reset_index(drop=True)
        comp_rows = [
            {
                "id": int(r["player_id"]),
                "values": [
                    r["player_name"],
                    r.get("team", ""),
                    _fmt(r["rapm_base"], 4),
                    f"{r['rapm_decay']:.6f}",
                    f"{r['toi_5v5']:.1f}",
                    _fmtbe(r["break_even"]),
                ],
                "pcts": [None, None, float(r["pct_rapm"]), float(r["pct_decay"]), float(r["pct_toi"]), None],
            }
            for _, r in worst_src.iterrows()
        ]
        clicked = clickable_player_table(rows=comp_rows,
                                         headers=["Player", "Team", "Base RAPM", "Decay", "TOI (min)", "Break-even"],
                                         key="overview_worst")
        if clicked is not None:
            st.session_state["selected_player_id"] = clicked
            st.rerun()

    with col_right:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Top Performers by Base RAPM</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
            'Best baseline value regardless of decay.</div>',
            unsafe_allow_html=True,
        )
        top_src = rapm.sort_values("rapm_base", ascending=False).head(10).reset_index(drop=True)
        comp_rows_top = [
            {
                "id": int(r["player_id"]),
                "values": [
                    r["player_name"],
                    r.get("team", ""),
                    _fmt(r["rapm_base"], 4),
                    f"{r['rapm_decay']:.6f}",
                    f"{r['toi_5v5']:.1f}",
                ],
                "pcts": [None, None, float(r["pct_rapm"]), float(r["pct_decay"]), float(r["pct_toi"])],
            }
            for _, r in top_src.iterrows()
        ]
        clicked2 = clickable_player_table(rows=comp_rows_top,
                                           headers=["Player", "Team", "Base RAPM", "Decay", "TOI (min)"],
                                           key="overview_top")
        if clicked2 is not None:
            st.session_state["selected_player_id"] = clicked2
            st.rerun()
