## Players page -- search for a player, see their decay curve and stats.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.rapm_reader import compute_decay_curve, get_break_even_second


def render(season: str, rapm: pd.DataFrame | None):
    st.title("Player Shift-Decay Profiles")

    if rapm is None:
        st.warning(f"No RAPM data for {season}.")
        return

    rapm = rapm[rapm["toi_5v5"] >= 10].copy()

    ## ---------------------------------------------------------------------------
    ## player selector
    ## ---------------------------------------------------------------------------

    col_search, col_filter = st.columns([2, 1])

    with col_filter:
        teams = sorted(rapm["team"].dropna().unique())
        team_filter = st.selectbox("Filter by team", ["All"] + list(teams))

    display_df = rapm if team_filter == "All" else rapm[rapm["team"] == team_filter]

    with col_search:
        player_options = display_df.sort_values("player_name")["player_name"].tolist()
        selected_name = st.selectbox("Select player", player_options)

    if not selected_name:
        return

    row = rapm[rapm["player_name"] == selected_name].iloc[0]

    ## ---------------------------------------------------------------------------
    ## metrics row
    ## ---------------------------------------------------------------------------

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base RAPM", f"{row['rapm_base']:.3f}", help="xG diff per 60 at shift start")
    m2.metric("Decay coef", f"{row['rapm_decay']:.4f}", help="Change per 5s bucket")
    m3.metric("5v5 TOI", f"{row['toi_5v5']:.1f} min")

    bev = get_break_even_second(row["rapm_base"], row["rapm_decay"])
    m4.metric(
        "Break-even",
        f"{bev}s" if bev is not None else "Never",
        help="Shift age when projected xG diff hits zero",
    )

    ## ---------------------------------------------------------------------------
    ## decay curve chart
    ## ---------------------------------------------------------------------------

    st.subheader(f"Shift Decay Curve -- {selected_name}")

    buckets, values = compute_decay_curve(row["rapm_base"], row["rapm_decay"], max_seconds=90)

    fig = go.Figure()

    ## shade area under/over zero
    fig.add_trace(go.Scatter(
        x=buckets, y=values,
        mode="lines+markers",
        name="Projected xGD/60",
        line=dict(color="#2c7bb6", width=2.5),
        marker=dict(size=6),
    ))

    ## fill positive region green, negative red
    pos_y = [max(v, 0) for v in values]
    neg_y = [min(v, 0) for v in values]

    fig.add_trace(go.Scatter(
        x=buckets, y=pos_y,
        fill="tozeroy",
        fillcolor="rgba(67,147,195,0.15)",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=buckets, y=neg_y,
        fill="tozeroy",
        fillcolor="rgba(214,96,77,0.15)",
        line=dict(width=0),
        showlegend=False,
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)

    if bev is not None and bev <= 90:
        fig.add_vline(
            x=bev,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Break-even: {bev}s",
            annotation_position="top right",
        )

    ## mark average NHL shift length
    fig.add_vline(
        x=45,
        line_dash="dot",
        line_color="gray",
        opacity=0.5,
        annotation_text="Avg shift",
        annotation_position="top left",
    )

    fig.update_layout(
        xaxis_title="Shift age (seconds)",
        yaxis_title="Projected xG diff per 60",
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    ## ---------------------------------------------------------------------------
    ## context vs rest of league
    ## ---------------------------------------------------------------------------

    st.subheader("Context vs League")

    col_a, col_b = st.columns(2)

    with col_a:
        pct_base = (rapm["rapm_base"] < row["rapm_base"]).mean() * 100
        st.metric("Base RAPM percentile", f"{pct_base:.0f}th")

    with col_b:
        pct_decay = (rapm["rapm_decay"] > row["rapm_decay"]).mean() * 100
        st.metric("Durability percentile", f"{pct_decay:.0f}th", help="Higher = holds up better late in shifts")

    ## quick comparison table: similar TOI players sorted by decay
    st.caption("Players with similar TOI, sorted by decay rate")
    toi_band = rapm[
        (rapm["toi_5v5"] >= row["toi_5v5"] * 0.7) &
        (rapm["toi_5v5"] <= row["toi_5v5"] * 1.3)
    ].sort_values("rapm_decay")[["player_name", "team", "rapm_base", "rapm_decay", "toi_5v5"]].head(10)

    st.dataframe(toi_band, use_container_width=True, hide_index=True)
