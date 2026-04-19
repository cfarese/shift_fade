## Overview page -- league-wide decay coefficient distribution and scatter.

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.rapm_reader import get_break_even_second


def render(season: str, rapm: pd.DataFrame | None):
    st.title("Shift Decay RAPM -- League Overview")

    if rapm is None:
        st.warning(f"No RAPM data found for {season}. Run the R model first.")
        return

    rapm = rapm.copy()
    rapm["break_even_sec"] = rapm.apply(
        lambda r: get_break_even_second(r["rapm_base"], r["rapm_decay"]), axis=1
    )

    ## ---------------------------------------------------------------------------
    ## top metrics
    ## ---------------------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players in sample", len(rapm))
    col2.metric("Overuse flags", int(rapm["overuse_flag"].sum()))
    col3.metric(
        "Avg decay coef",
        f"{rapm['rapm_decay'].mean():.3f}",
        help="Negative = players get worse as shifts age on average",
    )
    col4.metric(
        "Median break-even",
        f"{rapm['break_even_sec'].median():.0f}s"
        if rapm["break_even_sec"].notna().any()
        else "N/A",
    )

    st.markdown("---")

    ## ---------------------------------------------------------------------------
    ## decay coefficient histogram
    ## ---------------------------------------------------------------------------

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution of Decay Coefficients")
        fig = px.histogram(
            rapm,
            x="rapm_decay",
            nbins=40,
            color_discrete_sequence=["#2c7bb6"],
            labels={"rapm_decay": "Decay coefficient"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="zero")
        fig.add_vline(x=-0.05, line_dash="dot", line_color="orange", annotation_text="overuse threshold")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    ## ---------------------------------------------------------------------------
    ## base RAPM vs decay scatter
    ## ---------------------------------------------------------------------------

    with col_right:
        st.subheader("Base RAPM vs Decay Rate")
        plot_df = rapm[rapm["toi_5v5"] >= 50].copy()
        plot_df["label"] = plot_df.apply(
            lambda r: f"{r['player_name']}<br>Base: {r['rapm_base']:.2f} | Decay: {r['rapm_decay']:.3f}",
            axis=1,
        )

        fig2 = px.scatter(
            plot_df,
            x="rapm_decay",
            y="rapm_base",
            color="overuse_flag",
            size="toi_5v5",
            hover_name="player_name",
            hover_data={"rapm_base": ":.2f", "rapm_decay": ":.3f", "toi_5v5": ":.1f"},
            color_discrete_map={True: "#d6604d", False: "#4393c3"},
            labels={
                "rapm_decay": "Decay coefficient",
                "rapm_base": "Base RAPM",
                "overuse_flag": "Overuse flag",
            },
            height=350,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig2.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    ## ---------------------------------------------------------------------------
    ## worst decayers table
    ## ---------------------------------------------------------------------------

    st.subheader("Most Overused Players (fastest decay, min 50 min TOI)")
    worst = (
        rapm[rapm["toi_5v5"] >= 50]
        .sort_values("rapm_decay")
        .head(15)[["player_name", "team", "rapm_base", "rapm_decay", "toi_5v5", "break_even_sec", "overuse_flag"]]
        .rename(columns={
            "player_name": "Player",
            "team": "Team",
            "rapm_base": "Base RAPM",
            "rapm_decay": "Decay",
            "toi_5v5": "TOI (min)",
            "break_even_sec": "Break-even (s)",
            "overuse_flag": "Flagged",
        })
    )
    st.dataframe(worst, use_container_width=True, hide_index=True)
