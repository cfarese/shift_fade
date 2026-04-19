## Team Report page -- full breakdown of a team's deployment patterns.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from src.models.line_analysis import get_line_stats, get_overused_lines
from src.ingestion.roster import get_cached_names


def render(season: str, rapm: pd.DataFrame | None, stints: pd.DataFrame | None):
    st.title("Team Deployment Report")

    if rapm is None:
        st.warning(f"No RAPM data for {season}.")
        return

    teams = sorted(rapm["team"].dropna().unique())
    if not teams:
        st.warning("No team data in RAPM results. Player name lookup may not have run yet.")
        return

    selected_team = st.selectbox("Select team", teams)
    team_rapm = rapm[rapm["team"] == selected_team].copy()
    team_rapm["break_even_sec"] = team_rapm.apply(
        lambda r: get_break_even_second(r["rapm_base"], r["rapm_decay"]), axis=1
    )
    ## team summary metrics

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players in sample", len(team_rapm))
    m2.metric("Overuse flags", int(team_rapm["overuse_flag"].sum()))
    m3.metric("Team avg decay", f"{team_rapm['rapm_decay'].mean():.3f}")
    m4.metric(
        "Avg break-even",
        f"{team_rapm['break_even_sec'].mean():.0f}s"
        if team_rapm["break_even_sec"].notna().any()
        else "N/A",
    )

    ## player roster table with overuse highlight

    st.subheader("Roster Breakdown")

    roster_display = team_rapm[["player_name", "rapm_base", "rapm_decay", "toi_5v5", "break_even_sec", "overuse_flag"]].copy()
    roster_display.columns = ["Player", "Base RAPM", "Decay", "TOI (min)", "Break-even (s)", "Flagged"]
    roster_display = roster_display.sort_values("Decay")

    ## highlight flagged rows in the table
    def _highlight_flagged(row):
        color = "background-color: #fde8e4" if row["Flagged"] else ""
        return [color] * len(row)

    st.dataframe(
        roster_display.style.apply(_highlight_flagged, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    ## decay curves for all players on the team

    st.subheader("Decay Curves -- All Skaters")
    st.caption("Each line is one player. Red = overuse flagged.")

    fig = go.Figure()

    for _, row in team_rapm.iterrows():
        buckets, values = compute_decay_curve(row["rapm_base"], row["rapm_decay"], max_seconds=75)
        color = "#d6604d" if row["overuse_flag"] else "#4393c3"
        opacity = 0.9 if row["overuse_flag"] else 0.4

        fig.add_trace(go.Scatter(
            x=buckets,
            y=values,
            mode="lines",
            name=row["player_name"],
            line=dict(color=color, width=1.5),
            opacity=opacity,
            hovertemplate=f"{row['player_name']}<br>%{{y:.2f}} xGD/60 at %{{x}}s<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=45, line_dash="dot", line_color="gray", opacity=0.4, annotation_text="Avg shift")

    fig.update_layout(
        xaxis_title="Shift age (seconds)",
        yaxis_title="Projected xG diff per 60",
        height=450,
        hovermode="x",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    ## break-even bar chart

    st.subheader("Break-even Shift Length per Player")
    st.caption("Players who go negative before the average shift (45s) are overused.")

    bev_df = team_rapm[team_rapm["break_even_sec"].notna()].sort_values("break_even_sec")

    if bev_df.empty:
        st.info("No players with a break-even point (all have positive or zero decay).")
    else:
        fig2 = px.bar(
            bev_df,
            x="break_even_sec",
            y="player_name",
            orientation="h",
            color="overuse_flag",
            color_discrete_map={True: "#d6604d", False: "#4393c3"},
            labels={
                "break_even_sec": "Break-even (seconds)",
                "player_name": "",
                "overuse_flag": "Flagged",
            },
            height=max(300, len(bev_df) * 28),
        )
        fig2.add_vline(x=45, line_dash="dash", line_color="gray", annotation_text="Avg shift (45s)")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ## overused lines for this team
   
    if stints is not None:
        st.subheader("Line Combinations -- Overuse Check")
        name_map = get_cached_names()

        ## filter stints to this team's home games only
        ## TODO: need team ID in stints df to do this properly
        st.caption("Note: line filtering by team requires team ID in stint data. Showing all lines for now.")

        try:
            overuse = get_overused_lines(season, min_toi_min=3.0)
            if not overuse.empty:
                overuse["line"] = overuse["home_skaters"].apply(
                    lambda c: " / ".join(name_map.get(pid, {}).get("name", str(pid)) for pid in c)
                )
                flagged = overuse[overuse["overuse_flag"]][["line", "toi_min", "early_xgd60", "late_xgd60", "decay_delta"]]
                if not flagged.empty:
                    st.error(f"{len(flagged)} line combinations flagged for overuse")
                    st.dataframe(flagged.round(2), use_container_width=True, hide_index=True)
                else:
                    st.success("No line combinations flagged.")
        except FileNotFoundError:
            st.info("Run the ingestion pipeline to see line-level overuse data.")
