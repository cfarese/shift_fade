## Team Report page -- full breakdown of a team's deployment patterns.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from src.models.line_analysis import get_overused_lines
from src.ingestion.roster import get_cached_names
from dashboard.components.clickable_table import clickable_player_table


def render(season: str, rapm: pd.DataFrame | None, stints: pd.DataFrame | None):
    st.title("Team Deployment Report")

    if rapm is None:
        st.warning(f"No RAPM data for {season}.")
        return

    teams = sorted(rapm["team"].dropna().unique())
    if not teams:
        st.warning("No team data in RAPM results. Run resolve_names first: python3.11 -m src.ingestion.resolve_names --season " + season)
        return

    selected_team = st.selectbox("Select team", teams)
    team_rapm = rapm[rapm["team"] == selected_team].copy().reset_index(drop=True)
    team_rapm["break_even_sec"] = team_rapm.apply(
        lambda r: get_break_even_second(r["rapm_base"], r["rapm_decay"]), axis=1
    )

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

    ## player roster table with clickable names
    st.subheader("Roster Breakdown")
    st.caption("Click a player name to open their profile.")

    roster_src = team_rapm.sort_values("rapm_decay").reset_index(drop=True).copy()
    comp_rows = [
        {
            "id": int(r["player_id"]),
            "values": [
                r["player_name"],
                f"{r['rapm_base']:.4f}",
                f"{r['rapm_decay']:.4f}",
                f"{r['toi_5v5']:.1f}",
                f"{int(r['break_even_sec'])}s" if pd.notna(r["break_even_sec"]) else "Never",
                "Yes" if r["overuse_flag"] else "",
            ],
        }
        for _, r in roster_src.iterrows()
    ]
    clicked = clickable_player_table(
        rows=comp_rows,
        headers=["Player", "Base RAPM", "Decay", "TOI (min)", "Break-even", "Flagged"],
        key="team_roster_table",
    )
    if clicked is not None:
        st.session_state["selected_player_id"] = clicked
        st.rerun()

    ## decay curves overlay
    st.subheader("Decay Curves -- All Skaters")
    st.caption("Each line is one player. Red = overuse flagged.")

    fig = go.Figure()
    for _, row in team_rapm.iterrows():
        buckets, values = compute_decay_curve(row["rapm_base"], row["rapm_decay"], max_seconds=75)
        color   = "#d6604d" if row["overuse_flag"] else "#4393c3"
        opacity = 0.9       if row["overuse_flag"] else 0.4

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
            labels={"break_even_sec": "Break-even (seconds)", "player_name": "", "overuse_flag": "Flagged"},
            height=max(300, len(bev_df) * 28),
        )
        fig2.add_vline(x=45, line_dash="dash", line_color="gray", annotation_text="Avg shift (45s)")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ## overused lines
    if stints is not None:
        st.subheader("Line Combinations -- Overuse Check")
        name_map = get_cached_names()

        try:
            overuse = get_overused_lines(season, min_toi_min=3.0)
            if not overuse.empty:
                flagged = overuse[overuse["overuse_flag"]].copy()
                for c in ["toi_min", "early_xgd60", "late_xgd60", "decay_delta"]:
                    flagged[c] = flagged[c].round(2)

                if not flagged.empty:
                    st.error(f"{len(flagged)} line combinations flagged for overuse")
                    flagged_display = flagged.copy()
                    flagged_display["Line"] = flagged_display["home_skaters"].apply(
                        lambda c: " / ".join(name_map.get(pid, {}).get("name", str(pid)) for pid in c)
                    )
                    st.dataframe(
                        flagged_display[["Line", "toi_min", "early_xgd60", "late_xgd60", "decay_delta"]].rename(
                            columns={"toi_min": "TOI (min)", "early_xgd60": "Early xGD/60",
                                     "late_xgd60": "Late xGD/60", "decay_delta": "Decay"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("No line combinations flagged.")
        except FileNotFoundError:
            st.info("Run the ingestion pipeline to see line-level overuse data.")
