## Players page.
## Tab 1: empirical shift-decay (works with any sample size, no model needed)
## Tab 2: RAPM model output (needs 300+ games to be meaningful)

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ingestion.roster import get_cached_names
from src.models.player_decay import get_player_empirical_decay, get_league_decay_summary
from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from src.models.line_analysis import load_stints
from dashboard.components.clickable_table import clickable_player_table


def render(season: str, rapm: pd.DataFrame | None):
    st.title("Player Shift-Decay Profiles")

    name_map = get_cached_names()

    tab_empirical, tab_rapm = st.tabs(["Observed Decay", "RAPM Model"])

    ## ---------------------------------------------------------------------------
    ## tab 1: empirical -- no model, just raw stint data split by shift age
    ## ---------------------------------------------------------------------------

    with tab_empirical:
        st.subheader("Observed xG Differential by Shift Age")
        st.caption(
            "Splits each player's on-ice stints by how old their shift was and measures "
            "xGD/60 within each window. Early = 0-30s on ice, Late = 45s+."
        )

        try:
            stints_df = load_stints(season)
        except FileNotFoundError:
            st.warning("Run the ingestion pipeline first.")
            return

        ev5 = stints_df[stints_df["strength"] == "5v5"]
        all_players: set[int] = set()
        for s in ev5["home_skaters"]:
            all_players.update(int(i) for i in s)

        player_options = sorted(all_players)
        player_labels  = [
            f"{name_map.get(pid, {}).get('name', f'Player_{pid}')} "
            f"({name_map.get(pid, {}).get('team', '?')})"
            for pid in player_options
        ]
        label_to_id = dict(zip(player_labels, player_options))

        col_left, col_right = st.columns([2, 1])

        with col_left:
            selected_label = st.selectbox("Select player", player_labels, key="emp_player")
            if selected_label:
                pid = label_to_id[selected_label]

                bucket_df = get_player_empirical_decay(season, pid, bucket_size=10, min_toi_sec=30)

                if bucket_df.empty:
                    st.info("Not enough on-ice time at different shift ages to plot a curve.")
                else:
                    fig = go.Figure()

                    ## shaded error band
                    fig.add_trace(go.Scatter(
                        x=list(bucket_df["shift_bucket"]) + list(bucket_df["shift_bucket"])[::-1],
                        y=list(bucket_df["xg_diff_per60"] + bucket_df["se"]) +
                          list(bucket_df["xg_diff_per60"] - bucket_df["se"])[::-1],
                        fill="toself",
                        fillcolor="rgba(44,123,182,0.15)",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                    fig.add_trace(go.Scatter(
                        x=bucket_df["shift_bucket"],
                        y=bucket_df["xg_diff_per60"],
                        mode="lines+markers",
                        name="Observed xGD/60",
                        line=dict(color="#2c7bb6", width=2.5),
                        marker=dict(size=8),
                        customdata=bucket_df[["toi_min", "n_stints"]].values,
                        hovertemplate=(
                            "Shift age: %{x}s<br>"
                            "xGD/60: %{y:.2f}<br>"
                            "TOI in bucket: %{customdata[0]:.1f} min<br>"
                            "Stints: %{customdata[1]}<extra></extra>"
                        ),
                    ))

                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
                    fig.add_vline(
                        x=45, line_dash="dot", line_color="gray", opacity=0.4,
                        annotation_text="Avg shift (45s)", annotation_position="top left",
                    )

                    fig.update_layout(
                        xaxis_title="Shift age (seconds)",
                        yaxis_title="Observed xG diff per 60",
                        height=400,
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    early_rows = bucket_df[bucket_df["shift_bucket"] <= 20]
                    late_rows  = bucket_df[bucket_df["shift_bucket"] >= 40]
                    early = early_rows["xg_diff_per60"].mean() if not early_rows.empty else np.nan
                    late  = late_rows["xg_diff_per60"].mean()  if not late_rows.empty  else np.nan
                    m1.metric("Early xGD/60 (0-30s)", f"{early:.2f}" if not np.isnan(early) else "N/A")
                    m2.metric("Late xGD/60 (40s+)",   f"{late:.2f}"  if not np.isnan(late)  else "N/A")
                    if not np.isnan(early) and not np.isnan(late):
                        m3.metric("Drop", f"{late - early:.2f}", delta_color="inverse")

        with col_right:
            st.markdown("**Biggest early-to-late drops**")
            st.caption("Min 30 TOI minutes. Click a row to open the profile.")
            try:
                summary = get_league_decay_summary(season, min_toi_sec=1800)
            except FileNotFoundError:
                st.info("No stint data.")
                summary = pd.DataFrame()

            if not summary.empty:
                summary["player_name"] = summary["player_id"].apply(
                    lambda p: name_map.get(p, {}).get("name", f"Player_{p}")
                )
                summary["team"] = summary["player_id"].apply(
                    lambda p: name_map.get(p, {}).get("team", "")
                )
                summary_src = summary.head(20).reset_index(drop=True)
                comp_rows = [
                    {
                        "id": int(r["player_id"]),
                        "values": [
                            r["player_name"],
                            r.get("team", ""),
                            f"{r['early_xgd60']:.2f}",
                            f"{r['late_xgd60']:.2f}",
                            f"{r['decay_delta']:.2f}",
                        ],
                    }
                    for _, r in summary_src.iterrows()
                ]
                clicked = clickable_player_table(
                    rows=comp_rows,
                    headers=["Player", "Team", "Early", "Late", "Drop"],
                    key="players_decay_table",
                )
                if clicked is not None:
                    st.session_state["selected_player_id"] = clicked
                    st.rerun()

    ## ---------------------------------------------------------------------------
    ## tab 2: RAPM model output
    ## ---------------------------------------------------------------------------

    with tab_rapm:
        st.caption(
            "RAPM model coefficients. Requires 300+ games for stable estimates. "
            "With fewer games ridge regularization shrinks everything toward zero."
        )

        if rapm is None:
            st.warning("No RAPM data found. Run the R model first.")
            return

        if rapm["rapm_decay"].abs().max() < 0.001:
            st.warning(
                "Decay coefficients are near zero -- model is over-regularized. "
                "Pull more games (300+) and re-run rapm_model.R."
            )

        col_search, col_filter = st.columns([2, 1])
        with col_filter:
            teams = sorted(rapm["team"].dropna().unique())
            team_filter = st.selectbox("Filter by team", ["All"] + list(teams), key="rapm_team")

        display_df  = rapm if team_filter == "All" else rapm[rapm["team"] == team_filter]
        player_opts = display_df.sort_values("player_name")["player_name"].tolist()

        if not player_opts:
            st.info("No players found for this team.")
            return

        with col_search:
            selected_name = st.selectbox("Select player", player_opts, key="rapm_player")

        ## guard against stale session state value from a different team filter
        match = rapm[rapm["player_name"] == selected_name]
        if match.empty:
            st.info("Select a player above.")
            return

        row = match.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Base RAPM", f"{row['rapm_base']:.4f}")
        m2.metric("Decay coef", f"{row['rapm_decay']:.6f}")
        m3.metric("5v5 TOI", f"{row['toi_5v5']:.1f} min")
        bev = get_break_even_second(row["rapm_base"], row["rapm_decay"])
        m4.metric("Break-even", f"{bev}s" if bev is not None else "Never")

        buckets, values = compute_decay_curve(row["rapm_base"], row["rapm_decay"], max_seconds=90)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=buckets, y=values, mode="lines+markers",
            line=dict(color="#2c7bb6", width=2),
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig2.add_vline(x=45, line_dash="dot", line_color="gray", opacity=0.4)
        fig2.update_layout(
            xaxis_title="Shift age (seconds)",
            yaxis_title="Projected xGD/60 (model)",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
