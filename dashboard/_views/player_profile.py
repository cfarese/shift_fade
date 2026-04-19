## Full player profile view -- rendered when a player is selected from any table.
## Shows empirical decay curve, shift-age breakdown, league context, and RAPM if available.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ingestion.roster import get_cached_names
from src.models.player_decay import get_player_empirical_decay
from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from src.models.line_analysis import load_stints


def render(player_id: int, season: str, rapm: pd.DataFrame | None):
    name_map  = get_cached_names()
    info      = name_map.get(player_id, {})
    full_name = info.get("name", f"Player_{player_id}")
    team      = info.get("team", "")
    position  = info.get("position", "")

    if st.button("← Back"):
        st.session_state.selected_player_id = None
        st.session_state.pop("_player_page", None)
        st.rerun()

    st.title(full_name)
    st.caption(f"{team}  |  {position}  |  Season {season[:4]}-{season[4:]}")
    st.markdown("---")

    ## ---------------------------------------------------------------------------
    ## empirical decay curve
    ## ---------------------------------------------------------------------------

    try:
        bucket_df = get_player_empirical_decay(season, player_id, bucket_size=10, min_toi_sec=20)
    except FileNotFoundError:
        st.warning("Stint data not found. Run the pipeline first.")
        return

    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        st.subheader("On-Ice xGD/60 by Shift Age")
        st.caption("Observed, not modeled. Each bucket = stints where the shift was that old.")

        if bucket_df.empty:
            st.info("Not enough data at different shift ages to build a curve.")
        else:
            fig = go.Figure()

            ## error band
            fig.add_trace(go.Scatter(
                x=list(bucket_df["shift_bucket"]) + list(bucket_df["shift_bucket"])[::-1],
                y=list(bucket_df["xg_diff_per60"] + bucket_df["se"]) +
                  list(bucket_df["xg_diff_per60"] - bucket_df["se"])[::-1],
                fill="toself",
                fillcolor="rgba(44,123,182,0.12)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            ## main curve
            fig.add_trace(go.Scatter(
                x=bucket_df["shift_bucket"],
                y=bucket_df["xg_diff_per60"],
                mode="lines+markers",
                line=dict(color="#2c7bb6", width=2.5),
                marker=dict(size=9),
                name="xGD/60",
                customdata=bucket_df[["toi_min", "n_stints"]].values,
                hovertemplate=(
                    "<b>Shift age: %{x}s</b><br>"
                    "xGD/60: %{y:.2f}<br>"
                    "TOI in bucket: %{customdata[0]:.1f} min<br>"
                    "Stints: %{customdata[1]}<extra></extra>"
                ),
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(
                x=45, line_dash="dot", line_color="gray", opacity=0.4,
                annotation_text="Avg shift", annotation_position="top left",
            )

            ## shade positive green, negative red
            for i in range(len(bucket_df) - 1):
                y_val = bucket_df["xg_diff_per60"].iloc[i]
                color = "rgba(67,160,71,0.08)" if y_val >= 0 else "rgba(229,57,53,0.08)"

            fig.update_layout(
                xaxis_title="Shift age (seconds)",
                yaxis_title="Observed xG diff per 60",
                height=420,
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("Shift Breakdown")

        if not bucket_df.empty:
            early = bucket_df[bucket_df["shift_bucket"] <= 20]["xg_diff_per60"]
            mid   = bucket_df[(bucket_df["shift_bucket"] > 20) & (bucket_df["shift_bucket"] <= 40)]["xg_diff_per60"]
            late  = bucket_df[bucket_df["shift_bucket"] > 40]["xg_diff_per60"]

            early_val = early.mean() if not early.empty else np.nan
            mid_val   = mid.mean()   if not mid.empty   else np.nan
            late_val  = late.mean()  if not late.empty  else np.nan

            st.metric("Fresh (0-30s)",   f"{early_val:.2f}" if not np.isnan(early_val) else "N/A", help="xGD/60")
            st.metric("Mid (20-40s)",    f"{mid_val:.2f}"   if not np.isnan(mid_val)   else "N/A")
            st.metric("Late (40s+)",     f"{late_val:.2f}"  if not np.isnan(late_val)  else "N/A")

            if not np.isnan(early_val) and not np.isnan(late_val):
                drop = late_val - early_val
                st.metric("Early-to-late drop", f"{drop:.2f}", delta_color="inverse")

            st.markdown("---")
            total_toi = bucket_df["toi_min"].sum()
            st.metric("Total 5v5 TOI", f"{total_toi:.1f} min")
            st.metric("Total stints", int(bucket_df["n_stints"].sum()))

    ## ---------------------------------------------------------------------------
    ## shift-age bucket table
    ## ---------------------------------------------------------------------------

    st.subheader("Shift-Age Breakdown Table")
    if not bucket_df.empty:
        tbl = bucket_df[["shift_bucket", "xg_diff_per60", "toi_min", "n_stints"]].copy()
        tbl.columns = ["Shift age (s)", "xGD/60", "TOI (min)", "Stints"]
        tbl["xGD/60"]    = tbl["xGD/60"].round(2)
        tbl["TOI (min)"] = tbl["TOI (min)"].round(2)
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    ## ---------------------------------------------------------------------------
    ## league context
    ## ---------------------------------------------------------------------------

    st.subheader("Context vs League")

    try:
        from src.models.player_decay import get_league_decay_summary
        summary = get_league_decay_summary(season, min_toi_sec=600)
    except Exception:
        summary = pd.DataFrame()

    if not summary.empty and not bucket_df.empty:
        early_rows = bucket_df[bucket_df["shift_bucket"] <= 20]
        late_rows  = bucket_df[bucket_df["shift_bucket"] >= 40]
        p_early = early_rows["xg_diff_per60"].mean() if not early_rows.empty else np.nan
        p_late  = late_rows["xg_diff_per60"].mean()  if not late_rows.empty  else np.nan

        c1, c2, c3 = st.columns(3)

        if not np.isnan(p_early):
            pct = (summary["early_xgd60"] < p_early).mean() * 100
            c1.metric("Early xGD/60 percentile", f"{pct:.0f}th")

        if not np.isnan(p_late):
            pct = (summary["late_xgd60"] < p_late).mean() * 100
            c2.metric("Late xGD/60 percentile", f"{pct:.0f}th")

        if not np.isnan(p_early) and not np.isnan(p_late):
            drop = p_late - p_early
            pct_decay = (summary["decay_delta"] > drop).mean() * 100
            c3.metric("Durability percentile", f"{pct_decay:.0f}th", help="Higher = holds up better late in shifts")

    ## ---------------------------------------------------------------------------
    ## RAPM section if available
    ## ---------------------------------------------------------------------------

    if rapm is not None:
        row = rapm[rapm["player_id"] == player_id]
        if not row.empty:
            r = row.iloc[0]
            st.subheader("RAPM Model (Ridge Regression)")

            if abs(r["rapm_decay"]) < 0.001:
                st.caption("Coefficients near zero -- model needs more data (300+ games).")

            m1, m2, m3 = st.columns(3)
            m1.metric("Base RAPM", f"{r['rapm_base']:.4f}")
            m2.metric("Decay coef", f"{r['rapm_decay']:.6f}")
            bev = get_break_even_second(r["rapm_base"], r["rapm_decay"])
            m3.metric("Model break-even", f"{bev}s" if bev else "Never")
