## Lines page -- line combination stats and per-combo decay curves.

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.line_analysis import get_line_decay_by_bucket, get_line_stats, get_overused_lines
from src.ingestion.roster import get_cached_names


def _format_combo(skater_ids: tuple, name_map: dict) -> str:
    names = [name_map.get(pid, {}).get("name", str(pid)) for pid in skater_ids]
    return " / ".join(names)


def render(season: str, stints: pd.DataFrame | None):
    st.title("Line Combination Analysis")

    if stints is None:
        st.warning(f"No stint data found for {season}. Run the ingestion pipeline first.")
        return

    name_map = get_cached_names()

    min_toi = st.sidebar.slider("Min line TOI (minutes)", 1, 30, 5)

    tab1, tab2 = st.tabs(["Top Lines", "Overuse Report"])

    ## ---------------------------------------------------------------------------
    ## tab 1: top lines by xGD/60
    ## ---------------------------------------------------------------------------

    with tab1:
        st.subheader("Best Lines by xG Differential per 60 (5v5)")

        try:
            stats = get_line_stats(season, min_toi_sec=int(min_toi * 60))
        except FileNotFoundError as e:
            st.error(str(e))
            return

        if stats.empty:
            st.info("No line combinations met the minimum TOI threshold.")
            return

        stats["line"] = stats["home_skaters"].apply(lambda c: _format_combo(c, name_map))
        display_cols = ["line", "toi_min", "xg_diff_per60", "xgf_pct", "cf_pct", "avg_shift_age", "n_stints"]
        display = stats[display_cols].rename(columns={
            "line": "Line",
            "toi_min": "TOI (min)",
            "xg_diff_per60": "xGD/60",
            "xgf_pct": "xGF%",
            "cf_pct": "CF%",
            "avg_shift_age": "Avg Shift Age (s)",
            "n_stints": "Stints",
        })
        display["xGD/60"] = display["xGD/60"].round(2)
        display["xGF%"] = (display["xGF%"] * 100).round(1)
        display["CF%"] = (display["CF%"] * 100).round(1)
        display["TOI (min)"] = display["TOI (min)"].round(1)
        display["Avg Shift Age (s)"] = display["Avg Shift Age (s)"].round(1)

        st.dataframe(display.head(25), use_container_width=True, hide_index=True)

        ## decay curve for a selected line
        st.markdown("---")
        st.subheader("Shift Decay Curve for Selected Line")

        line_options = stats["home_skaters"].tolist()
        line_labels  = stats["line"].tolist()
        label_to_combo = dict(zip(line_labels, line_options))

        selected_label = st.selectbox("Select line", line_labels)
        if selected_label:
            combo = label_to_combo[selected_label]
            bucket_df = get_line_decay_by_bucket(season, combo)

            if bucket_df.empty:
                st.info("Not enough data for this combo.")
            else:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=bucket_df["shift_bucket"],
                    y=bucket_df["xg_diff_per60"],
                    mode="lines+markers",
                    name="Observed xGD/60",
                    line=dict(color="#2c7bb6", width=2.5),
                    error_y=dict(
                        type="data",
                        array=bucket_df["xg_diff_se"].tolist(),
                        visible=True,
                        color="#2c7bb6",
                        thickness=1.2,
                    ),
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
                fig.add_vline(x=45, line_dash="dot", line_color="gray", opacity=0.4, annotation_text="Avg shift")

                ## color background by positive/negative zone
                fig.update_layout(
                    xaxis_title="Shift age (seconds)",
                    yaxis_title="xG diff per 60",
                    height=380,
                    hovermode="x unified",
                    title=f"Decay Curve: {selected_label}",
                )

                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("TOI (min)", f"{stats[stats['home_skaters']==combo]['toi_min'].values[0]:.1f}")
                col2.metric("Avg shift age", f"{stats[stats['home_skaters']==combo]['avg_shift_age'].values[0]:.1f}s")
                col3.metric("Stints", int(stats[stats['home_skaters']==combo]['n_stints'].values[0]))

    ## ---------------------------------------------------------------------------
    ## tab 2: overuse report
    ## ---------------------------------------------------------------------------

    with tab2:
        st.subheader("Lines Showing Significant Decay Between Early and Late Shift")
        st.caption("Compares xGD/60 in first 30s of shift vs after 45s. Delta > 1 xGD/60 is flagged.")

        try:
            overuse_df = get_overused_lines(season, min_toi_min=min_toi)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        if overuse_df.empty:
            st.info("No line combinations met the threshold.")
            return

        overuse_df["line"] = overuse_df["home_skaters"].apply(lambda c: _format_combo(c, name_map))
        display = overuse_df[["line", "toi_min", "early_xgd60", "late_xgd60", "decay_delta", "overuse_flag"]].rename(columns={
            "line": "Line",
            "toi_min": "TOI (min)",
            "early_xgd60": "Early xGD/60 (0-30s)",
            "late_xgd60": "Late xGD/60 (>45s)",
            "decay_delta": "Decay",
            "overuse_flag": "Flagged",
        })

        for col in ["TOI (min)", "Early xGD/60 (0-30s)", "Late xGD/60 (>45s)", "Decay"]:
            display[col] = display[col].round(2)

        flagged = display[display["Flagged"]].drop(columns=["Flagged"])
        not_flagged = display[~display["Flagged"]].drop(columns=["Flagged"])

        if not flagged.empty:
            st.error(f"{len(flagged)} lines flagged for overuse")
            st.dataframe(flagged, use_container_width=True, hide_index=True)
        else:
            st.success("No lines flagged for overuse at this TOI threshold.")

        with st.expander("Show all lines"):
            st.dataframe(not_flagged, use_container_width=True, hide_index=True)
