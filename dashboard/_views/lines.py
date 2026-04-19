## Lines page -- line combination stats and per-combo decay curves.

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.line_analysis import get_line_decay_by_bucket, get_line_stats, get_overused_lines
from src.ingestion.roster import get_cached_names
from dashboard.components.clickable_table import clickable_player_table
from dashboard._views._theme import apply_chart_theme, pct_rank, BLUE


def _format_combo(skater_ids: tuple, name_map: dict) -> str:
    return " / ".join(name_map.get(pid, {}).get("name", str(pid)) for pid in skater_ids)


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
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Best Lines by xG Differential per 60 (5v5)</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
            'Select a line to view its decay curve. Click any player name to open their profile.</div>',
            unsafe_allow_html=True,
        )

        try:
            stats = get_line_stats(season, min_toi_sec=int(min_toi * 60))
        except FileNotFoundError as e:
            st.error(str(e))
            return

        if stats.empty:
            st.info("No line combinations met the minimum TOI threshold.")
            return

        display_src = stats.head(25).copy()
        display_src["Line"]              = display_src["home_skaters"].apply(lambda c: _format_combo(c, name_map))
        display_src["TOI (min)"]        = display_src["toi_min"].round(1)
        display_src["xGD/60"]           = display_src["xg_diff_per60"].round(2)
        display_src["xGF%"]             = (display_src["xgf_pct"] * 100).round(1)
        display_src["CF%"]              = (display_src["cf_pct"] * 100).round(1)
        display_src["Avg Shift Age (s)"] = display_src["avg_shift_age"].round(1)
        display_src["Stints"]           = display_src["n_stints"]

        # percentile ranks for coloring
        xgd_pcts   = pct_rank(display_src["xg_diff_per60"], higher_is_better=True).tolist()
        xgf_pcts   = pct_rank(display_src["xgf_pct"],       higher_is_better=True).tolist()
        cf_pcts    = pct_rank(display_src["cf_pct"],         higher_is_better=True).tolist()

        comp_rows = [
            {
                "players": [
                    {"id": int(pid), "name": name_map.get(int(pid), {}).get("name", str(pid))}
                    for pid in r["home_skaters"]
                ],
                "values": [
                    f"{r['toi_min']:.1f}",
                    f"{r['xg_diff_per60']:.2f}",
                    f"{r['xgf_pct'] * 100:.1f}",
                    f"{r['cf_pct'] * 100:.1f}",
                    f"{r['avg_shift_age']:.1f}",
                    str(int(r["n_stints"])),
                ],
                "pcts": [None, float(xgd_pcts[i]), float(xgf_pcts[i]), float(cf_pcts[i]), None, None],
            }
            for i, (_, r) in enumerate(display_src.iterrows())
        ]
        clicked = clickable_player_table(
            rows=comp_rows,
            headers=["Line", "TOI (min)", "xGD/60", "xGF%", "CF%", "Avg Shift Age (s)", "Stints"],
            key="lines_top",
        )
        if clicked is not None:
            st.session_state["selected_player_id"] = clicked
            st.rerun()

        ## decay curve for a selected line
        st.markdown("---")
        st.subheader("Shift Decay Curve for Selected Line")

        stats["_label"] = stats["home_skaters"].apply(lambda c: _format_combo(c, name_map))
        line_options    = stats["home_skaters"].tolist()
        line_labels     = stats["_label"].tolist()
        label_to_combo  = dict(zip(line_labels, line_options))

        selected_label = st.selectbox("Select line", line_labels)
        if selected_label:
            combo     = label_to_combo[selected_label]
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
                    line=dict(color=BLUE, width=2.5),
                    marker=dict(size=8, color=BLUE, line=dict(color="#fff", width=1.5)),
                    error_y=dict(
                        type="data",
                        array=bucket_df["xg_diff_se"].tolist(),
                        visible=True,
                        color=BLUE,
                        thickness=1.2,
                    ),
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
                fig.add_vline(x=45, line_dash="dot", line_color="#bbb", line_width=1,
                              annotation_text="avg shift", annotation_font=dict(size=10, color="#aaa"))
                apply_chart_theme(fig, height=360)
                fig.update_layout(title=f"Decay Curve: {selected_label}")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                row_match = stats[stats["home_skaters"] == combo]
                if not row_match.empty:
                    r = row_match.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("TOI (min)", f"{r['toi_min']:.1f}")
                    col2.metric("Avg shift age", f"{r['avg_shift_age']:.1f}s")
                    col3.metric("Stints", int(r["n_stints"]))

                st.caption("View individual player profiles:")
                btn_cols = st.columns(len(combo))
                for col, pid in zip(btn_cols, combo):
                    pname = name_map.get(pid, {}).get("name", str(pid))
                    if col.button(pname, key=f"line_player_{pid}"):
                        st.session_state["selected_player_id"] = pid
                        st.rerun()

    ## ---------------------------------------------------------------------------
    ## tab 2: overuse report
    ## ---------------------------------------------------------------------------

    with tab2:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Lines Showing Significant Decay</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
            'Compares xGD/60 in first 30s vs after 45s. Drop > 1.0 xGD/60 is flagged.</div>',
            unsafe_allow_html=True,
        )

        try:
            overuse_df = get_overused_lines(season, min_toi_min=min_toi)
        except FileNotFoundError as e:
            st.error(str(e))
            return

        if overuse_df.empty:
            st.info("No line combinations met the threshold.")
            return

        for c in ["toi_min", "early_xgd60", "late_xgd60", "decay_delta"]:
            overuse_df[c] = overuse_df[c].round(2)

        flagged     = overuse_df[overuse_df["overuse_flag"]].copy()
        not_flagged = overuse_df[~overuse_df["overuse_flag"]].copy()

        def _overuse_rows(df, key):
            if df.empty:
                return
            early_pcts  = pct_rank(df["early_xgd60"],  higher_is_better=True).tolist()
            late_pcts   = pct_rank(df["late_xgd60"],   higher_is_better=True).tolist()
            decay_pcts  = pct_rank(df["decay_delta"],   higher_is_better=False).tolist()
            rows = [
                {
                    "players": [
                        {"id": int(pid), "name": name_map.get(int(pid), {}).get("name", str(pid))}
                        for pid in r["home_skaters"]
                    ],
                    "values": [
                        f"{r['toi_min']:.2f}",
                        f"{r['early_xgd60']:.2f}",
                        f"{r['late_xgd60']:.2f}",
                        f"{r['decay_delta']:.2f}",
                    ],
                    "pcts": [None, float(early_pcts[i]), float(late_pcts[i]), float(decay_pcts[i])],
                }
                for i, (_, r) in enumerate(df.iterrows())
            ]
            clicked = clickable_player_table(
                rows=rows,
                headers=["Line", "TOI (min)", "Early xGD/60", "Late xGD/60", "Decay"],
                key=key,
            )
            if clicked is not None:
                st.session_state["selected_player_id"] = clicked
                st.rerun()

        if not flagged.empty:
            st.error(f"{len(flagged)} lines flagged for overuse")
            _overuse_rows(flagged, "lines_overuse_flagged")
        else:
            st.success("No lines flagged for overuse at this TOI threshold.")

        with st.expander("Show all lines"):
            _overuse_rows(not_flagged, "lines_overuse_all")
