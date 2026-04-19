## Team Report page -- full breakdown of a team's deployment patterns.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from src.models.line_analysis import get_overused_lines
from src.ingestion.roster import get_cached_names
from dashboard.components.clickable_table import clickable_player_table
from dashboard._views._theme import (
    apply_chart_theme, add_league_band, pct_bg, pct_fg, pct_rank, BLUE, RED,
)
from src.models.player_decay import get_league_curve_bands


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

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players in sample", len(team_rapm))
    m2.metric("Overuse flags", int(team_rapm["overuse_flag"].sum()))
    m3.metric("Team avg decay", f"{team_rapm['rapm_decay'].mean():.6f}")
    m4.metric(
        "Avg break-even",
        f"{team_rapm['break_even_sec'].mean():.0f}s"
        if team_rapm["break_even_sec"].notna().any()
        else "N/A",
    )

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # ── decay curves overlay + break-even bars side-by-side ──────────────────
    col_curves, col_be = st.columns([3, 2])

    with col_curves:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Decay Curves — All Skaters</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:4px">'
            'Each line = one player. Red = overuse flagged. Shaded = league P25–P75.</div>',
            unsafe_allow_html=True,
        )
        bands = get_league_curve_bands(season, min_toi_sec=600)
        fig = go.Figure()

        if bands:
            add_league_band(fig, bands["buckets"], bands["p25"], bands["med"], bands["p75"])

        for _, row in team_rapm.iterrows():
            buckets, values = compute_decay_curve(row["rapm_base"], row["rapm_decay"], max_seconds=75)
            color   = RED   if row["overuse_flag"] else BLUE
            opacity = 0.9   if row["overuse_flag"] else 0.4
            fig.add_trace(go.Scatter(
                x=buckets, y=values, mode="lines",
                name=row["player_name"],
                line=dict(color=color, width=1.5),
                opacity=opacity,
                hovertemplate=f"{row['player_name']}<br>%{{y:.2f}} xGD/60 at %{{x}}s<extra></extra>",
                showlegend=False,
            ))

        fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
        fig.add_vline(x=45, line_dash="dot", line_color="#bbb", line_width=1,
                      annotation_text="avg shift", annotation_font=dict(size=10, color="#aaa"))
        fig.update_layout(
            xaxis_title="Shift age (seconds)",
            yaxis_title="Projected xGD/60",
            legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
        )
        apply_chart_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_be:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'Break-even Distribution</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
            'Shift age where each player projects to 0 xGD/60.</div>',
            unsafe_allow_html=True,
        )
        be_rows = team_rapm.sort_values("break_even_sec", na_position="last")
        pct_decay_ranks = pct_rank(team_rapm["rapm_decay"], higher_is_better=True)
        pct_map = dict(zip(team_rapm["player_id"], pct_decay_ranks))

        bars_html = ""
        for _, r in be_rows.iterrows():
            name  = r["player_name"]
            pid   = int(r["player_id"])
            be    = r["break_even_sec"]
            pct   = float(pct_map.get(pid, 50))
            be_str = f"{int(be)}s" if pd.notna(be) and be <= 200 else (">200s" if pd.notna(be) else "Never")
            bar_w  = f"{min(100, (be / 120) * 100):.0f}%" if pd.notna(be) else "0%"
            bar_bg = pct_bg(pct) if pd.notna(be) else "#e5e5e5"
            bars_html += (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:9px">'
                f'<button onclick="void(0)" style="font-size:12px;font-weight:600;color:#1d4ed8;'
                f'background:none;border:none;cursor:pointer;min-width:110px;text-align:left;'
                f'padding:0;font-family:\'IBM Plex Sans\',sans-serif">{name}</button>'
                f'<div style="flex:1;height:6px;background:#f0f0f0;border-radius:3px;overflow:hidden">'
                f'<div style="width:{bar_w};height:100%;background:{bar_bg};border-radius:3px"></div>'
                f'</div>'
                f'<span style="font-size:11px;font-family:\'IBM Plex Mono\',monospace;'
                f'color:#888;min-width:44px;text-align:right">{be_str}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="padding-top:8px">{bars_html}</div>',
            unsafe_allow_html=True,
        )

    # ── roster breakdown table ────────────────────────────────────────────────
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
        'Roster Breakdown</div>'
        '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:12px">'
        'Click a player name to open their full profile.</div>',
        unsafe_allow_html=True,
    )

    roster_src = team_rapm.sort_values("rapm_decay").reset_index(drop=True).copy()
    roster_src["pct_rapm"]  = pct_rank(roster_src["rapm_base"],  higher_is_better=True)
    roster_src["pct_decay"] = pct_rank(roster_src["rapm_decay"],  higher_is_better=True)
    roster_src["pct_toi"]   = pct_rank(roster_src["toi_5v5"],    higher_is_better=True)

    comp_rows = [
        {
            "id": int(r["player_id"]),
            "values": [
                r["player_name"],
                f"{r['rapm_base']:.4f}",
                f"{r['rapm_decay']:.6f}",
                f"{r['toi_5v5']:.1f}",
                f"{int(r['break_even_sec'])}s" if pd.notna(r["break_even_sec"]) else "Never",
                "⚠ Yes" if r["overuse_flag"] else "—",
            ],
            "pcts": [
                None,
                float(r["pct_rapm"]),
                float(r["pct_decay"]),
                float(r["pct_toi"]),
                None,
                None,
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

    # ── overused lines ────────────────────────────────────────────────────────
    if stints is not None:
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:8px">'
            'Line Combinations — Overuse Check</div>',
            unsafe_allow_html=True,
        )
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
