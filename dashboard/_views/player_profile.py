## Full player profile view -- rendered when a player is selected from any table.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ingestion.roster import get_cached_names
from src.models.player_decay import get_player_empirical_decay, get_league_decay_summary, get_league_curve_bands
from src.models.rapm_reader import compute_decay_curve, get_break_even_second
from dashboard._views._theme import (
    apply_chart_theme, add_league_band, pct_badge_html, pct_rank, BLUE, RED,
)


def _fmt(v, d=2, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    s = f"{abs(v):.{d}f}"
    return ("+" if v >= 0 else "-") + s if sign else s


def render(player_id: int, season: str, rapm: pd.DataFrame | None):
    name_map  = get_cached_names()
    info      = name_map.get(player_id, {})
    full_name = info.get("name", f"Player_{player_id}")
    team      = info.get("team", "")
    position  = info.get("position", "")

    # ── back button + header ──────────────────────────────────────────────────
    if st.button("← Back"):
        st.session_state.selected_player_id = None
        st.session_state.pop("_player_page", None)
        st.rerun()

    rapm_row = rapm[rapm["player_id"] == player_id].iloc[0] if rapm is not None and not rapm[rapm["player_id"] == player_id].empty else None
    flagged  = bool(rapm_row["overuse_flag"]) if rapm_row is not None else False

    flag_html = (
        f' <span style="background:#fee2e2;color:#dc2626;border-radius:3px;padding:2px 8px;'
        f'font-size:11px;font-weight:700;vertical-align:middle">⚠ Overuse Flagged</span>'
        if flagged else ""
    )
    team_tag = (
        f'<span style="background:#f0f4ff;color:#1d4ed8;border-radius:3px;padding:2px 8px;'
        f'font-size:11px;font-weight:600">{team}</span> '
        if team else ""
    )
    st.markdown(
        f'<h1 style="font-size:28px;font-weight:700;letter-spacing:-0.02em;margin-bottom:6px">'
        f'{full_name}{flag_html}</h1>'
        f'<div style="font-size:13px;color:#888;margin-bottom:20px">'
        f'{team_tag}{position}{"&nbsp;·&nbsp;" if position else ""}'
        f'Season {season[:4]}–{season[4:]}</div>',
        unsafe_allow_html=True,
    )

    # ── empirical decay data ──────────────────────────────────────────────────
    try:
        bucket_df = get_player_empirical_decay(season, player_id, bucket_size=10, min_toi_sec=20)
    except FileNotFoundError:
        st.warning("Stint data not found. Run the pipeline first.")
        return

    bands = get_league_curve_bands(season, min_toi_sec=600)

    # ── decay curve + shift breakdown side-by-side ───────────────────────────
    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:2px">'
            'On-Ice xGD/60 by Shift Age</div>'
            '<div style="font-size:12px;color:#888;font-style:italic;margin-bottom:4px">'
            'Observed. Shaded band = league P25–P75. Dashed = league median.</div>',
            unsafe_allow_html=True,
        )

        if bucket_df.empty:
            st.info("Not enough data at different shift ages to build a curve.")
        else:
            fig = go.Figure()

            if bands:
                add_league_band(fig, bands["buckets"], bands["p25"], bands["med"], bands["p75"])

            # player error band
            fig.add_trace(go.Scatter(
                x=list(bucket_df["shift_bucket"]) + list(bucket_df["shift_bucket"])[::-1],
                y=list(bucket_df["xg_diff_per60"] + bucket_df["se"]) +
                  list(bucket_df["xg_diff_per60"] - bucket_df["se"])[::-1],
                fill="toself",
                fillcolor="rgba(29,78,216,0.10)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter(
                x=bucket_df["shift_bucket"],
                y=bucket_df["xg_diff_per60"],
                mode="lines+markers",
                line=dict(color=BLUE, width=2.5),
                marker=dict(size=8, color=BLUE, line=dict(color="#fff", width=1.5)),
                name=full_name,
                customdata=bucket_df[["toi_min", "n_stints"]].values,
                hovertemplate=(
                    "<b>Shift age: %{x}s</b><br>"
                    "xGD/60: %{y:.2f}<br>"
                    "TOI in bucket: %{customdata[0]:.1f} min<br>"
                    "Stints: %{customdata[1]}<extra></extra>"
                ),
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
            fig.add_vline(x=45, line_dash="dot", line_color="#bbb", line_width=1,
                          annotation_text="avg shift", annotation_position="top left",
                          annotation_font=dict(size=10, color="#aaa"))

            fig.update_layout(
                xaxis_title="Shift age (seconds)",
                yaxis_title="Observed xGD/60",
                legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
            )
            apply_chart_theme(fig, height=380)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_stats:
        st.markdown(
            '<div style="font-size:13px;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.06em;color:#888;margin-bottom:16px">Shift Breakdown</div>',
            unsafe_allow_html=True,
        )
        if not bucket_df.empty:
            early_rows = bucket_df[bucket_df["shift_bucket"] <= 20]["xg_diff_per60"]
            mid_rows   = bucket_df[(bucket_df["shift_bucket"] > 20) & (bucket_df["shift_bucket"] <= 40)]["xg_diff_per60"]
            late_rows  = bucket_df[bucket_df["shift_bucket"] > 40]["xg_diff_per60"]

            early_val = float(early_rows.mean()) if not early_rows.empty else np.nan
            mid_val   = float(mid_rows.mean())   if not mid_rows.empty   else np.nan
            late_val  = float(late_rows.mean())   if not late_rows.empty  else np.nan

            def _stat_block(label, val):
                v = f"{val:+.2f}" if not np.isnan(val) else "N/A"
                return (
                    f'<div style="margin-bottom:18px">'
                    f'<div style="font-size:11px;color:#999;margin-bottom:3px">{label}</div>'
                    f'<div style="font-size:22px;font-weight:700;font-family:\'IBM Plex Mono\',monospace;'
                    f'letter-spacing:-0.02em;color:#111">{v}</div>'
                    f'</div>'
                )

            html = (
                _stat_block("Fresh (0–30s)", early_val)
                + _stat_block("Mid (30–45s)", mid_val)
                + _stat_block("Late (45s+)", late_val)
                + '<hr style="border:none;border-top:1px solid #f0f0f0;margin:12px 0"/>'
            )
            if not np.isnan(early_val) and not np.isnan(late_val):
                drop = late_val - early_val
                html += _stat_block("Early→late drop", drop)

            total_toi = float(bucket_df["toi_min"].sum())
            total_stints = int(bucket_df["n_stints"].sum())
            html += (
                f'<div style="margin-bottom:12px">'
                f'<div style="font-size:11px;color:#999;margin-bottom:3px">Total 5v5 TOI</div>'
                f'<div style="font-size:22px;font-weight:700;font-family:\'IBM Plex Mono\',monospace">'
                f'{total_toi:.1f} <span style="font-size:13px;font-weight:400">min</span></div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size:11px;color:#999;margin-bottom:3px">Total stints</div>'
                f'<div style="font-size:22px;font-weight:700;font-family:\'IBM Plex Mono\',monospace">'
                f'{total_stints}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    # ── percentile badges vs league ───────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    try:
        summary = get_league_decay_summary(season, min_toi_sec=600)
    except Exception:
        summary = pd.DataFrame()

    if not summary.empty and not bucket_df.empty:
        early_rows2 = bucket_df[bucket_df["shift_bucket"] <= 20]["xg_diff_per60"]
        late_rows2  = bucket_df[bucket_df["shift_bucket"] > 40]["xg_diff_per60"]
        p_early = float(early_rows2.mean()) if not early_rows2.empty else np.nan
        p_late  = float(late_rows2.mean())  if not late_rows2.empty  else np.nan

        badges_html = '<div style="display:flex;flex-wrap:wrap;gap:0;margin-bottom:20px">'
        if not np.isnan(p_early):
            pct_e = float((summary["early_xgd60"] < p_early).mean() * 100)
            badges_html += pct_badge_html(pct_e, "Early xGD/60")
        if not np.isnan(p_late):
            pct_l = float((summary["late_xgd60"] < p_late).mean() * 100)
            badges_html += pct_badge_html(pct_l, "Late xGD/60")
        if not np.isnan(p_early) and not np.isnan(p_late):
            drop = p_late - p_early
            pct_d = float((summary["decay_delta"] > drop).mean() * 100)
            badges_html += pct_badge_html(pct_d, "Durability")

        if rapm_row is not None:
            rapm_all  = rapm.copy() if rapm is not None else pd.DataFrame()
            if not rapm_all.empty and "rapm_base" in rapm_all.columns:
                pct_r = float((rapm_all["rapm_base"] < rapm_row["rapm_base"]).mean() * 100)
                badges_html += pct_badge_html(pct_r, "Base RAPM")

        badges_html += "</div>"

        st.markdown(
            '<div style="font-size:13px;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.06em;color:#888;margin-bottom:12px">Context vs. League</div>'
            + badges_html,
            unsafe_allow_html=True,
        )

    # ── shift-age breakdown table ─────────────────────────────────────────────
    if not bucket_df.empty:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:8px">'
            'Shift-Age Breakdown Table</div>',
            unsafe_allow_html=True,
        )
        xgd_pcts = pct_rank(bucket_df["xg_diff_per60"], higher_is_better=True)
        rows_html = ""
        for i, (_, row) in enumerate(bucket_df.iterrows()):
            pct  = float(xgd_pcts.iloc[i])
            bg   = pct_bg(pct)
            from dashboard._views._theme import pct_fg as _pfg
            fg   = _pfg(pct)
            rows_html += (
                f'<tr>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                f'padding:7px 10px;border-bottom:1px solid #f0f0f0">{int(row["shift_bucket"])}s</td>'
                f'<td style="background:{bg};color:{fg};font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:12px;text-align:right;padding:7px 10px;border-bottom:1px solid #f0f0f0">'
                f'{row["xg_diff_per60"]:+.2f}</td>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                f'text-align:right;padding:7px 10px;border-bottom:1px solid #f0f0f0">'
                f'{row["toi_min"]:.1f}</td>'
                f'<td style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                f'text-align:right;padding:7px 10px;border-bottom:1px solid #f0f0f0">'
                f'{int(row["n_stints"])}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:#fff;border:1px solid #e8e8e8;border-radius:6px;overflow:hidden">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr>'
            f'<th style="text-align:left;padding:8px 10px;font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:#666;border-bottom:2px solid #e5e5e5">'
            f'Shift age</th>'
            f'<th style="text-align:right;padding:8px 10px;font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:#666;border-bottom:2px solid #e5e5e5">'
            f'xGD/60</th>'
            f'<th style="text-align:right;padding:8px 10px;font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:#666;border-bottom:2px solid #e5e5e5">'
            f'TOI (min)</th>'
            f'<th style="text-align:right;padding:8px 10px;font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:#666;border-bottom:2px solid #e5e5e5">'
            f'Stints</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    # ── RAPM section ──────────────────────────────────────────────────────────
    if rapm_row is not None:
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#111;margin-bottom:4px">'
            'RAPM Model (Ridge Regression)</div>',
            unsafe_allow_html=True,
        )
        r = rapm_row
        if abs(r["rapm_decay"]) < 0.001:
            st.caption("Coefficients near zero — model needs more data (300+ games).")

        m1, m2, m3 = st.columns(3)
        m1.metric("Base RAPM", f"{r['rapm_base']:.4f}")
        m2.metric("Decay coef", f"{r['rapm_decay']:.6f}")
        bev = get_break_even_second(r["rapm_base"], r["rapm_decay"])
        m3.metric("Model break-even", f"{bev}s" if bev else "Never")

        buckets, values = compute_decay_curve(r["rapm_base"], r["rapm_decay"], max_seconds=90)
        fig2 = go.Figure()
        if bands:
            add_league_band(fig2, bands["buckets"], bands["p25"], bands["med"], bands["p75"])
        fig2.add_trace(go.Scatter(
            x=buckets, y=values, mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=6, color=BLUE, line=dict(color="#fff", width=1)),
            name="RAPM model",
            showlegend=True,
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
        fig2.add_vline(x=45, line_dash="dot", line_color="#bbb", line_width=1)
        fig2.update_layout(
            xaxis_title="Shift age (seconds)",
            yaxis_title="Projected xGD/60 (model)",
            legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
        )
        apply_chart_theme(fig2, height=320)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
