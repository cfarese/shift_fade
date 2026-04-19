"""
FastAPI server for the Shift Decay RAPM dashboard.

Run from project root:
    uvicorn dashboard.web.server:app --reload --port 8080

Then open: http://localhost:8080
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from config.settings import cfg
from src.ingestion.roster import get_cached_names
from src.models.player_decay import _get_player_index, get_league_decay_summary
from src.models.line_analysis import get_line_stats, get_overused_lines

TEMPLATE = (Path(__file__).parent / "template.html").read_text()
AGES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
app = FastAPI()


def _pct_ranks(values: list[float], higher_is_better: bool = True) -> list[int]:
    import numpy as np
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n <= 1:
        return [50] * n
    order = arr.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    pcts = ranks / (n - 1) * 100
    if not higher_is_better:
        pcts = 100 - pcts
    return [int(round(p)) for p in pcts]


def _break_even(base: float, decay: float) -> int | None:
    if decay < -0.00002 and base > 0:
        return int(round(-base * 42 / (decay * 42)))
    if decay > 0.00002 and base < 0:
        return int(round(-base * 42 / (decay * 42)))
    return None


def _available_seasons() -> list[str]:
    return [
        s for s in cfg.seasons
        if (cfg.paths.processed / f"rapm_results_{s}.parquet").exists()
    ]


def _build_data(season: str) -> dict:
    import pandas as pd
    import numpy as np

    rapm_path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not rapm_path.exists():
        return {
            "players": [], "lines": [],
            "league_p25": [], "league_p75": [], "league_med": [],
            "seasons": _available_seasons(), "current_season": season,
        }

    rapm = pd.read_parquet(rapm_path)
    name_map = get_cached_names()

    # ── PLAYERS ───────────────────────────────────────────────────────────────
    players = []
    for _, row in rapm.iterrows():
        base  = float(row["rapm_base"])
        decay = float(row["rapm_decay"])
        # obs from RAPM model (same formula as design's fake data, minus noise)
        obs   = [round(base * 42 + decay * t * 42, 2) for t in AGES]
        early = round((obs[0]+obs[1]+obs[2]+obs[3]) / 4, 2)
        late  = round((obs[5]+obs[6]+obs[7]) / 3, 2)
        drop  = round(late - early, 2)
        pid   = int(row["player_id"])
        info  = name_map.get(pid, {})
        players.append({
            "id":         pid,
            "name":       str(row.get("player_name", info.get("name", f"Player_{pid}"))),
            "team":       str(row.get("team", info.get("team", "?"))),
            "pos":        str(info.get("position", "F")),
            "base_rapm":  round(base, 4),
            "decay_coef": round(decay, 6),
            "toi_min":    round(float(row["toi_5v5"]), 1),
            "stints":     0,
            "breakeven":  _break_even(base, decay),
            "obs":        obs,
            "early":      early,
            "late":       late,
            "drop":       drop,
            "flagged":    bool(row["overuse_flag"]),
        })

    if players:
        pct_rapm  = _pct_ranks([p["base_rapm"]  for p in players], True)
        pct_decay = _pct_ranks([p["decay_coef"] for p in players], True)
        pct_early = _pct_ranks([p["early"]      for p in players], True)
        pct_late  = _pct_ranks([p["late"]       for p in players], True)
        pct_drop  = _pct_ranks([p["drop"]       for p in players], False)
        pct_toi   = _pct_ranks([p["toi_min"]    for p in players], True)
        for i, p in enumerate(players):
            p["pct_rapm"]  = pct_rapm[i]
            p["pct_decay"] = pct_decay[i]
            p["pct_early"] = pct_early[i]
            p["pct_late"]  = pct_late[i]
            p["pct_drop"]  = pct_drop[i]
            p["pct_toi"]   = pct_toi[i]

    try:
        idx = _get_player_index(season)
        cnt = idx.groupby("player_id").size().to_dict()
        for p in players:
            p["stints"] = int(cnt.get(p["id"], 0))
    except Exception:
        pass

    # ── LEAGUE BANDS ─────────────────────────────────────────────────────────
    league_p25 = [-1.9 - 0.006*t for t in AGES]
    league_p75 = [ 1.9 - 0.006*t for t in AGES]
    league_med = [0.0 for _ in AGES]
    try:
        summary = get_league_decay_summary(season, min_toi_sec=600)
        if not summary.empty:
            ep25 = float(summary["early_xgd60"].quantile(0.25))
            ep75 = float(summary["early_xgd60"].quantile(0.75))
            emed = float(summary["early_xgd60"].median())
            lp25 = float(summary["late_xgd60"].quantile(0.25))
            lp75 = float(summary["late_xgd60"].quantile(0.75))
            lmed = float(summary["late_xgd60"].median())

            def interp(a: float, b: float, t: float) -> float:
                frac = max(0.0, min(1.0, (t - 10.0) / 50.0))
                return round(a + (b - a) * frac, 3)

            league_p25 = [interp(ep25, lp25, t) for t in AGES]
            league_p75 = [interp(ep75, lp75, t) for t in AGES]
            league_med = [interp(emed, lmed, t) for t in AGES]
    except Exception:
        pass

    # ── LINES ─────────────────────────────────────────────────────────────────
    lines = []
    try:
        stats   = get_line_stats(season, min_toi_sec=60)
        overuse = get_overused_lines(season, min_toi_min=1.0)

        ou_lookup: dict[tuple, dict] = {}
        if not overuse.empty:
            for _, row in overuse.iterrows():
                key = tuple(sorted(int(x) for x in row["home_skaters"]))
                ou_lookup[key] = {
                    "early_xgd": round(float(row.get("early_xgd60", 0)), 2),
                    "late_xgd":  round(float(row.get("late_xgd60",  0)), 2),
                    "decay":     round(float(row.get("decay_delta",  0)), 2),
                    "flagged":   bool(row.get("overuse_flag", False)),
                }

        for _, row in stats.head(200).iterrows():
            skaters = list(row["home_skaters"])
            key     = tuple(sorted(int(x) for x in skaters))
            label   = " / ".join(
                name_map.get(int(pid), {}).get("name", str(pid)) for pid in skaters
            )
            team_id = next(
                (name_map.get(int(pid), {}).get("team") for pid in skaters
                 if name_map.get(int(pid), {}).get("team")),
                "?"
            )
            ou      = ou_lookup.get(key, {})
            xgd60   = round(float(row["xg_diff_per60"]), 2)
            e_xgd   = ou.get("early_xgd", round(xgd60 + 1.0, 2))
            l_xgd   = ou.get("late_xgd",  round(xgd60 - 0.5, 2))
            lines.append({
                "players":   label,
                "team":      team_id,
                "toi_min":   round(float(row["toi_min"]), 1),
                "xgd60":     xgd60,
                "xgf":       round(float(row.get("xgf_pct", 0.5)) * 100, 1),
                "cf":        round(float(row.get("cf_pct",  0.5)) * 100, 1),
                "avg_shift": round(float(row.get("avg_shift_age", 40)), 1),
                "stints":    int(row.get("n_stints", 0)),
                "early_xgd": e_xgd,
                "late_xgd":  l_xgd,
                "decay":     round(l_xgd - e_xgd, 2),
                "flagged":   ou.get("flagged", False),
            })
    except Exception as exc:
        print(f"[lines] {exc}")

    return {
        "players":        players,
        "lines":          lines,
        "league_p25":     league_p25,
        "league_p75":     league_p75,
        "league_med":     league_med,
        "seasons":        _available_seasons(),
        "current_season": season,
    }


# cache per season so repeated loads are instant
_CACHE: dict[str, dict] = {}


def _get_data(season: str) -> dict:
    if season not in _CACHE:
        _CACHE[season] = _build_data(season)
    return _CACHE[season]


@app.get("/", response_class=HTMLResponse)
async def index(season: str | None = None):
    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")

    data      = _get_data(season)
    data_json = json.dumps(data, default=str)
    html      = TEMPLATE.replace("{{DATA_JSON}}", data_json)
    return HTMLResponse(html)
