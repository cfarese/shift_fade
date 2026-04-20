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
from fastapi.responses import JSONResponse

from src.ingestion.roster import get_cached_names
from src.models.player_decay import _get_player_index, get_league_decay_summary, get_player_rolling_decay
from src.models.line_analysis import get_line_stats, get_overused_lines, load_stints

TEMPLATE = (Path(__file__).parent / "template.html").read_text()
AGES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
MIN_TOI_MIN = 10.0        # players with less 5v5 TOI are excluded from all views
OVERUSE_TOI_MIN = 50.0    # minimum TOI to be considered for overuse flag
OVERUSE_DECAY_PCT = 15    # bottom N-th percentile of decay among eligible players
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
    """Seconds into a shift when projected xGD/60 crosses zero.
    Only meaningful when a player starts positive (base > 0) and fades (decay < 0).
    Returns None ('Never') for everyone else — positive decay or negative base."""
    if decay < -0.00002 and base > 0:
        return int(round(-base / decay))
    return None


def _available_seasons() -> list[str]:
    """Return seasons that have at least stints data available."""
    out = []
    for s in cfg.seasons:
        has_rapm   = (cfg.paths.processed / f"rapm_results_{s}.parquet").exists()
        has_stints = (cfg.paths.processed / f"stints_{s}.parquet").exists()
        if has_rapm or has_stints:
            out.append(s)
    return out


def _has_rapm(season: str) -> bool:
    return (cfg.paths.processed / f"rapm_results_{season}.parquet").exists()


def _build_data(season: str) -> dict:
    import pandas as pd
    import numpy as np

    rapm_path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not rapm_path.exists():
        # stints may exist — still serve lines and empty player list with a flag
        return {
            "players": [], "lines": _build_lines(season),
            "league_p25": [], "league_p75": [], "league_med": [],
            "seasons": _available_seasons(), "current_season": season,
            "rapm_ready": False,
        }

    rapm = pd.read_parquet(rapm_path)
    # filter out players with insufficient playing time before any analysis
    rapm = rapm[rapm["toi_5v5"] >= MIN_TOI_MIN].copy()
    name_map = get_cached_names()

    # ── PLAYERS ───────────────────────────────────────────────────────────────
    players = []
    for _, row in rapm.iterrows():
        base  = float(row["rapm_base"])
        decay = float(row["rapm_decay"])
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
            "flagged":    False,  # recomputed below from actual distribution
        })

    # recompute overuse flag: bottom OVERUSE_DECAY_PCT of decay among players
    # with enough TOI. The R model's threshold is in wrong units so we ignore it.
    if players:
        eligible_decays = [p["decay_coef"] for p in players if p["toi_min"] >= OVERUSE_TOI_MIN]
        if eligible_decays:
            threshold = float(np.percentile(eligible_decays, OVERUSE_DECAY_PCT))
            for p in players:
                p["flagged"] = p["toi_min"] >= OVERUSE_TOI_MIN and p["decay_coef"] <= threshold

    if players:
        pct_rapm  = _pct_ranks([p["base_rapm"]  for p in players], True)
        pct_decay = _pct_ranks([p["decay_coef"] for p in players], True)
        pct_early = _pct_ranks([p["early"]      for p in players], True)
        pct_late  = _pct_ranks([p["late"]       for p in players], True)
        pct_drop  = _pct_ranks([p["drop"]       for p in players], True)
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

        idx2 = idx.copy()
        idx2["shift_bucket"] = (idx2["shift_age"] // 10) * 10
        idx2["shift_bucket"] = idx2["shift_bucket"].clip(upper=90)
        bstats = idx2.groupby(["player_id", "shift_bucket"]).agg(
            toi_sec=("duration", "sum"),
            n_stints=("duration", "count"),
        ).reset_index()
        qual   = bstats[(bstats["toi_sec"] >= 120) & (bstats["n_stints"] >= 5)]
        bcnt   = qual.groupby("player_id").size().to_dict()

        for p in players:
            p["stints"]          = int(cnt.get(p["id"], 0))
            p["n_decay_buckets"] = int(bcnt.get(p["id"], 0))
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

    lines = _build_lines(season)

    return {
        "players":        players,
        "lines":          lines,
        "league_p25":     league_p25,
        "league_p75":     league_p75,
        "league_med":     league_med,
        "seasons":        _available_seasons(),
        "current_season": season,
        "rapm_ready":     True,
    }


def _build_lines(season: str) -> list[dict]:
    import pandas as pd
    name_map = get_cached_names()
    lines: list[dict] = []
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
            ou    = ou_lookup.get(key, {})
            xgd60 = round(float(row["xg_diff_per60"]), 2)
            e_xgd = ou.get("early_xgd", round(xgd60 + 1.0, 2))
            l_xgd = ou.get("late_xgd",  round(xgd60 - 0.5, 2))
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
    return lines


# cache per season so repeated loads are instant
_CACHE: dict[str, dict] = {}


def _get_data(season: str) -> dict:
    if season not in _CACHE:
        _CACHE[season] = _build_data(season)
    return _CACHE[season]


@app.get("/player/{player_id}/decay")
async def player_decay(player_id: int, season: str | None = None):
    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")
    try:
        df = get_player_rolling_decay(season, player_id, window=20, step=5, min_toi_sec=60, min_stint_sec=20)
        if df.empty:
            return JSONResponse({"buckets": [], "values": [], "max_age": 80})
        return JSONResponse({
            "buckets": df["age"].tolist(),
            "values":  df["xgd60"].tolist(),
            "max_age": int(df["age"].max()),
        })
    except Exception as exc:
        return JSONResponse({"buckets": [], "values": [], "se": [], "error": str(exc)})


## per-season cache of the exploded 5v5 stint frame with original row metadata
_STINT_META_CACHE: dict[str, "pd.DataFrame"] = {}


def _get_stint_meta(season: str) -> "pd.DataFrame":
    import pandas as pd
    import numpy as np

    if season in _STINT_META_CACHE:
        return _STINT_META_CACHE[season]

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()
    ev = ev.reset_index(drop=True)
    ev.index.name = "stint_idx"
    ev = ev.reset_index()

    keep = ["stint_idx", "game_id", "period", "start_sec", "duration",
            "home_shift_age", "away_shift_age", "zone_start", "score_state",
            "xg_for", "xg_against", "corsi_for", "corsi_against",
            "home_skaters", "away_skaters"]
    ev = ev[keep]

    # explode home side
    home = ev.explode("home_skaters").copy()
    home["player_id"] = home["home_skaters"].astype(int)
    home["is_home"]   = True
    home["shift_age"] = home["home_shift_age"]
    home["p_xgf"]     = home["xg_for"]
    home["p_xga"]     = home["xg_against"]

    # explode away side
    away = ev.explode("away_skaters").copy()
    away["player_id"] = away["away_skaters"].astype(int)
    away["is_home"]   = False
    away["shift_age"] = away["away_shift_age"]
    away["p_xgf"]     = away["xg_against"]  # flip so "for" = player's team
    away["p_xga"]     = away["xg_for"]

    cols = ["stint_idx", "player_id", "is_home", "game_id", "period",
            "start_sec", "duration", "shift_age", "zone_start", "score_state",
            "p_xgf", "p_xga"]
    combined = pd.concat([home[cols], away[cols]], ignore_index=True)
    combined["xgd"]   = combined["p_xgf"] - combined["p_xga"]

    _STINT_META_CACHE[season] = combined
    return combined


@app.get("/player/{player_id}/stints")
async def player_stints(player_id: int, season: str | None = None):
    import numpy as np

    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")
    try:
        meta   = _get_stint_meta(season)
        subset = meta[meta["player_id"] == player_id].sort_values(
            ["game_id", "period", "start_sec"]
        ).reset_index(drop=True)

        if subset.empty:
            return JSONResponse({"stints": [], "total": 0})

        # compute where each sub-stint ends on the clock
        subset["end_sec"] = subset["start_sec"] + subset["duration"]

        # a new shift starts when: different game, different period,
        # OR the gap between this sub-stint's start and the previous sub-stint's
        # end is more than 15 seconds (small gaps ≤15s are treated as same shift)
        GAP_TOLERANCE = 15
        prev_end    = subset["end_sec"].shift(1)
        prev_game   = subset["game_id"].shift(1)
        prev_period = subset["period"].shift(1)

        is_new = (
            (subset["game_id"]   != prev_game)   |
            (subset["period"]    != prev_period)  |
            ((subset["start_sec"] - prev_end) > GAP_TOLERANCE)
        )
        is_new.iloc[0] = True
        subset["shift_id"] = is_new.cumsum().astype(int)

        # aggregate sub-stints within each shift.
        # duration = end of last segment - start of first (includes tiny gaps).
        def first_zone(zones):
            for z in zones:
                if str(z) in ("O", "N", "D"):
                    return str(z)
            return "N"

        agg = (
            subset.groupby("shift_id", sort=False)
            .agg(
                game_id   =("game_id",    "first"),
                period    =("period",     "first"),
                start_sec =("start_sec",  "first"),
                end_sec   =("end_sec",    "last"),
                score     =("score_state","first"),
                p_xgf     =("p_xgf",      "sum"),
                p_xga     =("p_xga",      "sum"),
                n_segs    =("duration",   "count"),
                zones     =("zone_start", list),
            )
            .reset_index(drop=True)
        )
        agg["duration"] = agg["end_sec"] - agg["start_sec"]
        agg["zone"] = agg["zones"].apply(first_zone)
        agg["xgd"]  = agg["p_xgf"] - agg["p_xga"]

        records = []
        for _, r in agg.iterrows():
            records.append({
                "game_id":   int(r["game_id"]),
                "period":    int(r["period"]),
                "start_sec": int(r["start_sec"]),
                "duration":  int(r["duration"]),
                "zone":      r["zone"],
                "score":     int(r["score"]),
                "xgf":       round(float(r["p_xgf"]), 4),
                "xga":       round(float(r["p_xga"]), 4),
                "xgd":       round(float(r["xgd"]),   4),
                "n_segs":    int(r["n_segs"]),
            })
        return JSONResponse({"stints": records, "total": len(records)})
    except Exception as exc:
        return JSONResponse({"stints": [], "total": 0, "error": str(exc)})


@app.get("/", response_class=HTMLResponse)
async def index(season: str | None = None):
    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")

    data      = _get_data(season)
    data_json = json.dumps(data, default=str)
    html      = TEMPLATE.replace("{{DATA_JSON}}", data_json)
    return HTMLResponse(html)
