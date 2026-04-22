"""
FastAPI server for the Shift Decay RAPM dashboard.

Run from project root:
    uvicorn dashboard.web.server:app --reload --port 8080

Then open: http://localhost:8080
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from config.settings import cfg
from fastapi.responses import JSONResponse

from src.ingestion.roster import get_cached_names
from src.models.player_decay import _get_player_index, get_league_decay_summary, get_player_rolling_decay
from src.models.line_analysis import get_forward_line_stats, get_forward_line_overuse, load_stints

TEMPLATE = (Path(__file__).parent / "template.html").read_text()
WEB_DIR = Path(__file__).parent
APP_SOURCE = WEB_DIR / "app.jsx"
APP_BUNDLE = WEB_DIR / "app.bundle.js"
ESBUILD_BIN = Path(__file__).resolve().parent.parent.parent / "node_modules" / ".bin" / "esbuild"
AGES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
MIN_TOI_MIN = 10.0        # players with less 5v5 TOI are excluded from all views
OVERUSE_TOI_MIN = 50.0    # minimum TOI to be considered for overuse flag
OVERUSE_DECAY_PCT = 15    # bottom N-th percentile of decay among eligible players
SHIFT_GAP_TOLERANCE = 15
MIN_QUALIFYING_SHIFT_SECONDS = 10
MIN_QUALIFYING_SHIFTS = 200
app = FastAPI()


def _ensure_app_bundle() -> None:
    if not APP_SOURCE.exists():
        raise FileNotFoundError(f"Web app source not found: {APP_SOURCE}")

    needs_build = (
        not APP_BUNDLE.exists()
        or APP_BUNDLE.stat().st_mtime < APP_SOURCE.stat().st_mtime
        or APP_BUNDLE.stat().st_mtime < (WEB_DIR / "template.html").stat().st_mtime
    )
    if not needs_build:
        return

    if not ESBUILD_BIN.exists():
        raise FileNotFoundError(f"esbuild not found: {ESBUILD_BIN}")

    subprocess.run(
        [
            str(ESBUILD_BIN),
            str(APP_SOURCE),
            "--outfile=" + str(APP_BUNDLE),
            "--format=iife",
            "--target=es2018",
            "--minify",
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )


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
    Returns None ('Never') for everyone else: positive decay or negative base."""
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


def _normalize_team_code(team: str | None, season: str) -> str:
    code = str(team or "?").upper()
    if season >= "20242025" and code == "ARI":
        return "UTA"
    return code


def _window_overlap_stats(
    df,
    start_age: float,
    end_age: float | None = None,
) -> dict[str, float]:
    toi_sec = 0.0
    xg_for = 0.0
    xg_against = 0.0

    for row in df.itertuples(index=False):
        seg_start = float(getattr(row, "shift_age"))
        seg_end = seg_start + float(getattr(row, "duration"))
        win_end = seg_end if end_age is None else float(end_age)
        overlap = min(seg_end, win_end) - max(seg_start, float(start_age))
        if overlap <= 0:
            continue
        duration = float(getattr(row, "duration"))
        frac = overlap / duration if duration > 0 else 0.0
        toi_sec += overlap
        xg_for += float(getattr(row, "p_xgf")) * frac
        xg_against += float(getattr(row, "p_xga")) * frac

    xgd60 = None
    if toi_sec > 0:
        xgd60 = (xg_for - xg_against) / toi_sec * 3600

    return {
        "toi_sec": toi_sec,
        "xgd60": xgd60,
    }


def _build_data(season: str) -> dict:
    import numpy as np
    try:
        idx = _get_stint_meta(season)
    except Exception:
        return {
            "players": [], "lines": _build_lines(season),
            "league_p25": [], "league_p75": [], "league_med": [],
            "seasons": _available_seasons(), "current_season": season,
            "rapm_ready": False,
        }

    if idx.empty:
        return {
            "players": [], "lines": _build_lines(season),
            "league_p25": [], "league_p75": [], "league_med": [],
            "seasons": _available_seasons(), "current_season": season,
            "rapm_ready": False,
        }

    name_map = get_cached_names()
    season_team_map = _get_player_team_map(season)
    shift_counts = _get_player_shift_counts(season)
    qualified_idx = idx[idx["duration"] >= MIN_QUALIFYING_SHIFT_SECONDS].copy()

    # ── PLAYERS ───────────────────────────────────────────────────────────────
    players = []
    for pid, subset in idx.groupby("player_id"):
        pid = int(pid)
        total_toi_sec = float(subset["duration"].sum())
        if total_toi_sec < MIN_TOI_MIN * 60:
            continue

        xgd60 = (float(subset["p_xgf"].sum()) - float(subset["p_xga"].sum())) / total_toi_sec * 3600
        qualified_subset = qualified_idx[qualified_idx["player_id"] == pid].copy()
        early_stats = _window_overlap_stats(qualified_subset, 0, 30)
        mid_stats = _window_overlap_stats(qualified_subset, 30, 45)
        late_stats = _window_overlap_stats(qualified_subset, 45, None)
        early = float(early_stats["xgd60"]) if early_stats["xgd60"] is not None else 0.0
        mid = float(mid_stats["xgd60"]) if mid_stats["xgd60"] is not None else 0.0
        late = float(late_stats["xgd60"]) if late_stats["xgd60"] is not None else 0.0
        drop = late - early
        info  = name_map.get(pid, {})
        counts = shift_counts.get(pid, {})
        qualifying_shifts = int(counts.get("qualifying_shifts", 0))
        total_shifts = int(counts.get("total_shifts", 0))
        players.append({
            "id":         pid,
            "name":       str(info.get("name", f"Player_{pid}")),
            "team":       _normalize_team_code(season_team_map.get(pid) or info.get("team", "?"), season),
            "pos":        str(info.get("position", "F")),
            "overall_xgd": round(xgd60, 2),
            "durability": round(drop, 2),
            "base_rapm": round(xgd60, 2),
            "decay_coef": round(drop, 2),
            "toi_min":    round(total_toi_sec / 60, 1),
            "stints":     total_shifts,
            "qualifying_shifts_10s": qualifying_shifts,
            "eligible_for_graphs": qualifying_shifts >= MIN_QUALIFYING_SHIFTS,
            "breakeven":  None,
            "early":      round(early, 2),
            "mid":        round(mid, 2),
            "late":       round(late, 2),
            "drop":       round(drop, 2),
            "flagged":    False,  # recomputed below from actual distribution
        })

    # bottom OVERUSE_DECAY_PCT of empirical durability among higher-TOI players
    if players:
        eligible_drops = [p["drop"] for p in players if p["toi_min"] >= OVERUSE_TOI_MIN]
        if eligible_drops:
            threshold = float(np.percentile(eligible_drops, OVERUSE_DECAY_PCT))
            for p in players:
                p["flagged"] = p["toi_min"] >= OVERUSE_TOI_MIN and p["drop"] <= threshold

    if players:
        pct_overall = _pct_ranks([p["overall_xgd"] for p in players], True)
        pct_drop  = _pct_ranks([p["drop"]       for p in players], True)
        pct_early = _pct_ranks([p["early"]      for p in players], True)
        pct_mid   = _pct_ranks([p["mid"]        for p in players], True)
        pct_late  = _pct_ranks([p["late"]       for p in players], True)
        pct_toi   = _pct_ranks([p["toi_min"]    for p in players], True)
        for i, p in enumerate(players):
            p["pct_overall"] = pct_overall[i]
            p["pct_early"] = pct_early[i]
            p["pct_mid"]   = pct_mid[i]
            p["pct_late"]  = pct_late[i]
            p["pct_drop"]  = pct_drop[i]
            p["pct_toi"]   = pct_toi[i]
            p["pct_durability"] = pct_drop[i]

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
    league_empirical_curve = _get_league_empirical_curve(season)

    return {
        "players":        players,
        "lines":          lines,
        "league_p25":     league_p25,
        "league_p75":     league_p75,
        "league_med":     league_med,
        "league_empirical_curve": league_empirical_curve,
        "seasons":        _available_seasons(),
        "current_season": season,
        "rapm_ready":     True,
        "min_qualifying_shift_seconds": MIN_QUALIFYING_SHIFT_SECONDS,
        "min_qualifying_shifts": MIN_QUALIFYING_SHIFTS,
    }


def _build_lines(season: str) -> list[dict]:
    name_map = get_cached_names()
    pos_map  = {int(pid): str(info.get("position", "F")) for pid, info in name_map.items()}
    lines: list[dict] = []
    try:
        stats  = get_forward_line_stats(season, pos_map, min_toi_sec=600)
        overuse = get_forward_line_overuse(season, pos_map, min_toi_min=10.0)

        ou_lookup: dict[tuple, dict] = {}
        if not overuse.empty:
            for _, row in overuse.iterrows():
                key = tuple(row["fwd_line"])
                ou_lookup[key] = {
                    "early_xgd": round(float(row.get("early_xgd60", 0)), 2),
                    "late_xgd":  round(float(row.get("late_xgd60",  0)), 2),
                    "decay":     round(float(row.get("decay_delta",  0)), 2),
                    "flagged":   bool(row.get("overuse_flag", False)),
                }

        for _, row in stats.head(300).iterrows():
            skaters = list(row["fwd_line"])
            player_parts = [
                {
                    "id": int(pid),
                    "name": name_map.get(int(pid), {}).get("name", str(pid)),
                }
                for pid in skaters
            ]
            key     = tuple(skaters)
            label   = " / ".join(part["name"] for part in player_parts)
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
                "player_parts": player_parts,
                "team":      _normalize_team_code(team_id, season),
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


# cache per season -- keep only 1 to avoid holding 3x ~500KB dicts + deserialization overhead
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 1


def _web_payload_cache_path(season: str) -> Path:
    return cfg.paths.cache / f"web_payload_{season}.json"


def _web_payload_cache_is_fresh(season: str, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False

    cache_mtime = cache_path.stat().st_mtime
    deps = [
        cfg.paths.processed / f"stints_{season}.parquet",
        cfg.paths.cache / "player_names.json",
    ]
    for dep in deps:
        if dep.exists() and dep.stat().st_mtime > cache_mtime:
            return False
    return True


def _load_cached_web_payload(season: str) -> dict | None:
    cache_path = _web_payload_cache_path(season)
    if not _web_payload_cache_is_fresh(season, cache_path):
        return None
    try:
        return json.loads(cache_path.read_text())
    except Exception:
        return None


def _save_cached_web_payload(season: str, data: dict) -> None:
    cache_path = _web_payload_cache_path(season)
    try:
        cache_path.write_text(json.dumps(data))
    except Exception:
        pass


def _get_data(season: str) -> dict:
    if season not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            del _CACHE[next(iter(_CACHE))]
        cached = _load_cached_web_payload(season)
        if cached is not None:
            _CACHE[season] = cached
        else:
            _CACHE[season] = _build_data(season)
            _save_cached_web_payload(season, _CACHE[season])
    return _CACHE[season]


def _bucket_shift_age_overlap(df, bucket_size: int = 10, min_shift_age_sec: int = 0):
    import pandas as pd

    if df.empty:
        return pd.DataFrame(columns=["shift_bucket", "toi_sec", "xg_for", "xg_against", "n_segments"])

    rows = []
    for row in df.itertuples(index=False):
        start_age = float(max(min_shift_age_sec, getattr(row, "shift_age")))
        duration = float(getattr(row, "duration"))
        end_age = start_age + duration
        if end_age <= start_age:
            continue

        xg_for = float(getattr(row, "p_xgf"))
        xg_against = float(getattr(row, "p_xga"))
        bucket = int(start_age // bucket_size) * bucket_size
        while bucket < end_age:
            bucket_end = bucket + bucket_size
            overlap = min(end_age, bucket_end) - max(start_age, bucket)
            if overlap > 0:
                frac = overlap / duration if duration > 0 else 0
                rows.append({
                    "shift_bucket": bucket,
                    "toi_sec": overlap,
                    "xg_for": xg_for * frac,
                    "xg_against": xg_against * frac,
                    "n_segments": 1,
                })
            bucket += bucket_size

    if not rows:
        return pd.DataFrame(columns=["shift_bucket", "toi_sec", "xg_for", "xg_against", "n_segments"])

    return (
        pd.DataFrame(rows)
        .groupby("shift_bucket", as_index=False)
        .agg(
            toi_sec=("toi_sec", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            n_segments=("n_segments", "sum"),
        )
        .sort_values("shift_bucket")
    )


def _get_league_empirical_curve(
    season: str,
    bucket_size: int = 10,
    min_stint_sec: int = 10,
) -> dict:
    cache_key = (season, bucket_size, min_stint_sec)
    if cache_key in _LEAGUE_EMPIRICAL_CURVE_CACHE:
        return _LEAGUE_EMPIRICAL_CURVE_CACHE[cache_key]

    idx = _get_stint_meta(season).copy()
    if idx.empty:
        result = {
            "buckets": [],
            "values": [],
            "delta_values": [],
            "toi_sec": [],
        }
        _LEAGUE_EMPIRICAL_CURVE_CACHE[cache_key] = result
        return result

    if min_stint_sec > 0:
        idx = idx[idx["duration"] >= min_stint_sec]
    if idx.empty:
        result = {
            "buckets": [],
            "values": [],
            "delta_values": [],
            "toi_sec": [],
        }
        _LEAGUE_EMPIRICAL_CURVE_CACHE[cache_key] = result
        return result

    bucketed = _bucket_shift_age_overlap(idx, bucket_size=bucket_size)
    if bucketed.empty:
        result = {
            "buckets": [],
            "values": [],
            "delta_values": [],
            "toi_sec": [],
        }
        _LEAGUE_EMPIRICAL_CURVE_CACHE[cache_key] = result
        return result

    bucketed["xgd60"] = (bucketed["xg_for"] - bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
    fresh = float(bucketed.iloc[0]["xgd60"])
    bucketed["delta_xgd60"] = bucketed["xgd60"] - fresh

    result = {
        "buckets": [int(v) for v in bucketed["shift_bucket"].tolist()],
        "values": [round(float(v), 3) for v in bucketed["xgd60"].tolist()],
        "delta_values": [round(float(v), 3) for v in bucketed["delta_xgd60"].tolist()],
        "toi_sec": [round(float(v), 1) for v in bucketed["toi_sec"].tolist()],
    }
    _LEAGUE_EMPIRICAL_CURVE_CACHE[cache_key] = result
    return result


@app.get("/player/{player_id}/decay")
async def player_decay(player_id: int, season: str | None = None):
    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")
    try:
        disk = _load_player_decay_disk(season)
        if player_id in disk:
            return JSONResponse(disk[player_id])

        idx = _get_stint_meta(season)
        df = idx[idx["player_id"] == player_id].copy()
        if not df.empty:
            df = df[df["duration"] >= 10]
        shift_summary = _get_player_shift_summary(season, player_id)
        bucketed = _bucket_shift_age_overlap(df, bucket_size=10)
        if not bucketed.empty:
            bucketed = bucketed[(bucketed["toi_sec"] >= 60) & (bucketed["n_segments"] >= 3)].copy()
        if bucketed.empty:
            return JSONResponse({
                "buckets": [],
                "values": [],
                "delta_values": [],
                "toi_sec": [],
                "stints": [],
                "league": _get_league_empirical_curve(season),
                "max_age": 80,
                "max_shift_duration": shift_summary["max_shift_duration"],
            })
        bucketed["xgd60"] = (bucketed["xg_for"] - bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
        fresh = float(bucketed.iloc[0]["xgd60"])
        bucketed["delta_xgd60"] = bucketed["xgd60"] - fresh
        return JSONResponse({
            "buckets": [int(v) for v in bucketed["shift_bucket"].tolist()],
            "values": [round(float(v), 3) for v in bucketed["xgd60"].tolist()],
            "delta_values": [round(float(v), 3) for v in bucketed["delta_xgd60"].tolist()],
            "toi_sec": [round(float(v), 1) for v in bucketed["toi_sec"].tolist()],
            "stints": [int(v) for v in bucketed["n_segments"].tolist()],
            "league": _get_league_empirical_curve(season),
            "max_age": int(bucketed["shift_bucket"].max()),
            "max_shift_duration": shift_summary["max_shift_duration"],
        })
    except Exception as exc:
        return JSONResponse({"buckets": [], "values": [], "se": [], "error": str(exc)})


## per-season cache of the exploded 5v5 stint frame -- maxsize=1 to cap memory
_STINT_META_CACHE: dict[str, "pd.DataFrame"] = {}
_STINT_META_CACHE_MAX = 1
_PLAYER_SHIFT_COUNTS_CACHE: dict[str, dict[int, dict[str, int]]] = {}
_LEAGUE_EMPIRICAL_CURVE_CACHE: dict[tuple[str, int, int], dict] = {}
_PLAYER_TEAM_MAP_CACHE: dict[str, dict[int, str]] = {}

## disk-backed pre-computed player decay responses {season: {player_id: response_dict}}
_PLAYER_DECAY_DISK: dict[str, dict[int, dict]] = {}


def _player_decay_disk_path(season: str) -> Path:
    return cfg.paths.cache / f"player_decay_{season}.json"


def _load_player_decay_disk(season: str) -> dict[int, dict]:
    if season in _PLAYER_DECAY_DISK:
        return _PLAYER_DECAY_DISK[season]
    path = _player_decay_disk_path(season)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            _PLAYER_DECAY_DISK[season] = {int(k): v for k, v in raw.items()}
            return _PLAYER_DECAY_DISK[season]
        except Exception:
            pass
    return {}


def precompute_player_decay_all(season: str) -> None:
    import gc
    league_curve = _get_league_empirical_curve(season)
    meta = _get_stint_meta(season)
    all_pids = meta["player_id"].unique().tolist()
    out: dict[str, dict] = {}
    for pid in all_pids:
        df = meta[meta["player_id"] == pid].copy()
        df = df[df["duration"] >= 10]
        max_shift_duration = 0
        if not df.empty:
            df["end_sec"] = df["start_sec"] + df["duration"]
            prev_end = df["end_sec"].shift(1)
            is_new = (df["game_id"] != df["game_id"].shift(1)) | (df["period"] != df["period"].shift(1)) | ((df["start_sec"] - prev_end) > SHIFT_GAP_TOLERANCE)
            is_new.iloc[0] = True
            df = df.copy()
            df["shift_id"] = is_new.cumsum().astype(int)
            shifts = df.groupby("shift_id").agg(s=("start_sec","first"), e=("end_sec","last")).reset_index()
            max_shift_duration = int((shifts["e"] - shifts["s"]).max())
        bucketed = _bucket_shift_age_overlap(df, bucket_size=10)
        if not bucketed.empty:
            bucketed = bucketed[(bucketed["toi_sec"] >= 60) & (bucketed["n_segments"] >= 3)].copy()
        if bucketed.empty:
            out[str(pid)] = {"buckets": [], "values": [], "delta_values": [], "toi_sec": [], "stints": [], "league": league_curve, "max_age": 80, "max_shift_duration": max_shift_duration}
        else:
            bucketed["xgd60"] = (bucketed["xg_for"] - bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
            fresh = float(bucketed.iloc[0]["xgd60"])
            bucketed["delta_xgd60"] = bucketed["xgd60"] - fresh
            out[str(pid)] = {
                "buckets":      [int(v) for v in bucketed["shift_bucket"].tolist()],
                "values":       [round(float(v), 3) for v in bucketed["xgd60"].tolist()],
                "delta_values": [round(float(v), 3) for v in bucketed["delta_xgd60"].tolist()],
                "toi_sec":      [round(float(v), 1) for v in bucketed["toi_sec"].tolist()],
                "stints":       [int(v) for v in bucketed["n_segments"].tolist()],
                "league":       league_curve,
                "max_age":      int(bucketed["shift_bucket"].max()),
                "max_shift_duration": max_shift_duration,
            }
    _player_decay_disk_path(season).write_text(json.dumps(out))
    logger.info(f"Saved player decay cache for {season}: {len(out)} players")
    gc.collect()


## disk-backed pre-computed player stints {season: {player_id: [stints]}}
_PLAYER_STINTS_DISK: dict[str, dict[int, list]] = {}


def _player_stints_disk_path(season: str) -> Path:
    return cfg.paths.cache / f"player_stints_{season}.json"


def _load_player_stints_disk(season: str) -> dict[int, list]:
    if season in _PLAYER_STINTS_DISK:
        return _PLAYER_STINTS_DISK[season]
    path = _player_stints_disk_path(season)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            _PLAYER_STINTS_DISK[season] = {int(k): v for k, v in raw.items()}
            return _PLAYER_STINTS_DISK[season]
        except Exception:
            pass
    return {}


def precompute_player_stints_all(season: str) -> None:
    import gc
    meta = _get_stint_meta(season)
    all_pids = meta["player_id"].unique().tolist()
    out: dict[str, list] = {}

    for pid in all_pids:
        subset = meta[meta["player_id"] == pid].sort_values(
            ["game_id", "period", "start_sec"]
        ).reset_index(drop=True)
        if subset.empty:
            out[str(pid)] = []
            continue

        subset = subset.copy()
        subset["end_sec"] = subset["start_sec"] + subset["duration"]
        prev_end    = subset["end_sec"].shift(1)
        prev_game   = subset["game_id"].shift(1)
        prev_period = subset["period"].shift(1)
        is_new = (
            (subset["game_id"]   != prev_game)   |
            (subset["period"]    != prev_period)  |
            ((subset["start_sec"] - prev_end) > SHIFT_GAP_TOLERANCE)
        )
        is_new.iloc[0] = True
        subset["shift_id"] = is_new.cumsum().astype(int)

        def first_zone(zones):
            for z in zones:
                if str(z) in ("O", "N", "D"):
                    return str(z)
            return "N"

        agg = (
            subset.groupby("shift_id", sort=False)
            .agg(
                game_id   =("game_id",    "first"),
                game_date =("game_date",  "first"),
                home_team =("home_team",  "first"),
                away_team =("away_team",  "first"),
                is_home   =("is_home",    "first"),
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
            game_date = str(r.get("game_date", "") or "")
            home_team = str(r.get("home_team", "?") or "?")
            away_team = str(r.get("away_team", "?") or "?")
            is_home_val = r.get("is_home")
            is_home = None if is_home_val is None or is_home_val != is_home_val else bool(is_home_val)
            records.append({
                "game_id":   int(r["game_id"]),
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "game_label": _format_game_label(int(r["game_id"]), game_date, home_team, away_team, is_home),
                "period":    int(r["period"]),
                "start_sec": int(r["start_sec"]),
                "duration":  int(r["duration"]),
                "zone":      r["zone"],
                "score":     int(r["score"]) if r["score"] == r["score"] else 0,
                "xgf":       round(float(r["p_xgf"]), 4),
                "xga":       round(float(r["p_xga"]), 4),
                "xgd":       round(float(r["xgd"]),   4),
                "n_segs":    int(r["n_segs"]),
            })
        out[str(pid)] = records

    _player_stints_disk_path(season).write_text(json.dumps(out))
    logger.info(f"Saved player stints cache for {season}: {len(out)} players")
    gc.collect()


def _get_stint_meta(season: str) -> "pd.DataFrame":
    import pandas as pd
    import numpy as np

    if season in _STINT_META_CACHE:
        return _STINT_META_CACHE[season]

    if len(_STINT_META_CACHE) >= _STINT_META_CACHE_MAX:
        del _STINT_META_CACHE[next(iter(_STINT_META_CACHE))]

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()
    ev = ev.reset_index(drop=True)
    ev.index.name = "stint_idx"
    ev = ev.reset_index()

    for col, default in (("game_date", ""), ("home_team", "?"), ("away_team", "?")):
        if col not in ev.columns:
            ev[col] = default

    keep = ["stint_idx", "game_id", "game_date", "home_team", "away_team", "period", "start_sec", "duration",
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

    cols = ["stint_idx", "player_id", "is_home", "game_id", "game_date", "home_team", "away_team", "period",
            "start_sec", "duration", "shift_age", "zone_start", "score_state",
            "p_xgf", "p_xga"]
    combined = pd.concat([home[cols], away[cols]], ignore_index=True)
    combined["xgd"]   = combined["p_xgf"] - combined["p_xga"]
    combined = combined.sort_values(["player_id", "game_id", "period", "start_sec"]).reset_index(drop=True)
    combined["end_sec"] = combined["start_sec"] + combined["duration"]

    prev_player = combined["player_id"].shift(1)
    prev_game = combined["game_id"].shift(1)
    prev_period = combined["period"].shift(1)
    prev_end = combined["end_sec"].shift(1)
    is_new_shift = (
        (combined["player_id"] != prev_player) |
        (combined["game_id"] != prev_game) |
        (combined["period"] != prev_period) |
        ((combined["start_sec"] - prev_end) > SHIFT_GAP_TOLERANCE)
    )
    is_new_shift.iloc[0] = True
    combined["shift_id"] = is_new_shift.cumsum().astype(int)
    combined["shift_start_sec"] = combined.groupby(["player_id", "shift_id"])["start_sec"].transform("min")
    combined["shift_age"] = (combined["start_sec"] - combined["shift_start_sec"]).clip(lower=0)
    combined = combined.drop(columns=["end_sec", "shift_start_sec"])

    _STINT_META_CACHE[season] = combined
    return combined


def _get_player_team_map(season: str) -> dict[int, str]:
    if season in _PLAYER_TEAM_MAP_CACHE:
        return _PLAYER_TEAM_MAP_CACHE[season]

    meta = _get_stint_meta(season)
    if meta.empty:
        _PLAYER_TEAM_MAP_CACHE[season] = {}
        return _PLAYER_TEAM_MAP_CACHE[season]

    required_cols = {"home_team", "away_team", "is_home"}
    if not required_cols.issubset(meta.columns):
        _PLAYER_TEAM_MAP_CACHE[season] = {}
        return _PLAYER_TEAM_MAP_CACHE[season]

    team_series = meta["home_team"].where(meta["is_home"], meta["away_team"]).fillna("?")
    counts = (
        meta.assign(team_code=team_series)
        .loc[lambda df: df["team_code"].notna() & (df["team_code"] != "?") & (df["team_code"] != "")]
        .groupby(["player_id", "team_code"])
        .size()
        .reset_index(name="n")
        .sort_values(["player_id", "n", "team_code"], ascending=[True, False, True])
    )
    team_map = counts.groupby("player_id", sort=False)["team_code"].first().to_dict() if not counts.empty else {}
    _PLAYER_TEAM_MAP_CACHE[season] = {int(pid): str(team) for pid, team in team_map.items()}
    return _PLAYER_TEAM_MAP_CACHE[season]


def _get_player_shift_counts(season: str) -> dict[int, dict[str, int]]:
    import pandas as pd

    if season in _PLAYER_SHIFT_COUNTS_CACHE:
        return _PLAYER_SHIFT_COUNTS_CACHE[season]

    meta = _get_stint_meta(season)
    if meta.empty:
        _PLAYER_SHIFT_COUNTS_CACHE[season] = {}
        return _PLAYER_SHIFT_COUNTS_CACHE[season]

    subset = meta.sort_values(["player_id", "game_id", "period", "start_sec"]).copy()
    subset["end_sec"] = subset["start_sec"] + subset["duration"]

    prev_player = subset["player_id"].shift(1)
    prev_game = subset["game_id"].shift(1)
    prev_period = subset["period"].shift(1)
    prev_end = subset["end_sec"].shift(1)

    is_new = (
        (subset["player_id"] != prev_player) |
        (subset["game_id"] != prev_game) |
        (subset["period"] != prev_period) |
        ((subset["start_sec"] - prev_end) > SHIFT_GAP_TOLERANCE)
    )
    is_new.iloc[0] = True
    subset["shift_id"] = is_new.cumsum().astype(int)

    shifts = (
        subset.groupby(["player_id", "shift_id"], sort=False)
        .agg(
            start_sec=("start_sec", "first"),
            end_sec=("end_sec", "last"),
        )
        .reset_index()
    )
    shifts["duration"] = shifts["end_sec"] - shifts["start_sec"]
    shifts["qualifies"] = shifts["duration"] > MIN_QUALIFYING_SHIFT_SECONDS

    counts = (
        shifts.groupby("player_id")
        .agg(
            total_shifts=("shift_id", "count"),
            qualifying_shifts=("qualifies", "sum"),
        )
        .reset_index()
    )

    out = {
        int(row["player_id"]): {
            "total_shifts": int(row["total_shifts"]),
            "qualifying_shifts": int(row["qualifying_shifts"]),
        }
        for _, row in counts.iterrows()
    }
    _PLAYER_SHIFT_COUNTS_CACHE[season] = out
    return out


def _get_player_shift_summary(season: str, player_id: int) -> dict[str, int]:
    meta = _get_stint_meta(season)
    subset = meta[meta["player_id"] == player_id].sort_values(
        ["game_id", "period", "start_sec"]
    ).reset_index(drop=True)

    if subset.empty:
        return {"max_shift_duration": 0}

    subset["end_sec"] = subset["start_sec"] + subset["duration"]
    prev_end = subset["end_sec"].shift(1)
    prev_game = subset["game_id"].shift(1)
    prev_period = subset["period"].shift(1)

    is_new = (
        (subset["game_id"] != prev_game) |
        (subset["period"] != prev_period) |
        ((subset["start_sec"] - prev_end) > SHIFT_GAP_TOLERANCE)
    )
    is_new.iloc[0] = True
    subset["shift_id"] = is_new.cumsum().astype(int)

    shifts = (
        subset.groupby("shift_id", sort=False)
        .agg(
            start_sec=("start_sec", "first"),
            end_sec=("end_sec", "last"),
        )
        .reset_index()
    )
    shifts["duration"] = shifts["end_sec"] - shifts["start_sec"]
    max_shift_duration = int(shifts["duration"].max()) if not shifts.empty else 0
    return {"max_shift_duration": max_shift_duration}


def _format_game_label(game_id: int, game_date: str, home_team: str, away_team: str, is_home: bool | None) -> str:
    date_part = game_date or ""
    if is_home is True:
        opponent = away_team if away_team and away_team != "?" else ""
        if date_part and opponent:
            return f"{date_part} vs {opponent}"
        if opponent:
            return f"vs {opponent}"
    if is_home is False:
        opponent = home_team if home_team and home_team != "?" else ""
        if date_part and opponent:
            return f"{date_part} @ {opponent}"
        if opponent:
            return f"@ {opponent}"
    if date_part:
        return date_part
    return f"Game {game_id}"


@app.get("/player/{player_id}/stints")
async def player_stints(player_id: int, season: str | None = None):
    import numpy as np

    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")
    try:
        disk = _load_player_stints_disk(season)
        if player_id in disk:
            records = disk[player_id]
            return JSONResponse({"stints": records, "total": len(records)})

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
                game_date =("game_date",  "first"),
                home_team =("home_team",  "first"),
                away_team =("away_team",  "first"),
                is_home   =("is_home",    "first"),
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
            game_date = str(r.get("game_date", "") or "")
            home_team = str(r.get("home_team", "?") or "?")
            away_team = str(r.get("away_team", "?") or "?")
            is_home_val = r.get("is_home")
            is_home = None if is_home_val is None or is_home_val != is_home_val else bool(is_home_val)
            records.append({
                "game_id":   int(r["game_id"]),
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "game_label": _format_game_label(int(r["game_id"]), game_date, home_team, away_team, is_home),
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


@app.get("/app.bundle.js")
async def app_bundle():
    _ensure_app_bundle()
    return FileResponse(APP_BUNDLE, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def index(season: str | None = None):
    available = _available_seasons()
    if season is None:
        season = available[-1] if available else (cfg.seasons[-1] if cfg.seasons else "20232024")

    _ensure_app_bundle()
    data      = _get_data(season)
    data_json = json.dumps(data, default=str)
    html      = TEMPLATE.replace("{{DATA_JSON}}", data_json)
    return HTMLResponse(html)
