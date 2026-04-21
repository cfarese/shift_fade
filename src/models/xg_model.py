from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import cfg
from src.ingestion.nhl_client import NHLClient


UNBLOCKED_SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}
GOAL_X_COORD = 89.0
MAX_X_COORD = 89.0
MAX_Y_COORD = 42.5
DEFAULT_X_BIN = 10
DEFAULT_Y_BIN = 5
DEFAULT_MODEL_PATH = cfg.paths.processed / "xg_model_v1.json"

FEATURE_SPECS = [
    ("shot_type", 40.0, 0.65, 1.65),
    ("score_state", 60.0, 0.8, 1.2),
    ("rebound", 30.0, 0.75, 2.5),
    ("rush", 30.0, 0.8, 1.8),
    ("season", 150.0, 0.9, 1.1),
    ("rink", 150.0, 0.85, 1.15),
]


def _mmss_to_sec(mmss: str) -> int:
    parts = str(mmss or "").split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


def _clip_score_state(diff: int, limit: int = 2) -> int:
    return max(-limit, min(limit, diff))


def _normalize_shot_type(raw_type: Any) -> str:
    value = str(raw_type or "unknown").strip().lower().replace(" ", "-")
    return value if value else "unknown"


def _strength_state(situation_code: Any, is_home: bool) -> str:
    code = str(situation_code or "")
    if len(code) != 4 or not code.isdigit():
        return "OTHER"

    home_goalie, home_skaters, away_skaters, away_goalie = (int(ch) for ch in code)
    if home_goalie == 0 or away_goalie == 0:
        return "EN"

    team_skaters = home_skaters if is_home else away_skaters
    opp_skaters = away_skaters if is_home else home_skaters

    if team_skaters == opp_skaters and team_skaters in {3, 4, 5}:
        return f"{team_skaters}v{opp_skaters}"
    if team_skaters > opp_skaters:
        return "PP"
    if team_skaters < opp_skaters:
        return "SH"
    return "OTHER"


def _location_cell(x_abs: float | None, y_abs: float | None, x_bin: int, y_bin: int) -> str:
    if x_abs is None or y_abs is None or not np.isfinite(x_abs) or not np.isfinite(y_abs):
        return "UNK"

    xb = min(int(min(MAX_X_COORD, max(0.0, x_abs)) // x_bin) * x_bin, int(MAX_X_COORD // x_bin) * x_bin)
    yb = min(int(min(MAX_Y_COORD, max(0.0, y_abs)) // y_bin) * y_bin, int(MAX_Y_COORD // y_bin) * y_bin)
    return f"{xb}_{yb}"


def extract_shots_from_raw(
    raw: dict,
    game_id: int | None = None,
    x_bin: int = DEFAULT_X_BIN,
    y_bin: int = DEFAULT_Y_BIN,
) -> pd.DataFrame:
    """Build a shot-level table from NHL play-by-play.

    This mirrors the referenced article's approach: unblocked attempts only,
    a location-grid baseline, plus rebound/rush/shot-type/score/season/rink
    context for multiplicative adjustments.
    """
    plays = sorted(raw.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    home_team = raw.get("homeTeam", {}) or {}
    away_team = raw.get("awayTeam", {}) or {}
    home_team_id = home_team.get("id")
    away_team_id = away_team.get("id")
    home_abbrev = home_team.get("abbrev") or home_team.get("triCode") or "HOME"
    away_abbrev = away_team.get("abbrev") or away_team.get("triCode") or "AWAY"
    season = str(raw.get("season") or "")

    score_home = 0
    score_away = 0
    prev_team_event: dict[int, tuple[int, int, str]] = {}
    prev_team_shot: dict[int, tuple[int, int]] = {}

    rows: list[dict[str, Any]] = []

    for play in plays:
        event_type = play.get("typeDescKey", "")
        period = int(play.get("periodDescriptor", {}).get("number", 1) or 1)
        if period > 3:
            continue

        details = play.get("details", {}) or {}
        owner_team = details.get("eventOwnerTeamId")
        time_sec = _mmss_to_sec(play.get("timeInPeriod", "0:00"))

        if event_type in UNBLOCKED_SHOT_EVENTS and owner_team in {home_team_id, away_team_id}:
            is_home = owner_team == home_team_id
            score_state = _clip_score_state((score_home - score_away) if is_home else (score_away - score_home))
            raw_x = details.get("xCoord")
            raw_y = details.get("yCoord")
            x_abs = abs(float(raw_x)) if raw_x is not None else None
            y_abs = abs(float(raw_y)) if raw_y is not None else None
            dx = max(0.1, GOAL_X_COORD - min(MAX_X_COORD, x_abs if x_abs is not None else GOAL_X_COORD))
            dy = y_abs if y_abs is not None else 0.0
            distance = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx))
            prev_event = prev_team_event.get(int(owner_team))
            prev_shot = prev_team_shot.get(int(owner_team))
            rebound = int(prev_shot is not None and prev_shot[0] == period and 0 < time_sec - prev_shot[1] <= 2)
            rush = int(
                prev_event is not None
                and prev_event[0] == period
                and 0 < time_sec - prev_event[1] <= 4
                and prev_event[2] in {"N", "D"}
            )
            rows.append(
                {
                    "game_id": int(game_id or raw.get("id") or 0),
                    "season": season,
                    "event_id": int(play.get("eventId") or play.get("sortOrder") or 0),
                    "sort_order": int(play.get("sortOrder") or 0),
                    "period": period,
                    "time_sec": time_sec,
                    "event_type": event_type,
                    "goal": int(event_type == "goal"),
                    "team_id": int(owner_team),
                    "team_abbrev": home_abbrev if is_home else away_abbrev,
                    "rink": home_abbrev,
                    "is_home": int(is_home),
                    "strength_state": _strength_state(play.get("situationCode"), is_home),
                    "score_state": score_state,
                    "shot_type": _normalize_shot_type(details.get("shotType")),
                    "x_coord": float(raw_x) if raw_x is not None else np.nan,
                    "y_coord": float(raw_y) if raw_y is not None else np.nan,
                    "x_abs": float(x_abs) if x_abs is not None else np.nan,
                    "y_abs": float(y_abs) if y_abs is not None else np.nan,
                    "distance_ft": distance,
                    "angle_deg": angle,
                    "location_cell": _location_cell(x_abs, y_abs, x_bin=x_bin, y_bin=y_bin),
                    "rebound": rebound,
                    "rush": rush,
                }
            )

        if owner_team in {home_team_id, away_team_id}:
            prev_team_event[int(owner_team)] = (
                period,
                time_sec,
                str(details.get("zoneCode") or ""),
            )
            if event_type in UNBLOCKED_SHOT_EVENTS:
                prev_team_shot[int(owner_team)] = (period, time_sec)

        if event_type == "goal" and owner_team in {home_team_id, away_team_id}:
            if owner_team == home_team_id:
                score_home += 1
            else:
                score_away += 1

    return pd.DataFrame(rows)


def _smoothed_rate(goals: pd.Series, shots: pd.Series, prior_rate: float, prior_weight: float) -> pd.Series:
    raw_rate = goals / shots.replace(0, np.nan)
    weight = shots / (shots + prior_weight)
    return (prior_rate + weight * (raw_rate.fillna(prior_rate) - prior_rate)).clip(lower=1e-4, upper=0.6)


def _lookup_factor_map(values: pd.Series, mapping: dict[str, dict[str, float]], strengths: pd.Series) -> np.ndarray:
    out = np.ones(len(values), dtype=float)
    for i, (strength, value) in enumerate(zip(strengths.tolist(), values.tolist(), strict=False)):
        out[i] = float(mapping.get(str(strength), {}).get(str(value), 1.0))
    return out


def fit_xg_model(
    shots: pd.DataFrame,
    x_bin: int = DEFAULT_X_BIN,
    y_bin: int = DEFAULT_Y_BIN,
    iterations: int = 5,
    base_prior_weight: float = 75.0,
) -> dict[str, Any]:
    if shots.empty:
        raise ValueError("Shot table is empty")

    df = shots.copy()
    df["goal"] = df["goal"].astype(int)
    df["strength_state"] = df["strength_state"].fillna("OTHER").astype(str)
    df["shot_type"] = df["shot_type"].fillna("unknown").astype(str)
    df["score_state"] = df["score_state"].fillna(0).astype(int).astype(str)
    df["rebound"] = df["rebound"].fillna(0).astype(int).astype(str)
    df["rush"] = df["rush"].fillna(0).astype(int).astype(str)
    df["season"] = df["season"].fillna("").astype(str)
    df["rink"] = df["rink"].fillna("UNK").astype(str)
    df["location_cell"] = df["location_cell"].fillna("UNK").astype(str)

    strength_rates = (
        df.groupby("strength_state")
        .agg(goals=("goal", "sum"), shots=("goal", "size"))
        .reset_index()
    )
    global_rate = float(df["goal"].mean())
    base_global_rate = {row["strength_state"]: float(row["goals"] / max(row["shots"], 1)) for _, row in strength_rates.iterrows()}

    location_grouped = (
        df.groupby(["strength_state", "location_cell"])
        .agg(goals=("goal", "sum"), shots=("goal", "size"))
        .reset_index()
    )

    location_base: dict[str, dict[str, float]] = {}
    for strength, group in location_grouped.groupby("strength_state"):
        prior_rate = base_global_rate.get(strength, global_rate)
        group = group.copy()
        group["rate"] = _smoothed_rate(group["goals"], group["shots"], prior_rate=prior_rate, prior_weight=base_prior_weight)
        location_base[str(strength)] = {
            str(row["location_cell"]): round(float(row["rate"]), 6)
            for _, row in group.iterrows()
        }

    df["_base"] = [
        float(location_base.get(strength, {}).get(cell, base_global_rate.get(strength, global_rate)))
        for strength, cell in zip(df["strength_state"].tolist(), df["location_cell"].tolist(), strict=False)
    ]

    factors: dict[str, dict[str, dict[str, float]]] = {}
    factor_columns: list[str] = []

    for family, *_ in FEATURE_SPECS:
        factors[family] = {}
        col = f"_{family}_factor"
        factor_columns.append(col)
        df[col] = 1.0

    for _ in range(iterations):
        for family, prior, min_factor, max_factor in FEATURE_SPECS:
            current_col = f"_{family}_factor"
            other_cols = [col for col in factor_columns if col != current_col]
            pred_without = df["_base"].to_numpy(dtype=float)
            for col in other_cols:
                pred_without = pred_without * df[col].to_numpy(dtype=float)

            grouped = (
                df.assign(_pred_without=pred_without)
                .groupby(["strength_state", family])
                .agg(
                    goals=("goal", "sum"),
                    pred=("_pred_without", "sum"),
                    shots=("goal", "size"),
                )
                .reset_index()
            )

            family_map: dict[str, dict[str, float]] = {}
            for strength, group in grouped.groupby("strength_state"):
                strength_map: dict[str, float] = {}
                for _, row in group.iterrows():
                    pred = float(row["pred"])
                    shots_n = float(row["shots"])
                    raw_factor = (float(row["goals"]) / pred) if pred > 0 else 1.0
                    weight = shots_n / (shots_n + prior)
                    shrunk = 1.0 + weight * (raw_factor - 1.0)
                    strength_map[str(row[family])] = round(float(np.clip(shrunk, min_factor, max_factor)), 6)
                family_map[str(strength)] = strength_map

            factors[family] = family_map
            df[current_col] = _lookup_factor_map(df[family], family_map, df["strength_state"])

    df["_xg"] = df["_base"].to_numpy(dtype=float)
    for col in factor_columns:
        df["_xg"] = df["_xg"] * df[col].to_numpy(dtype=float)

    # final calibration so total expected goals matches total actual goals
    total_actual = float(df["goal"].sum())
    total_expected = float(df["_xg"].sum())
    calibration = total_actual / total_expected if total_expected > 0 else 1.0

    final_predictions = (df["_xg"] * calibration).clip(lower=1e-4, upper=0.95)
    strength_scale = (
        df.assign(_xg=final_predictions)
        .groupby("strength_state")
        .agg(actual=("goal", "sum"), expected=("_xg", "sum"))
        .reset_index()
    )
    strength_calibration = {
        str(row["strength_state"]): round(float(row["actual"] / row["expected"]) if row["expected"] > 0 else 1.0, 6)
        for _, row in strength_scale.iterrows()
    }

    return {
        "version": 1,
        "model_type": "fenwick_adjusted_grid",
        "x_bin": int(x_bin),
        "y_bin": int(y_bin),
        "global_rate": round(global_rate, 6),
        "strength_base_rate": {str(k): round(float(v), 6) for k, v in base_global_rate.items()},
        "location_base": location_base,
        "factors": factors,
        "global_calibration": round(float(calibration), 6),
        "strength_calibration": strength_calibration,
        "training_rows": int(len(df)),
        "training_goals": int(total_actual),
        "feature_families": [family for family, *_ in FEATURE_SPECS],
    }


def score_shots(shots: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    if shots.empty:
        out = shots.copy()
        out["xg"] = []
        return out

    df = shots.copy()
    df["strength_state"] = df["strength_state"].fillna("OTHER").astype(str)
    df["shot_type"] = df["shot_type"].fillna("unknown").astype(str)
    df["score_state"] = df["score_state"].fillna(0).astype(int).astype(str)
    df["rebound"] = df["rebound"].fillna(0).astype(int).astype(str)
    df["rush"] = df["rush"].fillna(0).astype(int).astype(str)
    df["season"] = df["season"].fillna("").astype(str)
    df["rink"] = df["rink"].fillna("UNK").astype(str)
    df["location_cell"] = df["location_cell"].fillna("UNK").astype(str)

    base_rates = model.get("strength_base_rate", {})
    global_rate = float(model.get("global_rate", 0.08))
    location_base = model.get("location_base", {})
    xg = np.array(
        [
            float(location_base.get(strength, {}).get(cell, base_rates.get(strength, global_rate)))
            for strength, cell in zip(df["strength_state"].tolist(), df["location_cell"].tolist(), strict=False)
        ],
        dtype=float,
    )

    for family, *_ in FEATURE_SPECS:
        xg = xg * _lookup_factor_map(df[family], model.get("factors", {}).get(family, {}), df["strength_state"])

    xg = xg * float(model.get("global_calibration", 1.0))
    strength_cal = model.get("strength_calibration", {})
    xg = xg * np.array([float(strength_cal.get(strength, 1.0)) for strength in df["strength_state"].tolist()], dtype=float)
    df["xg"] = np.clip(xg, 1e-4, 0.95)
    return df


def save_xg_model(model: dict[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_MODEL_PATH
    path.write_text(json.dumps(model, indent=2, sort_keys=True))
    return path


def load_xg_model(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_MODEL_PATH
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def get_default_xg_model() -> dict[str, Any] | None:
    if not DEFAULT_MODEL_PATH.exists():
        return None
    try:
        return load_xg_model(DEFAULT_MODEL_PATH)
    except Exception as exc:
        logger.warning(f"Failed loading xG model from {DEFAULT_MODEL_PATH}: {exc}")
        return None


def build_shot_dataset_for_season(
    season: str,
    force: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    out_path = cfg.paths.processed / f"shots_{season}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    rows: list[pd.DataFrame] = []
    with NHLClient() as client:
        game_ids = client.get_season_game_ids(season)
        if limit:
            game_ids = game_ids[:limit]

        for i, gid in enumerate(game_ids):
            try:
                raw = client.get_play_by_play(gid)
                if str(raw.get("gameState", "")).upper() in {"FUT", "PRE"}:
                    continue
                shot_df = extract_shots_from_raw(raw, game_id=gid)
                if not shot_df.empty:
                    rows.append(shot_df)
                if (i + 1) % 50 == 0:
                    logger.info(f"Shot dataset {season}: processed {i + 1}/{len(game_ids)} games")
            except Exception as exc:
                logger.warning(f"Shot dataset {season}: game {gid} failed: {exc}")

    if rows:
        combined = pd.concat(rows, ignore_index=True)
    else:
        combined = pd.DataFrame()
    combined.to_parquet(out_path, index=False)
    logger.success(f"Saved {len(combined)} shots to {out_path}")
    return combined


def train_and_save_xg_model(
    seasons: list[str],
    out_path: Path | None = None,
    force_shots: bool = False,
    limit: int | None = None,
    iterations: int = 5,
) -> Path:
    shot_frames = [build_shot_dataset_for_season(season, force=force_shots, limit=limit) for season in seasons]
    shots = pd.concat([df for df in shot_frames if not df.empty], ignore_index=True) if shot_frames else pd.DataFrame()
    if shots.empty:
        raise RuntimeError("No shots available to train xG model")

    model = fit_xg_model(shots, iterations=iterations)
    path = save_xg_model(model, out_path or DEFAULT_MODEL_PATH)
    logger.success(f"Saved xG model to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a v1 Fenwick-style xG model")
    parser.add_argument("--seasons", nargs="+", default=list(cfg.seasons))
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--force-shots", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Cap games per season for smoke tests")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out)
    train_and_save_xg_model(
        seasons=[str(s) for s in args.seasons],
        out_path=out_path,
        force_shots=bool(args.force_shots),
        limit=args.limit,
        iterations=int(args.iterations),
    )


if __name__ == "__main__":
    main()
