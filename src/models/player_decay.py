## Empirical shift-decay analysis at the player level.
##
## Core idea: build a player-stint index once per season by exploding
## home/away skater lists into individual rows (one per player per stint).
## All subsequent queries are fast groupby operations on this index
## instead of per-player full scans.

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.models.line_analysis import load_stints

## module-level cache keyed by season -- avoids rebuilding the exploded
## index on every dashboard interaction (lru_cache doesn't work on DataFrames)
_INDEX_CACHE: dict[str, pd.DataFrame] = {}


def _get_player_index(season: str) -> pd.DataFrame:
    ## explodes every 5v5 stint into one row per on-ice player
    ## columns: player_id, duration, p_xg_for, p_xg_against, shift_age

    if season in _INDEX_CACHE:
        return _INDEX_CACHE[season]

    df   = load_stints(season)
    ev   = df[df["strength"] == "5v5"].reset_index(drop=True)

    ## home side -- xg_for/against are already from home team's POV
    home = ev[["duration", "xg_for", "xg_against", "home_shift_age", "home_skaters"]].copy()
    home = home.explode("home_skaters")
    home = home.rename(columns={"home_skaters": "player_id", "home_shift_age": "shift_age"})
    home["p_xg_for"]     = home["xg_for"]
    home["p_xg_against"] = home["xg_against"]

    ## away side -- flip for/against so "for" always means the player's team
    away = ev[["duration", "xg_for", "xg_against", "away_shift_age", "away_skaters"]].copy()
    away = away.explode("away_skaters")
    away = away.rename(columns={"away_skaters": "player_id", "away_shift_age": "shift_age"})
    away["p_xg_for"]     = away["xg_against"]
    away["p_xg_against"] = away["xg_for"]

    cols   = ["player_id", "duration", "p_xg_for", "p_xg_against", "shift_age"]
    result = pd.concat([home[cols], away[cols]], ignore_index=True)
    result["player_id"] = result["player_id"].astype(int)

    _INDEX_CACHE[season] = result
    logger.debug(f"Built player index for {season}: {len(result)} player-stint rows")
    return result


def get_player_empirical_decay(
    season: str,
    player_id: int,
    bucket_size: int = 10,
    min_toi_sec: int = 60,
) -> pd.DataFrame:
    index  = _get_player_index(season)
    on_ice = index[index["player_id"] == player_id].copy()

    if on_ice.empty:
        return pd.DataFrame()

    on_ice["shift_bucket"] = (on_ice["shift_age"] // bucket_size) * bucket_size
    on_ice["shift_bucket"] = on_ice["shift_bucket"].clip(upper=90)

    bucketed = (
        on_ice.groupby("shift_bucket")
        .agg(
            toi_sec=("duration", "sum"),
            xg_for=("p_xg_for", "sum"),
            xg_against=("p_xg_against", "sum"),
            n_stints=("duration", "count"),
        )
        .reset_index()
    )

    bucketed = bucketed[bucketed["toi_sec"] >= min_toi_sec]
    if bucketed.empty:
        return pd.DataFrame()

    bucketed["xg_diff_per60"] = (bucketed["xg_for"] - bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
    bucketed["se"]            = np.sqrt(bucketed["xg_for"] + bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
    bucketed["toi_min"]       = bucketed["toi_sec"] / 60

    return bucketed.sort_values("shift_bucket")


def get_league_decay_summary(season: str, min_toi_sec: int = 1800) -> pd.DataFrame:
    index = _get_player_index(season)

    ## total TOI filter
    total = index.groupby("player_id")["duration"].sum().rename("total_toi")

    ## early shift (0-30s) and late shift (>45s) buckets
    early = (
        index[index["shift_age"] <= 30]
        .groupby("player_id")
        .agg(early_toi=("duration", "sum"), early_xgf=("p_xg_for", "sum"), early_xga=("p_xg_against", "sum"))
    )
    late = (
        index[index["shift_age"] > 45]
        .groupby("player_id")
        .agg(late_toi=("duration", "sum"), late_xgf=("p_xg_for", "sum"), late_xga=("p_xg_against", "sum"))
    )

    merged = total.to_frame().join(early, how="left").join(late, how="left").reset_index()
    merged = merged[
        (merged["total_toi"] >= min_toi_sec) &
        (merged["early_toi"] >= 30) &
        (merged["late_toi"]  >= 30)
    ]

    if merged.empty:
        return pd.DataFrame()

    merged["early_xgd60"]  = (merged["early_xgf"] - merged["early_xga"]) / merged["early_toi"] * 3600
    merged["late_xgd60"]   = (merged["late_xgf"]  - merged["late_xga"])  / merged["late_toi"]  * 3600
    merged["decay_delta"]  = merged["late_xgd60"] - merged["early_xgd60"]
    merged["toi_min"]      = merged["total_toi"] / 60
    merged["early_toi_min"] = merged["early_toi"] / 60
    merged["late_toi_min"]  = merged["late_toi"]  / 60

    cols = ["player_id", "toi_min", "early_xgd60", "late_xgd60", "decay_delta", "early_toi_min", "late_toi_min"]
    return merged[cols].sort_values("decay_delta").reset_index(drop=True)
