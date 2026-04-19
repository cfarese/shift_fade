## Empirical shift-decay analysis at the player level.
##
## This does NOT use the RAPM model -- it directly measures how a player's
## on-ice xGD/60 changes as their shift ages. Works with any sample size
## and is more interpretable than RAPM coefficients for coaching decisions.
##
## The approach: filter all 5v5 stints where player P was on ice, then
## bucket by home_shift_age and compute xGD/60 within each bucket.

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import cfg
from src.models.line_analysis import load_stints


def get_player_empirical_decay(
    season: str,
    player_id: int,
    bucket_size: int = 10,
    min_toi_sec: int = 60,
) -> pd.DataFrame:
    ## returns shift-age bucketed xGD/60 for one player
    ## min_toi_sec is the minimum seconds in a bucket to include it

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    ## find stints where this player was on ice (home or away)
    home_mask = ev["home_skaters"].apply(lambda s: player_id in s)
    away_mask = ev["away_skaters"].apply(lambda s: player_id in s)
    on_ice    = ev[home_mask | away_mask].copy()

    if on_ice.empty:
        return pd.DataFrame()

    ## from the player's perspective: on-ice xGF and xGA
    ## when the player is on the home team, for/against is as-is
    ## when away, flip it so "for" always means the player's team
    on_ice["player_home"] = home_mask[on_ice.index]
    on_ice["p_xg_for"]    = np.where(on_ice["player_home"], on_ice["xg_for"],    on_ice["xg_against"])
    on_ice["p_xg_against"] = np.where(on_ice["player_home"], on_ice["xg_against"], on_ice["xg_for"])

    ## use home_shift_age when player is home, away_shift_age when away
    on_ice["shift_age"] = np.where(
        on_ice["player_home"],
        on_ice["home_shift_age"],
        on_ice["away_shift_age"],
    )

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
    ## rough SE using Poisson assumption on xG totals
    bucketed["se"] = np.sqrt(bucketed["xg_for"] + bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
    bucketed["toi_min"] = bucketed["toi_sec"] / 60

    return bucketed.sort_values("shift_bucket")


def get_league_decay_summary(season: str, min_toi_sec: int = 1800) -> pd.DataFrame:
    ## for every player with enough ice time, compute their early vs late xGD/60
    ## returns a sortable table the dashboard can display directly

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    ## get all unique player IDs
    all_players: set[int] = set()
    for s in ev["home_skaters"]:
        all_players.update(int(i) for i in s)
    for s in ev["away_skaters"]:
        all_players.update(int(i) for i in s)

    results = []
    for pid in all_players:
        home_mask  = ev["home_skaters"].apply(lambda s: pid in s)
        away_mask  = ev["away_skaters"].apply(lambda s: pid in s)
        on_ice     = ev[home_mask | away_mask]

        total_toi = on_ice["duration"].sum()
        if total_toi < min_toi_sec:
            continue

        is_home     = home_mask[on_ice.index]
        xg_for      = np.where(is_home, on_ice["xg_for"],    on_ice["xg_against"])
        xg_against  = np.where(is_home, on_ice["xg_against"], on_ice["xg_for"])
        shift_age   = np.where(is_home, on_ice["home_shift_age"], on_ice["away_shift_age"])

        early = on_ice["duration"].values[shift_age <= 30]
        late  = on_ice["duration"].values[shift_age > 45]
        early_xgf = xg_for[shift_age <= 30]
        early_xga = xg_against[shift_age <= 30]
        late_xgf  = xg_for[shift_age > 45]
        late_xga  = xg_against[shift_age > 45]

        early_toi = early.sum()
        late_toi  = late.sum()

        if early_toi < 30 or late_toi < 30:
            continue

        early_xgd60 = (early_xgf.sum() - early_xga.sum()) / early_toi * 3600
        late_xgd60  = (late_xgf.sum()  - late_xga.sum())  / late_toi  * 3600

        results.append({
            "player_id":    pid,
            "toi_min":      total_toi / 60,
            "early_xgd60":  early_xgd60,
            "late_xgd60":   late_xgd60,
            "decay_delta":  late_xgd60 - early_xgd60,
            "early_toi_min": early_toi / 60,
            "late_toi_min":  late_toi  / 60,
        })

    return pd.DataFrame(results).sort_values("decay_delta") if results else pd.DataFrame()
