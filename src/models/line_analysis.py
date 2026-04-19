## Line-combination level analysis on top of the raw stint data.
##
## Individual RAPM tells you about a player, but coaches make deployment
## decisions at the line level. This module groups stints by the exact
## skater combination on ice and computes aggregate stats per line combo,
## including the shift-age breakdown so you can see where each specific
## unit starts to fall apart.

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import cfg


@lru_cache(maxsize=4)
def load_stints(season: str) -> pd.DataFrame:
    path = cfg.paths.processed / f"stints_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Stint data not found: {path}")
    return pd.read_parquet(path)


def get_line_stats(season: str, min_toi_sec: int = 300) -> pd.DataFrame:
    ## returns one row per unique home-skater combination with aggregate stats
    ## min_toi_sec filters out combos that barely played together

    df = load_stints(season)

    ## 5v5 only for line analysis
    ev = df[df["strength"] == "5v5"].copy()

    grouped = (
        ev.groupby("home_skaters")
        .agg(
            toi_sec=("duration", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            corsi_for=("corsi_for", "sum"),
            corsi_against=("corsi_against", "sum"),
            n_stints=("duration", "count"),
            avg_shift_age=("home_shift_age", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["toi_sec"] >= min_toi_sec].copy()
    grouped["toi_min"] = grouped["toi_sec"] / 60
    grouped["xgf_pct"] = grouped["xg_for"] / (grouped["xg_for"] + grouped["xg_against"]).replace(0, np.nan)
    grouped["xg_diff_per60"] = (grouped["xg_for"] - grouped["xg_against"]) / grouped["toi_sec"] * 3600
    grouped["cf_pct"] = grouped["corsi_for"] / (grouped["corsi_for"] + grouped["corsi_against"]).replace(0, np.nan)

    return grouped.sort_values("xg_diff_per60", ascending=False)


def get_line_decay_by_bucket(season: str, skater_combo: tuple[int, ...]) -> pd.DataFrame:
    ## for a specific line combo, returns xg stats broken out by shift-age bucket
    ## this is what powers the decay curve on the dashboard

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    ## match on home_skaters as a frozenset so ordering doesn't matter
    target = frozenset(skater_combo)
    mask = ev["home_skaters"].apply(lambda s: frozenset(s) == target)
    line_df = ev[mask].copy()

    if line_df.empty:
        logger.warning(f"No 5v5 stints found for combo {skater_combo}")
        return pd.DataFrame()

    line_df["shift_bucket"] = (line_df["home_shift_age"] // cfg.shift_age_bucket) * cfg.shift_age_bucket
    line_df["shift_bucket"] = line_df["shift_bucket"].clip(upper=180)

    bucketed = (
        line_df.groupby("shift_bucket")
        .agg(
            toi_sec=("duration", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            n_stints=("duration", "count"),
        )
        .reset_index()
    )

    bucketed["xg_diff_per60"] = (bucketed["xg_for"] - bucketed["xg_against"]) / bucketed["toi_sec"] * 3600
    ## simple confidence interval via Poisson approximation on xG counts
    bucketed["xg_diff_se"] = np.sqrt(
        bucketed["xg_for"] + bucketed["xg_against"]
    ) / bucketed["toi_sec"] * 3600

    return bucketed.sort_values("shift_bucket")


def get_top_lines(season: str, n: int = 20, min_toi_min: float = 5.0) -> pd.DataFrame:
    stats = get_line_stats(season, min_toi_sec=int(min_toi_min * 60))
    return stats.head(n)


def get_overused_lines(season: str, min_toi_min: float = 5.0) -> pd.DataFrame:
    ## "overused" lines: avg shift age is high relative to their xg_diff trend
    ## we flag lines where the last third of their avg shift shows declining xg
    ## TODO: make this smarter once the R decay coefficients are at line level

    stats = get_line_stats(season, min_toi_sec=int(min_toi_min * 60))
    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    results = []
    for _, row in stats.iterrows():
        combo = row["home_skaters"]
        target = frozenset(combo)
        mask = ev["home_skaters"].apply(lambda s: frozenset(s) == target)
        line_df = ev[mask].copy()

        if len(line_df) < 5:
            continue

        ## compare xg_diff in early shifts (0-30s) vs late shifts (>45s)
        early = line_df[line_df["home_shift_age"] <= 30]
        late  = line_df[line_df["home_shift_age"] > 45]

        if early.empty or late.empty:
            continue

        def _xgd60(d):
            sec = d["duration"].sum()
            if sec == 0:
                return np.nan
            return (d["xg_for"].sum() - d["xg_against"].sum()) / sec * 3600

        early_xgd = _xgd60(early)
        late_xgd  = _xgd60(late)

        results.append({
            "home_skaters":   combo,
            "toi_min":        row["toi_min"],
            "xg_diff_per60":  row["xg_diff_per60"],
            "early_xgd60":    early_xgd,
            "late_xgd60":     late_xgd,
            "decay_delta":    late_xgd - early_xgd,
            "overuse_flag":   late_xgd < early_xgd - 1.0,  ## >1 xGD/60 drop is meaningful
        })

    return pd.DataFrame(results).sort_values("decay_delta")
