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


@lru_cache(maxsize=1)
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

    ## parquet reads tuple columns back as numpy arrays which aren't hashable
    ## convert to tuple of plain ints so groupby works
    ev["home_skaters"] = ev["home_skaters"].apply(lambda x: tuple(int(i) for i in x))

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
    ev["home_skaters"] = ev["home_skaters"].apply(lambda x: tuple(int(i) for i in x))

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


def get_forward_line_stats(
    season: str,
    pos_map: dict[int, str],
    min_toi_sec: int = 1200,
) -> pd.DataFrame:
    ## groups by the 3 non-D home skaters; stints where home has != 3 forwards are dropped

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    def fwd_trio(skaters) -> tuple[int, ...] | None:
        fwds = tuple(sorted(int(p) for p in skaters if pos_map.get(int(p), "F") != "D"))
        return fwds if len(fwds) == 3 else None

    ev["fwd_line"] = ev["home_skaters"].apply(fwd_trio)
    ev = ev[ev["fwd_line"].notna()].copy()

    if ev.empty:
        return pd.DataFrame()

    grouped = (
        ev.groupby("fwd_line")
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
    total_xg = (grouped["xg_for"] + grouped["xg_against"]).replace(0, np.nan)
    grouped["xgf_pct"] = grouped["xg_for"] / total_xg
    grouped["xg_diff_per60"] = (grouped["xg_for"] - grouped["xg_against"]) / grouped["toi_sec"] * 3600
    total_cf = (grouped["corsi_for"] + grouped["corsi_against"]).replace(0, np.nan)
    grouped["cf_pct"] = grouped["corsi_for"] / total_cf

    return grouped.sort_values("xg_diff_per60", ascending=False)


def get_forward_line_overuse(
    season: str,
    pos_map: dict[int, str],
    min_toi_min: float = 20.0,
) -> pd.DataFrame:
    stats = get_forward_line_stats(season, pos_map, min_toi_sec=int(min_toi_min * 60))
    if stats.empty:
        return pd.DataFrame()

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()

    def fwd_trio(skaters) -> tuple[int, ...] | None:
        fwds = tuple(sorted(int(p) for p in skaters if pos_map.get(int(p), "F") != "D"))
        return fwds if len(fwds) == 3 else None

    ev["fwd_line"] = ev["home_skaters"].apply(fwd_trio)
    ev = ev[ev["fwd_line"].notna()].copy()

    valid = set(stats["fwd_line"].tolist())
    ev = ev[ev["fwd_line"].isin(valid)]

    early_ev = ev[ev["home_shift_age"] <= 30]
    late_ev  = ev[ev["home_shift_age"] > 45]

    early = early_ev.groupby("fwd_line").agg(
        early_toi=("duration", "sum"),
        early_xgf=("xg_for", "sum"),
        early_xga=("xg_against", "sum"),
    )
    late = late_ev.groupby("fwd_line").agg(
        late_toi=("duration", "sum"),
        late_xgf=("xg_for", "sum"),
        late_xga=("xg_against", "sum"),
    )

    merged = (
        stats.set_index("fwd_line")
        .join(early, how="left")
        .join(late, how="left")
        .reset_index()
        .dropna(subset=["early_toi", "late_toi"])
    )
    merged = merged[(merged["early_toi"] >= 30) & (merged["late_toi"] >= 30)]

    if merged.empty:
        return pd.DataFrame()

    merged["early_xgd60"] = (merged["early_xgf"] - merged["early_xga"]) / merged["early_toi"] * 3600
    merged["late_xgd60"]  = (merged["late_xgf"]  - merged["late_xga"])  / merged["late_toi"]  * 3600
    merged["decay_delta"] = merged["late_xgd60"] - merged["early_xgd60"]
    merged["overuse_flag"] = merged["late_xgd60"] < merged["early_xgd60"] - 1.0

    cols = ["fwd_line", "toi_min", "xg_diff_per60", "early_xgd60", "late_xgd60", "decay_delta", "overuse_flag"]
    return merged[cols].sort_values("decay_delta").reset_index(drop=True)


def get_overused_lines(season: str, min_toi_min: float = 5.0) -> pd.DataFrame:
    ## compares early (0-30s) vs late (>45s) xGD/60 for each line combination
    ## uses vectorized groupby instead of per-combo scans -- O(n_stints) not O(n_combos*n_stints)

    stats = get_line_stats(season, min_toi_sec=int(min_toi_min * 60))
    if stats.empty:
        return pd.DataFrame()

    df = load_stints(season)
    ev = df[df["strength"] == "5v5"].copy()
    ev["home_skaters"] = ev["home_skaters"].apply(lambda x: tuple(int(i) for i in x))

    ## restrict to combos that passed the min_toi filter
    valid = set(stats["home_skaters"].tolist())
    ev = ev[ev["home_skaters"].isin(valid)]

    early_ev = ev[ev["home_shift_age"] <= 30]
    late_ev  = ev[ev["home_shift_age"] > 45]

    early = early_ev.groupby("home_skaters").agg(
        early_toi=("duration", "sum"),
        early_xgf=("xg_for", "sum"),
        early_xga=("xg_against", "sum"),
    )
    late = late_ev.groupby("home_skaters").agg(
        late_toi=("duration", "sum"),
        late_xgf=("xg_for", "sum"),
        late_xga=("xg_against", "sum"),
    )

    merged = (
        stats.set_index("home_skaters")
        .join(early, how="left")
        .join(late, how="left")
        .reset_index()
        .dropna(subset=["early_toi", "late_toi"])
    )
    merged = merged[(merged["early_toi"] >= 30) & (merged["late_toi"] >= 30)]

    if merged.empty:
        return pd.DataFrame()

    merged["early_xgd60"] = (merged["early_xgf"] - merged["early_xga"]) / merged["early_toi"] * 3600
    merged["late_xgd60"]  = (merged["late_xgf"]  - merged["late_xga"])  / merged["late_toi"]  * 3600
    merged["decay_delta"] = merged["late_xgd60"] - merged["early_xgd60"]
    merged["overuse_flag"] = merged["late_xgd60"] < merged["early_xgd60"] - 1.0

    cols = ["home_skaters", "toi_min", "xg_diff_per60", "early_xgd60", "late_xgd60", "decay_delta", "overuse_flag"]
    return merged[cols].sort_values("decay_delta").reset_index(drop=True)
