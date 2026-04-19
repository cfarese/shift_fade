## Reads the R-generated RAPM results back into Python so the API
## and dashboard can query them without re-running the R model.
##
## Also handles the decay curve math since it's just arithmetic on top
## of the coefficients, no need to round-trip back to R for that.

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import cfg


@lru_cache(maxsize=4)
def load_rapm(season: str) -> pd.DataFrame:
    path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"RAPM results not found: {path}")
    df = pd.read_parquet(path)
    logger.debug(f"Loaded {len(df)} player RAPM rows for {season}")
    return df


def get_player_rapm(season: str, player_id: int) -> Optional[dict]:
    df = load_rapm(season)
    row = df[df["player_id"] == player_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def compute_decay_curve(
    rapm_base: float,
    rapm_decay: float,
    max_seconds: int = 90,
    bucket_size: int = 5,
) -> tuple[list[int], list[float]]:
    ## fitted xg_diff at time t is: rapm_base + rapm_decay * t
    ## this is the linear approximation, good enough for now
    buckets = list(range(0, max_seconds + 1, bucket_size))
    values  = [rapm_base + rapm_decay * t for t in buckets]
    return buckets, values


def get_break_even_second(rapm_base: float, rapm_decay: float) -> Optional[int]:
    ## when does rapm_base + rapm_decay * t == 0?
    ## t = -rapm_base / rapm_decay, only valid if decay is negative
    import math
    if math.isnan(rapm_base) or math.isnan(rapm_decay):
        return None
    if rapm_decay >= 0:
        return None
    t = -rapm_base / rapm_decay
    if t < 0 or math.isnan(t) or math.isinf(t):
        return None
    return int(t)


def get_overuse_report(season: str, min_toi: float = 50.0) -> pd.DataFrame:
    df = load_rapm(season)
    df = df[df["toi_5v5"] >= min_toi].copy()

    df["break_even_sec"] = df.apply(
        lambda r: get_break_even_second(r["rapm_base"], r["rapm_decay"]),
        axis=1,
    )

    ## flag anyone whose break-even is earlier than the league average shift
    avg_shift_sec = 45
    df["overused_at_avg"] = df["break_even_sec"].apply(
        lambda t: t is not None and t < avg_shift_sec
    )

    return df.sort_values("break_even_sec", na_position="last")
