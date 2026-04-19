## Feature engineering on top of the raw stint DataFrame.
##
## The main thing added here is shift age bucketing. We bin the continuous
## shift_age variable into discrete intervals so the R RAPM model can estimate
## per-bucket efficiency rather than assuming linear decay. Linear is probably
## fine as a first pass but the real curve is likely convex (players tire
## faster near the end of a long shift than at the start).

from __future__ import annotations

import pandas as pd
import numpy as np

from config.settings import cfg


def add_shift_age_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    bucket = cfg.shift_age_bucket

    out["home_shift_bucket"] = (out["home_shift_age"] // bucket) * bucket
    out["away_shift_bucket"] = (out["away_shift_age"] // bucket) * bucket

    ## shifts beyond 3 min are rare enough that we lump them into one bucket
    max_bucket = 180
    out["home_shift_bucket"] = out["home_shift_bucket"].clip(upper=max_bucket)
    out["away_shift_bucket"] = out["away_shift_bucket"].clip(upper=max_bucket)

    ## score state clipped to [-2, 2] since beyond that it is garbage time
    out["score_state"] = out["score_diff"].clip(-2, 2).astype(int)

    out["home_n"] = out["home_skaters"].apply(len)
    out["away_n"] = out["away_skaters"].apply(len)
    out["strength"] = out["home_n"].astype(str) + "v" + out["away_n"].astype(str)

    zone_map = {"O": 1, "N": 0, "D": -1}
    out["zone_start_num"] = out["zone_start"].map(zone_map).fillna(0)

    return out


def build_rapm_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ## expands stint rows into the design matrix for ridge regression
    ## player columns are +1 for home, -1 for away (standard RAPM formulation from Macdonald 2012)

    ## base model is 5v5 only
    ev_df = df[df["strength"] == "5v5"].copy()

    all_player_ids: set[int] = set()
    for skaters in ev_df["home_skaters"]:
        all_player_ids.update(skaters)
    for skaters in ev_df["away_skaters"]:
        all_player_ids.update(skaters)

    player_cols = [f"p_{pid}" for pid in sorted(all_player_ids)]
    pid_to_col = {pid: f"p_{pid}" for pid in all_player_ids}

    ## TODO: switch to sparse matrix if this blows up RAM on a full season
    matrix = pd.DataFrame(0, index=ev_df.index, columns=player_cols, dtype=np.int8)

    for idx, row in ev_df.iterrows():
        for pid in row["home_skaters"]:
            if pid in pid_to_col:
                matrix.at[idx, pid_to_col[pid]] = 1
        for pid in row["away_skaters"]:
            if pid in pid_to_col:
                matrix.at[idx, pid_to_col[pid]] = -1

    matrix["score_state"] = ev_df["score_state"].values
    matrix["zone_start_num"] = ev_df["zone_start_num"].values
    matrix["home_shift_bucket"] = ev_df["home_shift_bucket"].values
    matrix["away_shift_bucket"] = ev_df["away_shift_bucket"].values
    matrix["duration"] = ev_df["duration"].values
    matrix["xgf_pct"] = ev_df["xgf_pct"].values
    matrix["xg_diff_per60"] = (
        (ev_df["xg_for"] - ev_df["xg_against"]) / ev_df["duration"] * 3600
    ).values

    return matrix, player_cols
