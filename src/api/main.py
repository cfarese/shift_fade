## FastAPI backend for the dashboard.
##
## Most computation happens in R and gets cached to parquet. These endpoints
## just read those files back out. Stubs are defined upfront so the frontend
## has something to build against before the R models are wired in.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config.settings import cfg

app = FastAPI(
    title="Hockey RAPM Analytics API",
    version="0.1.0",
)

## wide open for local dev, lock this down before deploying anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerRAPM(BaseModel):
    player_id: int
    player_name: str
    rapm_base: float
    rapm_decay: float
    toi_5v5: float
    overuse_flag: bool


class LineDecayCurve(BaseModel):
    player_ids: list[int]
    player_names: list[str]
    shift_buckets: list[int]
    xg_diff_per60: list[float]
    confidence_low: list[float]
    confidence_high: list[float]
    ## second the line crosses zero xg diff, None if it never does
    break_even_second: Optional[int]


def _load_rapm_results(season: str) -> pd.DataFrame:
    path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"RAPM results for {season} not found. Run the R model first.",
        )
    return pd.read_parquet(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/players/rapm", response_model=list[PlayerRAPM])
def get_player_rapm(
    season: str = Query("20232024"),
    team: Optional[str] = Query(None),
    min_toi: float = Query(50.0, description="Minimum 5v5 TOI in minutes"),
):
    ## sorted: overuse flags first, then worst decayers
    df = _load_rapm_results(season)

    if team:
        df = df[df["team"].str.upper() == team.upper()]

    df = df[df["toi_5v5"] >= min_toi]

    results = []
    for _, row in df.iterrows():
        results.append(PlayerRAPM(
            player_id=int(row["player_id"]),
            player_name=row["player_name"],
            rapm_base=float(row["rapm_base"]),
            rapm_decay=float(row["rapm_decay"]),
            toi_5v5=float(row["toi_5v5"]),
            overuse_flag=bool(row["overuse_flag"]),
        ))

    results.sort(key=lambda x: (not x.overuse_flag, x.rapm_decay))
    return results


@app.get("/api/v1/lines/decay", response_model=LineDecayCurve)
def get_line_decay(
    player_ids: str = Query(..., description="Comma-separated player IDs"),
    season: str = Query("20232024"),
):
    ## TODO: wire up to R output
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/api/v1/teams/{team_abbrev}/overuse")
def get_team_overuse_report(
    team_abbrev: str,
    season: str = Query("20232024"),
):
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.get("/api/v1/stints/raw")
def get_raw_stints(
    season: str = Query("20232024"),
    game_id: Optional[int] = Query(None),
    limit: int = Query(1000, le=10000),
):
    ## debug endpoint
    path = cfg.paths.processed / f"stints_{season}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stint data not found")

    df = pd.read_parquet(path)
    if game_id:
        df = df[df["game_id"] == game_id]

    return df.head(limit).to_dict(orient="records")
