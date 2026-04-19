## FastAPI backend for the dashboard.
##
## Most computation happens in R and gets cached to parquet. These endpoints
## read those files back out or do lightweight math on top of the coefficients.

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from config.settings import cfg
from src.models.rapm_reader import (
    compute_decay_curve,
    get_break_even_second,
    get_overuse_report,
    load_rapm,
)

app = FastAPI(title="Hockey RAPM Analytics API", version="0.1.0")

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
    break_even_sec: Optional[int]


class PlayerDecayCurve(BaseModel):
    player_id: int
    player_name: str
    shift_buckets: list[int]
    xg_diff_per60: list[float]
    break_even_sec: Optional[int]


class OveruseEntry(BaseModel):
    player_id: int
    player_name: str
    rapm_base: float
    rapm_decay: float
    toi_5v5: float
    break_even_sec: Optional[int]
    overused_at_avg: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/players/rapm", response_model=list[PlayerRAPM])
def get_player_rapm(
    season: str = Query("20232024"),
    team: Optional[str] = Query(None),
    min_toi: float = Query(50.0),
):
    ## sorted: overuse flags first, then worst decayers
    try:
        df = load_rapm(season)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

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
            break_even_sec=get_break_even_second(row["rapm_base"], row["rapm_decay"]),
        ))

    results.sort(key=lambda x: (not x.overuse_flag, x.rapm_decay))
    return results


@app.get("/api/v1/players/{player_id}/decay", response_model=PlayerDecayCurve)
def get_player_decay(player_id: int, season: str = Query("20232024")):
    try:
        df = load_rapm(season)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    row = df[df["player_id"] == player_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")

    r = row.iloc[0]
    buckets, values = compute_decay_curve(r["rapm_base"], r["rapm_decay"])

    return PlayerDecayCurve(
        player_id=int(r["player_id"]),
        player_name=r["player_name"],
        shift_buckets=buckets,
        xg_diff_per60=values,
        break_even_sec=get_break_even_second(r["rapm_base"], r["rapm_decay"]),
    )


@app.get("/api/v1/teams/{team_abbrev}/overuse", response_model=list[OveruseEntry])
def get_team_overuse(
    team_abbrev: str,
    season: str = Query("20232024"),
    min_toi: float = Query(50.0),
):
    try:
        df = get_overuse_report(season, min_toi=min_toi)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    df = df[df["team"].str.upper() == team_abbrev.upper()]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for team {team_abbrev}")

    results = []
    for _, row in df.iterrows():
        results.append(OveruseEntry(
            player_id=int(row["player_id"]),
            player_name=row["player_name"],
            rapm_base=float(row["rapm_base"]),
            rapm_decay=float(row["rapm_decay"]),
            toi_5v5=float(row["toi_5v5"]),
            break_even_sec=int(row["break_even_sec"]) if pd.notna(row["break_even_sec"]) else None,
            overused_at_avg=bool(row["overused_at_avg"]),
        ))

    return results


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
