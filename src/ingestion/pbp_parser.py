## Converts raw NHL PBP JSON into a tidy DataFrame of stints.
##
## A stint is a contiguous stretch of play where the exact same set of
## skaters is on ice for both teams simultaneously. Every time there is
## a substitution, penalty, or stoppage that changes personnel we close
## the current stint and open a new one.
##
## Key output columns:
##   game_id, period, stint_start_sec, stint_end_sec,
##   home_skaters (frozenset of player IDs), away_skaters,
##   score_diff at stint start, corsi/xg for and against,
##   zone_start (O/N/D), shift_age (how long the line had been on already)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


## shot type codes that count for Corsi
_CORSI_EVENTS = {"shot-on-goal", "missed-shot", "blocked-shot"}

## rough xG values per shot type from Schuckers & Curro 2013, will upgrade later
_XG_BY_TYPE: dict[str, float] = {
    "shot-on-goal": 0.075,
    "missed-shot": 0.025,
    "blocked-shot": 0.015,
}


@dataclass
class Stint:
    game_id: int
    period: int
    start_sec: int
    end_sec: int
    home_skaters: frozenset[int]
    away_skaters: frozenset[int]
    score_diff: int
    zone_start: Optional[str]
    corsi_for: int = 0
    corsi_against: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    ## how long the home/away line combination had been on before this stint opened
    home_shift_age: int = 0
    away_shift_age: int = 0

    @property
    def duration(self) -> int:
        return self.end_sec - self.start_sec


class PBPParser:

    def __init__(self, game_id: int, raw: dict):
        self.game_id = game_id
        self.raw = raw
        ## tracks the second each player stepped on the ice
        self._shift_starts: dict[int, int] = {}

    def parse(self) -> list[Stint]:
        stints: list[Stint] = []

        plays = self.raw.get("plays", [])
        if not plays:
            logger.warning(f"Game {self.game_id}: no plays found in PBP")
            return stints

        home_on: frozenset[int] = frozenset()
        away_on: frozenset[int] = frozenset()
        period = 1
        score_diff = 0
        zone_start: Optional[str] = None
        stint_open_at: int = 0

        for play in plays:
            event_type = play.get("typeDescKey", "")
            time_in_period = self._to_seconds(play.get("timeInPeriod", "0:00"))
            period = play.get("periodDescriptor", {}).get("number", period)

            if event_type == "change" or event_type == "period-start":
                new_home, new_away = self._extract_on_ice(play)

                if (new_home != home_on or new_away != away_on) and home_on:
                    s = Stint(
                        game_id=self.game_id,
                        period=period,
                        start_sec=stint_open_at,
                        end_sec=time_in_period,
                        home_skaters=home_on,
                        away_skaters=away_on,
                        score_diff=score_diff,
                        zone_start=zone_start,
                        home_shift_age=self._line_age(home_on, stint_open_at),
                        away_shift_age=self._line_age(away_on, stint_open_at),
                    )
                    stints.append(s)

                self._update_shift_starts(home_on, away_on, new_home, new_away, time_in_period)
                home_on, away_on = new_home, new_away
                stint_open_at = time_in_period
                zone_start = None

            elif event_type == "faceoff":
                zone_start = play.get("details", {}).get("zoneCode")

            elif event_type in _CORSI_EVENTS:
                self._accumulate_shot(stints, play, home_on, event_type)

            elif event_type == "goal":
                self._accumulate_shot(stints, play, home_on, event_type)
                details = play.get("details", {})
                if details.get("eventOwnerTeamId") == self.raw.get("homeTeam", {}).get("id"):
                    score_diff += 1
                else:
                    score_diff -= 1

        ## drop micro-stints under 5 seconds, they are usually just change events firing twice
        return [s for s in stints if s.duration >= 5]

    def _to_seconds(self, mmss: str) -> int:
        parts = mmss.split(":")
        if len(parts) != 2:
            return 0
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return 0

    def _extract_on_ice(self, play: dict) -> tuple[frozenset[int], frozenset[int]]:
        home_ids: set[int] = set()
        away_ids: set[int] = set()
        for player in play.get("onIce", {}).get("home", []):
            pid = player.get("playerId")
            if pid:
                home_ids.add(int(pid))
        for player in play.get("onIce", {}).get("away", []):
            pid = player.get("playerId")
            if pid:
                away_ids.add(int(pid))
        return frozenset(home_ids), frozenset(away_ids)

    def _line_age(self, skaters: frozenset[int], now: int) -> int:
        ## group age = time since the last player stepped on (not the first)
        ## because the group wasn't complete until then
        if not skaters or not self._shift_starts:
            return 0
        starts = [self._shift_starts.get(p, now) for p in skaters]
        return now - max(starts)

    def _update_shift_starts(
        self,
        old_home: frozenset[int],
        old_away: frozenset[int],
        new_home: frozenset[int],
        new_away: frozenset[int],
        now: int,
    ):
        entering = (new_home | new_away) - (old_home | old_away)
        for pid in entering:
            self._shift_starts[pid] = now

    def _accumulate_shot(self, stints: list[Stint], play: dict, home_on: frozenset[int], event_type: str):
        if not stints:
            return
        s = stints[-1]
        shooter_team = play.get("details", {}).get("eventOwnerTeamId")
        home_team_id = self.raw.get("homeTeam", {}).get("id")
        is_home = shooter_team == home_team_id
        xg = _XG_BY_TYPE.get(event_type, 0.0)

        if is_home:
            s.corsi_for += 1
            s.xg_for += xg
        else:
            s.corsi_against += 1
            s.xg_against += xg


def stints_to_dataframe(stints: list[Stint]) -> pd.DataFrame:
    if not stints:
        return pd.DataFrame()

    rows = []
    for s in stints:
        rows.append({
            "game_id": s.game_id,
            "period": s.period,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "duration": s.duration,
            "home_skaters": tuple(sorted(s.home_skaters)),
            "away_skaters": tuple(sorted(s.away_skaters)),
            "score_diff": s.score_diff,
            "zone_start": s.zone_start,
            "home_shift_age": s.home_shift_age,
            "away_shift_age": s.away_shift_age,
            "corsi_for": s.corsi_for,
            "corsi_against": s.corsi_against,
            "xg_for": s.xg_for,
            "xg_against": s.xg_against,
        })

    df = pd.DataFrame(rows)
    df["cf_pct"] = df["corsi_for"] / (df["corsi_for"] + df["corsi_against"]).replace(0, np.nan)
    df["xgf_pct"] = df["xg_for"] / (df["xg_for"] + df["xg_against"]).replace(0, np.nan)

    return df
