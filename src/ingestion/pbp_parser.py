## Converts raw NHL PBP + shift chart data into a tidy DataFrame of stints.
##
## The v1 NHL API has no "change" events or on-ice player arrays per play.
## We reconstruct on-ice lineups using the shift chart endpoint, then
## cross-reference with PBP plays to attach shot/goal stats and zone starts.
##
## Key design decision: build shift intervals per-period, not globally.
## Periods share the same 0-1200 second range so you MUST separate them
## or you'll mix all three periods' shifts into every interval.
##
## situationCode in each play: [hGoalie][hSkaters][aSkaters][aGoalie]
##   "1551" = 5v5, "1451" = 4v5 (home PK), "1541" = PP, etc.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


_XG_BY_TYPE: dict[str, float] = {
    "shot-on-goal": 0.075,
    "missed-shot":  0.025,
    "blocked-shot": 0.015,
    "goal":         0.075,
}

_SHOT_EVENTS = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}


@dataclass
class Stint:
    game_id: int
    game_date: str
    home_team: str
    away_team: str
    period: int
    start_sec: int
    end_sec: int
    home_skaters: frozenset[int]
    away_skaters: frozenset[int]
    score_diff: int
    zone_start: Optional[str]
    strength: str
    corsi_for: int = 0
    corsi_against: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    home_shift_age: int = 0
    away_shift_age: int = 0

    @property
    def duration(self) -> int:
        return max(0, self.end_sec - self.start_sec)


def _mmss_to_sec(mmss: str) -> int:
    parts = mmss.split(":")
    if len(parts) != 2:
        return 0
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 0


class PBPParser:

    def __init__(self, game_id: int, raw: dict, shifts: list[dict]):
        self.game_id      = game_id
        self.raw          = raw
        self.shifts       = shifts
        self.home_team_id = raw.get("homeTeam", {}).get("id")
        self.away_team_id = raw.get("awayTeam", {}).get("id")
        self.game_date    = (
            raw.get("gameDate")
            or str(raw.get("gameDateTime", ""))[:10]
            or str(raw.get("startTimeUTC", ""))[:10]
            or ""
        )
        self.home_team_abbrev = self._team_abbrev(raw.get("homeTeam", {}))
        self.away_team_abbrev = self._team_abbrev(raw.get("awayTeam", {}))

        ## build goalie set from rosterSpots so we can exclude them
        self._goalie_ids: set[int] = {
            int(r["playerId"])
            for r in raw.get("rosterSpots", [])
            if r.get("positionCode") == "G"
        }

        ## build player -> team map for fast lookup
        self._player_team: dict[int, int] = {
            int(r["playerId"]): int(r["teamId"])
            for r in raw.get("rosterSpots", [])
        }

    @staticmethod
    def _team_abbrev(team: dict) -> str:
        if not team:
            return "?"
        return (
            team.get("abbrev")
            or team.get("triCode")
            or team.get("abbrevName", {}).get("default")
            or str(team.get("placeName", {}).get("default", "?"))[:3].upper()
        )

    def parse(self) -> list[Stint]:
        if not self.shifts:
            logger.warning(f"Game {self.game_id}: no shift data")
            return []

        ## group shifts by period -- critical to avoid cross-period time collisions
        by_period: dict[int, list[dict]] = {}
        for sh in self.shifts:
            per = sh.get("period", 0)
            if per < 1 or per > 3:
                continue
            by_period.setdefault(per, []).append(sh)

        all_stints: list[Stint] = []
        for period, period_shifts in sorted(by_period.items()):
            stints = self._build_period_stints(period, period_shifts)
            all_stints.extend(stints)

        ## attach PBP events after all periods are built
        all_stints = self._attach_pbp(all_stints)

        return [s for s in all_stints if s.duration >= 5]

    def _build_period_stints(self, period: int, shifts: list[dict]) -> list[Stint]:
        ## parse shifts into (player_id, team_id, start_sec, end_sec) -- skaters only
        records: list[tuple[int, int, int, int]] = []
        for sh in shifts:
            pid = sh.get("playerId")
            tid = sh.get("teamId")
            if None in (pid, tid):
                continue
            pid = int(pid)
            ## skip goalies -- we want skater-only sets for RAPM
            if pid in self._goalie_ids:
                continue
            s = _mmss_to_sec(sh.get("startTime", "0:00"))
            e = _mmss_to_sec(sh.get("endTime",   "0:00"))
            if e <= s:
                continue
            records.append((pid, int(tid), s, e))

        if not records:
            return []

        ## find all times where on-ice personnel changes within this period
        change_points = sorted(set(t for _, _, s, e in records for t in (s, e)))

        stints: list[Stint] = []
        group_start: dict[frozenset, int] = {}

        for i in range(len(change_points) - 1):
            t_start = change_points[i]
            t_end   = change_points[i + 1]
            mid     = (t_start + t_end) / 2.0

            home_on: set[int] = set()
            away_on: set[int] = set()

            for pid, tid, s, e in records:
                if s <= mid < e:
                    if tid == self.home_team_id:
                        home_on.add(pid)
                    elif tid == self.away_team_id:
                        away_on.add(pid)

            if not home_on or not away_on:
                continue

            h_set = frozenset(home_on)
            a_set = frozenset(away_on)

            if h_set not in group_start:
                group_start[h_set] = t_start
            if a_set not in group_start:
                group_start[a_set] = t_start

            home_shift_age = max(0, t_start - group_start[h_set])
            away_shift_age = max(0, t_start - group_start[a_set])
            strength = f"{len(home_on)}v{len(away_on)}"

            stints.append(Stint(
                game_id=self.game_id,
                game_date=self.game_date,
                home_team=self.home_team_abbrev,
                away_team=self.away_team_abbrev,
                period=period,
                start_sec=t_start,
                end_sec=t_end,
                home_skaters=h_set,
                away_skaters=a_set,
                score_diff=0,
                zone_start=None,
                strength=strength,
                home_shift_age=home_shift_age,
                away_shift_age=away_shift_age,
            ))

        return stints

    def _attach_pbp(self, stints: list[Stint]) -> list[Stint]:
        if not stints:
            return stints

        plays      = self.raw.get("plays", [])
        score_diff = 0

        for play in plays:
            event_type = play.get("typeDescKey", "")
            t          = _mmss_to_sec(play.get("timeInPeriod", "0:00"))
            period     = play.get("periodDescriptor", {}).get("number", 1)
            details    = play.get("details", {})

            if period > 3:
                continue

            idx = self._find_stint(stints, period, t)

            if event_type == "faceoff" and idx is not None:
                if stints[idx].zone_start is None:
                    stints[idx].zone_start = details.get("zoneCode")

            elif event_type in _SHOT_EVENTS:
                if idx is None:
                    continue
                s          = stints[idx]
                owner_team = details.get("eventOwnerTeamId")
                is_home    = owner_team == self.home_team_id
                xg         = _XG_BY_TYPE.get(event_type, 0.0)

                if is_home:
                    s.corsi_for  += 1
                    s.xg_for     += xg
                else:
                    s.corsi_against += 1
                    s.xg_against    += xg

                if event_type == "goal":
                    score_diff += 1 if is_home else -1
                    ## update score_diff on all subsequent stints
                    for j in range(idx + 1, len(stints)):
                        stints[j].score_diff += 1 if is_home else -1

        return stints

    def _find_stint(self, stints: list[Stint], period: int, t: int) -> Optional[int]:
        for i, s in enumerate(stints):
            if s.period == period and s.start_sec <= t < s.end_sec:
                return i
        return None


def stints_to_dataframe(stints: list[Stint]) -> pd.DataFrame:
    if not stints:
        return pd.DataFrame()

    rows = []
    for s in stints:
        rows.append({
            "game_id":        s.game_id,
            "game_date":      s.game_date,
            "home_team":      s.home_team,
            "away_team":      s.away_team,
            "period":         s.period,
            "start_sec":      s.start_sec,
            "end_sec":        s.end_sec,
            "duration":       s.duration,
            "home_skaters":   tuple(sorted(s.home_skaters)),
            "away_skaters":   tuple(sorted(s.away_skaters)),
            "score_diff":     s.score_diff,
            "zone_start":     s.zone_start,
            "strength":       s.strength,
            "home_shift_age": s.home_shift_age,
            "away_shift_age": s.away_shift_age,
            "corsi_for":      s.corsi_for,
            "corsi_against":  s.corsi_against,
            "xg_for":         s.xg_for,
            "xg_against":     s.xg_against,
        })

    df = pd.DataFrame(rows)
    cf_total = df["corsi_for"] + df["corsi_against"]
    xg_total = df["xg_for"] + df["xg_against"]
    df["cf_pct"]  = df["corsi_for"] / cf_total.replace(0, np.nan)
    df["xgf_pct"] = df["xg_for"]   / xg_total.replace(0, np.nan)

    return df
