## Thin wrapper around the NHL web API.
##
## The v1 API (api-web.nhle.com) is basically undocumented so some endpoint
## shapes here were reverse-engineered from sniffing the NHL app traffic.
## The older stats API is still live but has slightly different schemas for
## overlapping data, so everything goes through this one client.

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import cfg


_CURRENT_TEAM_CODES = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
    "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD",
    "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS",
    "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
]


class NHLClient:

    def __init__(self):
        self._http = httpx.Client(
            timeout=cfg.nhl_api.request_timeout,
            headers={"User-Agent": "hockey-analytics-research/0.1"},
        )
        self._last_call: float = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < cfg.nhl_api.rate_limit_sleep:
            time.sleep(cfg.nhl_api.rate_limit_sleep - elapsed)
        self._last_call = time.monotonic()

    @retry(
        stop=stop_after_attempt(cfg.nhl_api.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None = None) -> Any:
        self._throttle()
        logger.debug(f"GET {url} params={params}")
        r = self._http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def get_season_game_ids(self, season: str) -> list[int]:
        cache_path = cfg.paths.cache / f"season_game_ids_{season}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
                game_ids = sorted({int(gid) for gid in cached})
                if game_ids:
                    return game_ids
            except Exception as e:
                logger.warning(f"Failed reading cached game IDs from {cache_path}: {e}")

        try:
            game_ids = self._fetch_scheduled_game_ids(season)
            if game_ids:
                cache_path.write_text(json.dumps(game_ids))
                return game_ids
            logger.warning(f"Schedule lookup returned no game IDs for {season}; falling back to guessed IDs")
        except Exception as e:
            logger.warning(f"Schedule lookup failed for {season}: {e}; falling back to guessed IDs")

        ## fallback keeps older behavior if the schedule endpoints change
        ## game IDs are 10 digits: YYYY02NNNN
        ## e.g. 2023020001 is game 1 of the 2023-24 regular season
        season_prefix = int(season[:4]) * 1_000_000 + 20_000
        return [season_prefix + n for n in range(1, 1313)]

    def _fetch_scheduled_game_ids(self, season: str) -> list[int]:
        game_ids: set[int] = set()

        for team in self._season_team_codes(season):
            payload = self.get_team_schedule(team, season)
            team_ids = self._extract_game_ids_from_schedule(payload, season)
            if not team_ids:
                logger.warning(f"No regular-season game IDs found in {team} schedule for {season}")
                continue
            game_ids.update(team_ids)

        return sorted(game_ids)

    def get_team_schedule(self, team: str, season: str) -> dict:
        url = f"{cfg.nhl_api.base_url}/club-schedule-season/{team}/{season}"
        return self._get(url)

    @staticmethod
    def _season_team_codes(season: str) -> list[str]:
        teams = list(_CURRENT_TEAM_CODES)
        if season < "20242025":
            teams[teams.index("UTA")] = "ARI"
        return teams

    @staticmethod
    def _extract_game_ids_from_schedule(payload: Any, season: str) -> list[int]:
        ids: set[int] = set()
        prefix = f"{season[:4]}02"

        def visit(node: Any):
            if isinstance(node, list):
                for item in node:
                    visit(item)
                return

            if not isinstance(node, dict):
                return

            raw_gid = node.get("id", node.get("gameId"))
            if raw_gid is not None:
                try:
                    gid = int(raw_gid)
                except (TypeError, ValueError):
                    gid = None
                if gid is not None and str(gid).startswith(prefix):
                    ids.add(gid)

            for value in node.values():
                if isinstance(value, (list, dict)):
                    visit(value)

        visit(payload)
        return sorted(ids)

    def get_play_by_play(self, game_id: int) -> dict:
        ## returns the raw API dict, callers handle parsing
        url = f"{cfg.nhl_api.base_url}/gamecenter/{game_id}/play-by-play"
        return self._get(url)

    def get_shifts(self, game_id: int) -> list[dict]:
        ## shift chart: one row per player shift with start/end times
        ## this is how we reconstruct who was on ice at any given second
        url = f"{cfg.nhl_api.stats_url}/shiftcharts"
        data = self._get(url, params={"cayenneExp": f"gameId={game_id}"})
        shifts = data.get("data", [])
        if not shifts:
            shifts = self._get_shifts_from_html(game_id)
        return shifts

    def _get_shifts_from_html(self, game_id: int) -> list[dict]:
        ## fallback for games missing from the stats API
        ## scrapes the official NHL HTML time-on-ice reports
        import re
        try:
            pbp = self.get_play_by_play(game_id)
        except Exception:
            return []

        ## build jersey# -> (playerId, teamId) from rosterSpots
        jersey_map: dict[tuple[int, int], tuple[int, int]] = {}
        home_team_id = pbp.get("homeTeam", {}).get("id")
        away_team_id = pbp.get("awayTeam", {}).get("id")
        for spot in pbp.get("rosterSpots", []):
            pid  = spot.get("playerId")
            tid  = spot.get("teamId")
            num  = spot.get("sweaterNumber")
            if None not in (pid, tid, num):
                jersey_map[(int(tid), int(num))] = (int(pid), int(tid))

        ## HTML game number is the last 6 digits of the 10-digit game_id
        start_year = int(str(game_id)[:4])
        season_str = f"{start_year}{start_year + 1}"
        game_num   = str(game_id)[-6:]

        shifts: list[dict] = []
        for report_type, team_id in [("TH", home_team_id), ("TV", away_team_id)]:
            if team_id is None:
                continue
            url = f"https://www.nhl.com/scores/htmlreports/{season_str}/{report_type}{game_num}.HTM"
            try:
                self._throttle()
                r = self._http.get(url)
                r.raise_for_status()
            except Exception as e:
                logger.debug(f"HTML shift report unavailable for {game_id} {report_type}: {e}")
                continue

            ## each <td> is on its own line -- collect them into rows
            current_jersey: int | None = None
            in_row = False
            row_cells: list[str] = []
            is_heading_row = False

            def to_mmss(t: str) -> str:
                m2, s = t.split(":")
                return f"{int(m2):02d}:{s}"

            for line in r.text.splitlines():
                ## player heading: <td class="playerHeading...">4 BYRAM, BOWEN</td>
                if "playerHeading" in line:
                    m = re.search(r'>\s*(\d+)\s+[A-Z]', line)
                    if m:
                        current_jersey = int(m.group(1))
                    in_row = False
                    row_cells = []
                    continue

                ## start of a table row
                if re.match(r'\s*<tr[\s>]', line, re.IGNORECASE):
                    in_row = True
                    row_cells = []
                    is_heading_row = "heading" in line.lower()
                    continue

                if re.match(r'\s*</tr', line, re.IGNORECASE):
                    ## process completed row: shift#, period, start, end, duration, [event]
                    if not is_heading_row and len(row_cells) >= 5 and current_jersey is not None:
                        raw_per   = row_cells[1]
                        raw_start = row_cells[2]
                        raw_end   = row_cells[3]
                        if raw_per.isdigit():
                            period = int(raw_per)
                            start_part = raw_start.split("/")[0].strip()
                            end_part   = raw_end.split("/")[0].strip()
                            if 1 <= period <= 3 and re.match(r'^\d+:\d{2}$', start_part):
                                key = (int(team_id), current_jersey)
                                if key in jersey_map:
                                    pid, tid = jersey_map[key]
                                    shifts.append({
                                        "playerId":  pid,
                                        "teamId":    tid,
                                        "period":    period,
                                        "startTime": to_mmss(start_part),
                                        "endTime":   to_mmss(end_part),
                                    })
                    in_row = False
                    row_cells = []
                    continue

                if in_row and not is_heading_row:
                    m = re.search(r'<td[^>]*>(.*?)</td>', line, re.IGNORECASE)
                    if m:
                        cell = re.sub(r'<[^>]+>', '', m.group(1)).replace("&nbsp;", "").strip()
                        row_cells.append(cell)

        logger.debug(f"HTML fallback: {len(shifts)} shifts for game {game_id}")
        return shifts

    def get_game_roster(self, game_id: int) -> dict:
        ## game-level roster with jersey numbers and positions
        url = f"{cfg.nhl_api.base_url}/gamecenter/{game_id}/boxscore"
        return self._get(url)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
