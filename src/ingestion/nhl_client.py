## Thin wrapper around the NHL web API.
##
## The v1 API (api-web.nhle.com) is basically undocumented so some endpoint
## shapes here were reverse-engineered from sniffing the NHL app traffic.
## The older stats API is still live but has slightly different schemas for
## overlapping data, so everything goes through this one client.

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import cfg


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
        ## game IDs are 10 digits: YYYY02NNNN
        ## e.g. 2023020001 is game 1 of the 2023-24 regular season
        ## season[:4] gives the start year (2023 from "20232024")
        season_prefix = int(season[:4]) * 1_000_000 + 20_000
        ## 1312 regular season games per 32-team season
        return [season_prefix + n for n in range(1, 1313)]

    def get_play_by_play(self, game_id: int) -> dict:
        ## returns the raw API dict, callers handle parsing
        url = f"{cfg.nhl_api.base_url}/gamecenter/{game_id}/play-by-play"
        return self._get(url)

    def get_shifts(self, game_id: int) -> list[dict]:
        ## shift chart: one row per player shift with start/end times
        ## this is how we reconstruct who was on ice at any given second
        url = f"{cfg.nhl_api.stats_url}/shiftcharts"
        data = self._get(url, params={"cayenneExp": f"gameId={game_id}"})
        return data.get("data", [])

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
