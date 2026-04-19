## Fetches and caches player name/team info from the NHL API.
##
## The RAPM model only knows player IDs. This module resolves them to
## names and teams so the dashboard can display something readable.
## Results get cached to a JSON file so we don't re-hit the API every run.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import cfg
from src.ingestion.nhl_client import NHLClient

_CACHE_FILE = cfg.paths.cache / "player_names.json"


def _load_cache() -> dict[str, dict]:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text())
    return {}


def _save_cache(data: dict[str, dict]) -> None:
    _CACHE_FILE.write_text(json.dumps(data, indent=2))


def resolve_player_names(player_ids: list[int], game_ids: list[int]) -> dict[int, dict]:
    ## returns {player_id: {"name": "...", "team": "...", "position": "..."}}
    ## pulls from cache first, only hits API for missing IDs

    cache = _load_cache()
    missing = [pid for pid in player_ids if str(pid) not in cache]

    if missing:
        logger.info(f"Resolving {len(missing)} player IDs from API via game rosters")
        _fill_from_rosters(missing, game_ids, cache)
        _save_cache(cache)

    result = {}
    for pid in player_ids:
        entry = cache.get(str(pid))
        if entry:
            result[pid] = entry
        else:
            ## still missing after API call, use fallback
            result[pid] = {"name": f"Player_{pid}", "team": "UNK", "position": "UNK"}

    return result


def _fill_from_rosters(
    target_ids: list[int],
    game_ids: list[int],
    cache: dict[str, dict],
) -> None:
    ## scan game boxscores until we've found all the target IDs
    ## stops early once everyone is resolved so we don't pull 1312 games
    remaining = set(target_ids)

    with NHLClient() as client:
        for gid in game_ids:
            if not remaining:
                break
            try:
                boxscore = client.get_game_roster(gid)
                _parse_boxscore(boxscore, remaining, cache)
            except Exception as e:
                logger.debug(f"Boxscore fetch failed for game {gid}: {e}")
                continue

    if remaining:
        logger.warning(f"Could not resolve {len(remaining)} player IDs: {remaining}")


def _parse_boxscore(boxscore: dict, remaining: set[int], cache: dict[str, dict]) -> None:
    ## NHL boxscore has playerByGameStats nested under home/away
    for side in ("homeTeam", "awayTeam"):
        team_data = boxscore.get(side, {})
        team_abbrev = team_data.get("abbrev", "UNK")

        ## forwards, defense, goalies are separate lists
        for position_group in ("forwards", "defense", "goalies"):
            for player in team_data.get(position_group, []):
                pid = player.get("playerId")
                if pid is None:
                    continue
                pid = int(pid)
                if pid in remaining:
                    fname = player.get("firstName", {}).get("default", "")
                    lname = player.get("lastName", {}).get("default", "")
                    cache[str(pid)] = {
                        "name": f"{fname} {lname}".strip(),
                        "team": team_abbrev,
                        "position": player.get("position", "UNK"),
                    }
                    remaining.discard(pid)


def get_cached_names() -> dict[int, dict]:
    ## convenience wrapper for when you just want whatever is already on disk
    cache = _load_cache()
    return {int(k): v for k, v in cache.items()}
