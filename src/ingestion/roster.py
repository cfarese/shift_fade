## Resolves player IDs to full names and teams.
##
## Uses rosterSpots from the PBP endpoint (api-web.nhle.com/v1/gamecenter/{id}/play-by-play)
## which returns full first/last names and team info. Results are cached to
## data/cache/player_names.json so we don't re-hit the API on every run.

from __future__ import annotations

import json
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
    cache = _load_cache()
    missing = [pid for pid in player_ids if str(pid) not in cache]

    if missing:
        logger.info(f"Resolving {len(missing)} player IDs from PBP rosterSpots")
        _fill_from_pbp(missing, game_ids, cache)
        _save_cache(cache)

    result = {}
    for pid in player_ids:
        entry = cache.get(str(pid))
        result[pid] = entry if entry else {"name": f"Player_{pid}", "team": None, "position": None}

    return result


def _fill_from_pbp(
    target_ids: list[int],
    game_ids: list[int],
    cache: dict[str, dict],
) -> None:
    remaining = set(target_ids)

    with NHLClient() as client:
        for gid in game_ids:
            if not remaining:
                break
            try:
                raw = client.get_play_by_play(gid)
                _parse_roster_spots(raw, remaining, cache)
            except Exception as e:
                logger.debug(f"PBP fetch failed for game {gid}: {e}")
                continue

    if remaining:
        logger.warning(f"Could not resolve {len(remaining)} player IDs after scanning all games")


def _parse_roster_spots(raw: dict, remaining: set[int], cache: dict[str, dict]) -> None:
    ## rosterSpots has teamId, playerId, firstName/lastName, positionCode
    home_id = raw.get("homeTeam", {}).get("id")
    away_id = raw.get("awayTeam", {}).get("id")

    ## build a team id -> abbrev map from the response
    team_abbrev: dict[int, str] = {}
    for side in ("homeTeam", "awayTeam"):
        t = raw.get(side, {})
        if t.get("id") and t.get("abbrev"):
            team_abbrev[t["id"]] = t["abbrev"]

    for spot in raw.get("rosterSpots", []):
        pid = spot.get("playerId")
        if pid is None:
            continue
        pid = int(pid)
        if pid not in remaining:
            continue

        tid    = spot.get("teamId")
        fname  = spot.get("firstName", {}).get("default", "")
        lname  = spot.get("lastName",  {}).get("default", "")
        name   = f"{fname} {lname}".strip()
        team   = team_abbrev.get(tid, f"T{tid}")
        pos    = spot.get("positionCode", None)

        cache[str(pid)] = {"name": name, "team": team, "position": pos}
        remaining.discard(pid)


def get_cached_names() -> dict[int, dict]:
    cache = _load_cache()
    return {int(k): v for k, v in cache.items()}
