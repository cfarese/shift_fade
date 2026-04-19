## Joins player names and teams into the RAPM results parquet.
## Run this after rapm_model.R since R doesn't know player names.
##
## Usage:
##   python3.11 -m src.ingestion.resolve_names --season 20232024

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger

from config.settings import cfg
from src.ingestion.roster import resolve_player_names


def resolve(season: str) -> None:
    rapm_path = cfg.paths.processed / f"rapm_results_{season}.parquet"
    if not rapm_path.exists():
        logger.error(f"RAPM results not found: {rapm_path}")
        sys.exit(1)

    stint_path = cfg.paths.processed / f"stints_{season}.parquet"
    if not stint_path.exists():
        logger.error(f"Stint data not found: {stint_path}")
        sys.exit(1)

    rapm = pd.read_parquet(rapm_path)
    player_ids = rapm["player_id"].dropna().astype(int).tolist()
    logger.info(f"Resolving names for {len(player_ids)} players")

    ## get game IDs from stint data to scan boxscores
    stints = pd.read_parquet(stint_path)
    game_ids = stints["game_id"].unique().tolist()
    logger.info(f"Scanning {len(game_ids)} game boxscores")

    name_map = resolve_player_names(player_ids, game_ids)

    rapm["player_name"] = rapm["player_id"].apply(
        lambda pid: name_map.get(int(pid), {}).get("name", f"Player_{pid}")
    )
    rapm["team"] = rapm["player_id"].apply(
        lambda pid: name_map.get(int(pid), {}).get("team", None)
    )

    rapm.to_parquet(rapm_path, index=False)
    resolved = rapm[rapm["player_name"] != rapm["player_id"].apply(lambda p: f"Player_{p}")]
    logger.success(f"Resolved {len(resolved)}/{len(rapm)} player names")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="20232024")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)
    resolve(args.season)


if __name__ == "__main__":
    main()
