## Orchestrates the full ingestion pipeline for a given season.
##   1. fetch game IDs from the NHL API
##   2. for each game download PBP and parse into stints
##   3. save the aggregated stint DataFrame to parquet
##
## Usage:
##   python -m src.ingestion.pipeline --season 20232024 --limit 50

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from config.settings import cfg
from src.ingestion.nhl_client import NHLClient
from src.ingestion.pbp_parser import PBPParser, stints_to_dataframe
from src.features.stint_features import add_shift_age_features


def run_season(season: str, limit: int | None = None) -> Path:
    logger.info(f"Starting ingestion for season {season}")

    out_path = cfg.paths.processed / f"stints_{season}.parquet"
    if out_path.exists():
        logger.info(f"Found cached output at {out_path}, skipping re-ingestion")
        return out_path

    all_stints: list[pd.DataFrame] = []
    failed: list[int] = []

    with NHLClient() as client:
        game_ids = client.get_season_game_ids(season)

        if limit:
            game_ids = game_ids[:limit]
            logger.info(f"Limiting to {limit} games for dev run")

        for i, gid in enumerate(game_ids):
            try:
                raw    = client.get_play_by_play(gid)
                shifts = client.get_shifts(gid)
                parser = PBPParser(game_id=gid, raw=raw, shifts=shifts)
                stints = parser.parse()

                if stints:
                    df = stints_to_dataframe(stints)
                    all_stints.append(df)

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(game_ids)} games")

            except Exception as e:
                logger.warning(f"Game {gid} failed: {e}")
                failed.append(gid)
                continue

    if not all_stints:
        logger.error("No stints collected, check API or game IDs")
        sys.exit(1)

    combined = pd.concat(all_stints, ignore_index=True)
    combined = add_shift_age_features(combined)

    combined.to_parquet(out_path, index=False)
    logger.success(f"Saved {len(combined)} stints to {out_path}")

    if failed:
        fail_path = cfg.paths.cache / f"failed_games_{season}.txt"
        fail_path.write_text("\n".join(str(g) for g in failed))
        logger.warning(f"{len(failed)} games failed, see {fail_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="NHL stint ingestion pipeline")
    parser.add_argument("--season", default="20232024")
    parser.add_argument("--limit", type=int, default=None, help="Cap game count for testing")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    run_season(args.season, args.limit)


if __name__ == "__main__":
    main()
