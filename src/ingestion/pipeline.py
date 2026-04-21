## Orchestrates the full ingestion pipeline for a given season.
##   1. fetch game IDs from the NHL API
##   2. for each game download PBP and parse into stints
##   3. save the aggregated stint DataFrame to parquet
##
## Usage:
##   python -m src.ingestion.pipeline --season 20232024 --limit 50
##
## Checkpointing: a partial parquet is written every CHECKPOINT_EVERY games so
## an interrupted run leaves usable data. Re-running resumes from the checkpoint
## (already-processed games are skipped via a completed-IDs sidecar file).

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

CHECKPOINT_EVERY = 100


def _checkpoint_path(season: str) -> Path:
    return cfg.paths.cache / f"stints_checkpoint_{season}.parquet"


def _done_ids_path(season: str) -> Path:
    return cfg.paths.cache / f"done_game_ids_{season}.txt"


def _load_checkpoint(season: str) -> tuple[pd.DataFrame | None, set[int]]:
    cp = _checkpoint_path(season)
    done_path = _done_ids_path(season)
    done: set[int] = set()
    if done_path.exists():
        try:
            done = {int(x) for x in done_path.read_text().splitlines() if x.strip()}
        except Exception:
            pass
    df = None
    if cp.exists() and done:
        try:
            df = pd.read_parquet(cp)
            logger.info(f"Resuming from checkpoint: {len(done)} games already done, {len(df)} stints loaded")
        except Exception:
            df = None
            done = set()
    return df, done


def _save_checkpoint(season: str, stints: list[pd.DataFrame], done: set[int]) -> None:
    try:
        combined = pd.concat(stints, ignore_index=True)
        combined = add_shift_age_features(combined)
        combined.to_parquet(_checkpoint_path(season), index=False)
        _done_ids_path(season).write_text("\n".join(str(g) for g in sorted(done)))
    except Exception as e:
        logger.warning(f"Checkpoint save failed: {e}")


def run_season(season: str, limit: int | None = None) -> Path:
    logger.info(f"Starting ingestion for season {season}")

    out_path = cfg.paths.processed / f"stints_{season}.parquet"
    if out_path.exists():
        logger.info(f"Found cached output at {out_path}, skipping re-ingestion")
        return out_path

    checkpoint_df, done_ids = _load_checkpoint(season)
    all_stints: list[pd.DataFrame] = []
    if checkpoint_df is not None and not checkpoint_df.empty:
        all_stints.append(checkpoint_df)

    failed: list[int] = []

    with NHLClient() as client:
        game_ids = client.get_season_game_ids(season)

        if limit:
            game_ids = game_ids[:limit]
            logger.info(f"Limiting to {limit} games for dev run")

        remaining = [g for g in game_ids if g not in done_ids]
        if len(remaining) < len(game_ids):
            logger.info(f"Skipping {len(game_ids) - len(remaining)} already-processed games")

        for i, gid in enumerate(remaining):
            try:
                raw    = client.get_play_by_play(gid)
                shifts = client.get_shifts(gid)
                parser = PBPParser(game_id=gid, raw=raw, shifts=shifts)
                stints = parser.parse()

                if stints:
                    df = stints_to_dataframe(stints)
                    all_stints.append(df)

                done_ids.add(gid)

                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(remaining)} remaining games")

                if (i + 1) % CHECKPOINT_EVERY == 0 and all_stints:
                    _save_checkpoint(season, all_stints, done_ids)
                    logger.debug(f"Checkpoint saved at game {i+1}")

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

    ## clean up checkpoint files now that we have a complete output
    _checkpoint_path(season).unlink(missing_ok=True)
    _done_ids_path(season).unlink(missing_ok=True)

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
