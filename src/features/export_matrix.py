## Reads the processed stint parquet, builds the RAPM design matrix,
## and saves it so the R model can pick it up.
##
## This is the handoff point between the Python and R sides of the pipeline.
##
## Usage:
##   python -m src.features.export_matrix --season 20232024

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger

from config.settings import cfg
from src.features.stint_features import build_rapm_matrix


def export(season: str) -> None:
    stint_path = cfg.paths.processed / f"stints_{season}.parquet"
    if not stint_path.exists():
        logger.error(f"Stint file not found: {stint_path}")
        logger.error("Run src.ingestion.pipeline first")
        sys.exit(1)

    logger.info(f"Loading stints from {stint_path}")
    df = pd.read_parquet(stint_path)
    logger.info(f"Loaded {len(df)} stints")

    matrix, player_cols = build_rapm_matrix(df)
    logger.info(f"Matrix shape: {matrix.shape}, players: {len(player_cols)}")

    out_path = cfg.paths.processed / f"rapm_matrix_{season}.parquet"
    matrix.to_parquet(out_path, index=False)
    logger.success(f"Saved matrix to {out_path}")

    ## also write the player column list separately so R doesn't have to re-derive it
    col_path = cfg.paths.processed / f"player_cols_{season}.txt"
    col_path.write_text("\n".join(player_cols))
    logger.success(f"Saved {len(player_cols)} player column names to {col_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="20232024")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    export(args.season)


if __name__ == "__main__":
    main()
