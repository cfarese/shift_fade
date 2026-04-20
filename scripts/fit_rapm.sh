#!/usr/bin/env bash
## Run after ingestion is complete to produce RAPM results for a season.
## Usage: bash scripts/fit_rapm.sh 20252026

set -euo pipefail

SEASON="${1:-20252026}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Exporting RAPM matrix for $SEASON"
python3.11 -m src.features.export_matrix --season "$SEASON"

echo "==> Fitting RAPM model in R for $SEASON"
Rscript r/rapm/rapm_model.R --season "$SEASON" --alpha 25.0

echo "==> Resolving player names for $SEASON"
python3.11 -m src.ingestion.resolve_names --season "$SEASON"

echo "==> Done. Restart the dashboard server to load the new season."
