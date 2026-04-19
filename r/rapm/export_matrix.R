## export_matrix.R
##
## Reads the processed stint parquet and prints diagnostics before the RAPM fit.
## The actual matrix is built by Python (build_rapm_matrix in
## src/features/stint_features.py) and saved to
## data/processed/rapm_matrix_{season}.parquet.
## This script just validates it looks sane.

library(arrow)
library(dplyr)

season <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(season)) season <- "20232024"

path <- file.path("data", "processed", paste0("rapm_matrix_", season, ".parquet"))
if (!file.exists(path)) {
  stop(paste("Matrix not found:", path))
}

df <- read_parquet(path)
cat(sprintf("Matrix shape: %d rows x %d cols\n", nrow(df), ncol(df)))

player_cols <- grep("^p_\\d+$", names(df), value = TRUE)
cat(sprintf("Unique players: %d\n", length(player_cols)))
cat(sprintf("NA xg_diff_per60 rows: %d\n", sum(is.na(df$xg_diff_per60))))

cat("\nHome shift bucket distribution:\n")
print(table(df$home_shift_bucket))

cat("\nScore state distribution:\n")
print(table(df$score_state))
