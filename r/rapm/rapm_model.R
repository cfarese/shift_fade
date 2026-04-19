## rapm_model.R
##
## Fits the shift-decay RAPM model using ridge regression (glmnet).
##
## Instead of one RAPM coefficient per player we estimate a base RAPM
## and a decay coefficient that captures how marginal impact changes as
## the shift gets longer. The interaction term is player * shift_age_bucket.
##
## Inputs:  data/processed/rapm_matrix_{season}.parquet
## Outputs: data/processed/rapm_results_{season}.parquet
##
## Run: Rscript r/rapm/rapm_model.R --season 20232024

library(glmnet)
library(arrow)
library(dplyr)
library(tidyr)
library(argparse)

## ---------------------------------------------------------------------------
## args
## ---------------------------------------------------------------------------

parser <- ArgumentParser()
parser$add_argument("--season", default = "20232024")
parser$add_argument("--alpha", type = "double", default = 25.0)
args <- parser$parse_args()

season <- args$season
alpha  <- args$alpha

cat(sprintf("Fitting RAPM for season %s, alpha=%.1f\n", season, alpha))

## ---------------------------------------------------------------------------
## load data
## ---------------------------------------------------------------------------

matrix_path <- file.path("data", "processed", paste0("rapm_matrix_", season, ".parquet"))
if (!file.exists(matrix_path)) {
  stop(sprintf("Matrix not found: %s\nRun the Python pipeline first.", matrix_path))
}

df <- read_parquet(matrix_path)
cat(sprintf("Loaded %d stints\n", nrow(df)))

## ---------------------------------------------------------------------------
## split columns by role
## ---------------------------------------------------------------------------

player_cols  <- grep("^p_\\d+$", names(df), value = TRUE)
control_cols <- c("score_state", "zone_start_num", "home_shift_bucket", "away_shift_bucket")
response_col <- "xg_diff_per60"

cat(sprintf("Players: %d  Controls: %d\n", length(player_cols), length(control_cols)))

## ---------------------------------------------------------------------------
## design matrix
## ---------------------------------------------------------------------------

## weight by duration so longer stints have more influence (proportionally)
weights <- df$duration / mean(df$duration)

X_players  <- as.matrix(df[, player_cols])
X_controls <- as.matrix(df[, control_cols])
X  <- cbind(X_players, X_controls)
y  <- df[[response_col]]

## drop NA response rows (zero-shot stints produce NaN xg_diff_per60)
valid_rows <- !is.na(y)
X       <- X[valid_rows, ]
y       <- y[valid_rows]
weights <- weights[valid_rows]

cat(sprintf("After NA drop: %d stints\n", sum(valid_rows)))

## ---------------------------------------------------------------------------
## base RAPM fit
## ---------------------------------------------------------------------------

## penalty.factor = 0 means do not regularize that column
## controls should not be penalized, player columns should be
penalty_factors <- c(
  rep(1, length(player_cols)),
  rep(0, length(control_cols))
)

## let cv.glmnet search its own lambda sequence -- passing a single value breaks CV
## alpha here is the config value used to nudge lambda.min upward for more regularization
cv_fit <- cv.glmnet(
  X, y,
  weights        = weights,
  alpha          = 0,             ## ridge (alpha=0), not lasso
  penalty.factor = penalty_factors,
  nfolds         = 5,
  type.measure   = "mse"
)

lambda_opt <- cv_fit$lambda.1se
cat(sprintf("Optimal lambda (1se): %.6f\n", lambda_opt))

final_fit <- glmnet(
  X, y,
  weights        = weights,
  alpha          = 0,
  lambda         = lambda_opt,
  penalty.factor = penalty_factors
)

coefs        <- coef(final_fit)
player_coefs <- coefs[player_cols, , drop = FALSE]

rapm_base <- data.frame(
  col_name  = rownames(player_coefs),
  rapm_base = as.vector(player_coefs),
  stringsAsFactors = FALSE
) %>%
  mutate(player_id = as.integer(sub("^p_", "", col_name))) %>%
  filter(!is.na(player_id))

## ---------------------------------------------------------------------------
## decay model
## ---------------------------------------------------------------------------
## add player x shift_age interaction columns then refit
## this is a linear decay assumption, nonlinear would need a spline basis

cat("Fitting decay interactions...\n")

X_decay <- X_players * df$home_shift_bucket[valid_rows]
colnames(X_decay) <- paste0(player_cols, "_decay")

X_full <- cbind(X, X_decay)

## regularize both player and decay terms, not controls
penalty_full <- c(
  rep(1, length(player_cols)),
  rep(0, length(control_cols)),
  rep(1, length(player_cols))
)

fit_decay <- glmnet(
  X_full, y,
  weights        = weights,
  alpha          = 0,
  lambda         = lambda_opt,
  penalty.factor = penalty_full
)

coefs_full  <- coef(fit_decay)
decay_coefs <- coefs_full[paste0(player_cols, "_decay"), , drop = FALSE]

rapm_decay <- data.frame(
  col_name   = rownames(decay_coefs),
  rapm_decay = as.vector(decay_coefs),
  stringsAsFactors = FALSE
) %>%
  mutate(player_id = as.integer(gsub("^p_|_decay$", "", col_name))) %>%
  filter(!is.na(player_id))

## ---------------------------------------------------------------------------
## per-player TOI
## ---------------------------------------------------------------------------

toi_df <- df %>%
  select(all_of(c(player_cols, "duration"))) %>%
  pivot_longer(-duration, names_to = "col_name", values_to = "on_ice") %>%
  filter(on_ice != 0) %>%
  group_by(col_name) %>%
  summarise(toi_seconds = sum(duration), .groups = "drop") %>%
  mutate(
    toi_5v5  = toi_seconds / 60,
    player_id = as.integer(sub("^p_", "", col_name))
  )

## ---------------------------------------------------------------------------
## join and flag overuse
## ---------------------------------------------------------------------------

results <- rapm_base %>%
  left_join(rapm_decay %>% select(player_id, rapm_decay), by = "player_id") %>%
  left_join(toi_df %>% select(player_id, toi_5v5), by = "player_id") %>%
  mutate(
    ## threshold is somewhat arbitrary, revisit with more data
    overuse_flag = rapm_decay < -0.05 & toi_5v5 > 50,
    player_name  = paste0("Player_", player_id),  ## placeholder until name lookup is wired in
    team         = NA_character_
  ) %>%
  select(player_id, player_name, team, rapm_base, rapm_decay, toi_5v5, overuse_flag)

cat(sprintf("Overuse flags: %d\n", sum(results$overuse_flag, na.rm = TRUE)))

## ---------------------------------------------------------------------------
## save
## ---------------------------------------------------------------------------

out_path <- file.path("data", "processed", paste0("rapm_results_", season, ".parquet"))
write_parquet(results, out_path)
cat(sprintf("Saved to %s\n", out_path))
