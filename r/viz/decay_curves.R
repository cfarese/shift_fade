## decay_curves.R
##
## Plots shift-decay efficiency curves for individual lines or players.
## Reads the RAPM results parquet and the raw stint data.
##
## Usage:
##   Rscript r/viz/decay_curves.R --season 20232024 --out plots/

library(arrow)
library(dplyr)
library(ggplot2)
library(argparse)

parser <- ArgumentParser()
parser$add_argument("--season", default = "20232024")
parser$add_argument("--out", default = "plots")
args <- parser$parse_args()

dir.create(args$out, showWarnings = FALSE, recursive = TRUE)

## ---------------------------------------------------------------------------
## load data
## ---------------------------------------------------------------------------

rapm_path <- file.path("data", "processed", paste0("rapm_results_", args$season, ".parquet"))
stint_path <- file.path("data", "processed", paste0("stints_", args$season, ".parquet"))

if (!file.exists(rapm_path)) stop("Run rapm_model.R first")
if (!file.exists(stint_path)) stop("Run ingestion pipeline first")

rapm  <- read_parquet(rapm_path)
stints <- read_parquet(stint_path)

## ---------------------------------------------------------------------------
## plot 1: distribution of decay coefficients across all players
## ---------------------------------------------------------------------------

p1 <- ggplot(rapm, aes(x = rapm_decay)) +
  geom_histogram(bins = 40, fill = "#2c7bb6", color = "white", alpha = 0.85) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_vline(xintercept = -0.05, linetype = "dotted", color = "orange", linewidth = 0.7) +
  labs(
    title    = "Distribution of Shift-Decay Coefficients",
    subtitle = paste("Season", args$season, "| orange line = overuse threshold"),
    x        = "RAPM decay (xG diff per 60 change per 5s bucket)",
    y        = "Player count"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(args$out, "decay_distribution.png"), p1, width = 9, height = 5, dpi = 150)
cat("Saved decay_distribution.png\n")

## ---------------------------------------------------------------------------
## plot 2: base RAPM vs decay coefficient scatter
## this is the main diagnostic, good players in top-left, bad in bottom-right
## ---------------------------------------------------------------------------

p2 <- rapm %>%
  filter(!is.na(rapm_base), !is.na(rapm_decay), toi_5v5 >= 50) %>%
  ggplot(aes(x = rapm_decay, y = rapm_base, color = overuse_flag, size = toi_5v5)) +
  geom_point(alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(
    values = c("FALSE" = "#4393c3", "TRUE" = "#d6604d"),
    labels = c("FALSE" = "Normal", "TRUE" = "Overuse flag"),
    name   = NULL
  ) +
  scale_size_continuous(range = c(1.5, 5), name = "5v5 TOI (min)") +
  labs(
    title    = "Base RAPM vs Shift Decay Rate",
    subtitle = "Top-left quadrant: strong players who hold up late in shifts",
    x        = "Decay coefficient (negative = gets worse as shift ages)",
    y        = "Base RAPM (xG diff per 60)"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(args$out, "rapm_vs_decay.png"), p2, width = 10, height = 7, dpi = 150)
cat("Saved rapm_vs_decay.png\n")

## ---------------------------------------------------------------------------
## plot 3: fitted decay curves for top/bottom 5 decayers
## shows the actual projected efficiency curve across shift duration
## ---------------------------------------------------------------------------

## reconstruct fitted curve: xg_diff = rapm_base + rapm_decay * bucket
buckets <- seq(0, 60, by = 5)

## grab 5 worst decayers and 5 most durable (best decay coef) with enough TOI
worst_5 <- rapm %>%
  filter(toi_5v5 >= 50) %>%
  slice_min(rapm_decay, n = 5)

best_5 <- rapm %>%
  filter(toi_5v5 >= 50) %>%
  slice_max(rapm_decay, n = 5)

plot_players <- bind_rows(
  mutate(worst_5, group = "Worst decayers"),
  mutate(best_5,  group = "Most durable")
)

## expand each player across all shift buckets
curve_df <- plot_players %>%
  rowwise() %>%
  do(data.frame(
    player_id   = .$player_id,
    player_name = .$player_name,
    group       = .$group,
    bucket      = buckets,
    fitted_xgd  = .$rapm_base + .$rapm_decay * buckets
  )) %>%
  ungroup()

p3 <- ggplot(curve_df, aes(x = bucket, y = fitted_xgd,
                            group = player_name, color = player_name)) +
  geom_line(linewidth = 1.1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~group, ncol = 2) +
  labs(
    title    = "Fitted Shift-Decay Curves",
    subtitle = "How estimated xG differential per 60 changes as shifts age",
    x        = "Shift age (seconds)",
    y        = "Projected xG diff per 60",
    color    = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave(file.path(args$out, "decay_curves.png"), p3, width = 12, height = 6, dpi = 150)
cat("Saved decay_curves.png\n")

cat("All plots written to", args$out, "\n")
