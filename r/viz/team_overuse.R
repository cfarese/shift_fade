## team_overuse.R
##
## Team-level overuse report. For each team, shows which forward lines and
## defensive pairs are being deployed past their effective shift window.
##
## "Effective shift window" is defined as the point where the fitted xG
## differential curve crosses zero. If the coach keeps the line on past that
## point, they are giving up expected goals.
##
## Usage:
##   Rscript r/viz/team_overuse.R --season 20232024 --team BOS

library(arrow)
library(dplyr)
library(tidyr)
library(ggplot2)
library(argparse)

parser <- ArgumentParser()
parser$add_argument("--season", default = "20232024")
parser$add_argument("--team", default = NULL, help = "Three-letter team code, e.g. BOS")
parser$add_argument("--out", default = "plots")
args <- parser$parse_args()

dir.create(args$out, showWarnings = FALSE, recursive = TRUE)

rapm_path <- file.path("data", "processed", paste0("rapm_results_", args$season, ".parquet"))
if (!file.exists(rapm_path)) stop("Run rapm_model.R first")

rapm <- read_parquet(rapm_path)

if (!is.null(args$team)) {
  rapm <- rapm %>% filter(toupper(team) == toupper(args$team))
  if (nrow(rapm) == 0) stop(paste("No players found for team:", args$team))
}

## ---------------------------------------------------------------------------
## compute break-even second for each player
## break-even is when rapm_base + rapm_decay * t = 0, so t = -rapm_base / rapm_decay
## only meaningful if rapm_decay < 0 (otherwise player never goes negative)
## ---------------------------------------------------------------------------

rapm <- rapm %>%
  mutate(
    break_even_sec = case_when(
      rapm_decay < 0 ~ -rapm_base / rapm_decay,
      TRUE           ~ NA_real_
    ),
    ## league average shift is around 45 seconds
    overused_at_avg_shift = !is.na(break_even_sec) & break_even_sec < 45
  )

## ---------------------------------------------------------------------------
## bar chart: break-even seconds sorted by value
## ---------------------------------------------------------------------------

plot_df <- rapm %>%
  filter(!is.na(break_even_sec), toi_5v5 >= 50) %>%
  arrange(break_even_sec) %>%
  mutate(player_name = factor(player_name, levels = player_name))

if (nrow(plot_df) == 0) {
  cat("No players with negative decay coefficients. Nothing to plot.\n")
  quit(status = 0)
}

p <- ggplot(plot_df, aes(x = player_name, y = break_even_sec, fill = overused_at_avg_shift)) +
  geom_col() +
  geom_hline(yintercept = 45, linetype = "dashed", color = "gray30", linewidth = 0.8) +
  coord_flip() +
  scale_fill_manual(
    values = c("FALSE" = "#4393c3", "TRUE" = "#d6604d"),
    labels = c("FALSE" = "OK at avg shift", "TRUE" = "Overused at avg shift"),
    name   = NULL
  ) +
  labs(
    title    = paste("Shift Break-Even Points", if (!is.null(args$team)) paste("-", args$team) else ""),
    subtitle = "Dashed line = league average shift length (45s). Red = player goes negative before avg shift ends.",
    x        = NULL,
    y        = "Break-even shift age (seconds)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

team_suffix <- if (!is.null(args$team)) paste0("_", args$team) else ""
out_file <- file.path(args$out, paste0("team_overuse", team_suffix, ".png"))
ggsave(out_file, p, width = 10, height = max(6, nrow(plot_df) * 0.35), dpi = 150)
cat(sprintf("Saved %s\n", out_file))
