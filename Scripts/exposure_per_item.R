#!/usr/bin/env Rscript
# exposure_per_item.R
# -------------------------------------------------------------------------
# Is the beta2 crossover governed by cumulative exposure per pair rather than
# by training time or model size?
#
# Experiment 3 currently plots beta2 against training tokens, which puts BabyLM
# and Pythia on incomparable x-axes: a web-scale corpus delivers repeated
# exposure to any given binomial far faster than a 150M-token corpus cycled for
# 20 epochs. This script re-indexes the same beta2 estimates against how many
# times the model has actually encountered each pair by that checkpoint.
#
# If the two families collapse onto a common curve, the developmental claim
# stops being "memorization appears early" and becomes "idiosyncrasy overtakes
# abstraction at a characteristic level of per-item exposure."
#
# Exposure model
#   BabyLM  150M-token corpus, 20 epochs, so a pair with corpus count f has been
#           seen f * epochs_elapsed times, epochs_elapsed = eff_tokens / 150M.
#   Pythia  single pass over the ~300B-token Pile, so a pair with Pile count f
#           has been seen f * (tokens_seen / 300B) times.
#
# Both use the SAME ~49k attested pairs, counted in each model's own corpus,
# which is what makes the comparison meaningful.
#
# Outputs (Results/):
#   exposure_per_item.csv          beta2 + exposure quantiles per checkpoint
#   exposure_crossover.csv         interpolated beta2 = 0 point per model
#   exposure_per_item.png          beta2 vs exposure, both families overlaid
#
# Usage:
#   Rscript Scripts/exposure_per_item.R
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(tidyverse)
})

SEED <- 964
set.seed(SEED)

BASE <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=",
          commandArgs(trailingOnly = FALSE), value = TRUE)[1])), ".."),
          mustWork = FALSE)
if (is.na(BASE) || !dir.exists(BASE)) BASE <- normalizePath("..", mustWork = FALSE)

RES  <- file.path(BASE, "Results")
DATA <- file.path(BASE, "Data")

cat("BASE:", BASE, "\n\n")

# corpus sizes -------------------------------------------------------------
BABYLM_CORPUS_TOKENS <- 150e6      # one epoch
PILE_TOKENS          <- 300e9      # Pythia trains ~1 epoch over the Pile

# ── beta2 estimates ────────────────────────────────────────────────────────
brms_summary <- readRDS(file.path(DATA, "brms_summary.rds"))
cat("brms_summary columns:", paste(names(brms_summary), collapse = ", "), "\n")
cat("brms_summary rows:", nrow(brms_summary), "\n\n")

# ── checkpoint registries (mirrors writeup.qmd) ────────────────────────────
babylm_ckpt_registry <- bind_rows(
  tibble(label = "BabyLM-125M",
         step  = c(24, 48, 144, 384, 912, 2280),
         tokens_M = c(24, 48, 144, 384, 912, 2280) * 0.8192),
  tibble(label = "BabyLM-350M",
         step  = c(48, 96, 288, 768, 1824, 4560),
         tokens_M = c(48, 96, 288, 768, 1824, 4560) * 0.4096),
  tibble(label = "BabyLM-1.3B",
         step  = c(97, 194, 582, 1455, 3686, 9021),
         tokens_M = c(97, 194, 582, 1455, 3686, 9021) * 0.2048)
) |> mutate(family_lbl = "BabyLM")

pythia_ckpt_registry <- expand_grid(
  label = c("Pythia-160M", "Pythia-410M", "Pythia-1B"),
  step  = c(16, 32, 64, 256, 512, 1000)
) |> mutate(tokens_M = step * 2.097152, family_lbl = "Pythia")

ckpt_registry <- bind_rows(babylm_ckpt_registry, pythia_ckpt_registry)

# ── per-pair corpus counts ─────────────────────────────────────────────────
babylm_freq <- read_csv(file.path(DATA, "corpus_binomials.csv"),
                        show_col_types = FALSE) |>
  transmute(word1, word2, freq = freq_w1_w2 + freq_w2_w1) |>
  filter(freq > 0)

pile_freq <- read_csv(file.path(RES, "corpus_binomials_infinigram_piletrain.csv"),
                      show_col_types = FALSE) |>
  transmute(word1, word2, freq = freq_total) |>
  filter(freq > 0)

cat("BabyLM attested pairs with freq > 0:", nrow(babylm_freq), "\n")
cat("Pile   attested pairs with freq > 0:", nrow(pile_freq), "\n")
cat("BabyLM freq quantiles:",
    paste(sprintf("%.0f", quantile(babylm_freq$freq, c(.5, .9, .95, .99))),
          collapse = " / "), "(p50/p90/p95/p99)\n")
cat("Pile   freq quantiles:",
    paste(sprintf("%.0f", quantile(pile_freq$freq, c(.5, .9, .95, .99))),
          collapse = " / "), "(p50/p90/p95/p99)\n\n")

# ── exposure distribution per checkpoint ───────────────────────────────────
# beta2 curvature is driven by the high-frequency arm, so the informative
# summary is a high quantile of the exposure distribution, not its centre.
QS <- c(p50 = .50, p75 = .75, p90 = .90, p95 = .95, p99 = .99)

exposure_at <- function(freq_vec, multiplier) {
  q <- quantile(freq_vec * multiplier, QS)
  as_tibble_row(set_names(as.numeric(q), paste0("exp_", names(QS)))) |>
    mutate(exp_mean = mean(freq_vec * multiplier))
}

ckpt_exposure <- ckpt_registry |>
  rowwise() |>
  mutate(
    # BabyLM cycles its corpus, so the multiplier is epochs elapsed;
    # Pythia sees the Pile once, so it is the fraction of the corpus consumed.
    multiplier = if (family_lbl == "BabyLM") {
      tokens_M * 1e6 / BABYLM_CORPUS_TOKENS
    } else {
      tokens_M * 1e6 / PILE_TOKENS
    },
    exposure = list(exposure_at(
      if (family_lbl == "BabyLM") babylm_freq$freq else pile_freq$freq,
      multiplier
    ))
  ) |>
  ungroup() |>
  unnest(exposure)

# ── join beta2 to exposure ─────────────────────────────────────────────────
beta_cols <- intersect(c("beta2_mean", "beta2_ci025", "beta2_ci975"),
                       names(brms_summary))
stopifnot(length(beta_cols) > 0)

ckpt_beta <- brms_summary |>
  filter(!is.na(step),
         label %in% ckpt_registry$label) |>
  { \(d) if ("mode" %in% names(d)) filter(d, mode == "mean_pooled") else d }() |>
  select(label, step, condition, any_of(beta_cols), any_of("mode"))

dat <- ckpt_beta |>
  inner_join(ckpt_exposure, by = c("label", "step"))

cat("joined rows:", nrow(dat), " (models x steps x conditions)\n")
if (nrow(dat) == 0) stop("join produced no rows; check label/step keys")

write_csv(dat, file.path(RES, "exposure_per_item.csv"))

# ── crossover: interpolate where beta2 = 0 in exposure terms ───────────────
crossover <- dat |>
  arrange(label, condition, tokens_M) |>
  group_by(label, condition, family_lbl) |>
  group_modify(function(g, key) {
    b <- g$beta2_mean
    if (all(is.na(b)) || all(b > 0) || all(b < 0)) {
      # never crosses within the observed window
      return(tibble(cross_tokens_M = NA_real_, cross_exp_p90 = NA_real_,
                    cross_exp_p99 = NA_real_,
                    note = if (all(b > 0, na.rm = TRUE)) "positive throughout"
                           else "negative throughout"))
    }
    i <- which(diff(sign(b)) != 0)[1]
    w <- abs(b[i]) / (abs(b[i]) + abs(b[i + 1]))   # linear interp weight
    tibble(
      cross_tokens_M = g$tokens_M[i] + w * (g$tokens_M[i + 1] - g$tokens_M[i]),
      cross_exp_p90  = g$exp_p90[i]  + w * (g$exp_p90[i + 1]  - g$exp_p90[i]),
      cross_exp_p99  = g$exp_p99[i]  + w * (g$exp_p99[i + 1]  - g$exp_p99[i]),
      note           = "crosses"
    )
  }) |>
  ungroup()

write_csv(crossover, file.path(RES, "exposure_crossover.csv"))

cat("\n==================== CROSSOVER (beta2 = 0) ====================\n")
crossover |>
  mutate(across(where(is.numeric), \(x) signif(x, 3))) |>
  as.data.frame() |>
  print(row.names = FALSE)

cat("\n============ EXPOSURE AT EACH CHECKPOINT (p90 / p99) ============\n")
dat |>
  filter(condition == first(condition)) |>
  select(family_lbl, label, step, tokens_M, exp_p90, exp_p99, beta2_mean) |>
  arrange(family_lbl, label, tokens_M) |>
  mutate(across(where(is.numeric), \(x) signif(x, 3))) |>
  as.data.frame() |>
  print(row.names = FALSE)

# ── figure: beta2 against exposure, families overlaid ──────────────────────
p <- dat |>
  filter(exp_p90 > 0) |>
  ggplot(aes(x = exp_p90, y = beta2_mean, color = label, group = label)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50",
             linewidth = 0.3) +
  geom_line(linewidth = 0.5) +
  geom_point(size = 1.1) +
  facet_wrap(~ condition, scales = "free_y") +
  scale_x_log10() +
  labs(x = "Cumulative exposures per pair, 90th percentile (log scale)",
       y = expression(hat(beta)[2]),
       colour = "Model",
       title = "Does the crossover align on exposure rather than tokens?") +
  theme_minimal(base_size = 10)

ggsave(file.path(RES, "exposure_per_item.png"), p,
       width = 8, height = 4, dpi = 150)

cat("\nwrote:\n",
    " ", file.path(RES, "exposure_per_item.csv"), "\n",
    " ", file.path(RES, "exposure_crossover.csv"), "\n",
    " ", file.path(RES, "exposure_per_item.png"), "\n")
