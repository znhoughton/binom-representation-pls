#!/usr/bin/env Rscript
# quick_lm_relfreq.R
# -----------------
# Fast frequentist read on Eq. 1, for checking a trend before committing to the
# seven-hour brms sweep.
#
# Fits the same formula to the same data as corpus_relfreq_brms_all.R, with two
# differences, both of which affect inference rather than the point estimates
# this script exists to look at:
#
#   * lm, not brms. No priors, no posterior, so the intervals are Wald rather
#     than credible.
#   * no sigma ~ log_freq. The brms model lets residual scale vary with
#     frequency, which it needs because spread differs sharply between the
#     low-frequency mass and the tail. Omitting it leaves the coefficients
#     roughly where brms puts them but makes their standard errors optimistic
#     at the sparse end.
#
# So: use this to see whether a pattern is there. Quote the brms fits.
#
# Usage (from project root):
#   Rscript Scripts/analysis/quick_lm_relfreq.R
#   Rscript Scripts/analysis/quick_lm_relfreq.R znhoughton_opt-babylm-125m-20eps-seed964

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(purrr); library(tibble)
})

RESULTS <- "Results"
# Where to look for by_layer_corpus_pred.csv.xz. Overridable so the same
# estimator can be run against an archived set, which is the only way to
# compare coefficients before and after an item-set change without confounding
# the comparison with brms-versus-lm differences.
PRED_DIR <- Sys.getenv("PRED_DIR", unset = RESULTS)
MODE     <- "mean_pooled"

counts <- list(
  babylm = read_csv("Data/corpus_binomials.csv", show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1_w2, n_w2_w1 = freq_w2_w1),
  pile = read_csv(file.path(RESULTS, "corpus_binomials_infinigram_piletrain.csv"),
                  show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1w2, n_w2_w1 = freq_w2w1)
)

# Same preparation as the brms script: reduplicatives out (their two orderings
# are the same string, so y_true is identically zero while rel_freq is pinned
# at +0.5), rel_freq as the alphabetical share centred at zero, log_freq
# centred but not scaled, y_true and y_pred z-scored within the cell.
load_cell <- function(slug, corpus, cond) {
  xz <- file.path(PRED_DIR, slug, "by_layer_corpus_pred.csv.xz")
  if (!file.exists(xz)) return(NULL)
  con <- xzfile(xz, "rb"); pred <- read_csv(con, show_col_types = FALSE); close(con)
  fl <- max(suppressWarnings(as.integer(unique(pred$layer))), na.rm = TRUE)
  d <- pred |>
    filter(as.integer(layer) == fl, condition == cond, mode == MODE) |>
    inner_join(counts[[corpus]], by = c("word1", "word2")) |>
    filter(n_w1_w2 + n_w2_w1 > 0, word1 != word2) |>
    mutate(total = n_w1_w2 + n_w2_w1)
  if (nrow(d) == 0) return(NULL)
  transmute(d,
    y_true_z = c(scale(y_true)),
    y_pred_z = c(scale(y_pred)),
    log_freq = c(scale(log(total), scale = FALSE)),
    rel_freq = n_w1_w2 / total - 0.5)
}

args  <- commandArgs(trailingOnly = TRUE)
slugs <- if (length(args)) args else {
  list.dirs(PRED_DIR, recursive = FALSE, full.names = FALSE) |>
    setdiff(c("logs", "_pre_carriers")) |>
    keep(~ file.exists(file.path(PRED_DIR, .x, "by_layer_corpus_pred.csv.xz")))
}

rows <- list()
for (slug in slugs) {
  corpus <- if (grepl("babylm", slug)) "babylm" else "pile"
  for (cond in c("default", "attn_zeroed")) {
    d <- load_cell(slug, corpus, cond)
    if (is.null(d)) next
    m <- lm(y_true_z ~ y_pred_z * log_freq + rel_freq * log_freq, data = d)
    s <- summary(m)$coefficients
    g <- function(t) if (t %in% rownames(s)) s[t, "Estimate"] else NA_real_
    p <- function(t) if (t %in% rownames(s)) s[t, "Pr(>|t|)"] else NA_real_
    rows[[length(rows) + 1]] <- tibble(
      slug = slug, condition = cond, n = nrow(d),
      y_pred      = g("y_pred_z"),
      rel_freq    = g("rel_freq"),
      log_freq    = g("log_freq"),
      pred_x_freq = g("y_pred_z:log_freq"),
      rel_x_freq  = g("log_freq:rel_freq"),
      p_rel       = p("rel_freq"),
      p_rel_x     = p("log_freq:rel_freq"),
      r2          = summary(m)$r.squared)
    cat(sprintf("  %-44s %-12s n=%6d  y_pred %+.3f  rel_freq %+.3f  rel:lf %+.4f\n",
                slug, cond, nrow(d), g("y_pred_z"), g("rel_freq"),
                g("log_freq:rel_freq")))
  }
}

out <- bind_rows(rows)
if (nrow(out)) {
  write_csv(out, file.path(RESULTS, Sys.getenv("OUT_NAME", "quick_lm_relfreq.csv")))
  cat("\n")
  print(as.data.frame(out |> mutate(across(where(is.numeric), ~ round(.x, 4)))))
  cat(sprintf("\nwrote %s\n", file.path(RESULTS, "quick_lm_relfreq.csv")))
  cat("Point estimates only. The brms fits add sigma ~ log_freq; quote those.\n")
} else {
  cat("No cells with by_layer_corpus_pred.csv.xz found.\n")
}
