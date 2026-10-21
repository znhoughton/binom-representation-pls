#!/usr/bin/env Rscript
# pythia_exclusion_check.R
# -------------------------------------------------------------------------
# Appendix: the step-0 relative-frequency effect is a tokenisation artifact.
#
# About 2% of corpus binomials have carrier contexts that do not end in
# whitespace, so whichever word comes first loses its leading space and is
# segmented differently. The two orderings then sum different numbers of span
# tokens, and at initialisation each token costs roughly log|V|, so those items
# carry an ~11 nat difference that has nothing to do with ordering preference.
#
# This refits @eq-relfreq with those items dropped, for the three Pythia models
# at step 0 and at their final checkpoint. The unfiltered fits already exist in
# brms_relfreq_rawscale.rds and are reused rather than refitted, since they come
# from the same script with the same settings.
#
# Writes Data/derived/pythia_exclusion_summary.csv.
#
# Usage:  Rscript Scripts/analysis/pythia_exclusion_check.R

suppressMessages({
  library(dplyr); library(readr); library(tibble); library(purrr)
  library(brms); library(posterior)
})
options(contrasts = c("contr.sum", "contr.sum"))

RESULTS   <- "Results"
MODE      <- "mean_pooled"
SEED      <- 964
CHAINS    <- 4
ITER      <- 4000
WARMUP    <- 2000
MODEL_DIR <- file.path("Data", "brms_models_exclusion")
OUT       <- file.path("Data", "derived", "pythia_exclusion_summary.csv")
LOG       <- file.path(RESULTS, "pythia_exclusion_progress.log")
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) {
  m <- paste0(format(Sys.time(), "%H:%M:%S"), " | ", ...)
  cat(m, "\n", sep = ""); cat(m, "\n", sep = "", file = LOG, append = TRUE)
  flush.console()
}

# Items whose two orderings tokenise to different span lengths. The flag depends
# on the tokeniser, and all three Pythia models share the NeoX one.
flags <- read_csv("Data/derived/split_asym_pythia.csv", show_col_types = FALSE) |>
  filter(split_asym == 1) |>
  select(word1, word2)
log_msg("flagged items: ", nrow(flags))

counts <- read_csv(file.path(RESULTS, "corpus_binomials_infinigram_piletrain.csv"),
                   show_col_types = FALSE) |>
  transmute(word1, word2, n_w1_w2 = freq_w1w2, n_w2_w1 = freq_w2w1)

# Mirrors load_cell() in corpus_relfreq_brms_all.R with SCALE = "raw", plus the
# anti_join that drops the asymmetric-split items. Scaling is recomputed on the
# retained rows, which is what the analysis would have looked like had those
# items never been measured.
load_cell <- function(slug, cond, drop_split) {
  xz <- file.path(RESULTS, slug, "by_layer_corpus_pred.csv.xz")
  if (!file.exists(xz)) return(NULL)
  con <- xzfile(xz, "rb"); pred <- read_csv(con, show_col_types = FALSE); close(con)
  final_layer <- max(suppressWarnings(as.integer(unique(pred$layer))), na.rm = TRUE)

  d <- pred |>
    filter(as.integer(layer) == final_layer, condition == cond, mode == MODE) |>
    inner_join(counts, by = c("word1", "word2")) |>
    filter(n_w1_w2 + n_w2_w1 > 0) |>
    filter(word1 != word2)
  if (drop_split) d <- anti_join(d, flags, by = c("word1", "word2"))

  d |>
    mutate(total = n_w1_w2 + n_w2_w1) |>
    (\(x) { stopifnot(all(x$word1 < x$word2)); x })() |>
    transmute(
      y_true_z   = c(scale(y_true)),
      y_pred_z   = c(scale(y_pred)),
      log_freq_z = c(scale(log(total), scale = FALSE)),
      rel_freq_z = n_w1_w2 / total - 0.5
    )
}

FORMULA <- bf(y_true_z ~ y_pred_z * log_freq_z + rel_freq_z * log_freq_z,
              sigma ~ log_freq_z)

log_msg("compiling Stan template ...")
set.seed(SEED)
dummy <- tibble(y_true_z = rnorm(60), y_pred_z = rnorm(60),
                log_freq_z = rnorm(60), rel_freq_z = rnorm(60))
template <- brm(FORMULA, data = dummy, family = gaussian(), chains = 1,
                iter = 200, warmup = 100, refresh = 0, silent = 2,
                seed = SEED, backend = "cmdstanr")
log_msg("template compiled.")

cells <- tibble(
  label = rep(c("Pythia-160M", "Pythia-410M", "Pythia-1B"), each = 2),
  step  = rep(c(0, NA), times = 3),
  slug  = c("EleutherAI_pythia-160m_step0", "EleutherAI_pythia-160m",
            "EleutherAI_pythia-410m_step0", "EleutherAI_pythia-410m",
            "EleutherAI_pythia-1b_step0",   "EleutherAI_pythia-1b")
)
jobs <- tidyr::crossing(cells, condition = c("default", "attn_zeroed"))

rows <- list()
for (k in seq_len(nrow(jobs))) {
  j   <- jobs[k, ]
  tag <- if (is.na(j$step)) "final" else "step0"
  d   <- load_cell(j$slug, j$condition, drop_split = TRUE)
  if (is.null(d) || nrow(d) < 200) { log_msg("skip: ", j$slug); next }

  log_msg(sprintf("[%d/%d] %s %s | %s (n=%d)",
                  k, nrow(jobs), j$label, tag, j$condition, nrow(d)))
  fname <- file.path(MODEL_DIR, gsub("[^A-Za-z0-9]", "_",
                     paste(j$label, tag, j$condition, "excl", sep = "_")))
  fit <- tryCatch(
    update(template, newdata = d, recompile = FALSE, chains = CHAINS,
           iter = ITER, warmup = WARMUP, cores = CHAINS, refresh = 0,
           silent = 2, seed = SEED, file = fname, file_refit = "on_change"),
    error = function(e) { log_msg("  ERROR: ", conditionMessage(e)); NULL })
  if (is.null(fit)) next

  mx <- max(rhat(fit), na.rm = TRUE)
  dr <- as_draws_df(fit)
  grab <- function(term) {
    v <- dr[[paste0("b_", term)]]
    c(mean(v), unname(quantile(v, .025)), unname(quantile(v, .975)))
  }
  r <- grab("rel_freq_z"); p <- grab("y_pred_z")
  rows[[length(rows) + 1]] <- tibble(
    label = j$label, step = j$step, condition = j$condition,
    n = nrow(d), max_rhat = mx,
    rel_mean = r[1], rel_lo = r[2], rel_hi = r[3],
    pred_mean = p[1], pred_lo = p[2], pred_hi = p[3])
  log_msg(sprintf("    rel = %+.3f [%+.3f, %+.3f]  rhat=%.3f",
                  r[1], r[2], r[3], mx))
}

out <- bind_rows(rows)
write_csv(out, OUT)
log_msg("wrote ", OUT, " (", nrow(out), " rows)")
print(as.data.frame(out |> mutate(across(where(is.numeric), ~round(.x, 3)))),
      row.names = FALSE)
