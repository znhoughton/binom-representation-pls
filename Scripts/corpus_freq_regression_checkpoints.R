library(dplyr)
library(readr)
library(purrr)

BASE <- normalizePath(file.path(
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  ".."
))

argv <- commandArgs(trailingOnly = TRUE)
model_flag <- if (length(argv) >= 1) argv[1] else "125m"
stopifnot(model_flag %in% c("125m", "350m", "1.3b"))

# ── Checkpoint registry ──────────────────────────────────────────────────────
# Each list ends with the fully-trained final model (pct = 100.0, no step suffix
# in the slug).  The final model is treated as the last checkpoint in the
# training-dynamics trajectory.

CHECKPOINTS <- list(
  "125m" = list(
    out_csv    = "corpus_freq_regression_checkpoints_125m.csv",
    checkpoints = list(
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step24",   step = 24,   pct = 0.6),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step48",   step = 48,   pct = 1.2),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step144",  step = 144,  pct = 3.6),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step384",  step = 384,  pct = 9.6),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step912",  step = 912,  pct = 22.9),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964_step2280", step = 2280, pct = 57.2),
      list(slug = "znhoughton_opt-babylm-125m-20eps-seed964",          step = 3984, pct = 100.0)
    )
  ),
  "350m" = list(
    out_csv    = "corpus_freq_regression_checkpoints_350m.csv",
    checkpoints = list(
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step48",   step = 48,   pct = 0.6),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step96",   step = 96,   pct = 1.2),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step288",  step = 288,  pct = 3.6),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step768",  step = 768,  pct = 9.6),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step1824", step = 1824, pct = 22.9),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964_step4560", step = 4560, pct = 57.2),
      list(slug = "znhoughton_opt-babylm-350m-20eps-seed964",          step = 7968, pct = 100.0)
    )
  ),
  "1.3b" = list(
    out_csv    = "corpus_freq_regression_checkpoints_1.3b.csv",
    checkpoints = list(
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step97",   step = 97,   pct = 0.6),
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step194",  step = 194,  pct = 1.2),
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step582",  step = 582,  pct = 3.7),
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step1455", step = 1455, pct = 9.1),
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step3686", step = 3686, pct = 23.2),
      list(slug = "znhoughton_opt-babylm-1.3b-20eps-seed964_step9021", step = 9021, pct = 56.7),
      list(slug = "znhoughton_opt-babylm-1_3b-20eps-seed964",          step = 15908, pct = 100.0)
    )
  )
)[[model_flag]]

cat(sprintf("Model: OPT-BabyLM-%s\n", model_flag))
cat(sprintf("Output: %s\n\n", CHECKPOINTS$out_csv))

freq_df <- read_csv(file.path(BASE, "Data", "corpus_binomials.csv"),
                    show_col_types = FALSE) |>
  mutate(total_freq = freq_w1_w2 + freq_w2_w1) |>
  select(word1, word2, total_freq)

fit_one <- function(d) {
  if (sum(!is.na(d$log_freq)) < 3) return(tibble())
  m  <- lm(residual ~ log_freq, data = d)
  cf <- coef(summary(m))
  ci <- confint(m)
  tibble(
    n        = nrow(d),
    estimate = cf["log_freq", "Estimate"],
    se       = cf["log_freq", "Std. Error"],
    t        = cf["log_freq", "t value"],
    p        = cf["log_freq", "Pr(>|t|)"],
    ci_lo    = ci["log_freq", "2.5 %"],
    ci_hi    = ci["log_freq", "97.5 %"]
  )
}

all_results <- map_dfr(CHECKPOINTS$checkpoints, function(ckpt) {
  xz_path <- file.path(BASE, "Results", ckpt$slug, "by_layer_corpus_pred.csv.xz")
  if (!file.exists(xz_path)) {
    warning("Missing: ", xz_path)
    return(tibble())
  }
  cat("Reading", ckpt$slug, "...\n")
  con  <- xzfile(xz_path, "rb")
  pred <- read_csv(con, show_col_types = FALSE)
  close(con)

  pred |>
    left_join(freq_df, by = c("word1", "word2")) |>
    mutate(
      residual = abs(y_true - y_pred),
      log_freq = log(total_freq)
    ) |>
    group_by(condition, layer, mode) |>
    group_modify(~ fit_one(.x)) |>
    ungroup() |>
    mutate(step = ckpt$step, pct_training = ckpt$pct, .before = 1)
})

out_path <- file.path(BASE, "Results", CHECKPOINTS$out_csv)
write_csv(all_results, out_path)
cat("\nSaved:", out_path, "\n")

# ── Print summary tables ─────────────────────────────────────────────────────
fmt_sig <- function(p) case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "ns")

for (cond in c("default", "attn_zeroed")) {
  for (m in c("mean_pooled", "words_only")) {
    cat(sprintf("\n\n=== %s / %s ===\n", cond, m))
    cat(sprintf("  %-14s  %s\n", "step",
                paste(sprintf("L%02d", sort(unique(all_results$layer))), collapse = "  ")))
    steps <- sort(unique(all_results$step))
    for (s in steps) {
      row <- all_results |>
        filter(step == s, condition == cond, mode == m) |>
        arrange(layer)
      pct   <- unique(row$pct_training)
      label <- if (pct == 100) sprintf("%s (final)", s) else sprintf("%s (%.1f%%)", s, pct)
      vals  <- sprintf("%+.3f%s", row$estimate, fmt_sig(row$p))
      cat(sprintf("  %-14s  %s\n", label, paste(vals, collapse = "  ")))
    }
  }
}
