library(dplyr)
library(readr)
library(purrr)

BASE <- normalizePath(file.path(
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  ".."
))

argv       <- commandArgs(trailingOnly = TRUE)
model_flag <- if (length(argv) >= 1) argv[1] else "160m"
stopifnot(model_flag %in% c("160m", "410m", "1b", "2.8b"))

TOKENS_PER_STEP <- 2097152L

tokens_label <- function(step) {
  t <- step * TOKENS_PER_STEP
  if (t >= 1e9) sprintf("%.2fB", t / 1e9) else sprintf("%.0fM", t / 1e6)
}

# ── Checkpoint registry ──────────────────────────────────────────────────────
# Steps aligned to BabyLM token counts (step1000 ≈ BabyLM full training)
# plus extended steps up to full Pythia training (step143000 = 300B tokens).
# The final fully-trained checkpoint (step143000) is treated as the last point.

BABYLM_STEPS  <- c(16, 32, 64, 256, 512, 1000)
EXTENDED_STEPS <- c(2000, 5000, 14000, 33000, 143000)
ALL_STEPS     <- c(BABYLM_STEPS, EXTENDED_STEPS)

N_LAYERS <- list("160m" = 12L, "410m" = 24L, "1b" = 16L, "2.8b" = 32L)

out_csv <- sprintf("corpus_freq_regression_checkpoints_pythia_%s.csv",
                   gsub("\\.", "_", model_flag))

cat(sprintf("Model: Pythia-%s\n", model_flag))
cat(sprintf("Output: %s\n\n", out_csv))

# ── Frequency data (The Pile, Infinigram counts) ─────────────────────────────
freq_df <- read_csv(
  file.path(BASE, "Results", "corpus_binomials_infinigram_piletrain.csv"),
  show_col_types = FALSE
) |>
  mutate(total_freq = freq_w1_w2 + freq_w2_w1) |>
  select(word1, word2, total_freq)

# ── Regression helper ─────────────────────────────────────────────────────────
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

# ── Main loop ─────────────────────────────────────────────────────────────────
all_results <- map_dfr(ALL_STEPS, function(step) {
  slug     <- sprintf("EleutherAI_pythia-%s_step%d", model_flag, step)
  xz_path  <- file.path(BASE, "Results", slug, "by_layer_corpus_pred.csv.xz")
  tok_lab  <- tokens_label(step)
  is_final <- step == 143000L

  if (!file.exists(xz_path)) {
    warning("Missing: ", xz_path)
    return(tibble())
  }
  cat("Reading", slug, "...\n")
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
    mutate(
      step       = step,
      tokens     = step * TOKENS_PER_STEP,
      tokens_lab = tok_lab,
      is_final   = is_final,
      .before    = 1
    )
})

out_path <- file.path(BASE, "Results", out_csv)
write_csv(all_results, out_path)
cat("\nSaved:", out_path, "\n")

# ── Print summary tables ──────────────────────────────────────────────────────
fmt_sig <- function(p) case_when(
  p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "ns"
)

for (cond in c("default", "attn_zeroed")) {
  for (m in c("mean_pooled", "words_only")) {
    cat(sprintf("\n\n=== %s / %s ===\n", cond, m))
    layers <- sort(unique(all_results$layer))
    cat(sprintf("  %-18s  %s\n", "step (tokens)",
                paste(sprintf("L%02d", layers), collapse = "  ")))
    for (s in ALL_STEPS) {
      row <- all_results |>
        filter(step == s, condition == cond, mode == m) |>
        arrange(layer)
      if (nrow(row) == 0) next
      tok_lab <- unique(row$tokens_lab)
      label   <- if (s == 143000L) sprintf("%d/%s (final)", s, tok_lab) else
                                   sprintf("%d/%s",         s, tok_lab)
      vals    <- sprintf("%+.3f%s", row$estimate, fmt_sig(row$p))
      cat(sprintf("  %-18s  %s\n", label, paste(vals, collapse = "  ")))
    }
  }
}
