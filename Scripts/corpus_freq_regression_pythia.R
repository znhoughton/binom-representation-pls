library(dplyr)
library(readr)
library(purrr)

BASE <- normalizePath(file.path(
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  ".."
))

MODELS <- list(
  list(slug = "EleutherAI_pythia-160m",  label = "Pythia-160m"),
  list(slug = "EleutherAI_pythia-410m",  label = "Pythia-410m"),
  list(slug = "EleutherAI_pythia-1b",    label = "Pythia-1B"),
  list(slug = "EleutherAI_pythia-2.8b",  label = "Pythia-2.8B"),
  # GPT-2 family (OpenAI; trained on WebText — using Pile freqs as best available proxy)
  list(slug = "gpt2",                    label = "GPT-2"),
  list(slug = "gpt2-medium",             label = "GPT-2-medium"),
  list(slug = "gpt2-large",              label = "GPT-2-large")
)

# Pile corpus frequency (best available proxy for both Pythia and GPT-2)
freq_df <- read_csv(file.path(BASE, "Results", "corpus_binomials_infinigram_piletrain.csv"),
                    show_col_types = FALSE) |>
  select(word1, word2, freq_total)

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

all_results <- map_dfr(MODELS, function(mdl) {
  xz_path <- file.path(BASE, "Results", mdl$slug, "by_layer_corpus_pred.csv.xz")
  if (!file.exists(xz_path)) {
    message("Skipping (not found): ", mdl$slug)
    return(tibble())
  }
  cat("Reading", mdl$slug, "...\n")
  con  <- xzfile(xz_path, "rb")
  pred <- read_csv(con, show_col_types = FALSE)
  close(con)

  pred |>
    left_join(freq_df, by = c("word1", "word2")) |>
    mutate(
      residual = abs(y_true - y_pred),
      log_freq = ifelse(!is.na(freq_total) & freq_total > 0, log(freq_total), NA_real_)
    ) |>
    group_by(condition, layer, mode) |>
    group_modify(~ fit_one(.x)) |>
    ungroup() |>
    mutate(model = mdl$label, slug = mdl$slug, .before = 1)
})

out_path <- file.path(BASE, "Results", "corpus_freq_regression_pythia.csv")
write_csv(all_results, out_path)
cat("\nSaved:", out_path, "\n")

fmt_sig <- function(p) case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "ns")

for (mdl_label in unique(all_results$model)) {
  cat(sprintf("\n\n=== %s ===\n", mdl_label))
  for (cond in c("default", "attn_zeroed")) {
    for (m in c("mean_pooled", "words_only")) {
      sub <- all_results |> filter(model == mdl_label, condition == cond, mode == m)
      if (nrow(sub) == 0) next
      cat(sprintf("  %s / %s:  beta=%.4f  p=%.4f %s\n",
                  cond, m, sub$estimate, sub$p, fmt_sig(sub$p)))
    }
  }
}
