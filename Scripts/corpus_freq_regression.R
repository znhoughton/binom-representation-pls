library(dplyr)
library(readr)
library(purrr)

BASE <- normalizePath(file.path(
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  ".."
))

args_cli <- commandArgs(trailingOnly = TRUE)
model_slug <- if (length(args_cli) > 0) args_cli[1] else "znhoughton_opt-babylm-125m-20eps-seed964"

pred <- read_csv(file.path(BASE, "Results", model_slug,
                           "by_layer_corpus_pred.csv"), show_col_types = FALSE)

freq_df <- read_csv(file.path(BASE, "Data", "corpus_binomials.csv"), show_col_types = FALSE) |>
  mutate(total_freq = freq_w1_w2 + freq_w2_w1) |>
  select(word1, word2, total_freq)

df <- pred |>
  left_join(freq_df, by = c("word1", "word2")) |>
  mutate(
    residual = abs(y_true - y_pred),
    log_freq = log(total_freq)
  )

# Per condition × layer × mode: lm(residual ~ log_freq) with CIs and p-value
fit_one <- function(d) {
  m <- lm(residual ~ log_freq, data = d)
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

results <- df |>
  group_by(condition, layer, mode) |>
  group_modify(~ fit_one(.x)) |>
  ungroup()

out_path <- file.path(BASE, "Results", model_slug,
                      "corpus_freq_regression.csv")
write_csv(results, out_path)
cat("Saved:", out_path, "\n")

# Print by layer for each condition × mode
for (cond in unique(results$condition)) {
  for (m in unique(results$mode)) {
    cat(sprintf("\n--- %s / %s ---\n", cond, m))
    results |>
      filter(condition == cond, mode == m) |>
      select(layer, estimate, se, ci_lo, ci_hi, p) |>
      mutate(sig = case_when(p < .001 ~ "***", p < .01 ~ "**", p < .05 ~ "*", TRUE ~ "")) |>
      print(n = Inf)
  }
}
