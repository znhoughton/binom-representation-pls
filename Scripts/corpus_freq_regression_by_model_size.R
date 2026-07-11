library(dplyr)
library(readr)
library(purrr)

BASE <- normalizePath(file.path(
  dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  ".."
))

MODELS <- c(
  "125"  = "znhoughton_opt-babylm-125m-20eps-seed964",
  "350"  = "znhoughton_opt-babylm-350m-20eps-seed964",
  "1300" = "znhoughton_opt-babylm-1_3b-20eps-seed964"
)

freq_df <- read_csv(file.path(BASE, "Data", "corpus_binomials.csv"),
                    show_col_types = FALSE) |>
  mutate(total_freq = freq_w1_w2 + freq_w2_w1) |>
  select(word1, word2, total_freq)

parts <- list()
for (size_str in names(MODELS)) {
  slug <- MODELS[[size_str]]
  path <- file.path(BASE, "Results", slug, "by_layer_corpus_pred.csv")
  if (!file.exists(path)) { warning("Not found: ", path); next }
  df <- read_csv(path, show_col_types = FALSE) |>
    mutate(model_size = factor(size_str, levels = c("350", "125", "1300")))
  parts[[size_str]] <- df
}

combined <- bind_rows(parts) |>
  left_join(freq_df, by = c("word1", "word2")) |>
  mutate(
    residual = abs(y_true - y_pred),
    log_freq = log(total_freq)
  )

fit_one <- function(d) {
  m  <- lm(residual ~ log_freq * model_size, data = d)
  cf <- coef(summary(m))
  ci <- confint(m)

  # All terms except intercept
  terms <- rownames(cf)[rownames(cf) != "(Intercept)"]
  map_dfr(terms, function(term) {
    tibble(
      term     = term,
      n        = nrow(d),
      estimate = cf[term, "Estimate"],
      se       = cf[term, "Std. Error"],
      t        = cf[term, "t value"],
      p        = cf[term, "Pr(>|t|)"],
      ci_lo    = ci[term, "2.5 %"],
      ci_hi    = ci[term, "97.5 %"]
    )
  })
}

results <- combined |>
  group_by(condition, layer, mode) |>
  group_modify(~ fit_one(.x)) |>
  ungroup()

out_path <- file.path(BASE, "Results", "corpus_freq_regression_by_model_size.csv")
write_csv(results, out_path)
cat("Saved:", out_path, "\n")

# Effective log_freq slope per model size, derived from coefficients
# beta_125  = log_freq
# beta_350  = log_freq + log_freq:model_size350
# beta_1300 = log_freq + log_freq:model_size1300
cat("\n=== Effective log_freq slope by model size ===\n")
results |>
  filter(term %in% c("log_freq",
                     "log_freq:model_size125",
                     "log_freq:model_size1300")) |>
  select(condition, mode, layer, term, estimate, p) |>
  tidyr::pivot_wider(names_from = term, values_from = c(estimate, p)) |>
  mutate(
    beta_350  = estimate_log_freq,
    beta_125  = estimate_log_freq + `estimate_log_freq:model_size125`,
    beta_1300 = estimate_log_freq + `estimate_log_freq:model_size1300`,
    # p-values for contrasts vs 350M baseline
    p_125_vs_350  = `p_log_freq:model_size125`,
    p_1300_vs_350 = `p_log_freq:model_size1300`
  ) |>
  select(condition, mode, layer, beta_125, beta_350, beta_1300,
         p_125_vs_350, p_1300_vs_350) |>
  mutate(
    sig_125  = case_when(p_125_vs_350  < .001 ~ "***",
                         p_125_vs_350  < .01  ~ "**",
                         p_125_vs_350  < .05  ~ "*",  TRUE ~ ""),
    sig_1300 = case_when(p_1300_vs_350 < .001 ~ "***",
                         p_1300_vs_350 < .01  ~ "**",
                         p_1300_vs_350 < .05  ~ "*",  TRUE ~ "")
  ) |>
  arrange(condition, mode, layer) |>
  print(n = Inf)
