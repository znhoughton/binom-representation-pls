library(here)
library(tidyverse)

# Usage: Rscript feature_correlations.R [slug]
# Default slug: znhoughton_opt-babylm-125m-20eps-seed964
args  <- commandArgs(trailingOnly = TRUE)
slug  <- if (length(args) > 0) args[1] else "znhoughton_opt-babylm-125m-20eps-seed964"
layer <- if (length(args) > 1) args[2] else "last"

RESULTS_BASE <- here::here("Results")
RESULTS_DIR  <- file.path(RESULTS_BASE, slug, paste0("layer_", layer))
DELTA <- file.path(RESULTS_DIR, "delta_features.csv")
OUT   <- file.path(RESULTS_DIR, "feature_correlations.csv")

cat("Slug:", slug, "\n")
cat("Reading:", DELTA, "\n")

df <- read_csv(DELTA, show_col_types = FALSE)
cat("Loaded", nrow(df), "pairs\n")

components <- paste0("C", 1:15)
targets    <- c(components, "preference")
delta_cols <- names(df)[startsWith(names(df), "delta_") | names(df) == "pair_cosine_sim"]
cat("Delta features:", paste(delta_cols, collapse = ", "), "\n\n")

results <- list()
for (d in delta_cols) {
  feat <- sub("^delta_", "", d)
  valid <- df %>% select(all_of(c(d, targets))) %>% drop_na(all_of(d))
  n_all <- nrow(valid)

  for (tgt in targets) {
    sub_valid <- valid %>% drop_na(all_of(tgt))
    n <- nrow(sub_valid)
    if (n < 100) next

    x <- sub_valid[[d]]
    y <- sub_valid[[tgt]]

    r   <- cor(x, y, method = "pearson")
    rho <- cor(x, y, method = "spearman")

    # p-value for pearson r
    t_stat <- r * sqrt(n - 2) / sqrt(1 - r^2)
    p_val  <- 2 * pt(abs(t_stat), df = n - 2, lower.tail = FALSE)

    results[[length(results) + 1]] <- tibble(
      feature = feat, target = tgt, n = n,
      r = r, rho = rho, p = p_val
    )
  }
}

results_df <- bind_rows(results)

# Pearson r pivot
cat("=== Feature-component correlations (Pearson r) ===\n")
pivot_r <- results_df %>%
  select(feature, target, r) %>%
  pivot_wider(names_from = target, values_from = r) %>%
  select(feature, all_of(targets[targets %in% names(.)]))
print(pivot_r %>% mutate(across(where(is.numeric), ~round(.x, 3))), n = Inf)

cat("\n=== Feature-component correlations (Spearman rho) ===\n")
pivot_rho <- results_df %>%
  select(feature, target, rho) %>%
  pivot_wider(names_from = target, values_from = rho) %>%
  select(feature, all_of(targets[targets %in% names(.)]))
print(pivot_rho %>% mutate(across(where(is.numeric), ~round(.x, 3))), n = Inf)

# Which features most strongly predict preference?
cat("\n=== Strongest feature-preference correlations ===\n")
results_df %>%
  filter(target == "preference") %>%
  arrange(desc(abs(r))) %>%
  print(n = Inf)

write_csv(results_df, OUT)
cat("\nSaved to", OUT, "\n")