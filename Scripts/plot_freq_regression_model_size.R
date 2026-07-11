library(dplyr)
library(readr)
library(ggplot2)
library(purrr)

argv_full <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("^--file=", argv_full, value = TRUE)
BASE <- if (length(file_flag)) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_flag[1L])), ".."))
} else {
  normalizePath(getwd())
}

MODELS <- c(
  "125M"  = "znhoughton_opt-babylm-125m-20eps-seed964",
  "350M"  = "znhoughton_opt-babylm-350m-20eps-seed964",
  "1.3B"  = "znhoughton_opt-babylm-1_3b-20eps-seed964"
)

MODEL_PALETTE <- c("125M" = "#9ECAE1", "350M" = "#3182BD", "1.3B" = "#08306B")
MODEL_SHAPES  <- c("125M" = 17L,       "350M" = 16L,       "1.3B" = 15L)

MODE_LEVELS <- c("mean_pooled", "individual", "words_only")
MODE_LABELS <- c(mean_pooled = "Mean-pooled", individual = "Individual", words_only = "Words-only")

save_plot <- function(p, path_stem, width, height) {
  ggsave(paste0(path_stem, ".pdf"), p, width = width, height = height, device = "pdf")
  ggsave(paste0(path_stem, ".png"), p, width = width, height = height, dpi = 300)
  cat("  saved", basename(path_stem), "\n")
}

# Load per-model estimates (these have proper per-model CIs)
# corpus_freq_regression.csv: condition, layer, mode, n, estimate, se, t, p, ci_lo, ci_hi
parts <- list()
for (label in names(MODELS)) {
  slug <- MODELS[[label]]
  path <- file.path(BASE, "Results", slug, "corpus_freq_regression.csv")
  if (!file.exists(path)) { warning("Not found: ", path); next }
  df <- read_csv(path, show_col_types = FALSE) |>
    mutate(
      model = factor(label, levels = names(MODELS)),
      mode  = factor(mode, levels = MODE_LEVELS, labels = names(MODE_LABELS)),
      condition = factor(condition,
                         levels = c("default", "attn_zeroed"),
                         labels = c("Default", "Attn-zeroed"))
    )
  parts[[label]] <- df
}
df_all <- bind_rows(parts)

# Re-apply factor labels cleanly
df_all <- df_all |>
  mutate(mode = factor(mode, levels = MODE_LEVELS,
                       labels = c("Mean-pooled", "Individual", "Words-only")))

p <- ggplot(df_all, aes(x = layer, y = estimate,
                         colour = model, fill = model, shape = model)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.8) +
  facet_grid(mode ~ condition, scales = "free_y") +
  scale_colour_manual(values = MODEL_PALETTE, name = "Model") +
  scale_fill_manual(values = MODEL_PALETTE, name = "Model") +
  scale_shape_manual(values = MODEL_SHAPES, name = "Model") +
  labs(x     = "Layer",
       y     = expression(beta~"(|residual| ~ log freq)"),
       title = "Corpus frequency regression by model size") +
  theme_bw(base_size = 11L) +
  theme(panel.grid.minor = element_blank(),
        strip.text       = element_text(face = "bold"),
        legend.position  = "right")

out <- file.path(BASE, "Results", "freq_regression_by_model_size")
save_plot(p, out, width = 8L, height = 9L)
cat("Done.\n")
