#!/usr/bin/env Rscript
# Scripts/plot_freq_regression.R
#
# Plot beta-by-layer from corpus_freq_regression.csv for each model.
# Produces per-model plots (faceted by condition) and one combined plot.
#
# Usage:
#   Rscript Scripts/plot_freq_regression.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

argv_full <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("^--file=", argv_full, value = TRUE)
BASE <- if (length(file_flag)) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_flag[1L])), ".."))
} else {
  normalizePath(getwd())
}

MODELS <- c(
  "125M" = "znhoughton_opt-babylm-125m-20eps-seed964",
  "350M" = "znhoughton_opt-babylm-350m-20eps-seed964",
  "1.3B" = "znhoughton_opt-babylm-1_3b-20eps-seed964"
)

MODE_LEVELS  <- c("mean_pooled", "individual", "words_only")
MODE_PALETTE <- c(mean_pooled = "#1565C0", individual = "#42A5F5", words_only = "#00897B")
MODE_SHAPES  <- c(mean_pooled = 16L, individual = 17L, words_only = 15L)

COND_LABELS <- c(default = "Default", attn_zeroed = "Attn-zeroed")

save_plot <- function(p, path_stem, width, height) {
  ggsave(paste0(path_stem, ".pdf"), p, width = width, height = height, device = "pdf")
  ggsave(paste0(path_stem, ".png"), p, width = width, height = height, dpi = 300)
  cat("  saved", basename(path_stem), "\n")
}

# ---- load all models ----
parts <- list()
for (label in names(MODELS)) {
  slug <- MODELS[[label]]
  path <- file.path(BASE, "Results", slug, "corpus_freq_regression.csv")
  if (!file.exists(path)) { warning("Not found: ", path); next }
  df <- read_csv(path, show_col_types = FALSE) |>
    mutate(
      model     = factor(label, levels = names(MODELS)),
      mode      = factor(mode, levels = MODE_LEVELS),
      condition = factor(condition,
                         levels = c("default", "attn_zeroed"),
                         labels = c("Default", "Attn-zeroed"))
    )
  parts[[label]] <- df
}
df_all <- bind_rows(parts)

# ---- per-model plots ----
for (label in names(MODELS)) {
  slug      <- MODELS[[label]]
  plots_dir <- file.path(BASE, "Results", slug, "Plots")
  df        <- filter(df_all, model == label)

  p <- ggplot(df, aes(x = layer, y = estimate,
                      colour = mode, fill = mode, shape = mode)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               colour = "grey50", linewidth = 0.5) +
    geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
                alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.7) +
    geom_point(size = 1.8) +
    facet_wrap(~condition, ncol = 1L, scales = "free_y") +
    scale_colour_manual(values = MODE_PALETTE, name = "Mode") +
    scale_fill_manual(values = MODE_PALETTE, name = "Mode") +
    scale_shape_manual(values = MODE_SHAPES, name = "Mode") +
    labs(x     = "Layer",
         y     = expression(beta~"(|residual| ~ log freq)"),
         title = paste0("OPT-", label,
                        " — corpus frequency regression by layer")) +
    theme_bw(base_size = 11L) +
    theme(panel.grid.minor  = element_blank(),
          strip.text        = element_text(face = "bold"),
          legend.position   = "right")

  n_layers <- max(df$layer)
  save_plot(p,
            file.path(plots_dir, "freq_regression_by_layer"),
            width = 7L, height = 7L)
}

# ---- combined plot (all models × conditions) ----
p_combined <- ggplot(df_all, aes(x = layer, y = estimate,
                                  colour = mode, fill = mode, shape = mode)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             colour = "grey50", linewidth = 0.45) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi),
              alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  facet_grid(model ~ condition, scales = "free_y") +
  scale_colour_manual(values = MODE_PALETTE, name = "Mode") +
  scale_fill_manual(values = MODE_PALETTE, name = "Mode") +
  scale_shape_manual(values = MODE_SHAPES, name = "Mode") +
  labs(x     = "Layer",
       y     = expression(beta~"(|residual| ~ log freq)"),
       title = "Corpus frequency regression — all models") +
  theme_bw(base_size = 10L) +
  theme(panel.grid.minor  = element_blank(),
        strip.text        = element_text(face = "bold"),
        legend.position   = "right")

save_plot(p_combined,
          file.path(BASE, "Results", "freq_regression_combined"),
          width = 9L, height = 9L)

cat("Done.\n")
