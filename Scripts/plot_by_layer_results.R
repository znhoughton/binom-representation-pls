#!/usr/bin/env Rscript
# Scripts/plot_by_layer_results.R
#
# Generate r² by layer + observed vs. predicted scatter plots for one model.
# Reads Results/<slug>/by_layer_mlp.csv (and *_control.csv if present).
# For scatter plots, also reads cv_preds/ NPZ files via reticulate + numpy.
#
# Usage:
#   Rscript Scripts/plot_by_layer_results.R --model-slug <slug>
#   Rscript Scripts/plot_by_layer_results.R --model-slug <slug> --key-layers 0 8 24
#
# Outputs saved to Results/<slug>/Plots/ as both .pdf and .png:
#   r2_by_layer         r² vs layer, all conditions/modes; controls as dashed lines
#   scatter_pair_novel  obs vs pred scatter at key layers (pair_novel split)
#   scatter_word_novel  obs vs pred scatter at key layers (word_novel split)
#
# Set RETICULATE_PYTHON to a Python with numpy for scatter plots, e.g.:
#   RETICULATE_PYTHON=/opt/.../python Rscript ...
#   RETICULATE_PYTHON=C:\Users\zacha\anaconda3\python.exe Rscript ...

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# ---- locate project root ----
argv_full <- commandArgs(trailingOnly = FALSE)
file_flag <- grep("^--file=", argv_full, value = TRUE)
BASE <- if (length(file_flag)) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_flag[1L])), ".."))
} else {
  normalizePath(getwd())
}

# ---- argument parsing ----
argv <- commandArgs(trailingOnly = TRUE)

get_arg <- function(argv, flag, default = NULL, multi = FALSE) {
  i <- which(argv == flag)
  if (!length(i)) return(default)
  vals <- character(0L)
  j <- i[1L] + 1L
  while (j <= length(argv) && !startsWith(argv[[j]], "--")) {
    vals <- c(vals, argv[[j]]); j <- j + 1L
  }
  if (!length(vals)) return(default)
  if (multi) vals else vals[1L]
}

slug         <- get_arg(argv, "--model-slug")
key_layers_a <- get_arg(argv, "--key-layers", multi = TRUE)

if (is.null(slug)) stop("--model-slug is required")

results_dir <- file.path(BASE, "Results", slug)
plots_dir   <- file.path(results_dir, "Plots")
preds_dir   <- file.path(results_dir, "cv_preds")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

model_label <- sub("znhoughton_opt-babylm-", "OPT-", slug)
cat(sprintf("Model : %s\nOutput: %s\n", model_label, plots_dir))

# ---- helper: save pdf + png ----
save_plot <- function(p, stem, width, height) {
  base <- file.path(plots_dir, stem)
  ggsave(paste0(base, ".pdf"), p, width = width, height = height, device = "pdf")
  ggsave(paste0(base, ".png"), p, width = width, height = height, dpi = 300)
  cat("  saved", stem, "\n")
}

# ---- colour / linetype scheme ----
COND_MODE_LEVELS <- c(
  "default / mean_pooled",     "default / individual",     "default / words_only",
  "attn_zeroed / mean_pooled", "attn_zeroed / individual", "attn_zeroed / words_only"
)
PALETTE <- setNames(
  c("#1565C0", "#42A5F5", "#00897B",
    "#C62828", "#EF9A9A", "#AD1457"),
  COND_MODE_LEVELS
)
SHAPES <- setNames(c(16L, 17L, 15L, 16L, 17L, 15L), COND_MODE_LEVELS)

# ---- load summary CSVs ----
csv_main <- file.path(results_dir, "by_layer_mlp.csv")
csv_ctrl <- file.path(results_dir, "by_layer_mlp_control.csv")

if (!file.exists(csv_main)) stop("Not found: ", csv_main)

df <- read.csv(csv_main, stringsAsFactors = FALSE) %>%
  mutate(run_type = "observed")

if (file.exists(csv_ctrl)) {
  df <- bind_rows(df,
                  read.csv(csv_ctrl, stringsAsFactors = FALSE) %>%
                    mutate(run_type = "control"))
} else {
  message("No control CSV — control lines will be omitted.")
}

# keep only the main CV splits (exclude corpus_pred, pls_diff, etc.)
df <- df %>%
  filter(split %in% c("pair_novel", "word_novel"),
         mode  %in% c("mean_pooled", "individual", "words_only")) %>%
  mutate(
    cond_mode = factor(paste(condition, mode, sep = " / "),
                       levels = COND_MODE_LEVELS),
    split     = factor(split,
                       levels = c("pair_novel", "word_novel"),
                       labels = c("Pair-novel", "Word-novel")),
    run_type  = factor(run_type, levels = c("observed", "control"))
  )

df_obs <- filter(df, run_type == "observed")

# ---- plot 1: r² by layer ----
p_r2 <- ggplot(df,
               aes(x = layer, y = mean_r2,
                   colour   = cond_mode,
                   linetype = run_type,
                   group    = interaction(cond_mode, run_type))) +
  geom_line(linewidth = 0.65) +
  geom_point(data    = df_obs,
             mapping = aes(shape = cond_mode),
             size = 1.5, show.legend = FALSE) +
  facet_wrap(~split, ncol = 1L, scales = "fixed") +
  scale_colour_manual(values = PALETTE, name = "Condition / Mode", drop = FALSE) +
  scale_shape_manual(values = SHAPES, drop = FALSE) +
  scale_linetype_manual(
    values = c(observed = "solid", control = "dashed"),
    name   = "Run type",
    labels = c(observed = "Observed", control = "Control (shuffled labels)")
  ) +
  labs(x     = "Layer",
       y     = expression(r^2),
       title = paste0(model_label, " — r² by layer")) +
  guides(colour   = guide_legend(order = 1L, override.aes = list(linewidth = 1.2)),
         linetype = guide_legend(order = 2L, override.aes = list(linewidth = 1.2))) +
  theme_bw(base_size = 11L) +
  theme(legend.position   = "right",
        legend.key.width  = unit(1.6, "cm"),
        strip.text        = element_text(face = "bold"),
        panel.grid.minor  = element_blank())

save_plot(p_r2, "r2_by_layer", width = 10L, height = 7L)

# ---- obs vs predicted scatter plots ----
if (!dir.exists(preds_dir)) {
  cat("No cv_preds/ directory — skipping scatter plots.\n")
  quit(status = 0L)
}

# attempt to load numpy via reticulate
np <- NULL
if (requireNamespace("reticulate", quietly = TRUE)) {
  py_env <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nchar(py_env)) reticulate::use_python(py_env, required = FALSE)
  np <- tryCatch(reticulate::import("numpy"), error = function(e) NULL)
}

if (is.null(np)) {
  cat(paste0(
    "Cannot load numpy via reticulate — scatter plots skipped.\n",
    "Install the 'reticulate' package and set RETICULATE_PYTHON to a Python with numpy.\n"
  ))
  quit(status = 0L)
}

load_npz <- function(path) {
  d <- np$load(path, allow_pickle = TRUE)
  df <- data.frame(y_true = as.numeric(d[["y_true"]]),
                   y_pred = as.numeric(d[["y_pred"]]),
                   stringsAsFactors = FALSE)
  df[!is.nan(df$y_true) & !is.nan(df$y_pred), ]
}

# key layers (evenly spaced if not specified)
n_layers <- max(df$layer)
key_layers <- if (!is.null(key_layers_a)) {
  sort(unique(as.integer(key_layers_a)))
} else {
  sort(unique(c(0L,
                as.integer(round(n_layers / 3L)),
                as.integer(round(2L * n_layers / 3L)),
                n_layers)))
}
cat(sprintf("Scatter key layers: %s\n", paste(key_layers, collapse = ", ")))


for (sp in c("pair_novel", "word_novel")) {
  for (cond in c("default", "attn_zeroed")) {

    parts <- list()
    for (mode in c("mean_pooled", "individual", "words_only")) {
      for (lyr in key_layers) {
        fpath <- file.path(
          preds_dir,
          sprintf("%s_layer%d_%s_%s.npz", cond, lyr, mode, sp)
        )
        if (!file.exists(fpath)) next
        d           <- load_npz(fpath)
        d$mode      <- mode
        d$layer_lab <- sprintf("Layer %d", lyr)
        parts[[length(parts) + 1L]] <- d
      }
    }

    if (!length(parts)) {
      cat("No cv_preds found for split:", sp, "/ cond:", cond, "— skipping.\n")
      next
    }

    sdf <- bind_rows(parts) %>%
      mutate(
        mode      = factor(mode,
                           levels = c("mean_pooled", "individual", "words_only")),
        layer_lab = factor(layer_lab,
                           levels = paste("Layer", sort(key_layers)))
      )

    r_ann <- sdf %>%
      group_by(mode, layer_lab) %>%
      summarise(
        label  = sprintf("r = %.3f", cor(y_true, y_pred)),
        x_pos  = -Inf,
        y_pos  =  Inf,
        .groups = "drop"
      )

    cond_label <- if (cond == "default") "Default" else "Attn-zeroed"
    sp_label   <- sub("_", "-", sp)
    p_sc <- ggplot(sdf, aes(x = y_true, y = y_pred)) +
      geom_bin2d(bins = 60L) +
      geom_smooth(method  = "lm", formula = y ~ x,
                  se      = FALSE, colour = "#1565C0", linewidth = 0.7) +
      geom_text(data        = r_ann,
                mapping     = aes(x = x_pos, y = y_pos, label = label),
                hjust = -0.08, vjust = 1.4, size = 2.5,
                inherit.aes = FALSE) +
      scale_fill_viridis_c(option = "magma", trans = "sqrt",
                           name = "Count",
                           guide = guide_colourbar(barheight = 6L)) +
      facet_grid(mode ~ layer_lab) +
      labs(x     = "Observed preference",
           y     = "Predicted preference",
           title = paste0(model_label, " — obs vs. pred — ",
                          sp_label, " — ", cond_label)) +
      theme_bw(base_size = 9L) +
      theme(strip.text       = element_text(size = 7.5),
            panel.grid.minor = element_blank(),
            legend.position  = "right")

    n_col  <- length(key_layers)
    n_row  <- length(levels(sdf$mode))
    stem   <- paste0("scatter_", sp, "_", cond)
    save_plot(p_sc, stem,
              width  = 3.8 * n_col + 2.0,
              height = 3.2 * n_row + 1.5)
  }
}

cat("Done.\n")
