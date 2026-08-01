#!/usr/bin/env Rscript
# Scripts/plot_delta_by_model.R
#
# Plot the context-dependence delta (default/mp β − attn_zeroed/mp β) at the
# final layer for every available model. Models whose data is missing are
# silently skipped so the script is forward-compatible with Phase 3/4 additions.
#
# Delta = (default/mean_pooled β) − (attn_zeroed/mean_pooled β)
# Larger delta = more context-dependent (more memorisation via attention).
#
# Usage:
#   Rscript Scripts/plot_delta_by_model.R
#   Rscript Scripts/plot_delta_by_model.R --no-error-bars

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

argv_full <- commandArgs(trailingOnly = FALSE)
file_flag  <- grep("^--file=", argv_full, value = TRUE)
BASE <- if (length(file_flag)) {
  normalizePath(file.path(dirname(sub("^--file=", "", file_flag[1L])), ".."))
} else {
  normalizePath(getwd())
}

argv <- commandArgs(trailingOnly = TRUE)
show_errorbars <- !"--no-error-bars" %in% argv

# ── Model registry ─────────────────────────────────────────────────────────
# final_layer: which layer to extract delta from
# freq_csv:    path to the corpus-freq regression CSV
# freq_file:   which type of frequency data was used (babylm vs pile)
# group:       for faceting / colouring

babylm_freq  <- file.path(BASE, "Data",    "corpus_binomials.csv")
pile_freq    <- file.path(BASE, "Results", "corpus_binomials_infinigram_piletrain.csv")

MODELS <- list(
  # ── Phase 1/2: OPT-BabyLM ────────────────────────────────────────────────
  list(label = "OPT-125M",    group = "BabyLM (2B tok)",
       final_layer = 12,
       corpus_pred = file.path(BASE, "Results", "znhoughton_opt-babylm-125m-20eps-seed964",
                               "by_layer_corpus_pred.csv"),
       freq_csv    = file.path(BASE, "Results", "znhoughton_opt-babylm-125m-20eps-seed964",
                               "corpus_freq_regression.csv"),
       freq_data   = babylm_freq,
       freq_col    = "babylm"),

  list(label = "OPT-350M",    group = "BabyLM (2B tok)",
       final_layer = 24,
       corpus_pred = file.path(BASE, "Results", "znhoughton_opt-babylm-350m-20eps-seed964",
                               "by_layer_corpus_pred.csv"),
       freq_csv    = file.path(BASE, "Results", "znhoughton_opt-babylm-350m-20eps-seed964",
                               "corpus_freq_regression.csv"),
       freq_data   = babylm_freq,
       freq_col    = "babylm"),

  list(label = "OPT-1.3B",    group = "BabyLM (2B tok)",
       final_layer = 24,
       corpus_pred = file.path(BASE, "Results", "znhoughton_opt-babylm-1_3b-20eps-seed964",
                               "by_layer_corpus_pred.csv"),
       freq_csv    = file.path(BASE, "Results", "znhoughton_opt-babylm-1_3b-20eps-seed964",
                               "corpus_freq_regression.csv"),
       freq_data   = babylm_freq,
       freq_col    = "babylm"),

  # ── Phase 3: Pythia ───────────────────────────────────────────────────────
  # Pythia results come from the combined CSV; final_layer identifies the row.
  list(label = "Pythia-160m",  group = "Pythia (300B tok)",
       final_layer = 12,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "Pythia-160m"),

  list(label = "Pythia-410m",  group = "Pythia (300B tok)",
       final_layer = 24,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "Pythia-410m"),

  list(label = "Pythia-1B",    group = "Pythia (300B tok)",
       final_layer = 16,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "Pythia-1B"),

  list(label = "Pythia-2.8B",  group = "Pythia (300B tok)",
       final_layer = 32,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "Pythia-2.8B"),

  # ── Phase 3: GPT-2 ───────────────────────────────────────────────────────
  list(label = "GPT-2",        group = "GPT-2 (WebText)",
       final_layer = 12,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "GPT-2"),

  list(label = "GPT-2-medium", group = "GPT-2 (WebText)",
       final_layer = 24,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "GPT-2-medium"),

  list(label = "GPT-2-large",  group = "GPT-2 (WebText)",
       final_layer = 36,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "GPT-2-large"),

  list(label = "GPT-2-XL",    group = "GPT-2 (WebText)",
       final_layer = 48,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "GPT-2-XL"),

  # ── Phase 3: OLMo (AI2, trained on Dolma) ────────────────────────────────
  list(label = "OLMo-1B",     group = "OLMo (Dolma)",
       final_layer = 16,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "OLMo-1B"),

  list(label = "OLMo-7B",     group = "OLMo (Dolma)",
       final_layer = 32,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "OLMo-7B"),

  list(label = "OLMo-2-7B",   group = "OLMo (Dolma)",
       final_layer = 32,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "OLMo-2-7B"),

  # ── Phase 3: Llama 3 (Meta) ───────────────────────────────────────────────
  list(label = "Llama-3-8B",  group = "Llama 3 (Meta)",
       final_layer = 32,
       corpus_pred = NULL,
       freq_csv    = file.path(BASE, "Results", "corpus_freq_regression_pythia.csv"),
       freq_data   = pile_freq,
       freq_col    = "pile",
       pythia_label = "Llama-3-8B")
)

# ── Helper: run corpus-freq regression for a BabyLM model if CSV missing ──
run_regression <- function(pred_path, freq_data_path, freq_col) {
  pred <- read_csv(pred_path, show_col_types = FALSE)
  freq <- read_csv(freq_data_path, show_col_types = FALSE)

  if (freq_col == "babylm") {
    freq <- freq |> mutate(total_freq = freq_w1_w2 + freq_w2_w1) |>
      select(word1, word2, total_freq)
  } else {
    freq <- freq |> select(word1, word2, total_freq = freq_total)
  }

  fit_one <- function(d) {
    if (sum(!is.na(d$log_freq)) < 3) return(tibble())
    m  <- lm(residual ~ log_freq, data = d)
    cf <- coef(summary(m))
    ci <- confint(m)
    tibble(n = nrow(d),
           estimate = cf["log_freq", "Estimate"],
           se       = cf["log_freq", "Std. Error"],
           p        = cf["log_freq", "Pr(>|t|)"],
           ci_lo    = ci["log_freq", "2.5 %"],
           ci_hi    = ci["log_freq", "97.5 %"])
  }

  pred |>
    left_join(freq, by = c("word1", "word2")) |>
    mutate(residual = abs(y_true - y_pred),
           log_freq = ifelse(!is.na(total_freq) & total_freq > 0,
                             log(total_freq), NA_real_)) |>
    group_by(condition, layer, mode) |>
    group_modify(~ fit_one(.x)) |>
    ungroup()
}

# ── Collect delta estimates ─────────────────────────────────────────────────
collect_delta <- function(mdl) {
  L <- mdl$final_layer

  # Pythia: read from shared CSV, filter by model label
  if (!is.null(mdl$pythia_label)) {
    if (!file.exists(mdl$freq_csv)) return(NULL)
    reg <- read_csv(mdl$freq_csv, show_col_types = FALSE) |>
      filter(model == mdl$pythia_label)
  } else {
    # BabyLM: use cached CSV or recompute
    if (file.exists(mdl$freq_csv)) {
      reg <- read_csv(mdl$freq_csv, show_col_types = FALSE)
    } else if (!is.null(mdl$corpus_pred) && file.exists(mdl$corpus_pred)) {
      cat("Running regression for", mdl$label, "...\n")
      reg <- run_regression(mdl$corpus_pred, mdl$freq_data, mdl$freq_col)
      write_csv(reg, mdl$freq_csv)
      cat("  Saved:", mdl$freq_csv, "\n")
    } else {
      message("Skipping ", mdl$label, " — no data found")
      return(NULL)
    }
  }

  mp <- reg |> filter(mode == "mean_pooled", layer == L)
  def <- mp |> filter(condition == "default")    |> select(estimate, se, ci_lo, ci_hi)
  az  <- mp |> filter(condition == "attn_zeroed") |> select(estimate, se, ci_lo, ci_hi)

  if (nrow(def) == 0 || nrow(az) == 0) {
    message("Skipping ", mdl$label, " — missing conditions at layer ", L)
    return(NULL)
  }

  # Delta = default - attn_zeroed; propagate SE via sqrt(se_def^2 + se_az^2)
  delta    <- def$estimate - az$estimate
  delta_se <- sqrt(def$se^2 + az$se^2)

  tibble(
    label   = mdl$label,
    group   = mdl$group,
    layer   = L,
    def_b   = def$estimate,
    az_b    = az$estimate,
    delta   = delta,
    delta_se = delta_se,
    delta_lo = delta - 1.96 * delta_se,
    delta_hi = delta + 1.96 * delta_se
  )
}

results <- bind_rows(lapply(MODELS, collect_delta))

if (nrow(results) == 0) stop("No model data found — check Results/ directory.")

cat("\nDelta estimates:\n")
print(results |> select(label, group, def_b, az_b, delta, delta_se), n = Inf)

# ── Plot ────────────────────────────────────────────────────────────────────
# Fix label order: BabyLM first (ascending params), then Pythia (ascending)
label_order <- c("OPT-125M", "OPT-350M", "OPT-1.3B",
                 "Pythia-160m", "Pythia-410m", "Pythia-1B", "Pythia-2.8B",
                 "GPT-2", "GPT-2-medium", "GPT-2-large", "GPT-2-XL",
                 "OLMo-1B", "OLMo-7B", "OLMo-2-7B",
                 "Llama-3-8B")
results <- results |>
  mutate(label = factor(label, levels = intersect(label_order, label)))

group_colours <- c(
  "BabyLM (2B tok)"   = "#D55E00",
  "Pythia (300B tok)" = "#0072B2",
  "GPT-2 (WebText)"   = "#009E73",
  "OLMo (Dolma)"      = "#E69F00",
  "Llama 3 (Meta)"    = "#CC79A7"
)

p <- ggplot(results, aes(x = label, y = delta, fill = group)) +
  geom_col(width = 0.6, colour = "white", linewidth = 0.3) +
  { if (show_errorbars)
      geom_errorbar(aes(ymin = delta_lo, ymax = delta_hi),
                    width = 0.18, linewidth = 0.7, colour = "grey30")
    else list() } +
  geom_hline(yintercept = 0, linewidth = 0.4, linetype = "dashed", colour = "grey50") +
  scale_fill_manual(values = group_colours, name = NULL) +
  scale_y_continuous(
    name   = expression(Delta ~ "= default " * beta ~ "− attn_zeroed " * beta ~ "(mean_pooled, final layer)"),
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    title    = "Context-dependence (memorisation via attention) at the final layer",
    subtitle = "Larger delta = more memorisation encoded in cross-word attention",
    x        = NULL,
    caption  = paste0("Corpus-frequency regression: lm(|residual| ~ log_freq) ",
                      "at the final layer, mean_pooled mode.\n",
                      "BabyLM: BabyLM-corpus frequencies; Pythia: Pile frequencies.\n",
                      "Error bars: 95% CI propagated from individual β SEs.")
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position   = "top",
    plot.title        = element_text(face = "bold", size = 13),
    plot.subtitle     = element_text(colour = "grey40", size = 10),
    plot.caption      = element_text(colour = "grey50", size = 8, hjust = 0),
    panel.grid.major.x = element_blank(),
    axis.text.x       = element_text(size = 11)
  )

out_dir <- file.path(BASE, "Results")
ggsave(file.path(out_dir, "delta_by_model.pdf"), p, width = 7, height = 4.5)
ggsave(file.path(out_dir, "delta_by_model.png"), p, width = 7, height = 4.5, dpi = 150)
cat("\nSaved: Results/delta_by_model.{pdf,png}\n")
