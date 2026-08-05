# prepare_results.R
# Loads all brms model pairs (lin + quad) once, extracts fixef summaries,
# posterior beta2 stats, LOO comparisons, and fitted quadratic curves.
# Saves two lightweight files:
#   Data/brms_summary.rds  -- one row per model×cond×mode (fixef + LOO + beta2)
#   Data/brms_curves.rds   -- 100-point curve grid per model×cond×mode
#
# The writeup loads only these files — no brms models are loaded at render time.
#
# Usage (from project root):
#   Rscript Scripts/prepare_results.R

suppressPackageStartupMessages({
  library(brms)
  library(loo)
  library(dplyr)
  library(tidyr)
  library(parallel)
})

MODEL_DIR    <- file.path("Data", "brms_models")
OUT_SUMMARY  <- file.path("Data", "brms_summary.rds")
OUT_CURVES   <- file.path("Data", "brms_curves.rds")
N_PARALLEL   <- 4
N_CURVE_PTS  <- 100

log_msg <- function(...) {
  cat(paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", ..., "\n"))
}

# ── Filename convention (matches fitting script and writeup) ──────────────────
make_fname <- function(label, step, cond, mode, type) {
  step_str   <- if (is.na(step)) "final" else as.character(step)
  fname_base <- gsub("[^A-Za-z0-9]", "_", paste(label, step_str, cond, mode, sep = "_"))
  file.path(MODEL_DIR, paste0(fname_base, "_", type, ".rds"))
}

# ── Job registry ──────────────────────────────────────────────────────────────
build_registry <- function() {
  rows <- list()
  add  <- function(label, step) {
    rows[[length(rows) + 1]] <<- list(label = label, step = step)
  }
  for (lbl in c(
    "BabyLM-125M", "BabyLM-350M", "BabyLM-1.3B",
    "Pythia-160M", "Pythia-410M", "Pythia-1B", "Pythia-2.8B",
    "GPT-2", "GPT-2-M", "GPT-2-L", "GPT-2-XL",
    "OLMo-1B", "OLMo-7B", "OLMo-2-1B", "OLMo-2-7B",
    "Llama-1.3B", "Llama-3-8B"
  )) add(lbl, NA_integer_)

  for (s in c(24, 48, 144, 384, 912, 2280))   add("BabyLM-125M", s)
  for (s in c(48, 96, 288, 768, 1824, 4560))   add("BabyLM-350M", s)
  for (s in c(97, 194, 582, 1455, 3686, 9021)) add("BabyLM-1.3B", s)

  for (s in c(16, 32, 64, 256, 512, 1000)) add("Pythia-160M", s)
  for (s in c(16, 32, 64, 256, 512, 1000)) add("Pythia-410M", s)
  for (s in c(16, 32, 64, 256, 512, 1000)) add("Pythia-1B",   s)
  for (s in c(16, 32, 64, 256, 512, 1000)) add("Pythia-2.8B", s)

  bind_rows(rows)
}

REGISTRY  <- build_registry()
COND_MODE <- expand_grid(
  condition = c("default", "attn_zeroed"),
  mode      = c("mean_pooled", "words_only")
)
all_jobs <- expand_grid(i = seq_len(nrow(REGISTRY)), j = seq_len(nrow(COND_MODE)))
log_msg(sprintf("Total jobs: %d", nrow(all_jobs)))

# ── Per-job extraction ────────────────────────────────────────────────────────
process_job <- function(job_idx) {
  reg  <- REGISTRY[all_jobs$i[[job_idx]], ]
  cm   <- COND_MODE[all_jobs$j[[job_idx]], ]

  lbl  <- reg$label;    stp  <- reg$step
  cond <- cm$condition; mode <- cm$mode

  path_lin  <- make_fname(lbl, stp, cond, mode, "lin")
  path_quad <- make_fname(lbl, stp, cond, mode, "quad")
  if (!file.exists(path_lin) || !file.exists(path_quad)) return(NULL)

  fit_lin  <- tryCatch(readRDS(path_lin),  error = function(e) NULL)
  fit_quad <- tryCatch(readRDS(path_quad), error = function(e) NULL)
  if (is.null(fit_lin) || is.null(fit_quad)) return(NULL)

  # ── fixef ──────────────────────────────────────────────────────────────────
  fx    <- as.data.frame(fixef(fit_quad))
  sq_nm <- rownames(fx)[!rownames(fx) %in% c("Intercept", "log_freq")]

  # ── posterior draws for beta2 ──────────────────────────────────────────────
  draws  <- as_draws_df(fit_quad)
  b2_col <- names(draws)[grepl("Ilog_freq", names(draws))][1]
  b0_draws <- draws$b_Intercept
  b1_draws <- draws$b_log_freq
  b2_draws <- draws[[b2_col]]

  # ── LOO ───────────────────────────────────────────────────────────────────
  elpd_diff <- se_diff <- pct_pareto_bad_lin <- pct_pareto_bad_quad <- NA_real_
  loo_l <- tryCatch(loo(fit_lin),  error = function(e) NULL)
  loo_q <- tryCatch(loo(fit_quad), error = function(e) NULL)
  if (!is.null(loo_l) && !is.null(loo_q)) {
    elpd_diff <- loo_q$estimates["elpd_loo", "Estimate"] -
                 loo_l$estimates["elpd_loo",  "Estimate"]
    cmp       <- loo_compare(list(lin = loo_l, quad = loo_q))
    se_diff   <- cmp[2, "se_diff"]
    pct_pareto_bad_lin  <- mean(loo_l$diagnostics$pareto_k > 0.7)
    pct_pareto_bad_quad <- mean(loo_q$diagnostics$pareto_k > 0.7)
  }

  # ── fitted curve grid (from full posterior) ────────────────────────────────
  lf_range  <- range(fit_quad$data$log_freq)
  freq_grid <- seq(lf_range[1], lf_range[2], length.out = N_CURVE_PTS)
  # curve_mat[draw, grid_pt] = b0 + b1*x + b2*x^2
  curve_mat <- outer(b0_draws, rep(1, N_CURVE_PTS)) +
               outer(b1_draws, freq_grid) +
               outer(b2_draws, freq_grid^2)

  curve_tbl <- tibble(
    label     = lbl, step = stp, condition = cond, mode = mode,
    log_freq  = freq_grid,
    mean_fit  = colMeans(curve_mat),
    q025      = apply(curve_mat, 2, quantile, 0.025),
    q975      = apply(curve_mat, 2, quantile, 0.975)
  )

  rm(fit_lin, fit_quad, draws, curve_mat); gc(verbose = FALSE)

  list(
    summary = tibble(
      label = lbl, step = stp, condition = cond, mode = mode,
      b_intercept      = fx["Intercept", "Estimate"],
      b_intercept_lo   = fx["Intercept", "Q2.5"],
      b_intercept_hi   = fx["Intercept", "Q97.5"],
      b_quad_linear    = fx["log_freq",  "Estimate"],
      b_quad_linear_lo = fx["log_freq",  "Q2.5"],
      b_quad_linear_hi = fx["log_freq",  "Q97.5"],
      b_quad_sq        = fx[sq_nm, "Estimate"],
      b_quad_sq_lo     = fx[sq_nm, "Q2.5"],
      b_quad_sq_hi     = fx[sq_nm, "Q97.5"],
      vertex           = -fx["log_freq", "Estimate"] / (2 * fx[sq_nm, "Estimate"]),
      beta2_mean        = mean(b2_draws),
      beta2_se          = sd(b2_draws),
      beta2_ci025       = quantile(b2_draws, 0.025),
      beta2_ci975       = quantile(b2_draws, 0.975),
      beta2_pct_gt0     = mean(b2_draws > 0) * 100,
      elpd_diff           = elpd_diff,
      se_diff             = se_diff,
      pct_pareto_bad_lin  = pct_pareto_bad_lin,
      pct_pareto_bad_quad = pct_pareto_bad_quad
    ),
    curve = curve_tbl
  )
}

# ── Run ───────────────────────────────────────────────────────────────────────
log_msg("Starting extraction...")
raw <- mclapply(seq_len(nrow(all_jobs)), process_job, mc.cores = N_PARALLEL)
raw <- Filter(Negate(is.null), raw)

brms_summary <- bind_rows(lapply(raw, `[[`, "summary"))
brms_curves  <- bind_rows(lapply(raw, `[[`, "curve"))

log_msg(sprintf("Summary: %d rows; Curves: %d rows", nrow(brms_summary), nrow(brms_curves)))
saveRDS(brms_summary, OUT_SUMMARY)
saveRDS(brms_curves,  OUT_CURVES)
log_msg(sprintf("Saved:\n  %s\n  %s", OUT_SUMMARY, OUT_CURVES))
