# corpus_freq_regression_brms_quad_compare.R
# Fits linear and quadratic brms models for |residual| ~ log_freq and saves
# each fitted model to Results/brms_models/ as .rds files.
#
# Parallelism: N_PARALLEL models run simultaneously, each using 4 cores for
# Stan chains. Set N_PARALLEL = 4 for a 20-core server (4 × 4 = 16 cores).
#
# Resume-safe: skips any pair whose lin+quad .rds files already exist.
# Figures/tables are generated directly from the saved model objects.

suppressPackageStartupMessages({
  library(brms)
  library(parallel)
  library(dplyr)
  library(readr)
})

N_PARALLEL <- 4   # outer parallel workers; each uses CHAINS_PER_MODEL cores
CHAINS_PER_MODEL <- 4

RESULTS   <- "Results"
MODEL_DIR <- file.path("Data", "brms_models")
LOG       <- file.path(RESULTS, "brms_quad_compare_progress.log")

dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] [PID:", Sys.getpid(), "] ", ..., "\n")
  cat(msg)
  cat(msg, file = LOG, append = TRUE)
}

# ── Frequency data ─────────────────────────────────────────────────────────────
freq_babylm <- read_csv(file.path("Data", "corpus_binomials.csv"),
                        show_col_types = FALSE) |>
  mutate(freq_total = freq_w1_w2 + freq_w2_w1) |>
  select(word1, word2, freq_total)

freq_pile <- read_csv(
  file.path(RESULTS, "corpus_binomials_infinigram_piletrain.csv"),
  show_col_types = FALSE) |>
  select(word1, word2, freq_total)

# ── Model registry ──────────────────────────────────────────────────────────────
build_registry <- function() {
  rows <- list()
  add <- function(dir_slug, label, step, layer, freq_src) {
    rows[[length(rows) + 1]] <<- list(
      dir_slug = dir_slug, label = label, step = step,
      layer = as.character(layer), freq_src = freq_src
    )
  }

  # ── Final models ────────────────────────────────────────────────────────────
  add("znhoughton_opt-babylm-125m-20eps-seed964",  "BabyLM-125M",  NA, "last", "babylm")
  add("znhoughton_opt-babylm-350m-20eps-seed964",  "BabyLM-350M",  NA, "last", "babylm")
  add("znhoughton_opt-babylm-1_3b-20eps-seed964",  "BabyLM-1.3B",  NA, "last", "babylm")
  add("EleutherAI_pythia-160m",    "Pythia-160M",  NA, "last", "pile")
  add("EleutherAI_pythia-410m",    "Pythia-410M",  NA, "last", "pile")
  add("EleutherAI_pythia-1b",      "Pythia-1B",    NA, "last", "pile")
  add("EleutherAI_pythia-2.8b",    "Pythia-2.8B",  NA, "last", "pile")
  add("gpt2",                      "GPT-2",        NA, "last", "pile")
  add("gpt2-medium",               "GPT-2-M",      NA, "last", "pile")
  add("gpt2-large",                "GPT-2-L",      NA, "last", "pile")
  add("gpt2-xl",                   "GPT-2-XL",     NA, "last", "pile")
  add("allenai_OLMo-1B-hf",        "OLMo-1B",      NA, "last", "pile")
  add("allenai_OLMo-7B-hf",        "OLMo-7B",      NA, "last", "pile")
  add("allenai_OLMo-2-0425-1B",    "OLMo-2-1B",    NA, "last", "pile")
  add("allenai_OLMo-2-1124-7B",    "OLMo-2-7B",    NA, "last", "pile")
  add("meta-llama_Llama-3.2-1B",   "Llama-1.3B",   NA, "last", "pile")
  add("meta-llama_Meta-Llama-3-8B","Llama-3-8B",   NA, "last", "pile")

  # ── BabyLM checkpoints ──────────────────────────────────────────────────────
  for (s in c(24, 48, 144, 384, 912, 2280))
    add(paste0("znhoughton_opt-babylm-125m-20eps-seed964_step", s),
        "BabyLM-125M", s, "last", "babylm")
  for (s in c(48, 96, 288, 768, 1824, 4560))
    add(paste0("znhoughton_opt-babylm-350m-20eps-seed964_step", s),
        "BabyLM-350M", s, "last", "babylm")
  for (s in c(97, 194, 582, 1455, 3686, 9021))
    add(paste0("znhoughton_opt-babylm-1.3b-20eps-seed964_step", s),
        "BabyLM-1.3B", s, "last", "babylm")

  # ── Pythia checkpoints ──────────────────────────────────────────────────────
  for (s in c(16, 32, 64, 256, 512, 1000))
    add(paste0("EleutherAI_pythia-160m_step", s), "Pythia-160M", s, "last", "pile")
  for (s in c(16, 32, 64, 256, 512, 1000))
    add(paste0("EleutherAI_pythia-410m_step", s), "Pythia-410M", s, "last", "pile")
  for (s in c(16, 32, 64, 256, 512, 1000))
    add(paste0("EleutherAI_pythia-1b_step", s),   "Pythia-1B",   s, "last", "pile")
  for (s in c(16, 32, 64, 256, 512, 1000))
    add(paste0("EleutherAI_pythia-2.8b_step", s), "Pythia-2.8B", s, "last", "pile")

  bind_rows(rows)
}

REGISTRY  <- build_registry()
COND_MODE <- list(
  list(cond = "default",     mode = "mean_pooled"),
  list(cond = "default",     mode = "words_only"),
  list(cond = "attn_zeroed", mode = "mean_pooled"),
  list(cond = "attn_zeroed", mode = "words_only")
)

# ── Filename convention (must match the writeup's brms_path() helper) ──────────
make_fname <- function(label, step, cond, mode, type) {
  step_str   <- if (is.na(step)) "final" else as.character(step)
  fname_base <- gsub("[^A-Za-z0-9]", "_", paste(label, step_str, cond, mode, sep = "_"))
  file.path(MODEL_DIR, paste0(fname_base, "_", type))
}

# ── Data loader ────────────────────────────────────────────────────────────────
load_data <- function(dir_slug, layer, cond_val, mode_val, freq_src) {
  xz   <- file.path(RESULTS, dir_slug, "by_layer_corpus_pred.csv.xz")
  csv  <- file.path(RESULTS, dir_slug, "by_layer_corpus_pred.csv")
  path <- if (file.exists(xz)) xz else if (file.exists(csv)) csv else return(NULL)

  pred <- read_csv(path, show_col_types = FALSE)
  target_layer <- if (layer == "last") {
    as.character(max(suppressWarnings(as.integer(unique(pred$layer))), na.rm = TRUE))
  } else layer

  pred <- pred |>
    filter(as.character(.data$layer) == target_layer,
           .data$condition == cond_val,
           .data$mode      == mode_val)
  if (nrow(pred) == 0) return(NULL)

  freq_df <- if (freq_src == "babylm") freq_babylm else freq_pile
  pred |>
    inner_join(freq_df, by = c("word1", "word2")) |>
    filter(!is.na(freq_total), freq_total > 0) |>
    mutate(residual = abs(y_true - y_pred),
           log_freq = log(freq_total))
}

# ── Compile Stan templates in the parent process (inherited by children) ───────
log_msg("Compiling Stan templates...")
dummy <- data.frame(residual = c(1, 2, 3, 4, 5), log_freq = c(1, 2, 3, 4, 5))
template_lin <- brm(
  residual ~ log_freq,
  data = dummy, family = gaussian(), chains = 1, iter = 200, warmup = 100,
  refresh = 0, silent = 2
)
template_quad <- brm(
  residual ~ log_freq + I(log_freq^2),
  data = dummy, family = gaussian(), chains = 1, iter = 200, warmup = 100,
  refresh = 0, silent = 2
)
log_msg("Templates compiled.")

# ── Build job list ─────────────────────────────────────────────────────────────
all_jobs <- tidyr::expand_grid(
  i = seq_len(nrow(REGISTRY)),
  j = seq_along(COND_MODE)
)
log_msg(sprintf("Total jobs: %d; running %d in parallel (%d chains each)",
                nrow(all_jobs), N_PARALLEL, CHAINS_PER_MODEL))

# ── Fitting function (runs in each child process) ──────────────────────────────
fit_job <- function(job_idx) {
  i    <- all_jobs$i[[job_idx]]
  j    <- all_jobs$j[[job_idx]]
  reg  <- REGISTRY[i, ]
  cm   <- COND_MODE[[j]]

  lbl  <- reg$label
  stp  <- reg$step
  cond <- cm$cond
  mode <- cm$mode

  file_lin  <- make_fname(lbl, stp, cond, mode, "lin")
  file_quad <- make_fname(lbl, stp, cond, mode, "quad")

  if (file.exists(paste0(file_lin, ".rds")) && file.exists(paste0(file_quad, ".rds"))) {
    log_msg(sprintf("skip: %s|%s|%s|%s",
                    lbl, ifelse(is.na(stp), "final", stp), cond, mode))
    return(invisible(NULL))
  }

  log_msg(sprintf("fitting: %s|%s|%s|%s",
                  lbl, ifelse(is.na(stp), "final", stp), cond, mode))

  df <- load_data(reg$dir_slug, reg$layer, cond, mode, reg$freq_src)
  if (is.null(df) || nrow(df) < 50) {
    log_msg("  -> skipped (no data)")
    return(invisible(NULL))
  }

  check_convergence <- function(fit, name) {
    rh       <- rhat(fit)
    ess      <- neff_ratio(fit)
    divs     <- tryCatch(
      sum(nuts_params(fit, pars = "divergent__")$Value),
      error = function(e) NA_integer_
    )
    max_rh   <- max(rh,  na.rm = TRUE)
    min_ess  <- min(ess, na.rm = TRUE)
    ok <- max_rh <= 1.01 && min_ess >= 0.1 && (is.na(divs) || divs == 0)
    tag <- if (ok) "OK" else "WARNING"
    log_msg(sprintf("  [%s] %s: max_rhat=%.3f  min_ess_ratio=%.3f  divergent=%s",
                    tag, name, max_rh, min_ess,
                    ifelse(is.na(divs), "NA", as.character(divs))))
  }

  fit_lin <- tryCatch(
    update(template_lin, newdata = df, recompile = FALSE,
           chains = CHAINS_PER_MODEL, iter = 4000, warmup = 2000,
           cores  = CHAINS_PER_MODEL, refresh = 0, silent = 2,
           file   = file_lin, file_refit = "on_change"),
    error = function(e) { log_msg("  ERROR (lin): ", conditionMessage(e)); NULL }
  )
  if (is.null(fit_lin)) return(invisible(NULL))
  check_convergence(fit_lin, "lin")
  rm(fit_lin); gc(verbose = FALSE)

  fit_quad <- tryCatch(
    update(template_quad, newdata = df, recompile = FALSE,
           chains = CHAINS_PER_MODEL, iter = 4000, warmup = 2000,
           cores  = CHAINS_PER_MODEL, refresh = 0, silent = 2,
           file   = file_quad, file_refit = "on_change"),
    error = function(e) { log_msg("  ERROR (quad): ", conditionMessage(e)); NULL }
  )
  if (is.null(fit_quad)) return(invisible(NULL))
  check_convergence(fit_quad, "quad")
  rm(fit_quad); gc(verbose = FALSE)

  log_msg(sprintf("  -> saved: %s_{lin,quad}.rds", basename(file_lin)))
  invisible(NULL)
}

# ── Run ────────────────────────────────────────────────────────────────────────
mclapply(seq_len(nrow(all_jobs)), fit_job, mc.cores = N_PARALLEL)
log_msg("Done. All models saved to: ", MODEL_DIR)
