# corpus_relfreq_brms_all.R
# Experiments 1, 2 and 3 in one run: the ordering-source regression at every
# final checkpoint and every saved training checkpoint.
#
#   bf(y_true ~ y_pred * log_freq + rel_freq * log_freq, sigma ~ log_freq)
#
# ALL FOUR VARIABLES ARE Z-SCORED WITHIN EACH CELL, which is the change from the
# previous run. Two distinct scale problems have to be separated:
#
#   WITHIN a cell, the spread of y_true varies WITH FREQUENCY (elevated at both
#   tails). A slope is r * sd(y_true)/sd(x), so unmodelled this inflates the
#   moderation terms at one end of the range. sigma ~ log_freq handles it, by
#   making the frequency-dependence of the residual scale an estimated parameter.
#   Standardising y_true within frequency BINS would also remove it but is not
#   defensible: it rescales the outcome by a function of the moderator whose
#   interaction is the result being tested.
#
#   BETWEEN cells, the OVERALL spread of y_true differs -- across models, across
#   attention conditions, and across training checkpoints (sd(y_true) drifts from
#   2.02 to 2.85 over Pythia-160M's checkpoints alone). sigma ~ log_freq cannot
#   touch this: it models scatter around the line within a cell, and each cell is
#   a separate fit with no term linking one cell's scale to another's. Since the
#   predictors are z-scored per cell too, a coefficient is "log-odds per SD of
#   THIS cell's predictor", and both parts of that ratio move between cells.
#   Every contrast in Experiments 2 and 3 is a contrast across cells, so this
#   matters. Z-scoring y_true as well is a single constant per cell: it changes
#   no within-cell relationship and puts all cells in common units. Coefficients
#   are then standardised betas.
#
# SPEED. The distributional model evaluates sigma per observation over ~43k rows,
# which is what makes each Pile cell expensive. Three things address that:
#   * cmdstanr backend, generally well ahead of rstan on this kind of model
#   * within-chain threading via reduce_sum, which is close to linear in n
#   * one compiled template reused by update(recompile = FALSE) for every cell,
#     so Stan compilation happens once rather than 104 times
# Iterations stay at 4000/2000 to match every other regression in the paper;
# lowering them is the remaining lever if more speed is needed.
#
# rel_freq is the SHARE OF OCCURRENCES IN ALPHABETICAL ORDER, centred at zero
# (+0.5 always alphabetical, -0.5 always reversed, 0 no preference). This
# replaces the earlier log odds ratio, which was undefined when one ordering was
# unattested and so forced a both-orders-attested restriction that discarded 92%
# of BabyLM. The proportion is defined for every binomial, so only reduplicatives
# are dropped. The trade is that a binomial seen once lands at the extreme of the
# scale on a single observation: negligible for the large-scale corpora (0.9%
# singletons) but true of 71% of BabyLM, whose first seven frequency deciles are
# entirely singletons.
#
# Pythia-2.8B is excluded. Its checkpoint prediction files are byte-identical
# across all six steps -- the revision was not applied when they were extracted --
# and it is not among the 16 models the paper reports.
#
# Usage (from project root):
#   Rscript Scripts/analysis/corpus_relfreq_brms_all.R
# Resume-safe: skips any cell whose .rds already exists.
# Output: Data/derived/brms_<SUFFIX>.rds  (step = NA marks a final checkpoint)

suppressPackageStartupMessages({
  library(brms); library(dplyr); library(readr); library(tibble)
  library(purrr); library(posterior)
})

# Every path below is relative to the project root. Resolve it by walking up
# from this script rather than trusting the caller's working directory, so the
# script runs correctly from anywhere and survives being moved within Scripts/.
find_project_root <- function() {
  a <- commandArgs(FALSE)
  f <- sub("--file=", "", grep("--file=", a, value = TRUE)[1])
  d <- if (is.na(f)) getwd() else dirname(normalizePath(f))
  while (!all(dir.exists(file.path(d, c("Data", "Results")))) && dirname(d) != d) d <- dirname(d)
  if (!all(dir.exists(file.path(d, c("Data", "Results")))))
    stop("could not locate project root (no ancestor with Data/ and Results/)")
  d
}
setwd(find_project_root())


# ── Predictor scaling: "z" (default) or "raw" ────────────────────────────────
# Set by the SCALE environment variable; controls the output filenames too, so
# the two variants never overwrite each other's cached fits.
#
# "z"   every predictor z-scored within the cell. Coefficients are standardised
#       betas, comparable across cells, but "one SD of rel_freq" denotes a
#       different quantity in each corpus: BabyLM's SD is inflated because 92% of
#       its items sit at a boundary value, which is part of why its moderation
#       looks large.
#
# "raw" rel_freq is left on its natural -0.5..+0.5 proportion scale and log
#       frequency is centred but not scaled. A proportion is an interpretable
#       unit like miles, not an arbitrary one, so dividing it by a corpus-
#       specific SD makes the same physical difference count differently in
#       different corpora. On this scale the rel_freq coefficient reads as
#       "SDs of ordering preference per unit of proportion", one unit being the
#       span from always-reversed to always-alphabetical, and the log_freq
#       coefficient reads per e-fold of frequency. y_true and y_pred stay
#       z-scored either way: their spread genuinely differs across models,
#       conditions and checkpoints, and every cross-cell contrast in
#       Experiments 2 and 3 depends on putting them in common units.
SCALE <- tolower(Sys.getenv("SCALE", "z"))
stopifnot(SCALE %in% c("z", "raw"))
SUFFIX <- if (SCALE == "z") "relfreq_prop" else "relfreq_rawscale"

# Chains x threads = total cores. Defaults preserve the old 4x2=8 behaviour;
# set BRMS_CHAINS / BRMS_THREADS to use a bigger machine (e.g. 4 x 6 = 24).
CHAINS  <- as.integer(Sys.getenv("BRMS_CHAINS",  "4"))
THREADS <- as.integer(Sys.getenv("BRMS_THREADS", "2"))
message(sprintf("brms parallelism: %d chains x %d threads = %d cores",
                CHAINS, THREADS, CHAINS * THREADS))
ITER    <- 4000
WARMUP  <- 2000
SEED    <- 964

RESULTS   <- "Results"
MODEL_DIR <- file.path("Data", "brms_models")
OUT_RDS   <- file.path("Data", "derived", paste0("brms_", SUFFIX, ".rds"))
LOG       <- file.path(RESULTS, paste0("brms_", SUFFIX, "_progress.log"))
dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(OUT_RDS), showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) {
  m <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", ..., "\n")
  cat(m); cat(m, file = LOG, append = TRUE)
}

# Cells whose inputs are absent are skipped, so a run that refits one model would otherwise
# replace a file holding every model with one holding only that model. Anything already on disk
# is kept, except the cells this run refits, which supersede their previous values.
PRIOR <- if (file.exists(OUT_RDS)) readRDS(OUT_RDS) else NULL
if (!is.null(PRIOR))
  log_msg(sprintf("Found %d existing cells in %s; refitted cells will replace their rows.",
                  nrow(PRIOR), OUT_RDS))

# ── Registry: final checkpoints first, then training checkpoints ──────────────
final_reg <- tribble(
  ~slug,                                       ~label,        ~corpus,
  "znhoughton_opt-babylm-125m-20eps-seed964",  "BabyLM-125M", "babylm",
  "znhoughton_opt-babylm-350m-20eps-seed964",  "BabyLM-350M", "babylm",
  "znhoughton_opt-babylm-1_3b-20eps-seed964",  "BabyLM-1.3B", "babylm",
  "EleutherAI_pythia-160m",                    "Pythia-160M", "pile",
  "EleutherAI_pythia-410m",                    "Pythia-410M", "pile",
  "EleutherAI_pythia-1b",                      "Pythia-1B",   "pile",
  "gpt2",                                      "GPT-2",       "pile",
  "gpt2-medium",                               "GPT-2-M",     "pile",
  "gpt2-large",                                "GPT-2-L",     "pile",
  "gpt2-xl",                                   "GPT-2-XL",    "pile",
  "allenai_OLMo-1B-hf",                        "OLMo-1B",     "pile",
  "allenai_OLMo-7B-hf",                        "OLMo-7B",     "pile",
  "allenai_OLMo-2-0425-1B",                    "OLMo-2-1B",   "pile",
  "allenai_OLMo-2-1124-7B",                    "OLMo-2-7B",   "pile",
  "meta-llama_Llama-3.2-1B",                   "Llama-1.3B",  "pile",
  "meta-llama_Meta-Llama-3-8B",                "Llama-3-8B",  "pile"
) |> mutate(step = NA_integer_)

ckpt_reg <- bind_rows(
  tibble(label = "BabyLM-125M", step = c(24, 48, 144, 384, 912, 2280), corpus = "babylm",
         slug = paste0("znhoughton_opt-babylm-125m-20eps-seed964_step", step)),
  tibble(label = "BabyLM-350M", step = c(48, 96, 288, 768, 1824, 4560), corpus = "babylm",
         slug = paste0("znhoughton_opt-babylm-350m-20eps-seed964_step", step)),
  tibble(label = "BabyLM-1.3B", step = c(97, 194, 582, 1455, 3686, 9021), corpus = "babylm",
         slug = paste0("znhoughton_opt-babylm-1.3b-20eps-seed964_step", step)),
  tibble(label = "Pythia-160M", step = c(16, 32, 64, 256, 512, 1000), corpus = "pile",
         slug = paste0("EleutherAI_pythia-160m_step", step)),
  tibble(label = "Pythia-410M", step = c(16, 32, 64, 256, 512, 1000), corpus = "pile",
         slug = paste0("EleutherAI_pythia-410m_step", step)),
  tibble(label = "Pythia-1B",   step = c(16, 32, 64, 256, 512, 1000), corpus = "pile",
         slug = paste0("EleutherAI_pythia-1b_step", step))
) |> select(slug, label, corpus, step)

REGISTRY   <- bind_rows(final_reg, ckpt_reg)
CONDITIONS <- c("default", "attn_zeroed")
MODE       <- "mean_pooled"

counts <- list(
  babylm = read_csv("Data/corpus_binomials.csv", show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1_w2, n_w2_w1 = freq_w2_w1),
  pile = read_csv(file.path(RESULTS, "corpus_binomials_infinigram_piletrain.csv"),
                  show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1w2, n_w2_w1 = freq_w2w1)
)

make_fname <- function(label, step, cond) {
  tag  <- if (is.na(step)) "final" else as.character(step)
  base <- gsub("[^A-Za-z0-9]", "_", paste(label, tag, cond, MODE, sep = "_"))
  file.path(MODEL_DIR, paste0(base, "_", SUFFIX))
}


load_cell <- function(slug, corpus, cond) {
  xz <- file.path(RESULTS, slug, "by_layer_corpus_pred.csv.xz")
  if (!file.exists(xz)) return(NULL)
  con <- xzfile(xz, "rb"); pred <- read_csv(con, show_col_types = FALSE); close(con)
  final_layer <- max(suppressWarnings(as.integer(unique(pred$layer))), na.rm = TRUE)

  pred |>
    filter(as.integer(layer) == final_layer, condition == cond, mode == MODE) |>
    inner_join(counts[[corpus]], by = c("word1", "word2")) |>
    filter(n_w1_w2 + n_w2_w1 > 0) |>
    # Reduplicatives ("again and again") are excluded: the two orderings are the
    # same string, so y_true is exactly 0 for every one of them while the reverse
    # count is 0 by construction, which would put them at rel_freq = +0.5. That
    # is 472 items with no outcome variance parked at the extreme of the
    # predictor, which biases its slope toward zero. The previous log-odds scale
    # dropped them automatically by requiring both orders to be attested; on a
    # proportion scale they have to be removed by hand.
    filter(word1 != word2) |>
    # rel_freq is the share of occurrences in ALPHABETICAL order, centred at
    # zero: +0.5 means always alphabetical, -0.5 always reversed, 0 no
    # preference. Unlike a log odds ratio it is defined when one order is
    # unattested, so no other data has to be discarded.
    #
    # word1 is alphabetically first in every remaining row, so y_true, y_pred and
    # rel_freq already share an orientation and no sign flipping is needed. The
    # guard below makes that assumption fail loudly rather than silently if the
    # corpus files are ever regenerated differently.
    mutate(total = n_w1_w2 + n_w2_w1) |>
    (\(d) { stopifnot(all(d$word1 < d$word2)); d })() |>
    transmute(
      y_true_z   = c(scale(y_true)),
      y_pred_z   = c(scale(y_pred)),
      # column names are kept identical across both scalings so the formula,
      # the draw extraction and every downstream script stay unchanged
      # c() strips the 1-column matrix scale() returns back to a vector
      log_freq_z = if (SCALE == "z") c(scale(log(total)))
                   else              c(scale(log(total), scale = FALSE)),
      rel_freq_z = if (SCALE == "z") c(scale(n_w1_w2 / total - 0.5))
                   else              (n_w1_w2 / total - 0.5)
    )
}

FORMULA <- bf(y_true_z ~ y_pred_z * log_freq_z + rel_freq_z * log_freq_z,
              sigma ~ log_freq_z)

log_msg("Compiling Stan template (cmdstanr, threading = ", THREADS, ")...")
set.seed(SEED)
dummy <- tibble(y_true_z = rnorm(60), y_pred_z = rnorm(60),
                log_freq_z = rnorm(60), rel_freq_z = rnorm(60))
t0 <- Sys.time()
template <- brm(FORMULA, data = dummy, family = gaussian(), chains = 1,
                iter = 200, warmup = 100, refresh = 0, silent = 2, seed = SEED,
                backend = "cmdstanr", threads = threading(THREADS))
log_msg(sprintf("Template compiled in %.1f min.",
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))

check_convergence <- function(fit) {
  mx <- max(rhat(fit), na.rm = TRUE); mn <- min(neff_ratio(fit), na.rm = TRUE)
  dv <- tryCatch(sum(nuts_params(fit, pars = "divergent__")$Value),
                 error = function(e) NA_integer_)
  ok <- mx <= 1.01 && mn >= 0.1 && (is.na(dv) || dv == 0)
  log_msg(sprintf("  [%s] max_rhat=%.3f min_ess=%.3f divergent=%s",
                  if (ok) "OK" else "WARNING", mx, mn,
                  ifelse(is.na(dv), "NA", as.character(dv))))
}

jobs <- tidyr::expand_grid(i = seq_len(nrow(REGISTRY)), cond = CONDITIONS) |>
  arrange(i)
log_msg(sprintf("%d cells to fit (%d final, %d checkpoint)",
                nrow(jobs), 2 * nrow(final_reg), 2 * nrow(ckpt_reg)))

rows <- list()
for (k in seq_len(nrow(jobs))) {
  reg  <- REGISTRY[jobs$i[[k]], ]
  cond <- jobs$cond[[k]]
  tag  <- if (is.na(reg$step)) "final" else paste("step", reg$step)

  d <- load_cell(reg$slug, reg$corpus, cond)
  if (is.null(d) || nrow(d) < 200) {
    log_msg(sprintf("skip: %s %s | %s", reg$label, tag, cond)); next
  }
  log_msg(sprintf("[%d/%d] %s %s | %s (n=%d)", k, nrow(jobs), reg$label, tag, cond, nrow(d)))

  fit <- tryCatch(
    update(template, newdata = d, recompile = FALSE, chains = CHAINS,
           iter = ITER, warmup = WARMUP, cores = CHAINS, refresh = 0,
           silent = 2, seed = SEED, file = make_fname(reg$label, reg$step, cond),
           file_refit = "on_change"),
    error = function(e) { log_msg("  ERROR: ", conditionMessage(e)); NULL })
  if (is.null(fit)) next
  check_convergence(fit)

  dr <- as_draws_df(fit)
  grab <- function(term) {
    v <- dr[[paste0("b_", term)]]
    c(mean(v), unname(quantile(v, .025)), unname(quantile(v, .975)), mean(v > 0) * 100)
  }
  tm <- c(pred = "y_pred_z", freq = "log_freq_z", rel = "rel_freq_z",
          pred_x = "y_pred_z:log_freq_z", rel_x = "log_freq_z:rel_freq_z",
          sigma_freq = "sigma_log_freq_z")
  vals <- map(tm, grab)

  rows[[length(rows) + 1]] <- tibble(
    label = reg$label, step = reg$step, condition = cond, n = nrow(d),
    !!!set_names(map_dbl(vals, 1), paste0(names(tm), "_mean")),
    !!!set_names(map_dbl(vals, 2), paste0(names(tm), "_lo")),
    !!!set_names(map_dbl(vals, 3), paste0(names(tm), "_hi")),
    !!!set_names(map_dbl(vals, 4), paste0(names(tm), "_pgt0"))
  )
  log_msg(sprintf("  -> y_pred %+.3f | rel_freq %+.3f | yp:lf %+.3f | rf:lf %+.3f",
                  vals$pred[1], vals$rel[1], vals$pred_x[1], vals$rel_x[1]))

  # write incrementally so a long run is inspectable and survives interruption
  .new  <- bind_rows(rows)
  .keep <- if (is.null(PRIOR)) NULL
           else anti_join(PRIOR, .new, by = c("label", "step", "condition"))
  saveRDS(bind_rows(.keep, .new), OUT_RDS)
  rm(fit, dr); gc(verbose = FALSE)
}

log_msg(sprintf("Done. %d cells refitted; %d cells in %s",
                length(rows), nrow(readRDS(OUT_RDS)), OUT_RDS))
