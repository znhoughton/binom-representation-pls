# prepare_relfreq_slopes.R
# Model-implied simple slopes for the main-text figures.
#
# @eq-relfreq already says how each source's contribution changes with frequency:
#
#     slope of y_pred   at frequency f  =  beta_1 + beta_4 * f
#     slope of rel_freq at frequency f  =  beta_2 + beta_5 * f
#
# so the figures should be drawn from the posterior rather than by refitting an
# ordinary regression inside each frequency decile. Refitting per decile
# discards the pooling the model performs, produces a jagged line that invites
# reading noise as structure, and carries no credible intervals. It is a useful
# LINEARITY CHECK -- and that check was run: the per-decile slopes fall close to
# a straight line in every model -- but it is not the result.
#
# Slopes are emitted at two sets of frequencies, tagged by `kind`: the decile
# medians the prose quotes, and a dense grid spanning the fitted range, which is
# what the figures draw. Both carry a raw frequency (`mid_freq`) alongside the
# standardised one, so the figures can label an axis in word counts instead of
# SDs. See the two blocks below for why each is needed.
#
# The full posterior is used, not just the coefficient means: beta_1 and beta_4
# covary, so a credible interval for their sum has to be computed from the joint
# draws rather than from the two marginals.
#
# Usage (from project root):
#   Rscript Scripts/analysis/prepare_relfreq_slopes.R
# Output: Data/derived/relfreq_slope_curve.csv

# Every path below is relative to the project root, so resolve it by walking up
# from this script until the data directories appear. A fixed number of ".."
# hops would silently break the moment the script moves between Scripts/
# subdirectories, which is exactly what happened when Scripts/ was reorganised.
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
suppressPackageStartupMessages({
  library(brms); library(dplyr); library(readr); library(tibble); library(purrr)
})

NDEC <- 10
MODEL_DIR <- file.path("Data", "brms_models")
RESULTS   <- "Results"
MODE      <- "mean_pooled"

# Which parameterisation to read. "relfreq_rawscale" leaves rel_freq on its
# -0.5..+0.5 proportion scale and centres log frequency WITHOUT scaling;
# "relfreq_prop" z-scores both. The frequency back-conversion below differs
# between the two, so this must match the fits actually on disk.
SUFFIX <- "relfreq_rawscale"
RAW    <- SUFFIX == "relfreq_rawscale"
fits <- Sys.glob(file.path(MODEL_DIR, paste0("*_final_*_", SUFFIX, ".rds")))
cat(length(fits), "final-checkpoint fits found\n")

# ── Recovering raw frequencies ───────────────────────────────────────────────
# The fits store log_freq_z, a z-score, so an axis drawn from them is labelled in
# SDs and a reader cannot map a tick to a word count. Undoing the z-score needs
# the mean and SD of log(total) on the exact set each fit used, which the fit
# object does not keep, so recompute it from the source data.
#
# One pass per CORPUS is enough, not one per fit: log_freq is a property of the
# training corpus rather than of the model reading it, so every model sharing a
# corpus sees an identical frequency column. The assertion below enforces that
# rather than trusting it.
counts <- list(
  babylm = read_csv("Data/corpus_binomials.csv", show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1_w2, n_w2_w1 = freq_w2_w1),
  pile = read_csv(file.path(RESULTS, "corpus_binomials_infinigram_piletrain.csv"),
                  show_col_types = FALSE) |>
    transmute(word1, word2, n_w1_w2 = freq_w1w2, n_w2_w1 = freq_w2w1)
)
ref_slug <- c(babylm = "znhoughton_opt-babylm-125m-20eps-seed964", pile = "gpt2")

freq_scale <- lapply(names(ref_slug), function(cp) {
  xz  <- file.path(RESULTS, ref_slug[[cp]], "by_layer_corpus_pred.csv.xz")
  con <- xzfile(xz, "rb"); pred <- read_csv(con, show_col_types = FALSE); close(con)
  fl  <- max(suppressWarnings(as.integer(unique(pred$layer))), na.rm = TRUE)
  lt  <- pred |>
    filter(as.integer(layer) == fl, condition == "default", mode == MODE) |>
    inner_join(counts[[cp]], by = c("word1", "word2")) |>
    # the same two filters the fitting script applies, in the same order
    filter(n_w1_w2 + n_w2_w1 > 0, word1 != word2) |>
    transmute(lt = log(n_w1_w2 + n_w2_w1)) |>
    pull(lt)
  cat(sprintf("  %-7s n=%s  mean log freq=%.4f  sd=%.4f\n",
              cp, format(length(lt), big.mark = ","), mean(lt), sd(lt)))
  list(mu = mean(lt), sg = sd(lt), n = length(lt))
})
names(freq_scale) <- names(ref_slug)

corpus_of <- function(label) if (grepl("^BabyLM", label)) "babylm" else "pile"

parse_name <- function(f) {
  b <- sub(paste0("_", SUFFIX, "\\.rds$"), "", basename(f))
  b <- sub("_mean_pooled$", "", b)
  cond <- if (grepl("_attn_zeroed$", b)) "attn_zeroed" else "default"
  lab  <- sub("_(attn_zeroed|default)$", "", b)
  lab  <- sub("_final$", "", lab)
  lab  <- gsub("_", "-", lab)
  lab  <- sub("BabyLM-1-3B", "BabyLM-1.3B", lab)
  lab  <- sub("Pythia-2-8B", "Pythia-2.8B", lab)
  lab  <- sub("Llama-1-3B",  "Llama-1.3B",  lab)
  lab  <- sub("OLMo-2-1B",   "OLMo-2-1B",   lab)
  list(label = lab, condition = cond)
}

out <- map_dfr(fits, function(f) {
  nm  <- parse_name(f)
  fit <- readRDS(f)
  d   <- fit$data
  dr  <- as.data.frame(brms::as_draws_df(fit))

  b_pred   <- dr[["b_y_pred_z"]]
  b_rel    <- dr[["b_rel_freq_z"]]
  b_pred_x <- dr[["b_y_pred_z:log_freq_z"]]
  b_rel_x  <- dr[["b_log_freq_z:rel_freq_z"]]

  # decile midpoints on the standardised frequency scale the model was fit on
  dec <- dplyr::ntile(d$log_freq_z, NDEC)
  mid <- tapply(d$log_freq_z, dec, median)

  # the corpus constants must describe THIS fit's rows, or the raw frequencies
  # they produce are wrong; fail loudly rather than mislabel an axis
  fs <- freq_scale[[corpus_of(nm$label)]]
  stopifnot(nrow(d) == fs$n)

  # Two sets of evaluation points, distinguished by `kind`.
  #
  # "decile" reproduces the per-decile values the prose quotes. These are a poor
  # basis for the LINE, though: in BabyLM 70% of binomials occur exactly once, so
  # seven of the ten decile medians sit at a frequency of 1 and the ten points
  # collapse onto three distinct positions, spanning 1 to 5 occurrences when the
  # corpus actually reaches 1,095.
  #
  # "grid" therefore spans the 1st to 99th percentile of the frequency the model
  # was fit on, which is the range over which @eq-relfreq is being asserted. The
  # line and its interval are drawn from these; the decile points are overlaid to
  # show where the data mass really is.
  eval_at <- bind_rows(
    tibble(kind = "decile", f = as.numeric(mid), decile = seq_len(NDEC)),
    tibble(kind = "grid",
           f = seq(quantile(d$log_freq_z, .01), quantile(d$log_freq_z, .99),
                   length.out = 40),
           decile = NA_integer_)
  )

  map_dfr(seq_len(nrow(eval_at)), function(i) {
    f_at   <- eval_at$f[[i]]
    s_pred <- b_pred + b_pred_x * f_at        # simple slope, per draw
    s_rel  <- b_rel  + b_rel_x  * f_at
    tibble(
      label = nm$label, condition = nm$condition,
      kind = eval_at$kind[[i]], decile = eval_at$decile[[i]],
      mid_log_freq_z = f_at,
      # back on the count scale: z -> log count -> count
      # raw fits centre log frequency but do not scale it, so undoing the
      # transform is an addition; z-scored fits need the sd as well
      mid_log_freq   = if (RAW) f_at + fs$mu else f_at * fs$sg + fs$mu,
      mid_freq       = exp(if (RAW) f_at + fs$mu else f_at * fs$sg + fs$mu),
      term = c("y_pred", "rel_freq"),
      beta = c(mean(s_pred), mean(s_rel)),
      lo   = c(quantile(s_pred, .025), quantile(s_rel, .025)),
      hi   = c(quantile(s_pred, .975), quantile(s_rel, .975))
    )
  })
})

write_csv(out, "Data/derived/relfreq_slope_curve.csv")
cat(sprintf("wrote Data/derived/relfreq_slope_curve.csv: %d rows, %d models\n",
            nrow(out), length(unique(out$label))))

print(out |> filter(decile %in% c(1, 10), condition == "default") |>
        select(label, decile, term, beta) |>
        tidyr::pivot_wider(names_from = c(term, decile), values_from = beta) |>
        mutate(across(where(is.numeric), ~round(., 3))) |> as.data.frame(),
      row.names = FALSE)
