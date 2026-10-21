#!/usr/bin/env Rscript
# emergence_timing.R
# -------------------------------------------------------------------------
# When each of @eq-relfreq's two sources becomes measurable, on one scale.
#
# Both are coefficients in the same model, fit to the same items, so they can
# be timed against each other. Two criteria, which fail in opposite directions:
#
#   detectability  first checkpoint whose posterior is credibly above the SAME
#                  coefficient at step 0, in the same model and condition. Step
#                  0 cannot beat itself, so an untrained network never
#                  registers, which also discounts the tokenization artifact
#                  (Appendix: it inflates step 0 at least as much as the early
#                  checkpoints). Favours y_pred, which is a fitted readout.
#
#   trajectory     first checkpoint reaching a given fraction of that
#                  coefficient's own change from step 0 to its final value. No
#                  null and no power comparison, so it does not favour the
#                  fitted measure, but it needs a defined endpoint.
#
# Both are reported "sustained": the criterion must hold at that checkpoint and
# every later one, because significance flickers at noise level in the early
# window and a first-crossing rule picks up blips.
#
# The posterior draws for the current fits live on the compute server; only the
# summary came back. Sigma is therefore recovered from the credible interval,
# which is accurate for these posteriors (large n, regular model) and is the
# only approximation here.
#
# Writes Data/derived/emergence_timing.csv.

suppressMessages({library(dplyr); library(tidyr); library(readr)})
options(contrasts = c("contr.sum", "contr.sum"))

SUMMARY <- "Data/derived/brms_relfreq_rawscale.rds"
OUT     <- "Data/derived/emergence_timing.csv"
Z       <- qnorm(0.975)

# tokens per optimiser step, in millions
TOK <- c("BabyLM-125M" = 0.8192, "BabyLM-350M" = 0.4096, "BabyLM-1.3B" = 0.2048,
         "Pythia-160M" = 2.097152, "Pythia-410M" = 2.097152,
         "Pythia-1B"   = 2.097152)

d <- readRDS(SUMMARY) |> filter(!is.na(step), label %in% names(TOK)) |>
  mutate(tokens_M = step * TOK[label])

long <- bind_rows(
  d |> transmute(label, condition, step, tokens_M, term = "y_pred",
                 mu = pred_mean,  lo = pred_lo,  hi = pred_hi),
  d |> transmute(label, condition, step, tokens_M, term = "rel_freq",
                 mu = rel_mean,   lo = rel_lo,   hi = rel_hi),
  d |> transmute(label, condition, step, tokens_M, term = "rel_freq_x_log_freq",
                 mu = rel_x_mean, lo = rel_x_lo, hi = rel_x_hi)) |>
  mutate(sg = (hi - lo) / (2 * Z))

base <- long |> filter(step == 0) |>
  transmute(label, condition, term, mu0 = mu, sg0 = sg)
fin <- long |> group_by(label, condition, term) |> filter(step == max(step)) |>
  ungroup() |> transmute(label, condition, term, mufin = mu)

w <- long |> filter(step > 0) |>
  inner_join(base, by = c("label", "condition", "term")) |>
  inner_join(fin,  by = c("label", "condition", "term")) |>
  mutate(p_gt = pnorm((mu - mu0) / sqrt(sg^2 + sg0^2)),
         sig  = p_gt > 0.975,
         frac = (mu - mu0) / (mufin - mu0))

# first index from which a logical vector is TRUE and stays TRUE
sustained_at <- function(ok) {
  k <- which(rev(cumprod(rev(as.integer(ok)))) == 1)
  if (length(k)) k[1] else NA_integer_
}
pick <- function(df) {
  df <- arrange(df, step)
  i_det <- sustained_at(df$sig)
  fr <- function(f) {
    k <- sustained_at(df$frac >= f)
    if (is.na(k)) NA_real_ else df$tokens_M[k]
  }
  tibble(detect_step     = if (is.na(i_det)) NA_real_ else df$step[i_det],
         detect_tokens_M = if (is.na(i_det)) NA_real_ else df$tokens_M[i_det],
         frac25_tokens_M = fr(0.25),
         frac50_tokens_M = fr(0.50),
         frac75_tokens_M = fr(0.75))
}

res <- w |> group_by(label, condition, term) |> group_modify(~ pick(.x)) |>
  ungroup() |> arrange(condition, label, term)
write_csv(res, OUT)
cat("wrote", OUT, "\n\n")
print(as.data.frame(res |> mutate(across(where(is.numeric), ~ round(.x, 1)))),
      row.names = FALSE)

cat("\nranges, full context (M tokens):\n")
r <- res |> filter(condition == "default")
for (tm in c("y_pred", "rel_freq", "rel_freq_x_log_freq")) {
  s <- r |> filter(term == tm)
  cat(sprintf("  %-20s detect %5.0f-%5.0f   75%% of change %5.0f-%5.0f\n", tm,
              min(s$detect_tokens_M), max(s$detect_tokens_M),
              min(s$frac75_tokens_M), max(s$frac75_tokens_M)))
}
