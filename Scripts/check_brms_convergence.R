# check_brms_convergence.R
# Post-hoc convergence check for all saved brms models in Data/brms_models/.
# Run from the project root after corpus_freq_regression_brms_quad_compare.R
# completes (or at any point to check models fitted so far).
#
# Usage:  Rscript Scripts/check_brms_convergence.R
# Output: prints a summary table; any WARNING rows also written to
#         Results/brms_convergence_warnings.csv

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(readr)
})

MODEL_DIR <- file.path("Data", "brms_models")
OUT       <- file.path("Results", "brms_convergence_warnings.csv")

rds_files <- list.files(MODEL_DIR, pattern = "\\.rds$", full.names = TRUE)
if (length(rds_files) == 0) stop("No .rds files found in ", MODEL_DIR)

cat(sprintf("Checking %d model files...\n\n", length(rds_files)))

check_one <- function(path) {
  fit <- tryCatch(readRDS(path), error = function(e) NULL)
  if (is.null(fit)) {
    return(tibble(file = basename(path), max_rhat = NA, min_ess = NA,
                  divergent = NA, status = "ERROR (could not load)"))
  }
  rh   <- tryCatch(rhat(fit),        error = function(e) NULL)
  ess  <- tryCatch(neff_ratio(fit),  error = function(e) NULL)
  divs <- tryCatch(
    sum(nuts_params(fit, pars = "divergent__")$Value),
    error = function(e) NA_integer_
  )
  max_rh  <- if (!is.null(rh))  max(rh,  na.rm = TRUE) else NA_real_
  min_ess <- if (!is.null(ess)) min(ess, na.rm = TRUE) else NA_real_

  ok <- !is.na(max_rh)  && max_rh  <= 1.01 &&
        !is.na(min_ess) && min_ess >= 0.1  &&
        (is.na(divs)    || divs == 0)

  tibble(
    file      = basename(path),
    max_rhat  = round(max_rh,  4),
    min_ess   = round(min_ess, 4),
    divergent = divs,
    status    = if (ok) "OK" else "WARNING"
  )
}

results <- bind_rows(lapply(rds_files, check_one))

# Print full summary
print(results, n = Inf)

# Summary counts
n_ok      <- sum(results$status == "OK",      na.rm = TRUE)
n_warn    <- sum(results$status == "WARNING",  na.rm = TRUE)
n_err     <- sum(results$status == "ERROR (could not load)", na.rm = TRUE)

cat(sprintf("\n--- Summary ---\n  OK: %d  |  WARNING: %d  |  ERROR: %d  |  Total: %d\n",
            n_ok, n_warn, n_err, nrow(results)))

# Save warnings
warnings <- filter(results, status != "OK")
if (nrow(warnings) > 0) {
  write_csv(warnings, OUT)
  cat(sprintf("\nWARNING/ERROR rows written to: %s\n", OUT))
} else {
  cat("\nAll models passed convergence checks.\n")
}
