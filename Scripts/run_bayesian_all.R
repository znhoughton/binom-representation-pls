library(here)
library(parallel)

slugs  <- c(
  "znhoughton_opt-babylm-125m-20eps-seed964",
  "znhoughton_opt-babylm-350m-20eps-seed964",
  "znhoughton_opt-babylm-1_3b-20eps-seed964"
)
layers <- c("last", "second_to_last")

combos <- expand.grid(slug = slugs, layer = layers,
                      stringsAsFactors = FALSE)

RMD    <- here::here("Scripts", "bayesian_analysis.Rmd")
OUTDIR <- here::here("Results")

render_one <- function(i) {
  slug  <- combos$slug[i]
  layer <- combos$layer[i]
  label <- sprintf("%s_layer_%s", slug, layer)
  log_f <- file.path(OUTDIR, sprintf("bayesian_%s.log", label))

  cat(sprintf("[%s] Starting: %s\n", format(Sys.time(), "%H:%M:%S"), label))

  tryCatch({
    sink(log_f, append = FALSE, split = FALSE)
    rmarkdown::render(
      input       = RMD,
      params      = list(slug = slug, layer = layer),
      output_file = sprintf("bayesian_analysis_%s.html", label),
      output_dir  = OUTDIR,
      quiet       = FALSE
    )
    sink()
    cat(sprintf("[%s] Done: %s\n", format(Sys.time(), "%H:%M:%S"), label))
  }, error = function(e) {
    sink()
    msg <- sprintf("ERROR in %s: %s\n", label, conditionMessage(e))
    cat(msg)
    cat(msg, file = log_f, append = TRUE)
  })
}

# 7 slots × 4 chains = 28 cores; all 6 combos fit in one batch
mclapply(seq_len(nrow(combos)), render_one, mc.cores = 7)

cat("\nAll done.\n")
