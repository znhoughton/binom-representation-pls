#!/usr/bin/env bash
# =============================================================================
# run_stats.sh
#
# Stages 3 and 4 of the paper pipeline: everything left once the probe cells
# are done. Run this after run_paper_pipeline.sh --stages 0 1 has finished all
# 58 cells.
#
#   3  cross-model agreement        CPU, seconds      Appendix H, Discussion
#   4  brms Eq. 1, 116 cells        CPU, hours        Exp 1-3, Appendix F
#      merge shard summaries
#      simple slopes                CPU, minutes      the slope figures
#
# Parallelism
# -----------
# The brms script already ran its four chains in parallel, but cells went
# through a sequential loop, so a 24-core box used 8 cores and idled 16. This
# shards the cells across workers as well:
#
#   WORKERS x CHAINS x THREADS = cores
#          6 x      4 x       1 = 24
#
# THREADS is WITHIN-chain threading, so the three multiply. Raising THREADS
# without lowering WORKERS oversubscribes: 6 x 4 x 4 would be 96 threads on 24
# cores. With one chain per core there is nothing for within-chain threading
# to use, so THREADS=1 is the right setting here and the script skips the
# reduce_sum machinery entirely.
#
# What it leaves you
# ------------------
# The writeup reads two small tracked files, and this produces both:
#   Data/derived/brms_relfreq_rawscale.rds   ~0.02 MB
#   Data/derived/relfreq_slope_curve.csv     ~0.5 MB
# The 116 fitted models under Data/brms_models/ are a few hundred MB, stay
# gitignored, and are regenerable. --keep-fits leaves them; by default they
# are kept too, since deleting them costs you any later re-analysis that wants
# the posteriors. Delete by hand when you are sure.
#
# Usage
# -----
#   ./run_stats.sh                     # 6 workers, the default
#   ./run_stats.sh --workers 3         # if something else needs cores
#   ./run_stats.sh --stages 4          # skip cross-model agreement
#   ./run_stats.sh --dry-run
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
WORKERS=6
CHAINS=4
THREADS=1
SCALE=raw
STAGES="3 4"
DRY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --chains)  CHAINS="$2";  shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --scale)   SCALE="$2";   shift 2 ;;
    --stages)  shift; STAGES=""
               while [[ $# -gt 0 && "$1" != --* ]]; do STAGES="$STAGES $1"; shift; done ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="$ROOT/Results/logs"; mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/stats_$(date +%Y%m%d_%H%M%S).log"
say()  { printf '\n== %s  (%s)\n' "$1" "$(date +%H:%M:%S)" | tee -a "$RUN_LOG"; }
note() { printf '   %s\n' "$1" | tee -a "$RUN_LOG"; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }

say "binom-corpus-pls statistics"
note "stages: $STAGES   workers: $WORKERS x ${CHAINS} chains x ${THREADS} threads = $((WORKERS*CHAINS*THREADS)) cores"
note "scale: $SCALE   log: $RUN_LOG"
NPROC="$(nproc 2>/dev/null || echo '?')"
note "machine reports $NPROC cores"
if [[ "$NPROC" != "?" ]] && (( WORKERS*CHAINS*THREADS > NPROC )); then
  note "WARNING: that is more threads than cores; Stan degrades when oversubscribed."
fi
[[ -n "$DRY" ]] && note "DRY RUN: nothing will execute"

# ── Preflight ────────────────────────────────────────────────────────────────
say "preflight"
missing=0
n_pred=$(ls -d "$ROOT"/Results/*/by_layer_corpus_pred.csv.xz 2>/dev/null | wc -l)
note "cells with corpus predictions : $n_pred  (expect 58)"
(( n_pred < 58 )) && { note "Stage 1 is incomplete; Eq. 1 would be fitted on a subset."; missing=1; }
n_cv=$(ls -d "$ROOT"/Results/*/cv_preds 2>/dev/null | wc -l)
note "cells with cv_preds           : $n_cv"
# A stale summary is a silent hazard: cells already in it are anti-joined away
# and never refitted. The design tag now guards this, but say so out loud.
if [[ -f "$ROOT/Data/derived/brms_relfreq_rawscale.rds" ]]; then
  note "NOTE: Data/derived/brms_relfreq_rawscale.rds exists. Rows whose design"
  note "      column is not 'cross' are discarded, so this is safe, but if you"
  note "      want a clean refit move it aside first."
fi
(( missing )) && note "preflight raised concerns; continuing anyway"

# ── 3. Cross-model agreement ─────────────────────────────────────────────────
if has 3; then
  say "STAGE 3  cross-model agreement"
  if [[ -z "$DRY" ]]; then
    "$PYTHON" Scripts/supplementary/cross_model_agreement.py 2>&1 | tee -a "$RUN_LOG"
    [[ "${PIPESTATUS[0]}" -ne 0 ]] && { note "cross_model_agreement failed"; exit 1; }
  else
    note "\$ $PYTHON Scripts/supplementary/cross_model_agreement.py"
  fi
fi

# ── 4. brms, sharded ─────────────────────────────────────────────────────────
if has 4; then
  say "STAGE 4  brms Eq. 1, $WORKERS workers"
  pids=()
  for k in $(seq 0 $((WORKERS-1))); do
    out="$LOG_DIR/brms_shard$(printf '%02d' "$k").out"
    note "shard $k -> $out"
    if [[ -z "$DRY" ]]; then
      BRMS_SHARDS="$WORKERS" BRMS_SHARD="$k" SCALE="$SCALE" \
      BRMS_CHAINS="$CHAINS" BRMS_THREADS="$THREADS" \
        "$RSCRIPT" Scripts/analysis/corpus_relfreq_brms_all.R > "$out" 2>&1 &
      pids+=("$!")
    fi
  done

  rc=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      note "shard $i FAILED; see $LOG_DIR/brms_shard$(printf '%02d' "$i").out"
      rc=1
    else
      note "shard $i done"
    fi
  done
  # Merging a partial set would silently produce a summary missing cells, and
  # nothing downstream would notice.
  (( rc )) && { note "at least one shard failed; NOT merging"; exit 1; }

  say "merging shard summaries"
  if [[ -z "$DRY" ]]; then
    MERGE=1 SCALE="$SCALE" "$RSCRIPT" Scripts/analysis/corpus_relfreq_brms_all.R \
      2>&1 | tee -a "$RUN_LOG"
    [[ "${PIPESTATUS[0]}" -ne 0 ]] && { note "merge failed"; exit 1; }
  fi

  say "simple slopes"
  if [[ -z "$DRY" ]]; then
    "$RSCRIPT" Scripts/analysis/prepare_relfreq_slopes.R 2>&1 | tee -a "$RUN_LOG"
    [[ "${PIPESTATUS[0]}" -ne 0 ]] && { note "prepare_relfreq_slopes failed"; exit 1; }
  fi
fi

# ── What to push ─────────────────────────────────────────────────────────────
say "outputs the writeup reads"
for f in Data/derived/brms_relfreq_rawscale.rds \
         Data/derived/relfreq_slope_curve.csv \
         Results/cross_model_agreement_summary.csv; do
  if [[ -s "$ROOT/$f" ]]; then
    note "ok      $f  ($(du -h "$ROOT/$f" | cut -f1))"
  else
    note "MISSING $f"; missing=1
  fi
done

say "done"
note "log: $RUN_LOG"
if [[ "$missing" -eq 0 ]]; then
  note "to send these back:"
  note "  git add Data/derived/brms_relfreq_rawscale.rds \\"
  note "          Data/derived/relfreq_slope_curve.csv \\"
  note "          Results/cross_model_agreement_summary.csv"
  note "  git commit -m 'stats' && git push"
  note "Data/brms_models/ stays local: a few hundred MB, gitignored, regenerable."
  exit 0
fi
note "something is missing; do not render the paper from this"
exit 1
