#!/usr/bin/env bash
# =============================================================================
# run_paper_pipeline.sh
#
# Everything the paper reports, and nothing else.
#
# The stage list is derived from Scripts/PAPER_INVENTORY.md, which was built by
# tracing every file Writeup/writeup.qmd reads. Analyses that exist in the repo
# but appear nowhere in the document live in Scripts/archive/ and are not run.
#
# What changed from the previous pipeline
# ---------------------------------------
# The probe used to read its input and its target from the same forward pass,
# so it recovered the network's frozen arithmetic rather than anything learned.
# A randomly initialised BabyLM-125M scored 0.822 on word_novel/attn_zeroed,
# higher than any trained checkpoint. Two changes fix it, both validated on the
# 125M before this pipeline was written:
#
#   1. items must have real left context. 29% of the old novel set had the
#      binomial at sentence start, where every item shares one constant context
#      and the preference collapses to a single global linear functional of the
#      embeddings.
#   2. the target is measured in a SECOND occurrence of the binomial, which the
#      probe never sees.
#
#   word_novel, MLP, BabyLM-125M        step 0    final    shuffled control
#     old design                        0.8220   0.1430              0.0033
#     new design (filter + cross)       0.0011   0.1128              ~0.001
#
# Stages
# ------
#   0  harvest two carrier sentences per binomial from Wikipedia   ~40 min
#   1  probe cells: 16 final models + 42 checkpoints               GPU, long
#   2  ridge probe replication (Appendix D)                        GPU
#   3  cross-model agreement (Appendix H, Discussion)              CPU
#   4  brms Eq. 1 + simple slopes (Exp 1-3, Appendix F)            CPU, ~7 h
#
# Stages 1 and 2 are resume-safe: a cell whose outputs already exist is skipped,
# so an interrupted run continues where it stopped. Stage 4 caches per cell too.
#
# Usage
# -----
#   ./run_paper_pipeline.sh                      # everything
#   ./run_paper_pipeline.sh --stages 1 2         # just the GPU work
#   ./run_paper_pipeline.sh --skip-large         # omit the 7B/8B models
#   ./run_paper_pipeline.sh --gpu 1 --emb-dir /mnt/scratch/emb
#   ./run_paper_pipeline.sh --dry-run            # print the plan, run nothing
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
GPU=0
EMB_DIR=""
STAGES="0 1 2 3 4"
SKIP_LARGE=""
DRY=""
FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)        GPU="$2"; shift 2 ;;
    --emb-dir)    EMB_DIR="$2"; shift 2 ;;
    --stages)     shift; STAGES=""
                  while [[ $# -gt 0 && "$1" != --* ]]; do STAGES="$STAGES $1"; shift; done ;;
    --skip-large) SKIP_LARGE="--skip-large"; shift ;;
    --force)      FORCE="--force"; shift ;;
    --dry-run)    DRY="1"; shift ;;
    -h|--help)    sed -n '2,46p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOG_DIR="$ROOT/Results/logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

say()  { printf '\n\033[1m== %s\033[0m  (%s)\n' "$1" "$(date +%H:%M:%S)" | tee -a "$RUN_LOG"; }
note() { printf '   %s\n' "$1" | tee -a "$RUN_LOG"; }
has()  { [[ " $STAGES " == *" $1 "* ]]; }

# Run a command, tee its output, and stop the pipeline if it fails. Stages are
# ordered by dependency, so continuing past a failure would silently produce a
# paper built on partial data.
run() {
  note "\$ $*"
  [[ -n "$DRY" ]] && return 0
  if ! "$@" 2>&1 | tee -a "$RUN_LOG"; then
    echo "FAILED: $*" | tee -a "$RUN_LOG"
    exit 1
  fi
  return "${PIPESTATUS[0]}"
}

EMB_ARG=(); [[ -n "$EMB_DIR" ]] && EMB_ARG=(--embeddings-dir "$EMB_DIR")

# The canonical cell list: "<slug>\t<n_layers>", one reported cell per line.
# Everything downstream iterates this rather than globbing Results/, which
# still holds directories from the old item set and two models the paper
# does not report.
CELL_LIST() {
  "$PYTHON" Scripts/pipeline/run_paper_cells.py --list-slugs $SKIP_LARGE 2>/dev/null
}

say "binom-corpus-pls paper pipeline"
note "stages: $STAGES    gpu: $GPU    log: $RUN_LOG"
[[ -n "$DRY" ]] && note "DRY RUN: nothing will execute"

# ── Quarantine results from the previous item set ────────────────────────────
# Every Results/ and Data/supplementary/ directory on disk predates the carrier
# rebuild, so its numbers come from the design where the probe could recompute
# its own target. The resume logic in stages 1 and 2 skips any cell whose output
# already exists, which would silently mix the two designs inside one paper.
# Move them aside once, keeping them for reference rather than deleting.
STALE_DIR="$ROOT/Results/_pre_carriers"
if has 1 || has 2; then
  if [[ ! -d "$STALE_DIR" ]] && ls "$ROOT"/Results/*/by_layer_mlp.csv >/dev/null 2>&1; then
    say "quarantining results from the previous item set"
    note "Results/_pre_carriers/, Data/supplementary/_pre_carriers/, Data/brms_models_pre_carriers/"
    if [[ -z "$DRY" ]]; then
      mkdir -p "$STALE_DIR"
      for d in "$ROOT"/Results/*/; do
        b="$(basename "$d")"
        [[ "$b" == "logs" || "$b" == "_pre_carriers" ]] && continue
        [[ -f "$d/by_layer_mlp.csv" ]] && mv "$d" "$STALE_DIR/"
      done
      if [[ -d "$ROOT/Data/supplementary" ]]; then
        mkdir -p "$ROOT/Data/supplementary/_pre_carriers"
        for d in "$ROOT"/Data/supplementary/*/; do
          b="$(basename "$d")"
          [[ "$b" == "logs" || "$b" == "_pre_carriers" ]] && continue
          mv "$d" "$ROOT/Data/supplementary/_pre_carriers/"
        done
      fi
      # Embeddings from the previous pipeline are the same hazard in another
      # directory. run_model deletes them per cell, so they are normally
      # transient, but an interrupted run leaves them behind, and an npz
      # without preference_cross is useless under --target cross. OLMo-7B and
      # OLMo-2-7B each burned ~35 minutes of corpus extraction before dying on
      # that. run_bylayer now detects it and re-extracts, so this is belt and
      # braces; clearing up front just avoids reading large files to find out.
      for sub in embeddings novel_embeddings \
                 embeddings_attn_zeroed novel_embeddings_attn_zeroed; do
        if compgen -G "$ROOT/Data/$sub/*/" >/dev/null 2>&1; then
          note "clearing stale Data/$sub"
          rm -rf "${ROOT:?}/Data/$sub"
        fi
      done
      # The brms cache is keyed by model label and condition, not by item set,
      # so a stale fit would be reused verbatim. Same hazard, same fix.
      [[ -d "$ROOT/Data/brms_models" ]] && \
        mv "$ROOT/Data/brms_models" "$ROOT/Data/brms_models_pre_carriers"
      # Derived summaries the writeup reads directly.
      for f in brms_relfreq_rawscale.rds relfreq_slope_curve.csv; do
        [[ -f "$ROOT/Data/derived/$f" ]] && \
          mv "$ROOT/Data/derived/$f" "$ROOT/Data/derived/${f}.pre_carriers"
      done
    fi
  else
    note "no stale results to quarantine (or already done)"
  fi
fi

# ── 0. Carrier sentences ─────────────────────────────────────────────────────
# Two prefixed occurrences per binomial. Finding further occurrences of an
# already-validated pair needs a regex, not benepar, so this is a streaming
# scan: ~40 min for all of Wikipedia, yielding 28,164 attested and 218,725
# novel pairs.
if has 0; then
  say "STAGE 0  harvest carrier sentences"
  if [[ -s "$ROOT/Data/carrier_sentences.csv" && -z "$FORCE" ]]; then
    note "Data/carrier_sentences.csv exists; skipping (use --force to re-harvest)"
  else
    run "$PYTHON" Scripts/corpus/harvest_carrier_sentences.py
  fi
  run "$PYTHON" Scripts/corpus/assess_harvest.py
fi

# ── 1. Probe cells ───────────────────────────────────────────────────────────
# 58 cells. Each does extraction for both attention conditions, the MLP grid
# (2 conditions x 2 modes x 2 splits), the corpus-frequency predictions stage 4
# consumes, and the label-shuffle control, then frees its embeddings.
if has 1; then
  say "STAGE 1  probe cells (16 final models + 42 checkpoints)"
  run "$PYTHON" Scripts/pipeline/run_paper_cells.py \
      --gpu "$GPU" --target cross "${EMB_ARG[@]}" $SKIP_LARGE $FORCE
fi

# ── 2. Ridge probe, Appendix D ───────────────────────────────────────────────
# mean_pooled only: the appendix filters to it, and --skip-pls because no PLS
# output is read anywhere in the document.
if has 2; then
  say "STAGE 2  ridge probe replication (Appendix D)"
  # Iterate the canonical cell list, NOT Results/*/. Globbing would sweep in
  # pythia-2.8b and OLMo-2-1124-1B, which tbl-models does not report, and it
  # would also pick up directories left over from the previous item set.
  while IFS=$'\t' read -r slug nl; do
    [[ -n "$slug" ]] || continue
    out="$ROOT/Data/supplementary/$slug/linear_probe.csv"
    if [[ -s "$out" && -z "$FORCE" ]]; then
      note "ridge $slug: exists, skipping"
      continue
    fi
    run "$PYTHON" Scripts/supplementary/supplementary_probes.py \
        --model-slug "$slug" --num-layers "$nl" --target cross \
        --modes mean_pooled --skip-pls --gpu "$GPU" "${EMB_ARG[@]}"
  done < <(CELL_LIST)
fi

# ── 3. Cross-model agreement ─────────────────────────────────────────────────
# Appendix H and the Discussion's r = 0.45. Reads y_true from the saved CV
# predictions, so it needs no GPU and no re-extraction.
if has 3; then
  say "STAGE 3  cross-model agreement (Appendix H)"
  run "$PYTHON" Scripts/supplementary/cross_model_agreement.py
fi

# ── 4. Regressions ───────────────────────────────────────────────────────────
# SCALE=raw is what the paper reports. Resume-safe per cell; roughly 7 h cold.
if has 4; then
  say "STAGE 4  brms Eq. 1 and simple slopes"
  SCALE=raw run "$RSCRIPT" Scripts/analysis/corpus_relfreq_brms_all.R
  run "$RSCRIPT" Scripts/analysis/prepare_relfreq_slopes.R
fi

# ── Verification ─────────────────────────────────────────────────────────────
# Pythia-2.8B once produced six byte-identical "checkpoints" because the
# revision never reached extraction, and nothing caught it until the numbers
# were eyeballed months later. Two models trained to different points cannot
# give identical predictions, so this fingerprints every cell and fails if any
# two in a series match.
if has 1; then
  say "verifying checkpoints are distinct"
  if [[ -z "$DRY" ]]; then
    if ! "$PYTHON" Scripts/pipeline/verify_cells.py $SKIP_LARGE 2>&1 | tee -a "$RUN_LOG"; then
      note "verification failed; see above. Do not render the paper from this."
      exit 1
    fi
  fi
fi

# ── Completion check ─────────────────────────────────────────────────────────
say "verifying the outputs the writeup reads"
missing=0
for f in Data/derived/brms_relfreq_rawscale.rds \
         Data/derived/relfreq_slope_curve.csv \
         Results/cross_model_agreement_summary.csv \
         Results/novel_binomials_infinigram_dolma.csv \
         Results/corpus_binomials_infinigram_piletrain.csv; do
  if [[ -s "$ROOT/$f" ]]; then note "ok      $f"; else note "MISSING $f"; missing=1; fi
done
n_want=0; n_mlp=0; n_ctl=0; n_lin=0
while IFS=$'\t' read -r slug nl; do
  [[ -n "$slug" ]] || continue
  n_want=$((n_want + 1))
  [[ -s "$ROOT/Results/$slug/by_layer_mlp.csv" ]]            && n_mlp=$((n_mlp + 1))
  [[ -s "$ROOT/Results/$slug/by_layer_mlp_control.csv" ]]    && n_ctl=$((n_ctl + 1))
  [[ -s "$ROOT/Data/supplementary/$slug/linear_probe.csv" ]] && n_lin=$((n_lin + 1))
done < <(CELL_LIST)
note "by_layer_mlp.csv          : $n_mlp / $n_want reported cells"
note "by_layer_mlp_control.csv  : $n_ctl / $n_want"
note "linear_probe.csv          : $n_lin / $n_want"
[[ "$n_mlp" -lt "$n_want" ]] && missing=1

say "pipeline finished"
note "log: $RUN_LOG"
[[ "$missing" -eq 1 ]] && { note "some outputs are missing; the writeup will not render"; exit 1; }
exit 0
