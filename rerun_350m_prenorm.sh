#!/usr/bin/env bash
#
# TEMPORARY. Replaces the 350M results in this repo with results from the
# corrected pre-LN model. The old 350M was accidentally post-LN
# (do_layer_norm_before=False) while the 125M and 1.3B were pre-LN.
#
# WHY THE WRITEUP NEEDS NO EDITS: this repo separates the HuggingFace model id
# ("znhoughton/opt-babylm-350m-20eps-seed964", with a slash) from the output
# slug ("znhoughton_opt-babylm-350m-20eps-seed964", with an underscore).
# Writeup/writeup.qmd, the brms script and every supplementary script reference
# only the SLUG. So we change only the id: results are rewritten into the same
# directories and nothing downstream has to know. Verified: writeup.qmd
# contains 0 id-form and 2 slug-form occurrences.
#
# THIS REPLACES, IT DOES NOT ACCUMULATE. Same slug means same output paths, so
# --force is mandatory: without it the drivers see existing layer files and
# skip silently, and you would get the old numbers back with a clean exit.
#
# WHERE TO RUN: a GPU box. 48,966 corpus binomials x 7 model variants (final +
# 6 log-spaced step checkpoints) x 2 conditions. Hours on a 3060 Ti.
# The brms stage is CPU-only and can run anywhere.
#
# Usage:  bash rerun_350m_prenorm.sh
#         DRY_RUN=1 bash rerun_350m_prenorm.sh
#         SKIP_BRMS=1 bash rerun_350m_prenorm.sh     # GPU stages only
#         STAGE=brms bash rerun_350m_prenorm.sh      # brms + render only
set -uo pipefail

MODEL_ID="znhoughton/opt-babylm-350m-20eps-seed964"   # unchanged: the corrected model was promoted into this name
SLUG="znhoughton_opt-babylm-350m-20eps-seed964"      # must NOT change
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BRMS="${SKIP_BRMS:-0}"
STAGE="${STAGE:-all}"


# ── Interpreter ──────────────────────────────────────────────────────────────
# Override with PY=/path/to/python if the default is not the env you want.
PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in python3 python; do
        command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
    done
fi
[ -n "$PY" ] || { echo "FATAL: no python found. Set PY=/path/to/python." >&2; exit 1; }

"$PY" - <<'PROBE' || { echo "FATAL: $PY cannot import torch. Set PY= to the right env." >&2; exit 1; }
import sys, torch
print(f"interpreter: {sys.executable}")
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
if not torch.cuda.is_available():
    print("WARNING: no CUDA. GPU stages will run on CPU and take far longer.")
PROBE

# Checked at the brms step, not here: the GPU box need not have R installed.
RSCRIPT="${RSCRIPT:-Rscript}"

# ── Parallelism ──────────────────────────────────────────────────────────────
# BRMS_CORES is the total core budget for Stan sampling. Capped at the machine's
# real core count: oversubscribing Stan threads makes sampling slower, not faster.
DETECTED_CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
BRMS_CORES="${BRMS_CORES:-24}"
if [ "$BRMS_CORES" -gt "$DETECTED_CORES" ]; then
    echo "  BRMS_CORES=$BRMS_CORES exceeds $DETECTED_CORES detected cores; capping."
    BRMS_CORES="$DETECTED_CORES"
fi
export BRMS_CHAINS="${BRMS_CHAINS:-4}"
export BRMS_THREADS="${BRMS_THREADS:-$(( BRMS_CORES / BRMS_CHAINS ))}"
[ "$BRMS_THREADS" -lt 1 ] && BRMS_THREADS=1 && export BRMS_THREADS
echo "  brms: ${BRMS_CHAINS} chains x ${BRMS_THREADS} threads = $(( BRMS_CHAINS * BRMS_THREADS )) cores"


say () { echo; echo "=== $* ==="; }

# run(): a failing step must stop the run. Without this the script would sail on
# to re-render the paper after a failed extraction and produce confidently wrong
# output. (set -e is deliberately not used: the upstream runners here return
# non-zero for benign reasons, so failures are checked explicitly instead.)
run () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo >&2
        echo "FAILED (exit $rc): $*" >&2
        echo "Stopping before anything downstream consumes a half-finished result." >&2
        exit $rc
    fi
}

# run_soft(): for steps whose failure genuinely does not invalidate the run
# (cache purge, optional backup).
run_soft () {
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; return 0; fi
    "$@" || echo "  WARNING: non-fatal step failed: $*" >&2
}

# Verification below compares file mtimes against this, not a fixed window: a
# long run would otherwise report its own early outputs as stale.
RUN_STARTED_AT=$(date +%s)

[ -f Scripts/pipeline/run_bylayer.py ] || { echo "Run me from the repo root."; exit 1; }

say "0. Preconditions"
echo "  NOTE: this repo has no git remote. Results/ is your only copy."
git status --porcelain 2>/dev/null | grep -vE "^\?\?" | head -3

say "0b. Disk check"
# run_bylayer.py extracts BOTH the corpus split (48,965 pairs) and the novel
# split (340,042) across every layer and both conditions, and unlike
# run_babylm_checkpoints.py it does NOT delete the embeddings afterwards --
# it only clears the shard staging dir. Expect several hundred GB to persist,
# plus a transient checkpoint's worth on top. Override the threshold with
# MIN_FREE_GB= if you know better.
MIN_FREE_GB="${MIN_FREE_GB:-600}"
free_gb=$(df -Pk . 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
if [ -n "$free_gb" ]; then
    echo "  free space here: ${free_gb} GB (want >= ${MIN_FREE_GB} GB)"
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "  WARNING: this may not be enough. Extraction writes per-layer .npz for" >&2
        echo "           both splits x both conditions; a mid-run ENOSPC wastes the lot." >&2
        echo "           Set --embeddings-dir to a bigger volume, or MIN_FREE_GB= to override." >&2
    fi
else
    echo "  could not determine free space"
fi

if [ "$STAGE" = "all" ]; then
say "1. Confirm the replacement model is pre-LN"
if [ "$DRY_RUN" != "1" ]; then
"$PY" - "$MODEL_ID" <<'PY'
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
print(f"  do_layer_norm_before={c.do_layer_norm_before} hidden={c.hidden_size} layers={c.num_hidden_layers}")
if c.do_layer_norm_before is not True:
    print("  FATAL: not the corrected model."); sys.exit(1)
print("  OK: pre-LN.")
PY
[ $? -ne 0 ] && exit 1
fi

say "2. Purge the stale HF cache"
# The corrected model now lives at the ORIGINAL name, so no code changes are
# needed. But the cache is keyed by repo name, and it still holds the old
# post-LN weights under that name -- a re-run would silently use them.
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--znhoughton--opt-babylm-350m-20eps-seed964"
if [ -d "$CACHE_DIR" ]; then
    echo "  removing $CACHE_DIR"
    run_soft rm -rf "$CACHE_DIR"
else
    echo "  no cached copy at $CACHE_DIR"
fi

say "3. By-layer extraction + MLP probes for the 350M (--force: replace in place)"
run "$PY" Scripts/pipeline/run_bylayer.py --models 350m --gpu "$GPU" --force

say "4. The six log-spaced step checkpoints"
run "$PY" Scripts/pipeline/run_babylm_checkpoints.py --models 350m --gpu "$GPU" --force
fi

if [ "$SKIP_BRMS" != "1" ]; then
say "5. Re-fit the relative-frequency brms models (CPU, slow)"
command -v "$RSCRIPT" >/dev/null 2>&1 || [ -x "$RSCRIPT" ] || {
    echo "FATAL: Rscript not found. Install R here, set RSCRIPT=/path/to/Rscript," >&2
    echo "       or run the GPU stages only with SKIP_BRMS=1 and do brms elsewhere." >&2
    exit 1; }
# load_cell() returns NULL when by_layer_corpus_pred.csv.xz is missing, and the
# loop then logs "skip" and moves on -- brms would quietly produce no 350M rows
# at all rather than stale ones. Check the inputs exist before spending hours.
missing=0
for slug_dir in Results/${SLUG} Results/${SLUG}_step48 Results/${SLUG}_step96                 Results/${SLUG}_step288 Results/${SLUG}_step768                 Results/${SLUG}_step1824 Results/${SLUG}_step4560; do
    if [ ! -f "$slug_dir/by_layer_corpus_pred.csv.xz" ] && [ ! -f "$slug_dir/by_layer_corpus_pred.csv.gz" ]; then
        echo "  MISSING input: $slug_dir/by_layer_corpus_pred.csv.{xz,gz}" >&2
        missing=$((missing+1))
    fi
done
if [ "$missing" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
    echo "FATAL: $missing of 7 checkpoints lack by_layer_corpus_pred; brms would skip them" >&2
    echo "       silently. Re-run stages 3-4 first (they produce these)." >&2
    exit 1
fi

# corpus_relfreq_brms_all.R passes file=/file_refit="on_change" to brm(). That
# should refit when the data changes, but the whole point of this run is that
# the data changed, so do not depend on brms's change detection. Move the 350M
# fits aside; Data/brms_models is gitignored, so moved rather than deleted.
BRMS_ARCHIVE="Data/brms_models_pre_prenorm_$(date +%Y%m%d_%H%M%S)"
n_fits=$(ls Data/brms_models/ 2>/dev/null | grep -c "BabyLM_350M")
if [ "$n_fits" -gt 0 ]; then
    echo "  moving $n_fits cached 350M fits -> $BRMS_ARCHIVE"
    run mkdir -p "$BRMS_ARCHIVE"
    for f in Data/brms_models/*BabyLM_350M*; do
        [ -e "$f" ] || continue
        run mv "$f" "$BRMS_ARCHIVE/"
    done
else
    echo "  no cached 350M brms fits found"
fi

echo "  NOTE: bayes_R2 is still at ndraws=500 in this repo; raise to 8000 before submission."
run "$RSCRIPT" Scripts/analysis/corpus_relfreq_brms_all.R
fi

say "6. Re-render the writeup (no .qmd edits were needed)"
run quarto render Writeup/writeup.qmd

say "7. Verify the 350M results were actually rewritten"
if [ "$DRY_RUN" != "1" ]; then
    for d in Results/${SLUG} Results/${SLUG}_step48 Results/${SLUG}_step4560; do
        if [ -d "$d" ]; then
            newest=$(find "$d" -name "*.csv" -newermt "@$RUN_STARTED_AT" 2>/dev/null | wc -l)
            echo "  $d: $newest csv(s) written by this run"
        else
            echo "  $d: MISSING"
        fi
    done
fi

say "Done"
echo "No code edits were made: the corrected model carries the original name."
