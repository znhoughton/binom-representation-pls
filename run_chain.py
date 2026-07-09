"""
Unified by-layer analysis pipeline for all BabyLM models.

For each model, in sequence:
  1. Extract embeddings (one condition at a time to bound peak disk usage)
  2. Run MLP CV probes + save per-pair predictions
  3. Run corpus-freq transfer prediction (novel → corpus)
  4. Run control trials (shuffled-label selectivity check)
  5. Delete novel embeddings (largest files, ~100-175 GB per model/condition)
  6. Delete corpus embeddings
  7. After both conditions: run R corpus-frequency regression

Run with the correct Python environment (CUDA PyTorch required):
    python run_chain.py                          # all three models, GPU 0, default batch
    python run_chain.py --models 1.3b --gpu 0   # single model
    python run_chain.py --batch 32768            # explicit batch size
    python run_chain.py --skip-controls          # skip control runs (saves ~40%% compute)

The MLP batch size defaults are tuned for an 80 GB VRAM GPU. Reduce --batch if you
hit VRAM OOM (the GPU-cache path in by_layer_mlp.py falls back automatically, but
a smaller batch avoids allocation pressure).

Resume behaviour: by_layer_mlp.py skips rows already present in the summary CSV
AND whose per-pair prediction NPZ already exists. A clean rerun from scratch simply
finds nothing to skip.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Use cached models only — avoids silent hangs from HF Hub network checks
# on rate-limited / unauthenticated connections. Models must be pre-downloaded.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PYTHON  = sys.executable
RSCRIPT = shutil.which("Rscript")  # None if R is not installed
BASE    = Path(__file__).resolve().parent
SCRIPTS = BASE / "Scripts"

MODELS = [
    {
        "flag":          "125m",
        "id":            "znhoughton/opt-babylm-125m-20eps-seed964",
        "slug":          "znhoughton_opt-babylm-125m-20eps-seed964",
        "n_layers":      12,
        "mlp_batch":     32768,
    },
    {
        "flag":          "350m",
        "id":            "znhoughton/opt-babylm-350m-20eps-seed964",
        "slug":          "znhoughton_opt-babylm-350m-20eps-seed964",
        "n_layers":      24,
        "mlp_batch":     32768,
    },
    {
        "flag":          "1.3b",
        "id":            "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "slug":          "znhoughton_opt-babylm-1_3b-20eps-seed964",
        "n_layers":      24,
        "mlp_batch":     32768,
    },
]

CONDITIONS = [
    {
        "name":       "default",
        "novel_dir":  "novel_embeddings",
        "corpus_dir": "embeddings",
    },
    {
        "name":       "attn_zeroed",
        "novel_dir":  "novel_embeddings_attn_zeroed",
        "corpus_dir": "embeddings_attn_zeroed",
    },
]


def banner(msg):
    sep = "=" * 64
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run(cmd, label="", abort_on_fail=True):
    banner(label or " ".join(str(c) for c in cmd[:5]))
    t0 = time.perf_counter()
    rc = subprocess.call([str(c) for c in cmd])
    elapsed = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {status} in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)
    if rc != 0 and abort_on_fail:
        print("  *** ABORTING CHAIN ***", flush=True)
        sys.exit(rc)


def delete_dir(path, label=""):
    path = Path(path)
    if path.exists():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        banner(f"DELETE  {label or path.name}  ({size_gb:.1f} GB)")
        shutil.rmtree(path)
        print("  Deleted.", flush=True)
    else:
        banner(f"DELETE  {label or path.name}  (already gone)")
        print("  Skipping.", flush=True)


def do_extract(model, cond, gpu, emb_dir):
    run(
        [
            PYTHON, SCRIPTS / "run_by_layer_pipeline.py",
            "--models",         model["flag"],
            "--conditions",     cond["name"],
            "--gpu",            str(gpu),
            "--skip-mlp",
            "--embeddings-dir", str(emb_dir),
        ],
        label=f"EXTRACT  {model['flag']} / {cond['name']}",
    )


def do_mlp(model, cond, gpu, batch, emb_dir):
    run(
        [
            PYTHON, SCRIPTS / "by_layer_mlp.py",
            "--model-slug",     model["slug"],
            "--num-layers",     str(model["n_layers"]),
            "--conditions",     cond["name"],
            "--modes", "mean_pooled", "individual", "words_only",
            "--splits",         "pair_novel", "word_novel",
            "--gpu",            str(gpu),
            "--batch",          str(batch),
            "--embeddings-dir", str(emb_dir),
        ],
        label=f"MLP CV  {model['flag']} / {cond['name']}",
    )


def do_corpus_freq(model, cond, gpu, emb_dir):
    run(
        [
            PYTHON, SCRIPTS / "by_layer_mlp.py",
            "--model-slug",     model["slug"],
            "--num-layers",     str(model["n_layers"]),
            "--conditions",     cond["name"],
            "--modes", "mean_pooled", "individual", "words_only",
            "--corpus-freq",
            "--gpu",            str(gpu),
            "--embeddings-dir", str(emb_dir),
        ],
        label=f"CORPUS-FREQ PRED  {model['flag']} / {cond['name']}",
    )


def do_controls(model, cond, gpu, batch, emb_dir):
    run(
        [
            PYTHON, SCRIPTS / "by_layer_mlp.py",
            "--model-slug",     model["slug"],
            "--num-layers",     str(model["n_layers"]),
            "--conditions",     cond["name"],
            "--modes", "mean_pooled", "individual", "words_only",
            "--splits",         "pair_novel", "word_novel",
            "--gpu",            str(gpu),
            "--batch",          str(batch),
            "--control",
            "--embeddings-dir", str(emb_dir),
        ],
        label=f"CONTROLS  {model['flag']} / {cond['name']}",
    )


def do_r_regression(model):
    if RSCRIPT is None:
        banner(f"R REGRESSION  {model['flag']}  (SKIPPED — Rscript not found)")
        print("  Run Scripts/corpus_freq_regression.R and Scripts/plot_by_layer_results.R locally.", flush=True)
        return
    run(
        [RSCRIPT, SCRIPTS / "corpus_freq_regression.R", model["slug"]],
        label=f"R REGRESSION  {model['flag']}",
    )


def main():
    p = argparse.ArgumentParser(
        description="Full by-layer pipeline for all BabyLM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", default=None,
                   choices=["125m", "350m", "1.3b"],
                   help="Models to run (default: all three)")
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--batch", type=int, default=None,
                   help="MLP training batch size (default: per-model value; 32768 for 80 GB VRAM)")
    p.add_argument("--skip-controls", action="store_true", dest="skip_controls",
                   help="Skip control-trial runs")
    p.add_argument("--skip-corpus-freq", action="store_true", dest="skip_corpus_freq",
                   help="Skip novel→corpus transfer prediction and R regression")
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Root directory for embedding subdirs (default: <project>/Data)")
    args = p.parse_args()

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    model_list = [m for m in MODELS
                  if args.models is None or m["flag"] in args.models]

    t_chain = time.perf_counter()

    for model in model_list:
        banner(f"MODEL: {model['flag'].upper()}  ({model['id']})")
        batch = args.batch or model["mlp_batch"]

        for cond in CONDITIONS:
            # ── 1. Extract novel + corpus embeddings ────────────────────────
            do_extract(model, cond, args.gpu, emb_dir)

            # ── 2. MLP CV probes (saves summary CSV + per-pair pred NPZs) ──
            do_mlp(model, cond, args.gpu, batch, emb_dir)

            # ── 3. Novel → corpus transfer prediction ───────────────────────
            if not args.skip_corpus_freq:
                do_corpus_freq(model, cond, args.gpu, emb_dir)

            # ── 4. Control trials ───────────────────────────────────────────
            if not args.skip_controls:
                do_controls(model, cond, args.gpu, batch, emb_dir)

            # ── 5. Delete embeddings to free disk ───────────────────────────
            delete_dir(emb_dir / cond["novel_dir"]  / model["slug"],
                       label=f"{cond['novel_dir']}/{model['slug']}")
            delete_dir(emb_dir / cond["corpus_dir"] / model["slug"],
                       label=f"{cond['corpus_dir']}/{model['slug']}")

        # ── 6. R regression (both conditions now in CSV) ───────────────────
        if not args.skip_corpus_freq:
            do_r_regression(model)

        elapsed = time.perf_counter() - t_chain
        banner(f"MODEL {model['flag'].upper()} COMPLETE  "
               f"(chain running {elapsed / 3600:.1f}h total)")

    banner("PIPELINE COMPLETE")
    print(f"  All models done in {(time.perf_counter() - t_chain) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
