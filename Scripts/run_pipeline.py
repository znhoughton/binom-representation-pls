"""
Main analysis pipeline.

Runs all computations needed to produce the paper results, skipping steps
whose output already exists. R scripts are run separately afterward.

Phases:
  1  OPT-BabyLM full by-layer (125M / 350M / 1.3B)
       - All layers, both conditions (default + attn_zeroed)
       - MLP CV, corpus-freq, controls
       - Compress by_layer_corpus_pred.csv → .xz
       - Delete embeddings

  2  Training dynamics (OPT-BabyLM, log-spaced checkpoints)
       - Final layer only at 7 log-spaced training fractions per model size
       - Delegates entirely to run_checkpoint_pipeline.py

  3  New model families (Pythia / GPT-2 / OLMo / Llama)
       - Final layer only, final checkpoint
       - Delegates entirely to run_new_models.py

Usage:
    python Scripts/run_pipeline.py --embeddings-dir /path/to/embeddings --gpu 0
    python Scripts/run_pipeline.py --phases 1 2 --embeddings-dir /path/to/embeddings
    python Scripts/run_pipeline.py --phases 3 --models pythia gpt2
    python Scripts/run_pipeline.py --phases 3 --skip-large  # skip 7B/8B models
    python Scripts/run_pipeline.py --force                  # re-run all phases
"""

import argparse
import csv
import lzma
import shutil
import subprocess
import sys
import time
from pathlib import Path

PYTHON  = sys.executable
BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "Scripts"

# ── OPT-BabyLM model specs (Phase 1) ──────────────────────────────────────────
OPT_MODELS = [
    {
        "flag": "125m",
        "id":   "znhoughton/opt-babylm-125m-20eps-seed964",
        "slug": "znhoughton_opt-babylm-125m-20eps-seed964",
        "n_layers":  12,
        "mlp_batch": 262144,
    },
    {
        "flag": "350m",
        "id":   "znhoughton/opt-babylm-350m-20eps-seed964",
        "slug": "znhoughton_opt-babylm-350m-20eps-seed964",
        "n_layers":  24,
        "mlp_batch": 262144,
    },
    {
        "flag": "1.3b",
        "id":   "znhoughton/opt-babylm-1.3b-20eps-seed964",
        "slug": "znhoughton_opt-babylm-1_3b-20eps-seed964",
        "n_layers":  24,
        "mlp_batch": 262144,
    },
]

CONDITIONS = [
    {"name": "default",
     "corpus_dir": "embeddings",             "novel_dir": "novel_embeddings"},
    {"name": "attn_zeroed",
     "corpus_dir": "embeddings_attn_zeroed", "novel_dir": "novel_embeddings_attn_zeroed"},
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def banner(msg):
    sep = "=" * 66
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run(cmd, label="", abort_on_fail=True):
    banner(label or " ".join(str(c) for c in cmd[:6]))
    t0 = time.perf_counter()
    rc = subprocess.call([str(c) for c in cmd])
    elapsed = time.perf_counter() - t0
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"  {status} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if rc != 0 and abort_on_fail:
        sys.exit(rc)


def run_parallel(cmds_labels: list, abort_on_fail=True):
    """Launch all (cmd, label) pairs simultaneously and wait for all to finish."""
    procs = []
    for cmd, label in cmds_labels:
        banner(label)
        procs.append(subprocess.Popen([str(c) for c in cmd]))
    t0 = time.perf_counter()
    failed = False
    for p in procs:
        if p.wait() != 0:
            failed = True
    elapsed = time.perf_counter() - t0
    print(f"  {'OK' if not failed else 'FAILED'} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    if failed and abort_on_fail:
        sys.exit(1)


def delete_dir(path, label=""):
    path = Path(path)
    if path.exists():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        banner(f"DELETE  {label or path.name}  ({size_gb:.1f} GB)")
        shutil.rmtree(path)
        print("  Deleted.", flush=True)


def _pred_compressed(slug: str) -> bool:
    r = BASE / "Results" / slug
    return (r / "by_layer_corpus_pred.csv.gz").exists() or \
           (r / "by_layer_corpus_pred.csv.xz").exists()


def compress_corpus_pred(slug: str):
    src = BASE / "Results" / slug / "by_layer_corpus_pred.csv"
    dst = BASE / "Results" / slug / "by_layer_corpus_pred.csv.xz"
    if not src.exists():
        return
    if _pred_compressed(slug):
        # Already compressed (.gz or .xz); just remove the orphaned uncompressed copy
        src.unlink(missing_ok=True)
        return
    banner(f"COMPRESS  {slug}/by_layer_corpus_pred.csv")
    src_mb = src.stat().st_size / 1e6
    with open(src, "rb") as f_in, lzma.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    dst_mb = dst.stat().st_size / 1e6
    print(f"  {src_mb:.0f} MB → {dst_mb:.0f} MB ({dst_mb/src_mb*100:.0f}%)", flush=True)
    src.unlink()


def mlp_row_count(slug: str, cond_name: str) -> int:
    csv_path = BASE / "Results" / slug / "by_layer_mlp.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path, newline="") as f:
        return sum(1 for r in csv.DictReader(f)
                   if r["condition"] == cond_name
                   and r["split"] in ("pair_novel", "word_novel")
                   and r["mode"] in ("mean_pooled", "individual", "words_only"))


def corpus_freq_complete(slug: str, cond_name: str) -> bool:
    """True if corpus-freq predictions already written for this slug+condition."""
    src = BASE / "Results" / slug / "by_layer_corpus_pred.csv"
    gz  = BASE / "Results" / slug / "by_layer_corpus_pred.csv.gz"
    return gz.exists() or src.exists()


def phase1_complete(model: dict) -> bool:
    slug = model["slug"]
    expected_per_cond = (model["n_layers"] + 1) * 3 * 2  # layers × modes × splits
    for cond in CONDITIONS:
        if mlp_row_count(slug, cond["name"]) < expected_per_cond:
            return False
    return _pred_compressed(slug)


# ── Phase 1: OPT-BabyLM full by-layer ─────────────────────────────────────────

def run_phase1(models, gpu: int, emb_dir: Path, skip_controls: bool, force: bool = False):
    banner("PHASE 1: OPT-BabyLM full by-layer")

    for model in models:
        slug = model["slug"]

        if not force and phase1_complete(model):
            banner(f"PHASE 1  {model['flag']}  ALREADY COMPLETE — skipping")
            continue

        banner(f"PHASE 1  model={model['flag']}  slug={slug}")

        # ── Extract all layers + MLP CV ────────────────────────────────────
        # run_by_layer_pipeline.py handles its own skip logic internally
        run(
            [PYTHON, SCRIPTS / "run_by_layer_pipeline.py",
             "--models",    model["flag"],
             "--gpu",       str(gpu),
             "--embeddings-dir", str(emb_dir),
             "--mlp-batch", str(model["mlp_batch"])]
            + (["--force"] if force else []),
            label=f"PHASE 1  EXTRACT+MLP  {model['flag']}",
        )

        force_flag = ["--force"] if force else []

        def mlp_cmd(cond_name, extra_flags=()):
            return [PYTHON, SCRIPTS / "by_layer_mlp.py",
                    "--model-slug", slug,
                    "--num-layers", str(model["n_layers"]),
                    "--conditions", cond_name,
                    "--modes",      "mean_pooled", "individual", "words_only",
                    "--gpu",        str(gpu),
                    "--batch",      str(model["mlp_batch"]),
                    "--embeddings-dir", str(emb_dir),
                    *force_flag, *extra_flags]

        # ── Corpus-freq — both conditions in parallel ───────────────────────
        run_parallel([(mlp_cmd(c["name"], ["--corpus-freq"]),
                       f"PHASE 1  CORPUS-FREQ  {model['flag']} / {c['name']}")
                      for c in CONDITIONS])

        # ── Controls — both conditions in parallel ──────────────────────────
        if not skip_controls:
            run_parallel([(mlp_cmd(c["name"], ["--splits", "pair_novel", "word_novel", "--control"]),
                           f"PHASE 1  CONTROLS  {model['flag']} / {c['name']}")
                          for c in CONDITIONS])

        # ── Delete both conditions ───────────────────────────────────────────
        for cond in CONDITIONS:
            delete_dir(emb_dir / cond["novel_dir"]  / slug)
            delete_dir(emb_dir / cond["corpus_dir"] / slug)

        compress_corpus_pred(slug)
        banner(f"PHASE 1  {model['flag']}  DONE")


# ── Phase 2: Training dynamics ─────────────────────────────────────────────────

def run_phase2(models, gpu: int, emb_dir: Path, skip_controls: bool, force: bool = False):
    banner("PHASE 2: Training dynamics (OPT-BabyLM checkpoints)")
    flags = [m["flag"] for m in models]
    run(
        [PYTHON, SCRIPTS / "run_checkpoint_pipeline.py",
         "--models",       *flags,
         "--gpu",          str(gpu),
         "--embeddings-dir", str(emb_dir)]
        + (["--skip-controls"] if skip_controls else [])
        + (["--force"]         if force         else []),
        label="PHASE 2  run_checkpoint_pipeline.py",
    )


# ── Phase 3: New model families ────────────────────────────────────────────────

def run_phase3(model_groups, gpu: int, emb_dir: Path,
               skip_large: bool, skip_controls: bool, force: bool = False):
    banner("PHASE 3: New model families (Pythia / GPT-2 / OLMo / Llama)")
    cmd = [
        PYTHON, SCRIPTS / "run_new_models.py",
        "--gpu",           str(gpu),
        "--embeddings-dir", str(emb_dir),
    ]
    if model_groups:
        cmd += ["--models"] + model_groups
    if skip_large:
        cmd.append("--skip-large")
    if skip_controls:
        cmd.append("--skip-controls")
    if force:
        cmd.append("--force")
    run(cmd, label="PHASE 3  run_new_models.py")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Grand unified replication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phases", nargs="+", type=int, default=[1, 2, 3],
                   choices=[1, 2, 3],
                   help="Which phases to run (default: 1 2 3)")
    p.add_argument("--gpu",            type=int, default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Root directory for embedding subdirs "
                        "(e.g. /dpluth-data; default: <project>/Data)")
    # Phase 1 / 2 model filter
    p.add_argument("--opt-models", nargs="+", default=None, dest="opt_models",
                   choices=["125m", "350m", "1.3b"],
                   help="OPT-BabyLM model sizes to include in phases 1 and 2 (default: all)")
    # Phase 3 options
    p.add_argument("--models", nargs="+", default=None,
                   choices=["pythia", "gpt2", "olmo", "llama"],
                   help="New-model groups to include in phase 3 (default: all)")
    p.add_argument("--skip-large",    action="store_true", dest="skip_large",
                   help="Phase 3: skip models that need a large GPU (OLMo-7B, Llama-8B)")
    p.add_argument("--skip-controls", action="store_true", dest="skip_controls",
                   help="Skip shuffled-label control runs in all phases")
    p.add_argument("--force", action="store_true",
                   help="Re-run all computations even if output already exists")
    args = p.parse_args()

    emb_dir    = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    opt_models = [m for m in OPT_MODELS
                  if args.opt_models is None or m["flag"] in args.opt_models]

    print("Analysis pipeline", flush=True)
    print(f"  Phases:        {args.phases}", flush=True)
    print(f"  GPU:           {args.gpu}", flush=True)
    print(f"  Embeddings:    {emb_dir}", flush=True)
    print(f"  Skip controls: {args.skip_controls}", flush=True)
    print(flush=True)

    t0 = time.perf_counter()

    if 1 in args.phases:
        run_phase1(opt_models, args.gpu, emb_dir, args.skip_controls, args.force)

    if 2 in args.phases:
        run_phase2(opt_models, args.gpu, emb_dir, args.skip_controls, args.force)

    if 3 in args.phases:
        run_phase3(args.models, args.gpu, emb_dir, args.skip_large, args.skip_controls, args.force)

    elapsed = time.perf_counter() - t0
    banner(f"REPLICATION COMPLETE  ({elapsed/3600:.1f}h total)")


if __name__ == "__main__":
    main()
