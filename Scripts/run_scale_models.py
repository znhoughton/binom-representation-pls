"""
Analysis pipeline for additional model families (Pythia, GPT-2, OLMo, Llama 3).

Three modes:
  (default)       Final-layer probe at each model's final checkpoint.
  --checkpoints   Final-layer probe at BabyLM-aligned Pythia checkpoint steps.
  --by-layer      Full by-layer sweep at the final checkpoint (Pythia).

For each model in default/checkpoint mode:
  1. Extract last-layer embeddings (novel + corpus, both conditions)
  2. Run MLP CV probes (pair_novel + word_novel splits)
  3. Run novel→corpus transfer prediction (corpus-freq analysis)
  4. Run control probes (final models only, not checkpoints)
  5. Delete embeddings; compress by_layer_corpus_pred.csv → .xz

Large models (OLMo-7B, Llama-3-8B) require an A100 or similar. Mark them with
"large_gpu": True and pass --skip-large if running on a smaller machine.

Usage:
    python Scripts/run_scale_models.py [--models pythia gpt2 olmo llama]
                                      [--gpu 0]
                                      [--embeddings-dir /path]
                                      [--skip-large]
                                      [--skip-controls]
                                      [--checkpoints]   # Pythia checkpoint sweep
                                      [--by-layer]      # full layer sweep (Pythia)

Adding a model: append an entry to MODELS below. The slug is derived
automatically from the HF model ID (slashes → underscores).
"""

import argparse
import lzma
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PYTHON  = sys.executable
BASE    = Path(__file__).resolve().parent.parent
SCRIPTS = BASE / "Scripts"

# ── Model registry ─────────────────────────────────────────────────────────────
# n_layers is only used for by_layer_mlp.py's --num-layers arg; since we extract
# layer_last.npz the probe only ever sees that one layer regardless.

MODELS = [
    # ── Pythia (EleutherAI, trained on The Pile) ───────────────────────────
    {"group": "pythia", "id": "EleutherAI/pythia-160m",  "n_layers": 12,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "pythia", "id": "EleutherAI/pythia-410m",  "n_layers": 24,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "pythia", "id": "EleutherAI/pythia-1b",    "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "pythia", "id": "EleutherAI/pythia-2.8b",  "n_layers": 32,  "large_gpu": False, "shard_pairs": 200000, "batch_size": 6144},
    # ── GPT-2 (OpenAI) ─────────────────────────────────────────────────────
    {"group": "gpt2",   "id": "gpt2",                    "n_layers": 12,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "gpt2",   "id": "gpt2-medium",             "n_layers": 24,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "gpt2",   "id": "gpt2-large",              "n_layers": 36,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "gpt2",   "id": "gpt2-xl",                 "n_layers": 48,  "large_gpu": False, "shard_pairs": 200000, "batch_size": 6144},
    # ── OLMo (AI2, open data) ──────────────────────────────────────────────
    {"group": "olmo",   "id": "allenai/OLMo-1B-hf",     "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 512},
    {"group": "olmo",   "id": "allenai/OLMo-2-1124-1B",  "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 512},
    {"group": "olmo",   "id": "allenai/OLMo-7B-hf",     "n_layers": 32,  "large_gpu": True,  "shard_pairs": 100000, "batch_size": 4096},
    {"group": "olmo",   "id": "allenai/OLMo-2-1124-7B",  "n_layers": 32,  "large_gpu": True,  "shard_pairs": 100000, "batch_size": 4096},
    # ── Llama 3 (Meta, gated — requires HF token) ──────────────────────────
    {"group": "llama",  "id": "meta-llama/Llama-3.2-1B", "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "llama",  "id": "meta-llama/Meta-Llama-3-8B", "n_layers": 32, "large_gpu": True, "shard_pairs": 100000, "batch_size": 4096},
]

CONDITIONS = [
    {"name": "default",     "context": "binomial",    "extract": "word",
     "corpus_dir": "embeddings",             "novel_dir": "novel_embeddings"},
    {"name": "attn_zeroed", "context": "attn_zeroed", "extract": "word",
     "corpus_dir": "embeddings_attn_zeroed", "novel_dir": "novel_embeddings_attn_zeroed"},
]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def banner(msg):
    sep = "=" * 64
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
    return rc == 0


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


def calibrate_batch_size(model_id: str, max_bs: int, gpu: int,
                         revision: str = None) -> int:
    """
    Run extract_embeddings.py --calibrate as a subprocess to probe GPU memory
    and determine a safe starting batch_size for this model.  The subprocess
    exits after the probe, so its CUDA allocator state doesn't carry over into
    the real extraction.
    """
    cmd = [PYTHON, SCRIPTS / "extract_embeddings.py",
           "--model",      model_id,
           "--data",       "corpus",
           "--batch-size", str(max_bs),
           "--gpu",        str(gpu),
           "--calibrate"]
    if revision is not None:
        cmd += ["--revision", revision]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)
    for line in result.stdout.splitlines():
        if line.startswith("SAFE_BATCH_SIZE="):
            bs = int(line.split("=")[1])
            print(f"  → Calibrated batch_size={bs} for {model_id}", flush=True)
            return bs
    fallback = min(512, max_bs)
    print(f"  WARNING: calibration failed for {model_id}; using fallback bs={fallback}", flush=True)
    return fallback


def _pred_compressed(slug: str) -> bool:
    r = BASE / "Results" / slug
    return (r / "by_layer_corpus_pred.csv.gz").exists() or \
           (r / "by_layer_corpus_pred.csv.xz").exists()


def compress_corpus_pred(slug: str):
    """Compress by_layer_corpus_pred.csv → .gz and remove the original."""
    src = BASE / "Results" / slug / "by_layer_corpus_pred.csv"
    dst = BASE / "Results" / slug / "by_layer_corpus_pred.csv.xz"
    if not src.exists():
        return
    if _pred_compressed(slug):
        src.unlink(missing_ok=True)
        return
    banner(f"COMPRESS  {slug}/by_layer_corpus_pred.csv")
    src_mb = src.stat().st_size / 1e6
    with open(src, "rb") as f_in, lzma.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    dst_mb = dst.stat().st_size / 1e6
    print(f"  {src_mb:.0f} MB → {dst_mb:.0f} MB ({dst_mb/src_mb*100:.0f}% of original)", flush=True)
    src.unlink()


def mlp_complete(slug: str, cond_name: str) -> bool:
    csv_path = BASE / "Results" / slug / "by_layer_mlp.csv"
    if not csv_path.exists():
        return False
    with open(csv_path, newline="") as f:
        import csv
        rows = [r for r in csv.DictReader(f)
                if r["condition"] == cond_name
                and r["split"] in ("pair_novel", "word_novel")
                and r["mode"] in ("mean_pooled", "words_only")]
    # Last layer only → expect (1 layer) × 2 modes × 2 splits = 4 rows per condition
    return len(rows) >= 4


def model_complete(slug: str) -> bool:
    """True when both conditions' MLP rows exist and corpus-freq is compressed."""
    if not _pred_compressed(slug):
        return False
    return all(mlp_complete(slug, cond["name"]) for cond in CONDITIONS)


def run_model(model: dict, gpu: int, emb_dir: Path,
              skip_controls: bool, force: bool = False,
              revision: str = None):
    """Run final-layer extraction + probes for one model (or one checkpoint).

    revision: HF Hub revision string (e.g. "step256" for Pythia). When given,
              appended to the slug and passed to extraction with --revision.
    """
    base_slug = model_slug(model["id"])
    slug = f"{base_slug}_{revision}" if revision else base_slug

    if not force and model_complete(slug):
        banner(f"MODEL: {model['id']}  {revision or 'final'}  ALREADY COMPLETE — skipping")
        return
    banner(f"MODEL: {model['id']}  {revision or 'final'}  (slug={slug})")

    # ── 0. Calibrate batch_size (subprocess exits after probe — clean slate) ──
    banner(f"CALIBRATE  {model['id']}  {revision or ''}")
    effective_bs = calibrate_batch_size(model["id"], model["batch_size"], gpu,
                                        revision=revision)

    # ── 1. Extract sequentially (GPU-heavy; one condition at a time) ───────────
    for cond in CONDITIONS:
        if force or not mlp_complete(slug, cond["name"]):
            extract_cmd = [
                PYTHON, SCRIPTS / "run_bylayer.py",
                "--conditions",    cond["name"],
                "--gpu",           str(gpu),
                "--skip-mlp",
                "--layers",        "last",
                "--embeddings-dir", str(emb_dir),
                "--model-id",      model["id"],
                "--num-layers",    str(model["n_layers"]),
                "--slug",          slug,
                "--extract-batch-size", str(effective_bs),
            ]
            if revision is not None:
                extract_cmd += ["--revision", revision]
            if force:
                extract_cmd.append("--force")
            ok = run(extract_cmd,
                     label=f"EXTRACT  {slug} / {cond['name']}",
                     abort_on_fail=False)
            if not ok:
                print(f"  Extraction failed for {slug} / {cond['name']} — skipping model.", flush=True)
                return f"extract/{cond['name']}"
        else:
            banner(f"EXTRACT  {slug} / {cond['name']}  (skipped — MLP already complete)")

    force_flag = ["--force"] if force else []

    all_conds = [c["name"] for c in CONDITIONS]

    def mlp_cmd(*extra_flags):
        return [PYTHON, SCRIPTS / "by_layer_mlp.py",
                "--model-slug", slug,
                "--num-layers", str(model["n_layers"]),
                "--conditions", *all_conds,
                "--modes",      "mean_pooled", "words_only",
                "--gpu",        str(gpu),
                "--embeddings-dir", str(emb_dir),
                *force_flag, *extra_flags]

    # ── 2. MLP CV — both conditions in one process ───────────────────────────
    if not run(mlp_cmd("--splits", "pair_novel", "word_novel"),
               label=f"MLP CV  {slug}", abort_on_fail=False):
        return "mlp_cv"

    # ── 3. Corpus-freq — both conditions in one process ──────────────────────
    if not run(mlp_cmd("--corpus-freq"),
               label=f"CORPUS-FREQ  {slug}", abort_on_fail=False):
        return "corpus_freq"

    # ── 4. Controls — both conditions in one process ─────────────────────────
    if not skip_controls:
        if not run(mlp_cmd("--splits", "pair_novel", "word_novel", "--control"),
                   label=f"CONTROLS  {slug}", abort_on_fail=False):
            print("  Controls failed — continuing without control data.", flush=True)

    # ── 5. Delete embeddings ─────────────────────────────────────────────────
    for cond in CONDITIONS:
        delete_dir(emb_dir / cond["novel_dir"]  / slug)
        delete_dir(emb_dir / cond["corpus_dir"] / slug)

    # ── 6. Compress corpus predictions ───────────────────────────────────────
    compress_corpus_pred(slug)
    return None  # success


def run_by_layer(model: dict, gpu: int, emb_dir: Path,
                 skip_controls: bool, force: bool = False):
    """Run full by-layer extraction + MLP probes for a model's final checkpoint."""
    slug = model_slug(model["id"])
    banner(f"BY-LAYER: {model['id']}  (slug={slug})")

    effective_bs = calibrate_batch_size(model["id"], model["batch_size"], gpu)

    # Extraction + MLP for all layers via run_bylayer.py
    pipeline_cmd = [
        PYTHON, SCRIPTS / "run_bylayer.py",
        "--conditions", "default", "attn_zeroed",
        "--gpu",        str(gpu),
        "--embeddings-dir", str(emb_dir),
        "--model-id",   model["id"],
        "--num-layers", str(model["n_layers"]),
        "--slug",       slug,
        "--extract-batch-size", str(effective_bs),
    ]
    if force:
        pipeline_cmd.append("--force")
    if not run(pipeline_cmd, label=f"BY-LAYER PIPELINE  {slug}", abort_on_fail=False):
        print(f"  By-layer pipeline failed for {slug}.", flush=True)
        return "by_layer_pipeline"

    # Controls: run separately (run_bylayer.py doesn't include them)
    if not skip_controls:
        force_flag = ["--force"] if force else []
        ctrl_cmd = [
            PYTHON, SCRIPTS / "by_layer_mlp.py",
            "--model-slug", slug,
            "--num-layers", str(model["n_layers"]),
            "--conditions", "default", "attn_zeroed",
            "--modes",      "mean_pooled", "words_only",
            "--gpu",        str(gpu),
            "--embeddings-dir", str(emb_dir),
            "--splits",    "pair_novel", "word_novel",
            "--control",
            *force_flag,
        ]
        if not run(ctrl_cmd, label=f"BY-LAYER CONTROLS  {slug}", abort_on_fail=False):
            print("  Controls failed — continuing without control data.", flush=True)

    # Compress corpus predictions
    compress_corpus_pred(slug)
    return None  # success


# BabyLM-aligned Pythia checkpoint steps (step1000 ≈ 2.10B tokens = BabyLM full training)
PYTHIA_CHECKPOINT_STEPS = [16, 32, 64, 256, 512, 1000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   choices=["pythia", "gpt2", "olmo", "llama"],
                   help="Model groups to run (default: all)")
    p.add_argument("--gpu",          type=int, default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--skip-large",   action="store_true", dest="skip_large",
                   help="Skip models requiring a large GPU (OLMo-7B, Llama-3-8B)")
    p.add_argument("--skip-controls", action="store_true", dest="skip_controls")
    p.add_argument("--force",         action="store_true",
                   help="Re-run all computations even if output already exists")
    p.add_argument("--checkpoints",   action="store_true",
                   help="Run final-layer probe at BabyLM-aligned checkpoint steps "
                        "(Pythia only: step16/32/64/256/512/1000)")
    p.add_argument("--by-layer",      action="store_true", dest="by_layer",
                   help="Run full by-layer sweep at the final checkpoint "
                        "(any group; most useful for Pythia)")
    args = p.parse_args()

    if args.checkpoints and args.by_layer:
        p.error("--checkpoints and --by-layer are mutually exclusive")

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    model_list = [
        m for m in MODELS
        if (args.models is None or m["group"] in args.models)
        and not (args.skip_large and m["large_gpu"])
    ]

    t0 = time.perf_counter()
    failures: list[dict] = []

    def record(model_id, step, revision=None):
        failures.append({"model": model_id, "step": step,
                         "revision": revision or "final"})
        print(f"  !! FAILED: {model_id}  revision={revision or 'final'}  step={step}",
              flush=True)

    if args.checkpoints:
        # Final-layer probe at each BabyLM-aligned Pythia step
        pythia_list = [m for m in model_list if m["group"] == "pythia"]
        if not pythia_list:
            p.error("--checkpoints requires --models pythia (or default which includes pythia)")
        print(f"Checkpoint sweep: {len(pythia_list)} Pythia models × "
              f"{len(PYTHIA_CHECKPOINT_STEPS)} steps", flush=True)
        for model in pythia_list:
            for step in PYTHIA_CHECKPOINT_STEPS:
                revision = f"step{step}"
                err = run_model(model, args.gpu, emb_dir, args.skip_controls,
                                args.force, revision=revision)
                if err:
                    record(model["id"], err, revision)

    elif args.by_layer:
        # Full by-layer sweep at the final checkpoint
        print(f"By-layer sweep: {len(model_list)} models", flush=True)
        for m in model_list:
            print(f"  {m['id']}", flush=True)
        for model in model_list:
            err = run_by_layer(model, args.gpu, emb_dir, args.skip_controls, args.force)
            if err:
                record(model["id"], err)

    else:
        # Default: final-layer probe at the final checkpoint
        print(f"Running {len(model_list)} models:", flush=True)
        for m in model_list:
            print(f"  {m['id']}", flush=True)
        for model in model_list:
            err = run_model(model, args.gpu, emb_dir, args.skip_controls, args.force)
            if err:
                record(model["id"], err)

    if failures:
        log_path = BASE / "Results" / "pipeline_errors.log"
        with open(log_path, "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] run_scale_models.py\n")
            for fail in failures:
                f.write(f"  {fail['model']}  revision={fail['revision']}  step={fail['step']}\n")
        banner(f"COMPLETED WITH {len(failures)} FAILURE(S) — see Results/pipeline_errors.log")
        sys.exit(1)

    banner("SCALE MODELS PIPELINE COMPLETE")
    print(f"  Total: {(time.perf_counter()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
