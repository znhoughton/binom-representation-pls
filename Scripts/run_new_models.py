"""
Final-layer analysis pipeline for additional model families.

Runs only the last layer of each model's final checkpoint — no by-layer sweep,
no training dynamics. Designed for Pythia, GPT-2, OLMo, and Llama 3.

For each model:
  1. Extract last-layer embeddings (novel + corpus, both conditions)
     → saved to <embeddings_dir>/{condition}_dir/<slug>/layer_last.npz
  2. Run MLP CV probes (pair_novel + word_novel splits)
  3. Run novel→corpus transfer prediction (corpus-freq analysis)
  4. Compress by_layer_corpus_pred.csv → .gz  (gitignored uncompressed)
  5. Run control (shuffled-label) probes
  6. Delete embeddings to free disk

Large models (OLMo-7B, Llama-3-8B) require an A100 or similar. Mark them with
"large_gpu": True and pass --skip-large if running on a smaller machine.

Usage:
    python Scripts/run_new_models.py [--models pythia gpt2 olmo llama]
                                     [--gpu 0]
                                     [--embeddings-dir /dpluth-data]
                                     [--skip-large]
                                     [--skip-controls]

Adding a model: append an entry to MODELS below. The slug is derived
automatically from the HF model ID (slashes → underscores).
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import time
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
    {"group": "pythia", "id": "EleutherAI/pythia-2.8b",  "n_layers": 32,  "large_gpu": False, "shard_pairs": 200000, "batch_size": 4096},
    # ── GPT-2 (OpenAI) ─────────────────────────────────────────────────────
    {"group": "gpt2",   "id": "gpt2",                    "n_layers": 12,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "gpt2",   "id": "gpt2-medium",             "n_layers": 24,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "gpt2",   "id": "gpt2-large",              "n_layers": 36,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 4096},
    {"group": "gpt2",   "id": "gpt2-xl",                 "n_layers": 48,  "large_gpu": False, "shard_pairs": 200000, "batch_size": 4096},
    # ── OLMo (AI2, open data) ──────────────────────────────────────────────
    {"group": "olmo",   "id": "allenai/OLMo-1B-hf",     "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "olmo",   "id": "allenai/OLMo-2-1124-1B",  "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "olmo",   "id": "allenai/OLMo-7B-hf",     "n_layers": 32,  "large_gpu": True,  "shard_pairs": 100000, "batch_size": 2048},
    {"group": "olmo",   "id": "allenai/OLMo-2-1124-7B",  "n_layers": 32,  "large_gpu": True,  "shard_pairs": 100000, "batch_size": 2048},
    # ── Llama 3 (Meta, gated — requires HF token) ──────────────────────────
    {"group": "llama",  "id": "meta-llama/Llama-3.2-1B", "n_layers": 16,  "large_gpu": False, "shard_pairs": 500000, "batch_size": 8192},
    {"group": "llama",  "id": "meta-llama/Meta-Llama-3-8B", "n_layers": 32, "large_gpu": True, "shard_pairs": 100000, "batch_size": 2048},
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


def delete_dir(path, label=""):
    path = Path(path)
    if path.exists():
        size_gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
        banner(f"DELETE  {label or path.name}  ({size_gb:.1f} GB)")
        shutil.rmtree(path)
        print("  Deleted.", flush=True)


def compress_corpus_pred(slug: str):
    """Compress by_layer_corpus_pred.csv → .gz and remove the original."""
    src = BASE / "Results" / slug / "by_layer_corpus_pred.csv"
    dst = BASE / "Results" / slug / "by_layer_corpus_pred.csv.gz"
    if not src.exists():
        print(f"  by_layer_corpus_pred.csv not found, skipping compression.", flush=True)
        return
    if dst.exists():
        print(f"  .gz already exists, skipping.", flush=True)
        src.unlink(missing_ok=True)
        return
    banner(f"COMPRESS  {slug}/by_layer_corpus_pred.csv")
    src_mb = src.stat().st_size / 1e6
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=6) as f_out:
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
                and r["mode"] in ("mean_pooled", "individual", "words_only")]
    # Last layer only → expect (1 layer) × 3 modes × 2 splits = 6 rows per condition
    return len(rows) >= 6


def model_complete(slug: str) -> bool:
    """True when both conditions' MLP rows exist and corpus-freq .gz is present."""
    gz = BASE / "Results" / slug / "by_layer_corpus_pred.csv.gz"
    if not gz.exists():
        return False
    return all(mlp_complete(slug, cond["name"]) for cond in CONDITIONS)


def run_model(model: dict, gpu: int, emb_dir: Path,
              skip_controls: bool):
    slug = model_slug(model["id"])
    if model_complete(slug):
        banner(f"MODEL: {model['id']}  ALREADY COMPLETE — skipping")
        return
    banner(f"MODEL: {model['id']}  (slug={slug})")

    for cond in CONDITIONS:
        # ── 1. Extract last layer only ──────────────────────────────────────
        if not mlp_complete(slug, cond["name"]):
            run(
                [PYTHON, SCRIPTS / "run_by_layer_pipeline.py",
                 "--models",        "125m",    # flag ignored — overridden by --slug
                 "--conditions",    cond["name"],
                 "--gpu",           str(gpu),
                 "--skip-mlp",
                 "--layers",        "last",
                 "--embeddings-dir", str(emb_dir),
                 "--model-id",      model["id"],
                 "--slug",          slug,
                 "--extract-batch-size", str(model["batch_size"])],
                label=f"EXTRACT  {slug} / {cond['name']}",
            )
        else:
            banner(f"EXTRACT  {slug} / {cond['name']}  (skipped — MLP already complete)")

        # ── 2. MLP CV probes ────────────────────────────────────────────────
        run(
            [PYTHON, SCRIPTS / "by_layer_mlp.py",
             "--model-slug", slug,
             "--num-layers", str(model["n_layers"]),
             "--conditions", cond["name"],
             "--modes",      "mean_pooled", "individual", "words_only",
             "--splits",     "pair_novel", "word_novel",
             "--gpu",        str(gpu),
             "--embeddings-dir", str(emb_dir)],
            label=f"MLP CV  {slug} / {cond['name']}",
        )

        # ── 3. Novel→corpus transfer (corpus-freq) ──────────────────────────
        run(
            [PYTHON, SCRIPTS / "by_layer_mlp.py",
             "--model-slug", slug,
             "--num-layers", str(model["n_layers"]),
             "--conditions", cond["name"],
             "--modes",      "mean_pooled", "individual", "words_only",
             "--corpus-freq",
             "--gpu",        str(gpu),
             "--embeddings-dir", str(emb_dir)],
            label=f"CORPUS-FREQ  {slug} / {cond['name']}",
        )

        # ── 4. Controls ─────────────────────────────────────────────────────
        if not skip_controls:
            run(
                [PYTHON, SCRIPTS / "by_layer_mlp.py",
                 "--model-slug", slug,
                 "--num-layers", str(model["n_layers"]),
                 "--conditions", cond["name"],
                 "--modes",      "mean_pooled", "individual", "words_only",
                 "--splits",     "pair_novel", "word_novel",
                 "--gpu",        str(gpu),
                 "--control",
                 "--embeddings-dir", str(emb_dir)],
                label=f"CONTROLS  {slug} / {cond['name']}",
            )

        # ── 5. Delete embeddings ────────────────────────────────────────────
        delete_dir(emb_dir / cond["novel_dir"]  / slug)
        delete_dir(emb_dir / cond["corpus_dir"] / slug)

    # ── 6. Compress corpus predictions (after both conditions done) ─────────
    compress_corpus_pred(slug)


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
    args = p.parse_args()

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    model_list = [
        m for m in MODELS
        if (args.models is None or m["group"] in args.models)
        and not (args.skip_large and m["large_gpu"])
    ]

    print(f"Running {len(model_list)} models:", flush=True)
    for m in model_list:
        print(f"  {m['id']}", flush=True)

    t0 = time.perf_counter()
    for model in model_list:
        run_model(model, args.gpu, emb_dir, args.skip_controls)

    banner("NEW MODELS PIPELINE COMPLETE")
    print(f"  Total: {(time.perf_counter()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
