"""
Master pipeline: extraction → MLP analysis for the by-layer study.
Uses subprocess sharding to prevent memory accumulation.

Usage:
    python Scripts/run_by_layer_pipeline.py
    python Scripts/run_by_layer_pipeline.py --models 125m
"""
import argparse
import gc
import subprocess
import sys
import time
import shutil
import numpy as np
from pathlib import Path

PYTHON = sys.executable
BASE = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = str(BASE / "Scripts" / "extract_embeddings.py")

MODELS = [
    {"flag": "125m", "id": "znhoughton/opt-babylm-125m-20eps-seed964",
     "slug": "znhoughton_opt-babylm-125m-20eps-seed964",
     "num_layers": 12, "batch_size": 4096, "shard_pairs": 500000},
    {"flag": "350m", "id": "znhoughton/opt-babylm-350m-20eps-seed964",
     "slug": "znhoughton_opt-babylm-350m-20eps-seed964",
     "num_layers": 24, "batch_size": 8192, "shard_pairs": 500000},
    {"flag": "1.3b", "id": "znhoughton/opt-babylm-1.3b-20eps-seed964",
     "slug": "znhoughton_opt-babylm-1_3b-20eps-seed964",
     "num_layers": 24, "batch_size": 4096, "shard_pairs": 500000},
]

CONDITIONS = [
    {"name": "default",     "context": "binomial",    "extract": "word",
     "corpus_dir": "embeddings", "novel_dir": "novel_embeddings"},
    {"name": "attn_zeroed", "context": "attn_zeroed", "extract": "word",
     "corpus_dir": "embeddings_attn_zeroed", "novel_dir": "novel_embeddings_attn_zeroed"},
]

NOVEL_TOTAL = 340042
CORPUS_TOTAL = 48965
WORD_KEYS = ["alpha_w1", "alpha_and", "alpha_w2",
             "non_alpha_w1", "non_alpha_and", "non_alpha_w2"]


def run_label(label):
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)


def extract_sharded(model, cond, data_split, gpu, emb_root, force=False):
    """Extract embeddings using subprocess sharding to bound memory."""
    out_dir = str(emb_root / cond[f"{data_split}_dir"] / model["slug"])
    num_layers = model["num_layers"]
    layers = [str(i) for i in range(num_layers + 1)]
    bs = model["batch_size"]

    total_pairs = NOVEL_TOTAL if data_split == "novel" else CORPUS_TOTAL
    shard_size = model["shard_pairs"]

    # For corpus (small), use one shard
    if data_split == "corpus":
        n_shards = 1
    else:
        n_shards = (total_pairs + shard_size - 1) // shard_size

    label = f"{model['flag']} / {cond['name']} / {data_split} ({n_shards} shards, ~{shard_size} pairs each)"
    run_label(label)
    t0 = time.perf_counter()

    # Check if already done (all layer files exist)
    out_path = Path(out_dir)
    if not force:
        existing = [out_path / f"layer_{i}.npz" for i in range(num_layers + 1)]
        if all(f.exists() and f.stat().st_size > 1000 for f in existing):
            print("  All layers exist, skipping.", flush=True)
            return True

    # Run shards sequentially
    shard_dir = out_path / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for shard in range(n_shards):
        # Check if shard already done
        shard_check = shard_dir / f"layer_0_shard{shard}.npz"
        if shard_check.exists() and not force:
            print(f"  Shard {shard}/{n_shards} exists, skipping.", flush=True)
            continue

        cmd = [
            PYTHON, EXTRACT_SCRIPT,
            "--model", model["id"],
            "--data", data_split,
            "--layer", *layers,
            "--out", str(shard_dir),
            "--gpu", str(gpu),
            "--batch-size", str(bs),
            "--context", cond["context"],
            "--extract", cond["extract"],
            "--shard-index", str(shard),
            "--num-shards", str(n_shards),
        ]
        if force:
            cmd.append("--force")

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  Shard {shard} FAILED (exit {rc})", flush=True)
            return False
        print(f"  Shard {shard+1}/{n_shards} done", flush=True)
        # Prevent GPU driver crash (0xC0000409) from accumulated subprocess exits
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(3)

    # Merge shards into final layer files
    print(f"  Merging {n_shards} shards...", flush=True)
    out_path.mkdir(parents=True, exist_ok=True)

    for layer in range(num_layers + 1):
        final_path = out_path / f"layer_{layer}.npz"

        # Collect shard files for this layer
        shard_files = []
        for shard in range(n_shards):
            sf = shard_dir / f"layer_{layer}_shard{shard}.npz"
            if sf.exists():
                shard_files.append(sf)

        if not shard_files:
            continue

        # Merge one key at a time
        # First load metadata from first shard to get keys
        first = np.load(shard_files[0], allow_pickle=True)
        available_keys = [k for k in WORD_KEYS if k in first]
        has_meta = "word1" in first
        first.close()

        save_dict = {}
        if has_meta:
            w1_parts, w2_parts, pref_parts = [], [], []
            for sf in shard_files:
                d = np.load(sf, allow_pickle=True)
                w1_parts.append(d["word1"])
                w2_parts.append(d["word2"])
                pref_parts.append(d["preference"])
                d.close()
            save_dict["word1"] = np.concatenate(w1_parts)
            save_dict["word2"] = np.concatenate(w2_parts)
            save_dict["preference"] = np.concatenate(pref_parts)
            del w1_parts, w2_parts, pref_parts

        for k in available_keys:
            parts = []
            for sf in shard_files:
                d = np.load(sf)
                parts.append(d[k].copy())
                d.close()
            save_dict[k] = np.concatenate(parts)
            del parts
            gc.collect()

        np.savez(final_path, **save_dict)
        del save_dict
        gc.collect()

    print(f"  Merge done.", flush=True)

    # Clean up shard files
    shutil.rmtree(shard_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    print(f"  OK in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Conditions to extract (e.g. default attn_zeroed); omit for all")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--skip-extraction", action="store_true", dest="skip_extraction")
    p.add_argument("--skip-mlp", action="store_true", dest="skip_mlp")
    p.add_argument("--force", action="store_true")
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Root directory for embedding subdirs (default: <project>/Data)")
    args = p.parse_args()

    emb_root = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    total_t0 = time.perf_counter()

    # Step 1: Extraction
    if not args.skip_extraction:
        run_label("STEP 1: Extract embeddings at all layers")
        for model in MODELS:
            if args.models and not any(m in model["flag"] for m in args.models):
                continue
            for cond in CONDITIONS:
                if args.conditions and cond["name"] not in args.conditions:
                    continue
                for data_split in ["corpus", "novel"]:
                    ok = extract_sharded(model, cond, data_split, args.gpu, emb_root, args.force)
                    if not ok:
                        print(f"  FAILED. Continuing with next.", flush=True)

    # Step 2: MLP analysis
    if not args.skip_mlp:
        run_label("STEP 2: MLP analysis")
        for model in MODELS:
            if args.models and not any(m in model["flag"] for m in args.models):
                continue
            cmd = [
                PYTHON, str(BASE / "Scripts" / "by_layer_mlp.py"),
                "--model-slug", model["slug"],
                "--num-layers", str(model["num_layers"]),
                "--gpu", str(args.gpu),
                "--conditions", "default", "attn_zeroed",
                # modes run for both conditions:
                # words_only: cat(w1, w2) from full sentence — no "and", uses original sentence contexts
                "--modes", "mean_pooled", "individual", "words_only",
                "--splits", "pair_novel", "word_novel",
                "--freq-strata",
                "--freq-bootstrap", "50",
            ]
            run_label(f"MLP analysis: {model['flag']}")
            t0 = time.perf_counter()
            rc = subprocess.call(cmd)
            elapsed = time.perf_counter() - t0
            status = "OK" if rc == 0 else f"FAILED (exit {rc})"
            print(f"  {status} in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n{'='*60}", flush=True)
    print(f"  PIPELINE COMPLETE", flush=True)
    print(f"  Total time: {total_elapsed/3600:.1f} hours", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
