#!/usr/bin/env python3
"""
make_random_init.py
-------------------
Build "checkpoint 0" models: the architecture of a trained model, with freshly
initialized weights and no training at all.

Why this exists
---------------
Experiment 3 reports that probe R-squared is already high at the earliest saved
checkpoint (~20M tokens). That is only evidence about learning if an UNTRAINED
model of the same architecture does worse. Pythia publishes its initialization
as revision `step0`, so nothing has to be rebuilt there. The OPT-BabyLM models
were trained from scratch under the HF Trainer, which writes no checkpoint
before the first `save_steps`, so their step 0 has to be reconstructed here.

How the init is reproduced
--------------------------
train_autoreg.py calls transformers.set_seed(training_args.seed) before
AutoModelForCausalLM.from_config(...), and train_all_opt_babylm.sh passes
--seed 964. This script makes the same two calls in the same order, so the
weights should be the init those runs actually started from. That is not
guaranteed: anything between those two points in the training script that draws
from the torch RNG would shift the stream. It is also not required. The sanity
check needs a draw from the same init distribution, not the identical draw.
Pass several values to --seeds if one draw looks too convenient, and compare the
printed checksums across machines to confirm a draw is reproducible.

The config is pulled from the published repo rather than rebuilt from
facebook/opt-*, so the random init cannot disagree with the trained model about
architecture. The guard prints do_layer_norm_before and word_embed_proj_dim
explicitly, because silently inheriting those from facebook/opt-350m is exactly
how the post-LN bug happened.

Weights are saved in fp32. extract_embeddings.py casts to fp16 on load, the
same as for every trained checkpoint, so the step-0 run is numerically
comparable to the rest of the sweep.

Usage
-----
  # All three BabyLM step-0 models at seed 964, under Data/random_init/
  python Scripts/pipeline/make_random_init.py

  # One model, three independent draws
  python Scripts/pipeline/make_random_init.py --models 125m --seeds 964 1 2

  # Pythia reconstruction, to check it agrees with the published step0
  python Scripts/pipeline/make_random_init.py --models pythia-160m
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

HERE = Path(__file__).resolve().parent          # .../Scripts/pipeline
BASE = HERE.parents[1]                          # repo root
HF_CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))

# Seed 964 everywhere, per the training runs and the project convention.
DEFAULT_SEED = 964

MODELS = {
    "125m":        "znhoughton/opt-babylm-125m-20eps-seed964",
    "350m":        "znhoughton/opt-babylm-350m-20eps-seed964",
    "1.3b":        "znhoughton/opt-babylm-1.3b-20eps-seed964",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b":   "EleutherAI/pythia-1b",
}

BABYLM = ["125m", "350m", "1.3b"]

# Architecture fields worth printing before spending GPU hours. The first two
# are the ones the post-LN bug turned on.
GUARD_FIELDS = ["model_type", "num_hidden_layers", "hidden_size",
                "do_layer_norm_before", "word_embed_proj_dim",
                "vocab_size", "tie_word_embeddings", "init_std",
                "initializer_range", "max_position_embeddings"]


def out_dir_for(flag: str, seed: int) -> Path:
    tag = "step0" if seed == DEFAULT_SEED else f"step0seed{seed}"
    return BASE / "Data" / "random_init" / f"{MODELS[flag].replace('/', '_')}_{tag}"


def checksum(model) -> str:
    """Deterministic hash of every parameter, in sorted-key order."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        t = sd[k]
        if not torch.is_floating_point(t):
            t = t.to(torch.float32)
        h.update(k.encode())
        h.update(t.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build(flag: str, seed: int, force: bool) -> Path:
    model_id = MODELS[flag]
    out = out_dir_for(flag, seed)

    if out.exists() and (out / "config.json").exists() and not force:
        print(f"  exists, skipping: {out}   (--force to rebuild)", flush=True)
        return out

    config = AutoConfig.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)

    print(f"  config from {model_id}", flush=True)
    for f in GUARD_FIELDS:
        if hasattr(config, f):
            print(f"    {f:28s} {getattr(config, f)}", flush=True)

    # Initialize in fp32 regardless of what the published repo recorded. Some
    # transformers versions honour config.torch_dtype in from_config, which
    # would draw the weights in fp16 and give the untrained model a different
    # init distribution than the trained runs had. extract_embeddings.py casts
    # to fp16 on load anyway, the same as for every trained checkpoint.
    for field in ("torch_dtype", "dtype"):
        if getattr(config, field, None) is not None:
            setattr(config, field, "float32")

    # Same two calls, same order, as train_autoreg.py.
    set_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  initialized {n_params/1e6:.1f}M params  seed={seed}", flush=True)
    print(f"  weight checksum  {checksum(model)}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR,
                                        use_fast=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    size_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"  wrote {out}  ({size_gb:.2f} GB)", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=BABYLM,
                   choices=list(MODELS),
                   help="Which architectures to initialize (default: the three BabyLMs)")
    p.add_argument("--seeds", nargs="+", type=int, default=[DEFAULT_SEED],
                   help=f"Init seeds (default: {DEFAULT_SEED}, the training seed)")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if the output directory already exists")
    args = p.parse_args()

    print("Building random-init (checkpoint 0) models", flush=True)
    print(f"  models: {args.models}", flush=True)
    print(f"  seeds:  {args.seeds}\n", flush=True)

    built = []
    for flag in args.models:
        for seed in args.seeds:
            print(f"── {flag}  seed={seed} "
                  f"{'─' * max(0, 44 - len(flag))}", flush=True)
            built.append(build(flag, seed, args.force))
            print(flush=True)

    print("Done. Paths:", flush=True)
    for b in built:
        print(f"  {b}", flush=True)


if __name__ == "__main__":
    main()
