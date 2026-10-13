#!/usr/bin/env python3
"""
run_step0_baseline.py
---------------------
Checkpoint-0 sanity check for Experiment 3.

The question
------------
Experiment 3 reports that binomial-held-out probe R-squared is already high at
the earliest saved checkpoint, and reads that as abstract preferences being
present early in training. That reading requires a floor. If a model that has
seen zero tokens probes just as well, then what the probe recovers at the first
checkpoint is not something the model learned: it is a property of the
architecture, the tokenizer, and the initialization distribution.

This script produces that floor. Same final-layer probe, same two attention
conditions, same two cross-validation splits, run on untrained models.

  Pythia   revision `step0`, the initialization EleutherAI published.
  BabyLM   reconstructed by make_random_init.py, since the HF Trainer saved
           nothing before the first save_steps.

Output slugs follow the existing checkpoint convention, so a step-0 row drops
into the Experiment 3 figures and tables as one more checkpoint:

  Results/EleutherAI_pythia-160m_step0/
  Results/znhoughton_opt-babylm-125m-20eps-seed964_step0/

To pull them into the writeup and the brms sweep, add 0 to the step vectors in
Writeup/writeup.qmd (the *_ckpt_registry tibbles) and in
Scripts/analysis/corpus_relfreq_brms_all.R (CKPT_REGISTRY). Both are keyed by
slug and the brms script is resume-safe, so it will fit only the new cells.

How to read the result
----------------------
Two outcomes, both worth having:

  Step-0 R-squared is LOW. The early-checkpoint result stands as evidence about
      learning, and the paper gains a floor it currently lacks.

  Step-0 R-squared is HIGH. Then note what the probe target is. `preference` is
      the model's own log-probability difference between the two orderings, so
      at every checkpoint the probe is recovering the model's own scoring
      function from the hidden states that feed it. For an untrained net that
      function is arbitrary, but it is still smooth and still a function of
      exactly those representations, so it can be recoverable with no linguistic
      content whatsoever. A high step-0 number therefore does not settle whether
      the trained models' abstract preferences are real; it says probe
      generalization alone cannot establish them, which is the point
      Scripts/supplementary/cross_model_agreement.py already makes in its
      header. The discriminating follow-up is under --help-agreement: ask
      whether the untrained models agree WITH EACH OTHER. An arbitrary function
      of one random init has no reason to match another random init.

Cost
----
Per model: extraction of the corpus set (~49k pairs) and the novel set (~340k
pairs), twice, once per attention condition, then the probes. The same work as
any other checkpoint in the sweep, dominated by the novel-set forward passes.
Start with --models babylm-125m to time one cell before committing to all six.

Usage
-----
  # Everything: three BabyLM step-0 models, three Pythia step0 revisions
  python Scripts/pipeline/run_step0_baseline.py --gpu 0

  # One cell first, to time it
  python Scripts/pipeline/run_step0_baseline.py --models babylm-125m --gpu 0

  # Pythia only (no reconstruction needed, uses the published step0)
  python Scripts/pipeline/run_step0_baseline.py --models pythia --gpu 0

  # Keep embeddings on an external drive, as the other checkpoint runs do
  python Scripts/pipeline/run_step0_baseline.py --embeddings-dir /mnt/scratch/emb

  # Print the cross-model-agreement follow-up and exit
  python Scripts/pipeline/run_step0_baseline.py --help-agreement
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../Scripts/pipeline
BASE = HERE.parents[1]                          # repo root
sys.path.insert(0, str(HERE))

import run_scale_models as rsm                  # noqa: E402  (needs sys.path)
import make_random_init as mri                  # noqa: E402

PYTHON = sys.executable

# BabyLM entries carry the same n_layers / batch_size the rest of the pipeline
# uses for them. `slug` matches run_babylm_checkpoints.slug_for(flag, 0) exactly,
# including the "1.3b" spelling the checkpoint dirs use. The FINAL BabyLM dirs
# spell it "1_3b"; the checkpoint dirs do not. Do not "fix" this.
BABYLM_CELLS = [
    {"flag": "125m", "n_layers": 12, "batch_size": 8192, "shard_pairs": 500000,
     "large_gpu": False,
     "slug": "znhoughton_opt-babylm-125m-20eps-seed964_step0"},
    {"flag": "350m", "n_layers": 24, "batch_size": 8192, "shard_pairs": 500000,
     "large_gpu": False,
     "slug": "znhoughton_opt-babylm-350m-20eps-seed964_step0"},
    {"flag": "1.3b", "n_layers": 24, "batch_size": 8192, "shard_pairs": 500000,
     "large_gpu": False,
     "slug": "znhoughton_opt-babylm-1.3b-20eps-seed964_step0"},
]

# The three Pythia sizes the paper reports checkpoints for.
PYTHIA_IDS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
              "EleutherAI/pythia-1b"]

CHOICES = ["babylm", "pythia",
           "babylm-125m", "babylm-350m", "babylm-1.3b",
           "pythia-160m", "pythia-410m", "pythia-1b"]

AGREEMENT_HELP = """
Discriminating follow-up: do the untrained models agree with each other?
------------------------------------------------------------------------
Probe R-squared at step 0 cannot distinguish "the architecture already encodes
English ordering preferences" from "the probe can recover any smooth function of
its own input features". Cross-model agreement can, and the script for it
already exists. After this run finishes:

    python Scripts/supplementary/cross_model_agreement.py
        --slug-suffix _step0 --out-tag step0 --sets novel

  Agreement at step 0 is near zero, while trained models agree.
      The trained agreement reported in the paper is about language, and the
      step-0 probe R-squared is function-recoverability. The early-checkpoint
      claim survives, now with both a floor and a diagnosis.

  Agreement at step 0 is substantial.
      Something shared and pre-linguistic (tokenizer, word length, positional
      structure) is driving part of the measured agreement too. That belongs in
      the paper, and the residualised columns in that script's output, which
      partial out relative frequency and relative length, are where to look
      first.
"""


def banner(msg):
    print("\n" + "=" * 70 + "\n  " + msg + "\n" + "=" * 70, flush=True)


def _fmt_h(seconds: float) -> str:
    h, m = int(seconds // 3600), int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def preflight_pythia(model_ids: list) -> list:
    """Confirm each repo publishes a `step0` tag. Returns the ids that do."""
    banner("PREFLIGHT  does Pythia publish revision step0?")
    try:
        from huggingface_hub import list_repo_refs
    except ImportError:
        print("  huggingface_hub not importable; skipping the check. If a model "
              "below fails at extraction with a revision error, a missing step0 "
              "is the likely cause. Fall back to reconstructing it:\n"
              "    python Scripts/pipeline/make_random_init.py --models pythia-160m\n"
              "  and probe the reconstruction instead.", flush=True)
        return model_ids

    ok = []
    for mid in model_ids:
        short = mid.split("/")[-1]
        try:
            refs = list_repo_refs(mid)
            tags = {r.name for r in refs.branches} | {r.name for r in refs.tags}
            if "step0" in tags:
                print(f"  {mid:30s} step0 present", flush=True)
                ok.append(mid)
            else:
                print(f"  {mid:30s} NO step0, skipping. Reconstruct instead:\n"
                      f"      python Scripts/pipeline/make_random_init.py "
                      f"--models {short}", flush=True)
        except Exception as err:                        # noqa: BLE001
            print(f"  {mid:30s} could not check ({type(err).__name__}: {err}); "
                  f"attempting anyway", flush=True)
            ok.append(mid)
    return ok


def build_babylm_inits(cells: list, force: bool) -> dict:
    """Reconstruct the BabyLM initializations. Returns {flag: local path}."""
    banner("BUILD  BabyLM random initializations (seed 964)")
    cmd = [PYTHON, str(HERE / "make_random_init.py"),
           "--models"] + [c["flag"] for c in cells]
    if force:
        cmd.append("--force")
    if subprocess.call(cmd) != 0:
        sys.exit("make_random_init.py failed; cannot probe BabyLM step 0.")
    return {c["flag"]: mri.out_dir_for(c["flag"], mri.DEFAULT_SEED)
            for c in cells}


def summarise(slugs: list):
    """Print final-layer mean_pooled R-squared beside its control, per cell."""
    print("\n  Final-layer probe R-squared at step 0 "
          "(mean_pooled, the number Experiment 3 reports):", flush=True)
    print(f"    {'slug':52s} {'cond':12s} {'split':12s} {'R2':>7s} {'ctrl':>7s}",
          flush=True)
    for slug in slugs:
        rows = {}
        for fname, key in (("by_layer_mlp.csv", "r2"),
                           ("by_layer_mlp_control.csv", "ctrl")):
            fp = BASE / "Results" / slug / fname
            if not fp.exists():
                continue
            with open(fp, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["mode"] != "mean_pooled":
                        continue
                    rows.setdefault((r["condition"], r["split"]), {})[key] = r["mean_r2"]
        if not rows:
            print(f"    {slug:52s} (no results)", flush=True)
            continue
        for (cond, split), v in sorted(rows.items()):
            r2 = float(v.get("r2", "nan"))
            ct = float(v.get("ctrl", "nan"))
            print(f"    {slug:52s} {cond:12s} {split:12s} {r2:7.3f} {ct:7.3f}",
                  flush=True)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("--models", nargs="+", default=["babylm", "pythia"],
                   choices=CHOICES,
                   help="Cells to run (default: all six)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Root for the embedding subdirs (default: Data/)")
    p.add_argument("--skip-controls", action="store_true", dest="skip_controls",
                   help="Skip the label-shuffle control probes. Not advised: at "
                        "step 0 the control is what separates a real fit from "
                        "the probe memorising noise.")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells that already have results")
    p.add_argument("--help-agreement", action="store_true",
                   help="Print the cross-model-agreement follow-up and exit")
    args = p.parse_args()

    if args.help_agreement:
        print(AGREEMENT_HELP)
        return

    sel = set(args.models)
    babylm_cells = [c for c in BABYLM_CELLS
                    if "babylm" in sel or f"babylm-{c['flag']}" in sel]
    pythia_ids = [m for m in PYTHIA_IDS
                  if "pythia" in sel or f"pythia-{m.split('-')[-1]}" in sel]

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    banner("CHECKPOINT-0 BASELINE")
    print(f"  BabyLM (reconstructed init): {[c['flag'] for c in babylm_cells]}",
          flush=True)
    print(f"  Pythia (published step0):    {pythia_ids}", flush=True)
    print(f"  embeddings dir: {emb_dir}", flush=True)
    print(f"  controls: {'skipped' if args.skip_controls else 'on'}", flush=True)

    if pythia_ids:
        pythia_ids = preflight_pythia(pythia_ids)

    init_paths = build_babylm_inits(babylm_cells, args.force) if babylm_cells else {}

    t0 = time.perf_counter()
    failures = []

    # BabyLM: local random-init dirs, explicit slug.
    for c in babylm_cells:
        model = {"id": str(init_paths[c["flag"]]),
                 "n_layers": c["n_layers"],
                 "batch_size": c["batch_size"],
                 "shard_pairs": c["shard_pairs"],
                 "large_gpu": c["large_gpu"]}
        err = rsm.run_model(model, args.gpu, emb_dir, args.skip_controls,
                            args.force, revision=None, slug_override=c["slug"])
        if err:
            failures.append((c["slug"], err))

    # Pythia: published step0 revision, slug derived the same way as elsewhere.
    for mid in pythia_ids:
        model = next(m for m in rsm.MODELS if m["id"] == mid)
        err = rsm.run_model(model, args.gpu, emb_dir, args.skip_controls,
                            args.force, revision="step0")
        if err:
            failures.append((f"{rsm.model_slug(mid)}_step0", err))

    banner("CHECKPOINT-0 BASELINE COMPLETE")
    print(f"  Total: {_fmt_h(time.perf_counter() - t0)}", flush=True)

    if failures:
        print("\n  FAILURES:", flush=True)
        for slug, err in failures:
            print(f"    {slug}  at {err}", flush=True)

    summarise([c["slug"] for c in babylm_cells]
              + [f"{rsm.model_slug(m)}_step0" for m in pythia_ids])

    print("\n  Compare against the earliest TRAINED checkpoint in the paper:",
          flush=True)
    print("    BabyLM pair_novel 0.65-0.72 at ~20M tokens;  "
          "Pythia pair_novel 0.68-0.74 at 33.6M tokens.", flush=True)
    print("\n  Then the follow-up:  "
          "python Scripts/pipeline/run_step0_baseline.py --help-agreement",
          flush=True)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
