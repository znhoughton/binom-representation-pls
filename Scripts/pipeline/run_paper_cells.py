#!/usr/bin/env python3
"""
run_paper_cells.py
------------------
Every probe cell the paper reports, and nothing else.

The cell list is derived from Scripts/PAPER_INVENTORY.md, which was built by
tracing each file Writeup/writeup.qmd actually reads. Two models that exist in
Results/ from earlier exploration are deliberately absent, because tbl-models
does not list them: EleutherAI/pythia-2.8b and allenai/OLMo-2-1124-1B.

Each cell runs extraction for both attention conditions, then the MLP probes
over the full reported grid (2 conditions x 2 modes x 2 splits), the
corpus-frequency predictions the brms fits consume, and the label-shuffle
control, then deletes its embeddings and compresses its predictions. That is
run_scale_models.run_model, reused unchanged apart from the two new arguments.

What is new here
----------------
`--carriers` makes extraction read Data/carrier_sentences.csv, so each pair has
two prefixed occurrences: the first supplies the representations and the
same-context preference, the second supplies the cross-context target.
`--target cross` then has the probe predict that second preference.

That combination is what repairs the measure. Under the old design a randomly
initialised BabyLM-125M scored 0.822 on word_novel/attn_zeroed, higher than any
trained checkpoint, because the probe's input and its target came from the same
forward pass and it was recovering the network's frozen arithmetic. Restricting
to items with real left context and predicting a preference measured elsewhere
puts step 0 at 0.003, its shuffled-label control, while the trained model holds
0.19.

Checkpoint 0 is included as a reported cell. Pythia publishes its
initialisation as revision step0; the BabyLM models have no such tag, so
make_random_init.py reconstructs theirs and this driver probes the local copy.

Usage
-----
  python Scripts/pipeline/run_paper_cells.py --gpu 0
  python Scripts/pipeline/run_paper_cells.py --stage finals
  python Scripts/pipeline/run_paper_cells.py --stage checkpoints --skip-large
  python Scripts/pipeline/run_paper_cells.py --list        # print cells, run nothing
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE))

import make_random_init as mri                      # noqa: E402
import run_scale_models as rsm                      # noqa: E402

PYTHON = sys.executable

# ── The 16 final-checkpoint models of tbl-models ─────────────────────────────
PAPER_FINALS = [
    "znhoughton/opt-babylm-125m-20eps-seed964",
    "znhoughton/opt-babylm-350m-20eps-seed964",
    "znhoughton/opt-babylm-1.3b-20eps-seed964",
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
    "allenai/OLMo-1B-hf", "allenai/OLMo-2-0425-1B",
    "allenai/OLMo-7B-hf", "allenai/OLMo-2-1124-7B",
    "meta-llama/Llama-3.2-1B", "meta-llama/Meta-Llama-3-8B",
]

# The BabyLM finals are registered in run_bylayer.py, not run_scale_models.py,
# and their final-checkpoint slugs spell the 1.3B with an underscore while the
# checkpoint slugs use a dot. Both spellings are load-bearing: Results/ and the
# writeup registries already use them.
BABYLM_FINAL = {
    "znhoughton/opt-babylm-125m-20eps-seed964":
        ("znhoughton_opt-babylm-125m-20eps-seed964", 12, 8192),
    "znhoughton/opt-babylm-350m-20eps-seed964":
        ("znhoughton_opt-babylm-350m-20eps-seed964", 24, 8192),
    "znhoughton/opt-babylm-1.3b-20eps-seed964":
        ("znhoughton_opt-babylm-1_3b-20eps-seed964", 24, 8192),
}

# ── Experiment 3 checkpoints, including step 0 ───────────────────────────────
BABYLM_STEPS = {
    "125m": [0, 24, 48, 144, 384, 912, 2280],
    "350m": [0, 48, 96, 288, 768, 1824, 4560],
    "1.3b": [0, 97, 194, 582, 1455, 3686, 9021],
}
BABYLM_LAYERS = {"125m": 12, "350m": 24, "1.3b": 24}
PYTHIA_STEPS = [0, 16, 32, 64, 256, 512, 1000]
PYTHIA_CKPT = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1b"]

LARGE = {"allenai/OLMo-7B-hf", "allenai/OLMo-2-1124-7B",
         "meta-llama/Meta-Llama-3-8B"}


def babylm_ckpt_slug(flag, step):
    return f"znhoughton_opt-babylm-{flag}-20eps-seed964_step{step}"


def enumerate_cells(stage, skip_large):
    """(label, model dict, revision, slug_override) for every reported cell."""
    cells = []

    if stage in ("all", "finals"):
        for mid in PAPER_FINALS:
            if skip_large and mid in LARGE:
                continue
            if mid in BABYLM_FINAL:
                slug, n_layers, bs = BABYLM_FINAL[mid]
                cells.append((f"final {mid}",
                              {"id": mid, "n_layers": n_layers, "batch_size": bs,
                               "shard_pairs": 500000, "large_gpu": False},
                              None, slug))
            else:
                m = next(x for x in rsm.MODELS if x["id"] == mid)
                cells.append((f"final {mid}", m, None, None))

    if stage in ("all", "checkpoints"):
        for flag, steps in BABYLM_STEPS.items():
            for st in steps:
                slug = babylm_ckpt_slug(flag, st)
                if st == 0:
                    # No step-0 tag exists on the Hub for models trained from
                    # scratch under the HF Trainer, so probe the reconstruction.
                    mid = str(mri.out_dir_for(flag, mri.DEFAULT_SEED))
                    rev = None
                else:
                    mid = f"znhoughton/opt-babylm-{flag}-20eps-seed964"
                    rev = f"step-{st}"
                cells.append((f"babylm-{flag} step{st}",
                              {"id": mid, "n_layers": BABYLM_LAYERS[flag],
                               "batch_size": 8192, "shard_pairs": 500000,
                               "large_gpu": False},
                              rev, slug))
        for mid in PYTHIA_CKPT:
            m = next(x for x in rsm.MODELS if x["id"] == mid)
            for st in PYTHIA_STEPS:
                cells.append((f"{mid} step{st}", m, f"step{st}", None))

    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["all", "finals", "checkpoints"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--target", default="cross", choices=["same", "cross"])
    p.add_argument("--no-carriers", action="store_true", dest="no_carriers",
                   help="Use the old single-sentence item set. Only for "
                        "reproducing the pre-repair numbers.")
    p.add_argument("--skip-large", action="store_true", dest="skip_large",
                   help="Skip the 7B/8B models, which need an A100")
    p.add_argument("--force", action="store_true")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards",
                   help="Split the cell list across this many processes, for "
                        "running on several GPUs at once. Cells are "
                        "independent: each writes its own Results/<slug> and "
                        "Data/embeddings/<slug>, so there is nothing to race "
                        "on. Build the step-0 weights once beforehand with "
                        "make_random_init.py, since every shard would "
                        "otherwise try to write them.")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index",
                   help="Which shard this process handles, 0-based.")
    p.add_argument("--list", action="store_true", dest="just_list")
    p.add_argument("--list-slugs", action="store_true", dest="list_slugs",
                   help="Print one result slug per line and exit. The driver "
                        "script uses this so later stages iterate exactly the "
                        "reported cells rather than globbing Results/, which "
                        "still holds directories from the old item set.")
    args = p.parse_args()

    carriers = not args.no_carriers
    if carriers and not (BASE / "Data" / "carrier_sentences.csv").exists():
        sys.exit("Data/carrier_sentences.csv not found. Run "
                 "Scripts/corpus/harvest_carrier_sentences.py first, "
                 "or pass --no-carriers.")

    cells = enumerate_cells(args.stage, args.skip_large)

    # Interleave rather than taking contiguous blocks. The list is ordered by
    # model, so a block split would hand one process every BabyLM-1.3B
    # checkpoint while another got only 125M cells.
    if args.num_shards > 1:
        cells = [c for i, c in enumerate(cells)
                 if i % args.num_shards == args.shard_index]

    if args.list_slugs:
        for _, model, rev, slug in cells:
            name = slug or (f"{rsm.model_slug(model['id'])}_{rev}" if rev
                            else rsm.model_slug(model["id"]))
            # slug and layer count: the ridge stage needs both and cannot
            # recover the layer count from the directory name.
            print(f"{name}	{model['n_layers']}")
        return

    print(f"{len(cells)} cells  (stage={args.stage}, target={args.target}, "
          f"carriers={carriers})", flush=True)
    for lbl, _, rev, slug in cells:
        print(f"  {lbl:46s} rev={rev or '-':10s} slug={slug or 'auto'}", flush=True)
    if args.just_list:
        return

    # The BabyLM step-0 cells need their weights reconstructed before probing.
    if args.stage in ("all", "checkpoints"):
        need = [f for f in BABYLM_STEPS
                if not (mri.out_dir_for(f, mri.DEFAULT_SEED) / "config.json").exists()]
        if need:
            print(f"\nBuilding random-init weights for {need}", flush=True)
            if subprocess.call([PYTHON, str(HERE / "make_random_init.py"),
                                "--models", *need]) != 0:
                sys.exit("make_random_init.py failed")

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    t0, failures = time.perf_counter(), []
    for i, (lbl, model, rev, slug) in enumerate(cells, 1):
        print(f"\n{'=' * 70}\n  [{i}/{len(cells)}]  {lbl}\n{'=' * 70}", flush=True)
        err = rsm.run_model(model, args.gpu, emb_dir, skip_controls=False,
                            force=args.force, revision=rev, slug_override=slug,
                            carriers=carriers, target=args.target)
        if err:
            failures.append((lbl, err))
            print(f"  !! FAILED at {err}", flush=True)

    el = time.perf_counter() - t0
    print(f"\n{'=' * 70}\n  DONE  {len(cells) - len(failures)}/{len(cells)} cells "
          f"in {el/3600:.1f}h\n{'=' * 70}", flush=True)
    if failures:
        log = BASE / "Results" / "pipeline_errors.log"
        with open(log, "a") as f:
            f.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] run_paper_cells.py\n")
            for lbl, err in failures:
                f.write(f"  {lbl}  at {err}\n")
                print(f"  FAILED: {lbl}  at {err}", flush=True)
        print(f"  see {log}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
