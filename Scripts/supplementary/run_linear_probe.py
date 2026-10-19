#!/usr/bin/env python3
"""
run_linear_probe.py
-------------------
Appendix D: the ridge replication, re-extracting the embeddings it needs.

Why this is separate
--------------------
The ridge probe reads the feature matrix, which only exists in the embedding
npz files, and run_model deletes those at the end of each cell. cv_preds keeps
only y_true and y_pred, so there is no shortcut. Running the ridge probe inside
run_model, before the delete, would have avoided this entirely; it does not,
so these cells have to be extracted again.

What it does NOT re-extract
---------------------------
Only the novel set. The ridge probe cross-validates on novel binomials and
never touches the attested set, which is what --corpus-freq is for, so the
corpus half of each cell is skipped. That is roughly a 10% saving, the novel
set being 216,523 pairs against 27,835.

Folds
-----
Extraction is of the FULL novel set, not a subsample, so the fold assignment is
byte-identical to the one stage 1 used: same seed, same items, same modulo.
That matters because the appendix compares ridge against the MLP numbers
already on disk and claims both used "the same cross-validation folds". A
subsample would break that claim and require refitting the MLP too.

Usage
-----
  # the 16 final models, which is what the appendix text covers
  python Scripts/supplementary/run_linear_probe.py --stage finals --gpu 0

  # a named subset
  python Scripts/supplementary/run_linear_probe.py --slugs gpt2 EleutherAI_pythia-1b

  python Scripts/supplementary/run_linear_probe.py --stage finals --list
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / "Scripts" / "pipeline"))

import run_paper_cells as rpc                      # noqa: E402
import run_scale_models as rsm                     # noqa: E402

PYTHON = sys.executable
CONDS = [("default", "novel_embeddings", "binomial"),
         ("attn_zeroed", "novel_embeddings_attn_zeroed", "attn_zeroed")]


def banner(msg):
    print("\n" + "=" * 70 + "\n  " + msg + "\n" + "=" * 70, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="finals",
                   choices=["finals", "checkpoints", "all"])
    p.add_argument("--slugs", nargs="+", default=None,
                   help="Explicit slugs, overriding --stage")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--keep-embeddings", action="store_true",
                   dest="keep_embeddings",
                   help="Do not delete after probing. Only for debugging: the "
                        "novel set is ~40 GB per cell across both conditions.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true", dest="just_list")
    args = p.parse_args()

    cells = rpc.enumerate_cells(args.stage if args.stage != "finals" else "finals",
                                skip_large=False)
    # (slug, n_layers, model id, revision) per cell
    todo = []
    for _, model, rev, slug in cells:
        name = slug or (f"{rsm.model_slug(model['id'])}_{rev}" if rev
                        else rsm.model_slug(model["id"]))
        todo.append((name, model["n_layers"], model["id"], rev,
                     model.get("batch_size", 8192)))
    if args.slugs:
        want = set(args.slugs)
        todo = [t for t in todo if t[0] in want]

    emb = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    pending = [t for t in todo
               if args.force or not (BASE / "Data" / "supplementary" / t[0]
                                     / "linear_probe.csv").exists()]
    print(f"{len(todo)} cells in scope, {len(pending)} still need probing",
          flush=True)
    for t in pending:
        print(f"  {t[0]}", flush=True)
    if args.just_list:
        return

    t0, failed = time.perf_counter(), []
    for i, (slug, n_layers, mid, rev, bs) in enumerate(pending, 1):
        banner(f"[{i}/{len(pending)}]  {slug}")
        ok = True
        for cond, subdir, ctx in CONDS:
            out = emb / subdir / slug
            cmd = [PYTHON, str(BASE / "Scripts" / "pipeline" / "extract_embeddings.py"),
                   "--model", mid, "--data", "novel", "--carriers",
                   "--layer", "last", "--context", ctx, "--extract", "word",
                   "--batch-size", str(bs), "--gpu", str(args.gpu),
                   "--out", str(out)]
            if rev:
                cmd += ["--revision", rev]
            if subprocess.call(cmd) != 0:
                print(f"  extraction failed: {slug} / {cond}", flush=True)
                ok = False
                break
        if ok:
            rc = subprocess.call(
                [PYTHON, str(BASE / "Scripts" / "supplementary" / "supplementary_probes.py"),
                 "--model-slug", slug, "--num-layers", str(n_layers),
                 "--target", "cross", "--modes", "mean_pooled", "--skip-pls",
                 "--gpu", str(args.gpu), "--embeddings-dir", str(emb)])
            ok = rc == 0
        if not ok:
            failed.append(slug)
        if not args.keep_embeddings:
            for _, subdir, _ in CONDS:
                d = emb / subdir / slug
                if d.exists():
                    shutil.rmtree(d)

    el = (time.perf_counter() - t0) / 3600
    banner(f"DONE  {len(pending) - len(failed)}/{len(pending)} cells in {el:.1f}h")
    if failed:
        for s in failed:
            print(f"  FAILED: {s}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
