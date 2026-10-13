#!/usr/bin/env python3
"""
verify_cells.py
---------------
Catch silently duplicated checkpoints.

Why this exists
---------------
Pythia-2.8B's checkpoint prediction files once came out byte-identical across
all six steps: the revision was never applied during extraction, so every
"checkpoint" was really the final model. It was caught by eye, long after the
fact, and the response was to drop that model from the fits
(corpus_relfreq_brms_all.R, "the revision was not applied when they were
extracted"). The model went away; the hazard did not.

The new pipeline runs 42 checkpoint cells, each depending on a revision
resolving correctly. A missing Hub tag raises, so that case is loud. The
dangerous case is quiet: a driver that forgets to pass --revision, or a cache
that serves the wrong weights, yields a complete and plausible-looking cell
holding another checkpoint's numbers.

Two models trained to different points cannot produce identical predictions, so
identical outputs mean the same weights were loaded twice. That is what this
checks, by fingerprinting each cell's y_true.

It also verifies that every reported cell exists, and that step 0 differs from
the trained checkpoints, which is the specific failure that would quietly
restore the artifact the carrier rebuild removed.

Run with base conda, not PRenv (see the spearmanr DLL crash).

Usage
-----
  python Scripts/pipeline/verify_cells.py
  python Scripts/pipeline/verify_cells.py --skip-large
"""

import argparse
import hashlib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RES = BASE / "Results"


def fingerprint(slug):
    """Hash y_true from this cell's saved CV predictions, or None if absent."""
    d = RES / slug / "cv_preds"
    if not d.is_dir():
        return None
    pat = re.compile(r"^default_layer(\d+)_mean_pooled_pair_novel\.npz$")
    best, path = -1, None
    for f in d.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best:
            best, path = int(m.group(1)), f
    if path is None:
        return None
    z = np.load(path, allow_pickle=True)
    y = np.asarray(z["y_true"], dtype=np.float64)
    y = y[np.isfinite(y)]
    return hashlib.sha256(y.tobytes()).hexdigest()[:16] if y.size else None


def condition_parity(slug):
    """Do both attention conditions cover the same items?

    The paper compares default against attn_zeroed directly, and Appendix E
    states the folds are shared across conditions, so the two must be built
    from the same binomials. A condition-specific extraction failure breaks
    that silently: both cells complete, both look plausible, and the contrast
    between them is confounded. This happened once, when an OOM in the
    cross-context pass dropped 2,248 pairs from one attn_zeroed run.

    Returns (n_default, n_attn, n_shared) or None if either is unreadable.
    """
    d = RES / slug / "cv_preds"
    if not d.is_dir():
        return None
    keys = {}
    for cond in ("default", "attn_zeroed"):
        pat = re.compile(rf"^{cond}_layer(\d+)_mean_pooled_pair_novel\.npz$")
        best, path = -1, None
        for f in d.iterdir():
            m = pat.match(f.name)
            if m and int(m.group(1)) > best:
                best, path = int(m.group(1)), f
        if path is None:
            return None
        z = np.load(path, allow_pickle=True)
        keys[cond] = set(np.char.add(np.char.add(
            z["word1"].astype(str), "|"), z["word2"].astype(str)))
    a, b = keys["default"], keys["attn_zeroed"]
    return len(a), len(b), len(a & b)


def series_of(slug):
    """Group checkpoints of one model together; '' for final checkpoints."""
    m = re.match(r"^(.*)_step\d+$", slug)
    return m.group(1) if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-large", action="store_true", dest="skip_large")
    args = p.parse_args()

    cmd = [sys.executable, str(HERE / "run_paper_cells.py"), "--list-slugs"]
    if args.skip_large:
        cmd.append("--skip-large")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    slugs = [ln.split("\t")[0] for ln in out.splitlines() if ln.strip()]
    print(f"{len(slugs)} reported cells\n")

    missing, prints = [], {}
    for s in slugs:
        if not (RES / s / "by_layer_mlp.csv").exists():
            missing.append(s)
            continue
        fp = fingerprint(s)
        if fp is None:
            print(f"  no cv_preds, cannot fingerprint: {s}")
            continue
        prints[s] = fp

    bad = False
    if missing:
        bad = True
        print(f"MISSING {len(missing)} cells:")
        for s in missing:
            print(f"    {s}")
        print()

    # Both attention conditions must cover the same binomials.
    parity_bad = []
    for s_ in sorted(prints):
        pr = condition_parity(s_)
        if pr is None:
            continue
        nd, na, ns = pr
        if not (nd == na == ns):
            parity_bad.append((s_, nd, na, ns))
    if parity_bad:
        bad = True
        print("CONDITION MISMATCH (default vs attn_zeroed cover different items):")
        for s_, nd, na, ns in parity_bad:
            print(f"    {s_}")
            print(f"        default {nd:,}   attn_zeroed {na:,}   shared {ns:,}")
        print("    The attention contrast is confounded. Re-extract with --force.\n")

    # Identical fingerprints within one model's checkpoint series mean the same
    # weights were loaded twice. This is the Pythia-2.8B failure.
    by_series = defaultdict(list)
    for s, fp in prints.items():
        by_series[series_of(s)].append((s, fp))

    for series, items in sorted(by_series.items()):
        if not series or len(items) < 2:
            continue
        seen = defaultdict(list)
        for s, fp in items:
            seen[fp].append(s)
        dupes = {fp: ss for fp, ss in seen.items() if len(ss) > 1}
        if dupes:
            bad = True
            print(f"DUPLICATE CHECKPOINTS in {series}:")
            for fp, ss in dupes.items():
                print(f"    fingerprint {fp} shared by {len(ss)} cells:")
                for s in sorted(ss):
                    print(f"        {s}")
            print("    The revision did not apply. Re-extract with --force.\n")
        else:
            print(f"  ok  {series}: {len(items)} checkpoints, all distinct")

    # A final checkpoint matching one of its own earlier steps is the same bug
    # wearing a different hat.
    for series, items in sorted(by_series.items()):
        if not series:
            continue
        fin = prints.get(series)
        if fin is None:
            continue
        clash = [s for s, fp in items if fp == fin]
        if clash:
            bad = True
            print(f"FINAL MODEL matches a checkpoint in {series}: {clash}")

    # Step 0 is the floor the repaired measure rests on. If it is identical to a
    # trained checkpoint, the floor is not a floor.
    for series, items in sorted(by_series.items()):
        z = [fp for s, fp in items if s.endswith("_step0")]
        if not z:
            continue
        same = [s for s, fp in items if fp == z[0] and not s.endswith("_step0")]
        if same:
            bad = True
            print(f"STEP 0 matches trained checkpoints in {series}: {same}")

    print()
    if bad:
        print("VERIFICATION FAILED")
        sys.exit(1)
    print(f"all {len(prints)} fingerprinted cells are distinct within their series")


if __name__ == "__main__":
    main()
