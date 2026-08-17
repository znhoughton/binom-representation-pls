"""
supplementary_backfill_markers.py
---------------------------------
Write .done / .done_nosteer markers for cells that already finished, so a run
started before the resume logic existed is not repeated from scratch.

A cell is judged complete by checking linear_probe.csv against the job spec: it
must contain a row for every (condition, layer, mode, split) combination the
cell was supposed to produce. That is exact rather than a row-count heuristic,
so a cell interrupted mid-write is correctly judged incomplete.

  .done_nosteer  written when the probe output is complete
  .done          also written when steering.csv exists and is non-trivial, or
                 when the cell has fewer than 3 layers, in which case
                 supplementary_analyses.sh skips steering by design

Dry run by default; pass --write to create the markers.

Usage:
    python Scripts/supplementary_backfill_markers.py
    python Scripts/supplementary_backfill_markers.py --write
"""
import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_ROOT = BASE / "Data" / "supplementary"

CONDITIONS = ["default", "attn_zeroed"]
MODES = ["mean_pooled", "words_only"]
SPLITS = ["pair_novel", "word_novel"]
MIN_LAYERS_FOR_STEERING = 3


def job_list(extra_args):
    out = subprocess.run(
        [sys.executable, str(BASE / "Scripts" / "supplementary_jobs.py")] + extra_args,
        capture_output=True, text=True, check=True).stdout
    jobs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        slug, hf_id, rev, nlayers, batch, layers = line.split("\t")
        jobs.append((slug, [int(x) for x in layers.split()]))
    return jobs


def probe_complete(cell_dir, layers):
    path = cell_dir / "linear_probe.csv"
    if not path.exists():
        return False, "no linear_probe.csv"
    seen = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                seen.add((row["condition"], int(row["layer"]),
                          row["mode"], row["split"]))
            except (KeyError, ValueError):
                continue
    want = set(itertools.product(CONDITIONS, layers, MODES, SPLITS))
    missing = want - seen
    if missing:
        return False, f"{len(missing)}/{len(want)} rows missing"
    return True, f"all {len(want)} rows present"


def steering_done(cell_dir, layers):
    if len(layers) < MIN_LAYERS_FOR_STEERING:
        return True, "steering not applicable (<3 layers)"
    p = cell_dir / "steering.csv"
    if p.exists() and sum(1 for _ in open(p)) > 1:
        return True, "steering.csv present"
    return False, "no steering.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually create the markers (default is a dry run)")
    ap.add_argument("--jobs-args", nargs=argparse.REMAINDER, default=[],
                    help="passed through to supplementary_jobs.py")
    args = ap.parse_args()

    jobs = job_list(args.jobs_args)
    n_full = n_nosteer = n_incomplete = n_absent = 0

    print(f"{len(jobs)} cells in job list; output root {OUT_ROOT}\n")
    for slug, layers in jobs:
        cell = OUT_ROOT / slug
        if not cell.is_dir():
            n_absent += 1
            continue
        ok, why = probe_complete(cell, layers)
        if not ok:
            print(f"  INCOMPLETE  {slug:<52s} {why}")
            n_incomplete += 1
            continue
        st_ok, st_why = steering_done(cell, layers)
        mark = cell / (".done" if st_ok else ".done_nosteer")
        label = "DONE      " if st_ok else "DONE(-st) "
        print(f"  {label}  {slug:<52s} {why}; {st_why}")
        if st_ok:
            n_full += 1
        else:
            n_nosteer += 1
        if args.write:
            mark.touch()

    print(f"\ncomplete (with steering):    {n_full}")
    print(f"complete (steering skipped): {n_nosteer}")
    print(f"incomplete, will re-run:     {n_incomplete}")
    print(f"never started:               {n_absent}")
    if not args.write:
        print("\nDRY RUN. Re-run with --write to create the markers.")


if __name__ == "__main__":
    main()
