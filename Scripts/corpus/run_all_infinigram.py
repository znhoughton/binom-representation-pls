"""
Run all four infini-gram query jobs sequentially:
    1. corpus_binomials    × Dolma  (~11h)   ← priority
    2. corpus_binomials    × Pile   (~11h)
    3. wikipedia_novel     × Dolma  (~75h)
    4. wikipedia_novel     × Pile   (~75h)

Each job has built-in resume logic, so this script is safe to interrupt and
restart — already-completed pairs are skipped automatically.
"""

import subprocess
import sys
import time
from pathlib import Path

PY      = sys.executable
SCRIPT  = Path(__file__).parent / "query_infinigram.py"
BASE    = Path(__file__).parent.parent
CORPUS  = BASE / "Data" / "wikipedia_novel_binomials.csv"  # 340k novel pairs
NOVEL   = BASE / "Data" / "corpus_binomials.csv"           # 49k corpus pairs

JOBS = [
    {
        "label":  "corpus × Dolma",
        "args":   ["--index", "v4_dolma-v1_7_llama"],
    },
    {
        "label":  "corpus × Pile",
        "args":   ["--index", "v4_piletrain_llama"],
    },
    {
        "label":  "novel × Dolma",
        "args":   [
            "--input",  str(BASE / "Data" / "wikipedia_novel_binomials.csv"),
            "--index",  "v4_dolma-v1_7_llama",
            "--output", str(BASE / "Results" / "novel_binomials_infinigram_dolma.csv"),
        ],
    },
    {
        "label":  "novel × Pile",
        "args":   [
            "--input",  str(BASE / "Data" / "wikipedia_novel_binomials.csv"),
            "--index",  "v4_piletrain_llama",
            "--output", str(BASE / "Results" / "novel_binomials_infinigram_pile.csv"),
        ],
    },
]


def run_job(job: dict, job_num: int, total: int):
    print(f"\n{'='*60}", flush=True)
    print(f"Job {job_num}/{total}: {job['label']}", flush=True)
    print(f"{'='*60}", flush=True)
    cmd = [PY, str(SCRIPT)] + job["args"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: job exited with code {result.returncode}", flush=True)
    return result.returncode


def main():
    print(f"Starting all {len(JOBS)} infini-gram jobs.", flush=True)
    print(f"Each job resumes automatically if previously interrupted.\n", flush=True)

    t0 = time.time()
    for i, job in enumerate(JOBS, 1):
        run_job(job, i, len(JOBS))

    elapsed = (time.time() - t0) / 3600
    print(f"\nAll jobs complete. Total elapsed: {elapsed:.1f}h", flush=True)


if __name__ == "__main__":
    main()
