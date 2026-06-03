"""
run_full_pipeline.py
-----------------------------------
Run all post-embedding analysis steps for one or more model slugs.
Assumes embeddings have already been extracted (see Scripts/preprocessing/).

Steps:
  pls          lib/pls_analysis.py           fit PLS, save corpus/novel scores
  cv_pair      lib/cross_validation.py       10-fold pair-level CV (novel)
  cv_word_nov  lib/cross_validation.py       10-fold word-level CV (novel)
  cv_word_cor  lib/cross_validation.py       10-fold word-level CV (corpus)
  features     lib/compute_delta_features.py feature differences CSV
  semantic     lib/semantic_analysis.py      direction + clustering (spaCy)
  freq_stratum lib/frequency_analysis.py     per-stratum PLS → novel transfer
  freq_holdout lib/frequency_analysis.py     low-freq train → high-freq test
  freq_boot    lib/frequency_analysis.py     bootstrap equalized-N stratum
  correlations feature_correlations.R        Pearson/Spearman r table

Usage:
  python Scripts/run_full_pipeline.py
  python Scripts/run_full_pipeline.py --slugs znhoughton_opt-babylm-350m-20eps-seed964
  python Scripts/run_full_pipeline.py --steps pls cv_pair features
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE    = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls")
PYTHON  = r"C:\Users\zacha\anaconda3\envs\PRenv\python.exe"
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
SCRIPTS = BASE / "Scripts"
LIB     = SCRIPTS / "lib"

ALL_SLUGS = [
    "znhoughton_opt-babylm-125m-20eps-seed964",
    "znhoughton_opt-babylm-350m-20eps-seed964",
    "znhoughton_opt-babylm-1_3b-20eps-seed964",
]

# (step_name, script_path, extra_args)
STEPS = [
    ("pls",          LIB / "pls_analysis.py",          []),
    ("cv_pair",      LIB / "cross_validation.py",       ["--mode", "pair_novel"]),
    ("cv_word_nov",  LIB / "cross_validation.py",       ["--mode", "word_novel"]),
    ("cv_word_cor",  LIB / "cross_validation.py",       ["--mode", "word_corpus"]),
    ("features",     LIB / "compute_delta_features.py", []),
    ("semantic",     LIB / "semantic_analysis.py",      []),
    ("freq_stratum", LIB / "frequency_analysis.py",     ["--mode", "stratum"]),
    ("freq_holdout", LIB / "frequency_analysis.py",     ["--mode", "holdout"]),
    ("freq_boot",    LIB / "frequency_analysis.py",     ["--mode", "bootstrap"]),
    ("correlations", SCRIPTS / "feature_correlations.R", None),  # None = R script
]

BLAS_ENV = os.environ.copy()
BLAS_ENV.update({
    "OMP_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS":      "1",
    "NUMEXPR_NUM_THREADS":  "1",
})


def run(cmd, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False, env=BLAS_ENV)
    if result.returncode != 0:
        print(f"ERROR: {label} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slugs", nargs="+", default=ALL_SLUGS)
    parser.add_argument("--steps", nargs="+", default=None,
                        help="Subset of steps by name (e.g. pls cv_pair features)")
    args = parser.parse_args()

    step_names = {name for name, _, _ in STEPS}
    if args.steps:
        unknown = set(args.steps) - step_names
        if unknown:
            print(f"Unknown steps: {unknown}\nValid: {sorted(step_names)}")
            sys.exit(1)
        active = set(args.steps)
    else:
        active = step_names

    for slug in args.slugs:
        print(f"\n{'#'*60}\nSlug: {slug}\n{'#'*60}")

        for name, script, extra in STEPS:
            if name not in active:
                continue

            if extra is None:
                # R script — no --slug, takes slug as positional arg
                run([RSCRIPT, str(script), slug], f"{script.name} {slug}")
            else:
                run([PYTHON, str(script), "--slug", slug] + extra,
                    f"{script.name} {' '.join(extra)} --slug {slug}")

    print("\nAll slugs complete.")


if __name__ == "__main__":
    main()
