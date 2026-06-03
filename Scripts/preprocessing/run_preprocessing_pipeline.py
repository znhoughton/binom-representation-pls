"""
run_preprocessing_pipeline.py
-----------------------------------
Run all data preparation steps, from raw corpora to extracted embeddings.
Only needs to be run once; outputs are stored in Data/.

Steps:
  1. extract_corpus_binomials.py   extract N+N binomials from BabyLM corpus
  2. extract_wikipedia_binomials.py extract novel pairs from Wikipedia
  3. filter_strict_nouns_v3.py     filter out verb-ambiguous words via WordNet
  4. score_and_extract.py          score corpus pairs + extract diff_vecs
  5. score_and_extract_novel.py    score novel pairs  + extract diff_vecs

After these steps complete, run Scripts/run_full_pipeline.py for analysis.

Note: Steps 1-3 have no --slug argument (they produce corpus-level data).
      Steps 4-5 take --models (HuggingFace model IDs, not local slugs).

Usage:
  python Scripts/preprocessing/run_preprocessing_pipeline.py
  python Scripts/preprocessing/run_preprocessing_pipeline.py --steps score score_novel
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

BASE    = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls")
PYTHON  = r"C:\Users\zacha\anaconda3\envs\PRenv\python.exe"
HERE    = Path(__file__).parent

ALL_MODELS = [
    "znhoughton/opt-babylm-125m-20eps-seed964",
    "znhoughton/opt-babylm-350m-20eps-seed964",
    "znhoughton/opt-babylm-1.3b-20eps-seed964",
]

# (step_name, script, takes_models_arg)
STEPS = [
    ("extract_corpus",   HERE / "extract_corpus_binomials.py",   False),
    ("extract_wiki",     HERE / "extract_wikipedia_binomials.py", False),
    ("filter_nouns",     HERE / "filter_strict_nouns_v3.py",      False),
    ("score",            HERE / "score_and_extract.py",           True),
    ("score_novel",      HERE / "score_and_extract_novel.py",     True),
]


def run(cmd, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {label} exited with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--steps",  nargs="+", default=None)
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

    for name, script, takes_models in STEPS:
        if name not in active:
            continue
        if takes_models:
            run([PYTHON, str(script), "--models"] + args.models, script.name)
        else:
            run([PYTHON, str(script)], script.name)

    print("\nPreprocessing complete. Run Scripts/run_full_pipeline.py next.")


if __name__ == "__main__":
    main()
