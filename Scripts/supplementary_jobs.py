"""
supplementary_jobs.py
---------------------
Emit the job list for supplementary_analyses.sh.

Rather than re-deriving which models, checkpoints and layers the paper covers
(BabyLM log-spaced steps, Pythia step tags, all-layer vs final-layer only), we
read it off what the main pipeline actually produced. Every directory under
Results/ containing by_layer_mlp.csv is a cell reported in the paper, and the
distinct `layer` values in that CSV say how deep it was run.

That guarantees the supplementary analyses replicate the MLP coverage exactly:
  - final checkpoint of all 17 models
  - all layers for BabyLM and Pythia, final layer only for GPT-2 / OLMo / Llama
  - every checkpoint that was run, at whatever depth it was run

Output is TSV on stdout:
    slug <TAB> hf_id <TAB> revision <TAB> n_layers <TAB> batch <TAB> layers

`revision` is "-" when the job is a final checkpoint. `layers` is a
space-separated list.

Usage:
    python Scripts/supplementary_jobs.py
    python Scripts/supplementary_jobs.py --only babylm pythia
    python Scripts/supplementary_jobs.py --final-only
"""
import argparse
import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Cells that exist under Results/ but are not reported in the paper. Matched as
# prefixes, so excluding a base slug also drops its checkpoint cells.
#   OLMo-2-1124-1B : not in run_scale_models.py's registry; leftover from an
#                    earlier sweep, and absent from the behavioural results.
#   pythia-2.8b    : the writeup lists Pythia as 160M/410M/1B only.
# Pass --include-unreported to run them anyway.
DEFAULT_EXCLUDE = ["allenai_OLMo-2-1124-1B", "EleutherAI_pythia-2.8b"]

# Extraction batch sizes, mirroring run_scale_models.py / run_bylayer.py.
BATCH = {
    "opt-babylm-125m": 8192, "opt-babylm-350m": 8192, "opt-babylm-1.3b": 8192,
    "pythia-160m": 8192, "pythia-410m": 8192, "pythia-1b": 8192, "pythia-2.8b": 6144,
    "gpt2": 8192, "gpt2-medium": 8192, "gpt2-large": 8192, "gpt2-xl": 6144,
    "OLMo-1B-hf": 512, "OLMo-2-0425-1B": 512,
    "OLMo-7B-hf": 4096, "OLMo-2-1124-7B": 4096,
    "Llama-3.2-1B": 8192, "Meta-Llama-3-8B": 4096,
}


def parse_slug(slug):
    """slug -> (hf_id, revision, family). revision is None for final checkpoints.

    BabyLM checkpoint tags are 'step-N' (hyphen); Pythia's are 'stepN' (no
    hyphen). The BabyLM 1.3B slug is also inconsistent between the final model
    (`1_3b`) and its checkpoints (`1.3b`), so both are normalised here.
    """
    m = re.search(r"_step(\d+)$", slug)
    step = int(m.group(1)) if m else None
    base = slug[: m.start()] if m else slug

    if base.startswith("znhoughton_"):
        name = base[len("znhoughton_"):].replace("1_3b", "1.3b")
        return f"znhoughton/{name}", (f"step-{step}" if step else None), "babylm"
    if base.startswith("EleutherAI_"):
        return f"EleutherAI/{base[len('EleutherAI_'):]}", (f"step{step}" if step else None), "pythia"
    if base.startswith("allenai_"):
        return f"allenai/{base[len('allenai_'):]}", (f"step{step}" if step else None), "olmo"
    if base.startswith("meta-llama_"):
        return f"meta-llama/{base[len('meta-llama_'):]}", (f"step{step}" if step else None), "llama"
    if base.startswith("gpt2"):
        return base, (f"step{step}" if step else None), "gpt2"
    return base, (f"step{step}" if step else None), "other"


def batch_for(hf_id):
    for key, b in BATCH.items():
        if key in hf_id:
            return b
    return 4096


def layers_from_csv(path):
    """Distinct layer indices the MLP was actually run at, sorted."""
    seen = set()
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    seen.add(int(row["layer"]))
                except (KeyError, ValueError):
                    continue
    except OSError:
        return []
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to families: babylm pythia gpt2 olmo llama")
    ap.add_argument("--final-only", action="store_true", dest="final_only",
                    help="skip checkpoint cells")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="additional slugs (or slug prefixes) to drop")
    ap.add_argument("--include-unreported", action="store_true",
                    dest="include_unreported",
                    help=f"also run cells excluded by default: {DEFAULT_EXCLUDE}")
    ap.add_argument("--results-dir", default=None, dest="results_dir")
    args = ap.parse_args()

    exclude = list(args.exclude)
    if not args.include_unreported:
        exclude += DEFAULT_EXCLUDE

    res = Path(args.results_dir) if args.results_dir else BASE / "Results"
    rows = []
    for csv_path in sorted(res.glob("*/by_layer_mlp.csv")):
        slug = csv_path.parent.name
        if any(slug.startswith(x) for x in exclude):
            continue
        hf_id, revision, family = parse_slug(slug)
        if args.only and family not in args.only:
            continue
        if args.final_only and revision is not None:
            continue
        layers = layers_from_csv(csv_path)
        if not layers:
            continue
        rows.append((slug, hf_id, revision or "-", max(layers),
                     batch_for(hf_id), " ".join(str(x) for x in layers)))

    for r in rows:
        print("\t".join(str(x) for x in r))


if __name__ == "__main__":
    main()
