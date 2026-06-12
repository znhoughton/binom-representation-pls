"""
run_pipeline.py
---------------
Main pipeline entry point: extract embeddings → run analysis.

Extraction conditions
---------------------
  default     binomial context, span mean-pooled        (current behavior)
  last_token  binomial context, last token of span
  isolated    isolated context ("the {word}"), word-pooled

  Each condition produces embeddings in its own directory and results in its own
  Results/{slug}/layer_{layer}[_{condition}]/ subdirectory.

Analysis steps (core)
---------------------
  pls              PLS transfer + corpus/novel projections
  cv_pair          PLS pair-level CV (novel)
  cv_word_nov      PLS word-level CV (novel)
  cv_word_cor      PLS word-level CV (corpus)
  mlp_diff_*       MLP-diff for all four splits
  mlp_concat_*     MLP-concat for all four splits

Analysis steps (auxiliary — default condition only)
-----------------------------------------------------
  features         compute_delta_features.py
  semantic         semantic_analysis.py
  freq_stratum     frequency_analysis.py --mode stratum
  freq_holdout     frequency_analysis.py --mode holdout
  freq_boot        frequency_analysis.py --mode bootstrap
  permutation      permutation_test.py
  correlations     feature_correlations.R  (R script)

Control condition
-----------------
  --run-control adds a Hewitt & Liang control pass for every core analysis step:
  labels are globally shuffled (seed 964) before fitting; outputs are prefixed
  control_ in the same results directory.

Usage
-----
  # Full default pipeline (extract + analyze, all models, both layers):
  python Scripts/run_pipeline.py

  # All extraction conditions + default analysis + control:
  python Scripts/run_pipeline.py --conditions default last_token isolated --run-control

  # Extract new conditions only (no analysis):
  python Scripts/run_pipeline.py --conditions last_token isolated --skip-analysis

  # Analyze only for isolated condition (embeddings already exist):
  python Scripts/run_pipeline.py --conditions isolated --skip-extraction

  # Subset of models/layers/steps:
  python Scripts/run_pipeline.py --models 350m --layers last \\
      --steps pls cv_pair mlp_concat_pair_novel
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE    = Path(__file__).resolve().parents[1]
PYTHON  = sys.executable
RSCRIPT = shutil.which("Rscript") or "Rscript"
SCRIPTS = BASE / "Scripts"
LIB     = SCRIPTS / "lib"

MODELS = {
    "125m": "znhoughton/opt-babylm-125m-20eps-seed964",
    "350m": "znhoughton/opt-babylm-350m-20eps-seed964",
    "1.3b": "znhoughton/opt-babylm-1.3b-20eps-seed964",
}

# Extraction condition → kwargs for extract_embeddings.py
EXTRACT_CONDITIONS = {
    "default": {
        "context": "binomial",
        "extract": "word",
        "pool":    "mean",
    },
    "last_token": {
        "context": "binomial",
        "extract": "last",
    },
    "isolated": {
        "context":  "isolated",
        "extract":  "word",
        "pool":     "mean",
        "template": "the {word}",
    },
}

# Embedding output directories per condition
EMBED_DIRS = {
    "default": {
        "corpus": BASE / "Data" / "embeddings",
        "novel":  BASE / "Data" / "novel_embeddings",
    },
    "last_token": {
        "corpus": BASE / "Data" / "embeddings_last",
        "novel":  BASE / "Data" / "novel_embeddings_last",
    },
    "isolated": {
        "corpus": BASE / "Data" / "embeddings_isolated",
        "novel":  BASE / "Data" / "novel_embeddings_isolated",
    },
}

# Results subdirectory tag per condition (empty string = Results/{slug}/layer_{layer})
CONDITION_TAG = {
    "default":    "",
    "last_token": "_last_token",
    "isolated":   "_isolated",
}

# Core analysis steps: (name, script, extra_args)
CORE_STEPS = [
    ("pls",                    LIB / "pls_analysis.py",   []),
    ("cv_pair",                LIB / "cross_validation.py", ["--mode", "pair_novel"]),
    ("cv_word_nov",            LIB / "cross_validation.py", ["--mode", "word_novel"]),
    ("cv_word_cor",            LIB / "cross_validation.py", ["--mode", "word_corpus"]),
    ("mlp_diff_transfer",      LIB / "mlp_comparison.py",  ["--input", "diff",   "--split", "transfer"]),
    ("mlp_diff_pair_novel",    LIB / "mlp_comparison.py",  ["--input", "diff",   "--split", "pair_novel"]),
    ("mlp_diff_word_novel",    LIB / "mlp_comparison.py",  ["--input", "diff",   "--split", "word_novel"]),
    ("mlp_diff_word_strict",   LIB / "mlp_comparison.py",  ["--input", "diff",   "--split", "word_strict"]),
    ("mlp_concat_transfer",    LIB / "mlp_comparison.py",  ["--input", "concat", "--split", "transfer"]),
    ("mlp_concat_pair_novel",  LIB / "mlp_comparison.py",  ["--input", "concat", "--split", "pair_novel"]),
    ("mlp_concat_word_novel",  LIB / "mlp_comparison.py",  ["--input", "concat", "--split", "word_novel"]),
    ("mlp_concat_word_strict", LIB / "mlp_comparison.py",  ["--input", "concat", "--split", "word_strict"]),
]

# Auxiliary steps (default condition only; no --embed-dir support)
AUX_STEPS = [
    ("features",     LIB / "compute_delta_features.py", []),
    ("semantic",     LIB / "semantic_analysis.py",      []),
    ("freq_stratum", LIB / "frequency_analysis.py",     ["--mode", "stratum"]),
    ("freq_holdout", LIB / "frequency_analysis.py",     ["--mode", "holdout"]),
    ("freq_boot",    LIB / "frequency_analysis.py",     ["--mode", "bootstrap"]),
    ("permutation",  LIB / "permutation_test.py",       []),
    ("correlations", SCRIPTS / "feature_correlations.R", None),  # None = R script
]

ALL_STEP_NAMES = (
    {name for name, _, _ in CORE_STEPS} |
    {name for name, _, _ in AUX_STEPS} |
    {"extract"}
)

BLAS_ENV = os.environ.copy()
BLAS_ENV.update({
    "OMP_NUM_THREADS":      "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS":      "1",
    "NUMEXPR_NUM_THREADS":  "1",
    "PYTHONUNBUFFERED":     "1",
})


def run(cmd, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False, env=BLAS_ENV)
    if result.returncode != 0:
        print(f"ERROR: {label} exited with code {result.returncode}")
        sys.exit(result.returncode)


def slug_for(model_id: str) -> str:
    return model_id.replace("/", "_").replace(".", "_")


def results_dir(slug: str, layer: str, condition: str) -> Path:
    tag = CONDITION_TAG[condition]
    return BASE / "Results" / slug / f"layer_{layer}{tag}"


def run_extraction(model_key, model_id, layer, data_split, condition, gpu, force):
    cond    = EXTRACT_CONDITIONS[condition]
    slug    = slug_for(model_id)
    out_dir = EMBED_DIRS[condition][data_split] / slug

    cmd = [
        PYTHON, str(SCRIPTS / "extract_embeddings.py"),
        "--model", model_id,
        "--data",  data_split,
        "--layer", layer,
        "--out",   str(out_dir),
        "--gpu",   str(gpu),
        "--context", cond["context"],
        "--extract", cond["extract"],
    ]
    if "pool" in cond:
        cmd += ["--pool", cond["pool"]]
    if "template" in cond:
        cmd += ["--template", cond["template"]]
    if force:
        cmd.append("--force")

    if cond["context"] == "isolated":
        pref_src = (EMBED_DIRS["default"][data_split] / slug /
                    f"layer_{layer}.npz")
        cmd += ["--pref-source", str(pref_src)]

    run(cmd, f"extract  model={model_key}  layer={layer}  "
             f"data={data_split}  condition={condition}")


def run_core_analysis(slug, layer, condition, active_steps, gpu, run_control, control_only=False):
    ed_corpus = str(EMBED_DIRS[condition]["corpus"] / slug)
    ed_novel  = str(EMBED_DIRS[condition]["novel"]  / slug)
    out       = str(results_dir(slug, layer, condition))

    for name, script, extra in CORE_STEPS:
        if name not in active_steps:
            continue

        base_cmd = [
            PYTHON, str(script),
            "--slug",             slug,
            "--layer",            layer,
            "--gpu",              str(gpu),
            "--embed-dir-corpus", ed_corpus,
            "--embed-dir-novel",  ed_novel,
            "--out-dir",          out,
        ] + extra

        # Real run (skipped when --control-only)
        if not control_only:
            run(base_cmd, f"{script.name}  {' '.join(extra)}  slug={slug}  "
                          f"layer={layer}  condition={condition}")

        # Control run
        if run_control:
            run(base_cmd + ["--control"],
                f"{script.name}  {' '.join(extra)}  slug={slug}  "
                f"layer={layer}  condition={condition}  [CONTROL]")


def run_aux_analysis(slug, layer, active_steps, gpu):
    """Auxiliary steps only make sense for the default (binomial) condition."""
    for name, script, extra in AUX_STEPS:
        if name not in active_steps:
            continue
        if extra is None:
            # R script: takes slug and layer as positional args
            run([RSCRIPT, str(script), slug, layer],
                f"{script.name}  slug={slug}  layer={layer}")
        else:
            run([PYTHON, str(script),
                 "--slug", slug, "--layer", layer, "--gpu", str(gpu)] + extra,
                f"{script.name}  {' '.join(extra)}  slug={slug}  layer={layer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",       nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--layers",       nargs="+", default=["last", "second_to_last"])
    parser.add_argument("--data",         nargs="+", default=["corpus", "novel"],
                        choices=["corpus", "novel"])
    parser.add_argument("--conditions",   nargs="+", default=["default"],
                        choices=list(EXTRACT_CONDITIONS.keys()))
    parser.add_argument("--steps",        nargs="+", default=None,
                        help="Subset of steps by name, or 'extract' to run only extraction. "
                             f"Valid: {sorted(ALL_STEP_NAMES)}")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-analysis",   action="store_true")
    parser.add_argument("--run-control",     action="store_true",
                        help="Run Hewitt & Liang control alongside each core analysis step")
    parser.add_argument("--control-only",    action="store_true",
                        help="Run ONLY the Hewitt & Liang control (skip real analysis passes)")
    parser.add_argument("--gpu",  type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if output files already exist")
    args = parser.parse_args()

    if args.steps:
        unknown = set(args.steps) - ALL_STEP_NAMES
        if unknown:
            print(f"Unknown steps: {unknown}\nValid: {sorted(ALL_STEP_NAMES)}")
            sys.exit(1)
        active_steps = set(args.steps)
        do_extraction = "extract" in active_steps and not args.skip_extraction
        active_steps.discard("extract")
    else:
        active_steps  = ALL_STEP_NAMES - {"extract"}
        do_extraction = not args.skip_extraction

    core_names = {name for name, _, _ in CORE_STEPS}
    aux_names  = {name for name, _, _ in AUX_STEPS}
    active_core = active_steps & core_names
    active_aux  = active_steps & aux_names

    # ── Extraction ────────────────────────────────────────────────────────────
    if do_extraction:
        for condition in args.conditions:
            for model_key in args.models:
                model_id = MODELS[model_key]
                for data_split in args.data:
                    for layer in args.layers:
                        run_extraction(
                            model_key, model_id, layer, data_split,
                            condition, args.gpu, args.force
                        )

    # ── Analysis ──────────────────────────────────────────────────────────────
    if not args.skip_analysis:
        for condition in args.conditions:
            for model_key in args.models:
                slug = slug_for(MODELS[model_key])
                for layer in args.layers:
                    print(f"\n{'#'*60}\n"
                          f"slug={slug}  layer={layer}  condition={condition}\n"
                          f"{'#'*60}")

                    if active_core:
                        run_core_analysis(
                            slug, layer, condition,
                            active_core, args.gpu,
                            run_control=args.run_control or args.control_only,
                            control_only=args.control_only,
                        )

                    if active_aux and condition == "default":
                        run_aux_analysis(slug, layer, active_aux, args.gpu)
                    elif active_aux and condition != "default":
                        print(f"  (skipping auxiliary steps for condition={condition}; "
                              f"these only apply to 'default')")

    print("\nAll done.")


if __name__ == "__main__":
    main()
