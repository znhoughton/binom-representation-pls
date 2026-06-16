"""
make_plots.py
-------------
Generate predicted-vs-observed scatter plots organized by model size, eval split,
embedding condition, and probe type. Panels with missing results show "pending".

Usage
-----
  python Scripts/make_plots.py              # layer=last
  python Scripts/make_plots.py last
  python Scripts/make_plots.py second_to_last

Output
------
  Plots/pred_vs_obs_by_split.png      - models x eval splits  (MLP-concat, default)
  Plots/pred_vs_obs_by_condition.png  - models x embed conds  (MLP-concat, pair-CV)
  Plots/pred_vs_obs_by_probe.png      - models x probe types  (default, pair-CV/transfer)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    _here = Path(__file__).resolve()
    BASE = next(p for p in [_here.parent, _here.parent.parent]
                if (p / "Results").exists())
except (NameError, StopIteration):
    BASE = Path.cwd()
PLOTS = BASE / "Plots"
PLOTS.mkdir(exist_ok=True)

LAYER = sys.argv[1] if len(sys.argv) > 1 else "last"

MODELS = {
    "125m": "znhoughton_opt-babylm-125m-20eps-seed964",
    "350m": "znhoughton_opt-babylm-350m-20eps-seed964",
    "1.3b": "znhoughton_opt-babylm-1_3b-20eps-seed964",
}

CONDITION_TAG = {
    "default":    "",
    "last_token": "_last_token",
    "isolated":   "_isolated",
}

CONDITION_LABEL = {
    "default":    "Default\n(binomial, mean-pool)",
    "last_token": "Last token\n(binomial, last)",
    "isolated":   "Isolated\n(\"the {word}\")",
}

SPLIT_LABEL = {
    "transfer":    "Transfer\n(corpus→novel)",
    "pair_novel":  "Pair-CV\n(novel 10-fold)",
    "word_novel":  "Word-CV novel\n(word-split)",
    "word_corpus": "Word-CV corpus\n(word-split)",
    "word_strict": "Word-strict\n(corpus->novel strict)",
}

PROBE_LABEL = {
    "pls":        "PLS",
    "mlp_diff":   "MLP-diff",
    "mlp_concat": "MLP-concat",
}

MAX_POINTS = 5_000
RNG = np.random.default_rng(964)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── data loading ──────────────────────────────────────────────────────────────

def _results_dir(slug, condition):
    tag = CONDITION_TAG[condition]
    return BASE / "Results" / slug / f"layer_{LAYER}{tag}"


def load_preds(slug, condition, probe, split):
    """
    Return DataFrame with columns ['obs', 'pred'], or None if file missing.

    probe : 'pls' | 'mlp_diff' | 'mlp_concat'
    split : 'transfer' | 'pair_novel' | 'word_novel' | 'word_corpus' | 'word_strict'
    """
    d = _results_dir(slug, condition)

    if probe == "pls":
        pls_files = {
            "transfer":    ("novel_pls_scores.csv",             "pls_pred"),
            "pair_novel":  ("novel_cv_predictions.csv",         "cv_pred"),
            "word_novel":  ("novel_wordcv_predictions.csv",     "cv_pred"),
            "word_corpus": ("corpus_wordcv_predictions.csv",    "cv_pred"),
            "word_strict": ("pls_word_strict_predictions.csv",  "pls_pred"),
        }
        if split not in pls_files:
            return None
        fname, pred_col = pls_files[split]
        path = d / fname
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if pred_col not in df.columns:
            return None
        return df[["preference", pred_col]].rename(
            columns={"preference": "obs", pred_col: "pred"})

    else:
        input_type = probe.replace("mlp_", "")   # "diff" or "concat"
        path = d / f"mlp_{input_type}_{split}_preds.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "mlp_pred" not in df.columns:
            return None
        return df[["preference", "mlp_pred"]].rename(
            columns={"preference": "obs", "mlp_pred": "pred"})


# ── panel rendering ───────────────────────────────────────────────────────────

def plot_panel(ax, df, xlabel=True, ylabel=True):
    """Scatter of pred vs obs with OLS line and r² annotation."""
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "pending", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888888",
                fontstyle="italic")
        ax.set_xticks([]); ax.set_yticks([])
        return

    df = df.dropna()
    if len(df) < 5:
        ax.text(0.5, 0.5, f"n = {len(df)}\n(too few)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])
        return

    obs  = df["obs"].values
    pred = df["pred"].values
    r, _ = stats.pearsonr(obs, pred)
    r2   = r ** 2

    if len(obs) > MAX_POINTS:
        idx  = RNG.choice(len(obs), MAX_POINTS, replace=False)
        obs_plot, pred_plot = obs[idx], pred[idx]
    else:
        obs_plot, pred_plot = obs, pred

    ax.scatter(obs_plot, pred_plot, alpha=0.12, s=3, color="#2166ac",
               linewidths=0, rasterized=True)

    slope, intercept, *_ = stats.linregress(obs, pred)
    lo, hi = obs.min(), obs.max()
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, slope * xs + intercept, color="#d73027", lw=1.5, zorder=5)

    ax.text(0.05, 0.96, f"r2 ={r2:.3f}", transform=ax.transAxes,
            fontsize=8, va="top", ha="left")

    if xlabel:
        ax.set_xlabel("Observed (log-pref)", fontsize=7)
    if ylabel:
        ax.set_ylabel("Predicted", fontsize=7)
    ax.tick_params(labelsize=7)


def make_grid(nrows, ncols, figsize, row_labels, col_labels, suptitle):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              squeeze=False, constrained_layout=True)
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=9, fontweight="bold", pad=6)
    for i, label in enumerate(row_labels):
        axes[i, 0].annotate(label, xy=(0, 0.5), xytext=(-0.28, 0.5),
                             xycoords="axes fraction", textcoords="axes fraction",
                             ha="right", va="center", fontsize=9, fontweight="bold",
                             rotation=0)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    return fig, axes


# ── Figure 1: models × eval splits (MLP-concat, default) ─────────────────────

def fig_by_split(condition="default", probe="mlp_concat"):
    splits  = ["transfer", "pair_novel", "word_novel", "word_strict", "word_corpus"]
    models  = list(MODELS.keys())
    fig, axes = make_grid(
        nrows=len(models), ncols=len(splits),
        figsize=(3.8 * len(splits), 3.3 * len(models)),
        row_labels=models,
        col_labels=[SPLIT_LABEL[s] for s in splits],
        suptitle=f"{PROBE_LABEL[probe]} ({CONDITION_LABEL[condition].replace(chr(10), ', ')}) "
                  f"— predicted vs observed by eval split  [layer: {LAYER}]",
    )
    for i, model_key in enumerate(models):
        for j, split in enumerate(splits):
            plot_panel(axes[i, j], load_preds(MODELS[model_key], condition, probe, split),
                       xlabel=(i == len(models) - 1), ylabel=(j == 0))
    out = PLOTS / f"pred_vs_obs_by_split_{LAYER}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: models × embedding conditions (MLP-concat, pair-CV) ────────────

def fig_by_condition(split="pair_novel", probe="mlp_concat"):
    conditions = ["default", "last_token", "isolated"]
    models     = list(MODELS.keys())
    fig, axes  = make_grid(
        nrows=len(models), ncols=len(conditions),
        figsize=(3.8 * len(conditions), 3.3 * len(models)),
        row_labels=models,
        col_labels=[CONDITION_LABEL[c] for c in conditions],
        suptitle=f"{PROBE_LABEL[probe]} ({SPLIT_LABEL[split].replace(chr(10), ', ')}) "
                  f"— predicted vs observed by embedding condition  [layer: {LAYER}]",
    )
    for i, model_key in enumerate(models):
        for j, cond in enumerate(conditions):
            plot_panel(axes[i, j], load_preds(MODELS[model_key], cond, probe, split),
                       xlabel=(i == len(models) - 1), ylabel=(j == 0))
    out = PLOTS / f"pred_vs_obs_by_condition_{LAYER}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: models × probe types (default, pair-CV or transfer) ────────────

def fig_by_probe(condition="default"):
    # PLS: show pair-CV (most comparable to MLP CV). Transfer shown separately as col 1.
    probes_splits = [
        ("pls",        "transfer",   "PLS\n(transfer)"),
        ("pls",        "pair_novel", "PLS\n(pair-CV)"),
        ("mlp_diff",   "pair_novel", "MLP-diff\n(pair-CV)"),
        ("mlp_concat", "pair_novel", "MLP-concat\n(pair-CV)"),
        ("mlp_concat", "word_novel", "MLP-concat\n(word-CV novel)"),
    ]
    models = list(MODELS.keys())
    fig, axes = make_grid(
        nrows=len(models), ncols=len(probes_splits),
        figsize=(3.6 * len(probes_splits), 3.3 * len(models)),
        row_labels=models,
        col_labels=[label for _, _, label in probes_splits],
        suptitle=f"Predicted vs observed by probe type "
                  f"({CONDITION_LABEL[condition].replace(chr(10), ', ')})  [layer: {LAYER}]",
    )
    for i, model_key in enumerate(models):
        for j, (probe, split, _) in enumerate(probes_splits):
            plot_panel(axes[i, j], load_preds(MODELS[model_key], condition, probe, split),
                       xlabel=(i == len(models) - 1), ylabel=(j == 0))
    out = PLOTS / f"pred_vs_obs_by_probe_{LAYER}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_by_split()
    fig_by_condition()
    fig_by_probe()
    print("Done.")
