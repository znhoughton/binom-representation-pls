"""
generate_comparison_plots.py
----------------------------
Pred-vs-observed scatter plots for PLS and MLP-concat across all three models.
Uses last layer for all models.

Outputs (in Plots/):
  pred_vs_obs_pls.png         -- PLS: 3 models × 4 conditions
  pred_vs_obs_mlp_concat.png  -- MLP-concat: 3 models × 4 splits (if preds exist)
  pls_vs_mlp_pair_novel.png   -- Direct PLS vs MLP-concat for pair CV, 3 models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE  = Path(__file__).resolve().parents[1]
LAYER = "last"
COMPS = [f"C{i}" for i in range(1, 16)]
PLOTS = BASE / "Plots"
PLOTS.mkdir(exist_ok=True)

MODELS = [
    ("125m", "znhoughton_opt-babylm-125m-20eps-seed964"),
    ("350m", "znhoughton_opt-babylm-350m-20eps-seed964"),
    ("1.3b", "znhoughton_opt-babylm-1_3b-20eps-seed964"),
]

N_SAMPLE = 5000
rng = np.random.default_rng(964)


def sample(df, n=N_SAMPLE):
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[idx]


def r2_label(y, yhat):
    r2 = float(np.corrcoef(np.asarray(y), np.asarray(yhat))[0, 1] ** 2)
    return f"r² = {r2:.3f}"


def scatter_panel(ax, y, yhat, color, title, row_label=None):
    df_s = sample(pd.DataFrame({"y": np.asarray(y), "yhat": np.asarray(yhat)}))
    ax.scatter(df_s["y"], df_s["yhat"], alpha=0.15, s=2, color=color, rasterized=True)
    m, b = np.polyfit(df_s["y"], df_s["yhat"], 1)
    xs = np.linspace(df_s["y"].min(), df_s["y"].max(), 200)
    ax.plot(xs, m * xs + b, color="firebrick", linewidth=1.2)
    ax.text(0.05, 0.93, r2_label(y, yhat), transform=ax.transAxes,
            fontsize=7.5, va="top", color="firebrick")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    if row_label:
        ax.set_ylabel(row_label, fontsize=8, fontweight="bold")
    else:
        ax.set_ylabel("Predicted", fontsize=7)
    ax.set_xlabel("Observed preference", fontsize=7)


# ── Plot 1: PLS pred vs observed — 3 models × 4 conditions ───────────────────
PLS_CONDITIONS = [
    "Corpus (in-sample)",
    "Novel (transfer)",
    "Novel pair CV",
    "Novel word CV",
]

fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
fig.suptitle("PLS predicted vs. observed preference", fontsize=12, fontweight="bold")

for row, (mlabel, slug) in enumerate(MODELS):
    res = BASE / "Results" / slug / f"layer_{LAYER}"
    corpus = pd.read_csv(res / "corpus_pls_scores.csv")
    novel  = pd.read_csv(res / "novel_pls_scores.csv")

    X_corp = np.c_[np.ones(len(corpus)), corpus[COMPS].values]
    beta   = np.linalg.lstsq(X_corp, corpus["preference"].values, rcond=None)[0]
    corpus["predicted"] = X_corp @ beta
    novel["predicted"]  = np.c_[np.ones(len(novel)), novel[COMPS].values] @ beta

    pair_cv = pd.read_csv(res / "novel_cv_predictions.csv")
    word_cv = pd.read_csv(res / "novel_wordcv_predictions.csv")

    panels = [
        (corpus["preference"], corpus["predicted"]),
        (novel["preference"],  novel["predicted"]),
        (pair_cv["preference"], pair_cv["cv_pred"]),
        (word_cv["preference"], word_cv["cv_pred"]),
    ]

    for col, ((y, yhat), cond_label) in enumerate(zip(panels, PLS_CONDITIONS)):
        ax = axes[row, col]
        row_label = mlabel if col == 0 else None
        scatter_panel(ax, y, yhat, "steelblue", cond_label if row == 0 else "", row_label)

fig.savefig(PLOTS / "pred_vs_obs_pls.png", dpi=150)
plt.close(fig)
print("Saved pred_vs_obs_pls.png")


# ── Plot 2: MLP-concat pred vs observed — 3 models × 4 splits ────────────────
MLP_SPLITS = [
    ("transfer",   "Transfer (corpus→novel)"),
    ("pair_novel", "Pair-level CV"),
    ("word_novel", "Word-level CV"),
    ("word_strict","Word-strict"),
]

# Check which models have MLP preds
available = {}
for mlabel, slug in MODELS:
    res = BASE / "Results" / slug / f"layer_{LAYER}"
    available[mlabel] = {s: (res / f"mlp_concat_{s}_preds.csv").exists()
                         for s, _ in MLP_SPLITS}

any_available = any(any(v.values()) for v in available.values())

if any_available:
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    fig.suptitle("MLP-concat predicted vs. observed preference", fontsize=12, fontweight="bold")

    for row, (mlabel, slug) in enumerate(MODELS):
        res = BASE / "Results" / slug / f"layer_{LAYER}"
        for col, (split, cond_label) in enumerate(MLP_SPLITS):
            ax = axes[row, col]
            f  = res / f"mlp_concat_{split}_preds.csv"
            if f.exists():
                df = pd.read_csv(f)
                row_label = mlabel if col == 0 else None
                scatter_panel(ax, df["preference"], df["mlp_pred"], "darkorange",
                              cond_label if row == 0 else "", row_label)
            else:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(cond_label if row == 0 else "", fontsize=8)
                if col == 0:
                    ax.set_ylabel(mlabel, fontsize=8, fontweight="bold")

    fig.savefig(PLOTS / "pred_vs_obs_mlp_concat.png", dpi=150)
    plt.close(fig)
    print("Saved pred_vs_obs_mlp_concat.png")


# ── Plot 3: PLS vs MLP-concat side-by-side for pair CV — 3 models ────────────
fig, axes = plt.subplots(3, 2, figsize=(9, 12), constrained_layout=True)
fig.suptitle("Pair-level CV: PLS vs MLP-concat", fontsize=12, fontweight="bold")

for row, (mlabel, slug) in enumerate(MODELS):
    res = BASE / "Results" / slug / f"layer_{LAYER}"
    corpus = pd.read_csv(res / "corpus_pls_scores.csv")
    novel  = pd.read_csv(res / "novel_pls_scores.csv")
    X_corp = np.c_[np.ones(len(corpus)), corpus[COMPS].values]
    beta   = np.linalg.lstsq(X_corp, corpus["preference"].values, rcond=None)[0]
    pair_cv = pd.read_csv(res / "novel_cv_predictions.csv")

    # PLS
    ax = axes[row, 0]
    scatter_panel(ax, pair_cv["preference"], pair_cv["cv_pred"], "steelblue",
                  "PLS" if row == 0 else "", mlabel)

    # MLP-concat
    ax = axes[row, 1]
    f = res / "mlp_concat_pair_novel_preds.csv"
    if f.exists():
        df = pd.read_csv(f)
        scatter_panel(ax, df["preference"], df["mlp_pred"], "darkorange",
                      "MLP-concat" if row == 0 else "")
    else:
        ax.text(0.5, 0.5, "pending", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title("MLP-concat" if row == 0 else "", fontsize=8)

fig.savefig(PLOTS / "pls_vs_mlp_pair_novel.png", dpi=150)
plt.close(fig)
print("Saved pls_vs_mlp_pair_novel.png")

print("Done.")
