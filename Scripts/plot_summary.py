import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

base = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls\Results")
out_dir = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls\Plots")

models_map = {
    "OPT-125M": "znhoughton_opt-babylm-125m-20eps-seed964",
    "OPT-350M": "znhoughton_opt-babylm-350m-20eps-seed964",
    "OPT-1.3B": "znhoughton_opt-babylm-1_3b-20eps-seed964",
}
conds = {"Default": "layer_last", "Last-token": "layer_last_last_token", "Isolated": "layer_last_isolated"}
splits_order = ["Transfer", "Pair-CV", "Word-CV\n(novel)", "Corpus\nword-CV", "Word-strict"]
splits_keys = ["transfer", "pair_novel", "word_novel", "word_corpus", "word_strict"]

def get_mlp(d, prefix, inp, split, col):
    fs = d / f"{prefix}mlp_{inp}_{split}_fold_stats.csv"
    if fs.exists():
        rows = list(csv.DictReader(open(fs)))
        if col in rows[0]:
            vals = [float(r[col]) for r in rows]
            return np.mean(vals), np.std(vals, ddof=1)
    f = d / f"{prefix}mlp_{inp}_{split}.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f)))
        if rows and col in rows[-1]:
            return float(rows[-1][col]), 0
    return None, None

def get_pls_r2(d, split_key, split_label):
    if split_label == "Transfer":
        f = d / "novel_pls_scores.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
            pref = np.array([float(r["preference"]) for r in rows])
            pred = np.array([float(r["pls_pred"]) for r in rows])
            ss_res = np.sum((pref - pred)**2)
            ss_tot = np.sum((pref - np.mean(pref))**2)
            return max(1 - ss_res/ss_tot, 0), 0
        return None, None
    elif split_label == "Word-strict":
        f = d / "pls_word_strict.csv"
        if f.exists():
            row = list(csv.DictReader(open(f)))[0]
            return float(row["r2"]), 0
        return None, None
    else:
        summ_map = {"pair_novel": "novel_cv_summary.csv",
                    "word_novel": "novel_wordcv_summary.csv",
                    "word_corpus": "corpus_wordcv_summary.csv"}
        fold_map = {"pair_novel": "novel_cv_fold_stats.csv",
                    "word_novel": "novel_wordcv_fold_stats.csv",
                    "word_corpus": "corpus_wordcv_fold_stats.csv"}
        summ_f = d / summ_map[split_key]
        fold_f = d / fold_map[split_key]
        if summ_f.exists():
            row = list(csv.DictReader(open(summ_f)))[0]
            mean = float(row["cv_r2"])
            sd = 0
            if fold_f.exists():
                rows = list(csv.DictReader(open(fold_f)))
                sd = np.std([float(r["r2"]) for r in rows], ddof=1)
            return mean, sd
        return None, None

probes = ["PLS", "MLP-diff", "MLP-concat"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey='row')

for ci, (cond_label, cond_dir) in enumerate(conds.items()):
    for mi, (model_label, model_slug) in enumerate(models_map.items()):
        ax = axes[ci, mi]
        d = base / model_slug / cond_dir

        vals = {p: [] for p in probes}
        errs = {p: [] for p in probes}

        for si, (skey, slabel_raw) in enumerate(zip(splits_keys, splits_order)):
            slabel = slabel_raw.replace("\n", " ").replace("  ", " ")
            if slabel == "Word-CV (novel)":
                slabel_for_func = "Word-CV (novel)"
            elif slabel == "Corpus word-CV":
                slabel_for_func = "Corpus word-CV"
            else:
                slabel_for_func = slabel

            pls_r2, pls_sd = get_pls_r2(d, skey, slabel_for_func)
            diff_r2, diff_sd = get_mlp(d, "h128_", "diff", skey, "test_r2")
            conc_r2, conc_sd = get_mlp(d, "h128_", "concat", skey, "test_r2")

            vals["PLS"].append(pls_r2 or 0)
            vals["MLP-diff"].append(diff_r2 or 0)
            vals["MLP-concat"].append(conc_r2 or 0)
            errs["PLS"].append(pls_sd or 0)
            errs["MLP-diff"].append(diff_sd or 0)
            errs["MLP-concat"].append(conc_sd or 0)

        x = np.arange(len(splits_order))
        width = 0.25

        for pi, probe in enumerate(probes):
            ax.bar(x + (pi - 1) * width, vals[probe], width,
                   yerr=errs[probe], label=probe, color=colors[pi],
                   capsize=2, error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(splits_order, fontsize=8)
        ax.set_ylim(0, 0.45)

        if ci == 0:
            ax.set_title(model_label, fontsize=12, fontweight="bold")
        if mi == 0:
            ax.set_ylabel(f"{cond_label}\nr²", fontsize=10)
        if ci == 2 and mi == 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2),
                     ncol=3, fontsize=9, frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

fig.suptitle("Probe Performance (r²) Across Conditions and Splits", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(out_dir / "summary_probe_performance.png", dpi=200, bbox_inches="tight")
print(f"Saved to {out_dir / 'summary_probe_performance.png'}")
