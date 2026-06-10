"""
frequency_analysis.py
---------------------------------------
Frequency-based PLS analyses.

Modes:
  stratum    — fit PLS on each frequency stratum, test transfer to novel pairs
  holdout    — train on low-frequency complement, test on held-out high-freq stratum
  bootstrap  — bootstrap (B=500) equalized-N stratum analysis; controls for
               sample-size differences across strata

Strata based on BabyLM total frequency (freq_w1_w2 + freq_w2_w1):
  freq=1, freq=2-5, freq=6-20, freq>20

Output (per slug):
  stratum_summary.csv, stratum_{name}_{corpus,novel}.csv     (stratum)
  freq_holdout_summary.csv                                   (holdout)
  stratum_bootstrap_{summary,diffs}.csv, stratum_bootstrap_r2.npy  (bootstrap)

Usage:
  python Scripts/lib/frequency_analysis.py --mode stratum
  python Scripts/lib/frequency_analysis.py --mode holdout --slug ...
  python Scripts/lib/frequency_analysis.py --mode bootstrap --B 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pls_utils import nipals_pls, pearsonr, compute_scale, apply_scale, load_device

BASE = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls")
K    = 15

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--mode",  choices=["stratum", "holdout", "bootstrap"], required=True)
parser.add_argument("--B",     type=int, default=500, help="Bootstrap iterations (bootstrap mode only)")
parser.add_argument("--layer", default="last", choices=["last", "second_to_last"])
args = parser.parse_args()

device  = load_device(args.gpu)
out_dir = BASE / "Results" / args.slug / f"layer_{args.layer}"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Slug: {args.slug}  mode: {args.mode}  device: {device}")

# -- Load corpus embeddings and frequencies ----------------------------------
corpus_npz = np.load(BASE / "Data/embeddings" / args.slug / f"layer_{args.layer}.npz", allow_pickle=True)
X_corpus   = torch.from_numpy(corpus_npz["diff_vecs"].astype(np.float32))
y_corpus   = torch.from_numpy(corpus_npz["preference"].astype(np.float32))
w1_corpus  = corpus_npz["word1"].astype(str)
w2_corpus  = corpus_npz["word2"].astype(str)

freq_df    = pd.read_csv(BASE / "Data/corpus_binomials.csv")
freq_df["total_freq"] = freq_df["freq_w1_w2"] + freq_df["freq_w2_w1"]
corpus_df  = pd.DataFrame({"word1": w1_corpus, "word2": w2_corpus})
corpus_df  = corpus_df.merge(freq_df[["word1","word2","total_freq"]], on=["word1","word2"], how="left")
total_freq = corpus_df["total_freq"].fillna(1).values.astype(int)

STRATA = {
    "freq_1":    total_freq == 1,
    "freq_2_5":  (total_freq >= 2) & (total_freq <= 5),
    "freq_6_20": (total_freq >= 6) & (total_freq <= 20),
    "freq_gt20": total_freq > 20,
}
LABELS = ["freq=1", "freq=2-5", "freq=6-20", "freq>20"]


# -- stratum mode ------------------------------------------------------------
def run_stratum():
    novel_npz = np.load(BASE / "Data/novel_embeddings" / args.slug / f"layer_{args.layer}.npz",
                        allow_pickle=True)
    X_novel   = torch.from_numpy(novel_npz["diff_vecs"].astype(np.float32))
    y_novel   = torch.from_numpy(novel_npz["preference"].astype(np.float32))
    w1_novel  = novel_npz["word1"].astype(str)
    w2_novel  = novel_npz["word2"].astype(str)

    comp_cols    = [f"C{k+1}" for k in range(K)]
    summary_rows = []

    for (name, mask), label in zip({**STRATA, "all": np.ones(len(total_freq), dtype=bool)}.items(),
                                   LABELS + ["all"]):
        n = int(mask.sum())
        print(f"\n{'='*50}\nStratum: {name}  n={n:,}")

        X_s = X_corpus[torch.from_numpy(mask)]
        y_s = y_corpus[torch.from_numpy(mask)]

        X_s_sc, mean_s, std_s = compute_scale(X_s)
        X_n_sc = apply_scale(X_novel, mean_s, std_s)

        T_s, W_star, b = nipals_pls(X_s_sc, y_s, K, device)
        T_nov = (X_n_sc.to(device) @ W_star.to(device)).cpu()
        y_pred = T_nov @ b

        r = pearsonr(y_novel, y_pred)
        print(f"  Novel r={r:.4f}  r²={r**2:.4f}")

        df_s = pd.DataFrame(T_s.numpy(), columns=comp_cols)
        df_s["preference"] = y_s.numpy()
        df_s["word1"] = w1_corpus[mask]; df_s["word2"] = w2_corpus[mask]
        df_s.to_csv(out_dir / f"stratum_{name}_corpus.csv", index=False)

        df_n = pd.DataFrame(T_nov.numpy(), columns=comp_cols)
        df_n["preference"] = y_novel.numpy()
        df_n["word1"] = w1_novel; df_n["word2"] = w2_novel
        df_n.to_csv(out_dir / f"stratum_{name}_novel.csv", index=False)

        summary_rows.append({"stratum": name, "label": label, "n_corpus": n,
                             "novel_r": round(r,6), "novel_r2": round(r**2,6)})

    pd.DataFrame(summary_rows).to_csv(out_dir / "stratum_summary.csv", index=False)
    print("\nStratum summary saved.")


# -- holdout mode ------------------------------------------------------------
def run_holdout():
    HOLDOUTS = [
        ("freq>20", total_freq > 20,  total_freq <= 20),
        ("freq>5",  total_freq > 5,   total_freq <= 5),
        ("freq>1",  total_freq > 1,   total_freq == 1),
    ]
    summary_rows = []

    for label, test_mask, train_mask in HOLDOUTS:
        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())
        print(f"\n{'='*50}\nHoldout: {label}  n_train={n_train:,}  n_test={n_test:,}")

        X_tr = X_corpus[torch.from_numpy(train_mask)]
        y_tr = y_corpus[torch.from_numpy(train_mask)]
        X_te = X_corpus[torch.from_numpy(test_mask)]
        y_te = y_corpus[torch.from_numpy(test_mask)]

        X_tr_sc, mean, std = compute_scale(X_tr)
        X_te_sc = apply_scale(X_te, mean, std)

        _, W_star, b = nipals_pls(X_tr_sc, y_tr, K, device)
        y_pred = (X_te_sc.to(device) @ W_star.to(device)).cpu() @ b

        r = pearsonr(y_te, y_pred)
        print(f"  r={r:.4f}  r²={r**2:.4f}")
        summary_rows.append({"holdout": label, "n_train": n_train, "n_test": n_test,
                             "r": round(r,6), "r2": round(r**2,6)})

    pd.DataFrame(summary_rows).to_csv(out_dir / "freq_holdout_summary.csv", index=False)
    print("\nHoldout summary saved.")


# -- bootstrap mode ----------------------------------------------------------
def run_bootstrap():
    novel_npz = np.load(BASE / "Data/novel_embeddings" / args.slug / f"layer_{args.layer}.npz",
                        allow_pickle=True)
    X_novel   = torch.from_numpy(novel_npz["diff_vecs"].astype(np.float32))
    y_novel   = torch.from_numpy(novel_npz["preference"].astype(np.float32))

    B    = args.B
    SEED = 964
    rng  = np.random.default_rng(SEED)
    N    = min(int(m.sum()) for m in STRATA.values())
    print(f"N per bootstrap sample = {N:,}  B={B}")

    stratum_names = list(STRATA.keys())
    stratum_idx   = {name: np.where(mask)[0] for name, mask in STRATA.items()}
    bootstrap_r2  = np.zeros((B, len(STRATA)))

    for b in range(B):
        if (b + 1) % 50 == 0:
            print(f"  Bootstrap {b+1}/{B}")
        for s, name in enumerate(stratum_names):
            idx_boot = rng.choice(stratum_idx[name], size=N, replace=True)
            X_s = X_corpus[torch.from_numpy(idx_boot)]
            y_s = y_corpus[torch.from_numpy(idx_boot)]

            X_s_sc, mean_s, std_s = compute_scale(X_s)
            X_n_sc = apply_scale(X_novel, mean_s, std_s)

            _, W_star, b_coef = nipals_pls(X_s_sc, y_s, K, device)
            y_pred = (X_n_sc.to(device) @ W_star.to(device)).cpu() @ b_coef
            bootstrap_r2[b, s] = pearsonr(y_novel, y_pred) ** 2

    print(f"\nBootstrap summary (B={B}, N={N}):")
    summary_rows, diff_rows = [], []
    for s, (name, label) in enumerate(zip(stratum_names, LABELS)):
        vals = bootstrap_r2[:, s]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {label:12s}  mean_r²={vals.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        summary_rows.append({"stratum": name, "label": label,
                             "n_stratum": len(stratum_idx[name]), "n_sampled": N,
                             "mean_r2": round(vals.mean(),6),
                             "ci_lo": round(lo,6), "ci_hi": round(hi,6)})

    pairs = [("freq_1","freq_2_5"), ("freq_2_5","freq_6_20"),
             ("freq_6_20","freq_gt20"), ("freq_1","freq_gt20")]
    print("\nPairwise differences (lower - higher freq):")
    for a, b_name in pairs:
        sa, sb = stratum_names.index(a), stratum_names.index(b_name)
        diff = bootstrap_r2[:, sa] - bootstrap_r2[:, sb]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        sig = "*" if lo > 0 else ""
        print(f"  {a} - {b_name}: mean={diff.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]  {sig}")
        diff_rows.append({"contrast": f"{a}_vs_{b_name}",
                         "mean_diff": round(diff.mean(),6),
                         "ci_lo": round(lo,6), "ci_hi": round(hi,6),
                         "significant": lo > 0})

    pd.DataFrame(summary_rows).to_csv(out_dir / "stratum_bootstrap_summary.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out_dir / "stratum_bootstrap_diffs.csv", index=False)
    np.save(out_dir / "stratum_bootstrap_r2.npy", bootstrap_r2)
    print(f"\nBootstrap outputs saved to {out_dir}")


if args.mode == "stratum":
    run_stratum()
elif args.mode == "holdout":
    run_holdout()
elif args.mode == "bootstrap":
    run_bootstrap()
