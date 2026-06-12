"""
cross_validation.py
-------------------
10-fold cross-validation for PLS ordering preferences.

Modes
-----
  pair_novel   pair-level CV on novel pairs (random KFold split)
  word_novel   word-level CV on novel pairs (split by unique words)
  word_corpus  word-level CV on corpus pairs (split by unique words)

In word-level CV, a pair is testable only when both words fall in the same
held-out fold (~10% of pairs). This tests generalization to new word pairs.

New args
--------
  --embed-dir-corpus   path to dir containing corpus layer_*.npz
  --embed-dir-novel    path to dir containing novel layer_*.npz
  --out-dir            output directory
  --control            Hewitt & Liang control: shuffle preference labels
                       once globally before CV; outputs prefixed 'control_'

Outputs
-------
  {out_dir}/novel_cv_{predictions,fold_stats,summary}.csv      (pair_novel)
  {out_dir}/novel_wordcv_{predictions,fold_stats,summary}.csv  (word_novel)
  {out_dir}/corpus_wordcv_{predictions,fold_stats,summary}.csv (word_corpus)
  (control runs: control_ prefix on each file)

Usage
-----
  python Scripts/lib/cross_validation.py --mode pair_novel --slug ...
  python Scripts/lib/cross_validation.py --mode word_novel --slug ... --control
  python Scripts/lib/cross_validation.py --mode pair_novel --slug ... \\
    --embed-dir-novel Data/novel_embeddings_isolated/{slug} \\
    --out-dir Results/{slug}/layer_last_isolated
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from pls_utils import nipals_pls, pearsonr, spearmanr, compute_scale, apply_scale, load_device

BASE  = Path(__file__).resolve().parents[2]
K_PLS = 15
FOLDS = 10
SEED  = 964

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--mode",  choices=["pair_novel", "word_novel", "word_corpus"],
                    required=True)
parser.add_argument("--layer", default="last")
parser.add_argument("--embed-dir-corpus", dest="embed_dir_corpus", default=None)
parser.add_argument("--embed-dir-novel",  dest="embed_dir_novel",  default=None)
parser.add_argument("--out-dir", dest="out_dir", default=None)
parser.add_argument("--control", action="store_true",
                    help="Hewitt & Liang control: shuffle labels before CV")
args = parser.parse_args()

SLUG  = args.slug
LAYER = args.layer
NPZ   = f"layer_{LAYER}.npz"

corpus_dir = Path(args.embed_dir_corpus) if args.embed_dir_corpus \
             else BASE / "Data" / "embeddings" / SLUG
novel_dir  = Path(args.embed_dir_novel)  if args.embed_dir_novel  \
             else BASE / "Data" / "novel_embeddings" / SLUG
out_dir    = Path(args.out_dir)          if args.out_dir          \
             else BASE / "Results" / SLUG / f"layer_{LAYER}"
prefix     = "control_" if args.control else ""

out_dir.mkdir(parents=True, exist_ok=True)
device = load_device(args.gpu)
print(f"Slug: {SLUG}  mode: {args.mode}  layer: {LAYER}  "
      f"control: {args.control}  device: {device}")


def _load_npz(path):
    npz = np.load(path, allow_pickle=True)
    X   = torch.from_numpy(npz["diff_vecs"].astype(np.float32))
    y   = torch.from_numpy(npz["preference"].astype(np.float32))
    w1  = npz["word1"].astype(str)
    w2  = npz["word2"].astype(str)
    return X, y, w1, w2


def _shuffle_labels(y: torch.Tensor) -> torch.Tensor:
    perm = np.random.default_rng(SEED).permutation(len(y))
    return y[torch.from_numpy(perm)]


# ── pair-level CV (novel) ─────────────────────────────────────────────────────
def run_pair_novel():
    X, y, w1, w2 = _load_npz(novel_dir / NPZ)
    if args.control:
        y = _shuffle_labels(y)
    print(f"Novel pairs: {len(y):,}  dim={X.shape[1]}")

    kf        = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    all_preds = torch.zeros(len(y))
    fold_rows = []

    for fi, (tr, te) in enumerate(kf.split(X)):
        tr_t = torch.tensor(tr); te_t = torch.tensor(te)
        X_tr_sc, mean, std = compute_scale(X[tr_t])
        X_te_sc = apply_scale(X[te_t], mean, std)
        y_tr = y[tr_t]; y_te = y[te_t]

        _, W_star, b = nipals_pls(X_tr_sc, y_tr, K_PLS, device)
        pred_te = (X_te_sc.to(device) @ W_star.to(device)).cpu() @ b

        all_preds[te_t] = pred_te
        r = pearsonr(y_te, pred_te)
        fold_rows.append({"fold": fi+1, "n_test": len(te), "r": round(r,6), "r2": round(r**2,6)})
        print(f"  Fold {fi+1:2d}: n={len(te):,}  r={r:.4f}  r²={r**2:.4f}")

    r_cv   = pearsonr(y, all_preds)
    rho_cv = spearmanr(y, all_preds)
    print(f"\nOverall CV  r={r_cv:.4f}  r²={r_cv**2:.4f}  rho={rho_cv:.4f}")

    pd.DataFrame({"word1": w1, "word2": w2,
                  "preference": y.numpy(), "cv_pred": all_preds.numpy()}).to_csv(
        out_dir / f"{prefix}novel_cv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{prefix}novel_cv_fold_stats.csv", index=False)
    pd.DataFrame([{"k_folds": FOLDS, "k_pls": K_PLS, "n": len(y),
                   "cv_r": round(r_cv,6), "cv_r2": round(r_cv**2,6),
                   "cv_rho": round(rho_cv,6)}]).to_csv(
        out_dir / f"{prefix}novel_cv_summary.csv", index=False)
    print(f"Saved pair-level CV outputs to {out_dir}")


# ── word-level CV (shared for novel and corpus) ───────────────────────────────
def run_word_cv(data_path, file_prefix):
    X, y, w1, w2 = _load_npz(data_path)
    if args.control:
        y = _shuffle_labels(y)
    rng = np.random.default_rng(SEED)

    all_words = sorted(set(w1) | set(w2))
    shuf      = rng.permutation(len(all_words))
    word_fold = {all_words[i]: int(shuf[i] % FOLDS) for i in range(len(all_words))}

    w1_fold = np.array([word_fold[w] for w in w1])
    w2_fold = np.array([word_fold[w] for w in w2])
    same    = (w1_fold == w2_fold).sum()

    print(f"Pairs: {len(y):,}  dim={X.shape[1]}")
    print(f"Unique words: {len(all_words):,}")
    print(f"Same-fold (testable): {same:,}  ({100*same/len(y):.1f}%)\n")

    all_pred_idx, all_pred_val, fold_rows = [], [], []

    for fk in range(FOLDS):
        test_mask  = (w1_fold == fk) & (w2_fold == fk)
        train_mask = (w1_fold != fk) & (w2_fold != fk)
        n_te, n_tr = test_mask.sum(), train_mask.sum()
        print(f"Fold {fk+1}: train={n_tr:,}  test={n_te:,}  "
              f"excl={len(y)-n_te-n_tr:,}", end="")
        if n_te == 0:
            print("  (skip)"); continue

        X_tr = X[torch.from_numpy(train_mask)]
        X_te = X[torch.from_numpy(test_mask)]
        y_tr = y[torch.from_numpy(train_mask)]
        y_te = y[torch.from_numpy(test_mask)]

        X_tr_sc, mean, std = compute_scale(X_tr)
        X_te_sc = apply_scale(X_te, mean, std)

        _, W_star, b = nipals_pls(X_tr_sc, y_tr, K_PLS, device)
        pred_te = (X_te_sc.to(device) @ W_star.to(device)).cpu() @ b

        r = pearsonr(y_te, pred_te)
        print(f"  r={r:.4f}  r²={r**2:.4f}")
        all_pred_idx.extend(np.where(test_mask)[0].tolist())
        all_pred_val.extend(pred_te.tolist())
        fold_rows.append({"fold": fk+1, "n_train": int(n_tr), "n_test": int(n_te),
                          "r": round(r,6), "r2": round(r**2,6)})

    pred_idx = np.array(all_pred_idx)
    pred_val = torch.tensor(all_pred_val)
    y_tested = y[torch.tensor(pred_idx)]
    r_cv     = pearsonr(y_tested, pred_val)
    rho_cv   = spearmanr(y_tested, pred_val)

    print(f"\nTested: {len(pred_idx):,} / {len(y):,}  ({100*len(pred_idx)/len(y):.1f}%)")
    print(f"Word-level CV  r={r_cv:.4f}  r²={r_cv**2:.4f}  rho={rho_cv:.4f}")

    pd.DataFrame({
        "original_idx": pred_idx, "word1": w1[pred_idx], "word2": w2[pred_idx],
        "preference": y_tested.numpy(), "cv_pred": pred_val.numpy(),
        "fold": [word_fold[w1[i]]+1 for i in pred_idx]
    }).to_csv(out_dir / f"{prefix}{file_prefix}_wordcv_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        out_dir / f"{prefix}{file_prefix}_wordcv_fold_stats.csv", index=False)
    pd.DataFrame([{"k_folds": FOLDS, "k_pls": K_PLS, "n_tested": len(pred_idx),
                   "n_total": len(y), "cv_r": round(r_cv,6), "cv_r2": round(r_cv**2,6),
                   "cv_rho": round(rho_cv,6)}]).to_csv(
        out_dir / f"{prefix}{file_prefix}_wordcv_summary.csv", index=False)
    print(f"Saved word-level CV outputs to {out_dir}")


if args.mode == "pair_novel":
    run_pair_novel()
elif args.mode == "word_novel":
    run_word_cv(novel_dir / NPZ, "novel")
elif args.mode == "word_corpus":
    run_word_cv(corpus_dir / NPZ, "corpus")
