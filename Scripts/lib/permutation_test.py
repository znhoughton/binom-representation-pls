"""
permutation_test.py
---------------------------------------
Permutation test for the PLS corpus→novel transfer.

For each of B iterations:
  1. Permute the corpus ordering preferences y
  2. Fit PLS (K=15, NIPALS) on (X_corpus, y_perm)
  3. Apply frozen W*, β to X_novel
  4. Record r² against the *real* y_novel

The observed r² is compared to this null distribution.
p-value = fraction of permuted r² ≥ observed r².

Output (per slug):
  Results/{slug}/permutation_test.csv   — full distribution (B rows)
  Results/{slug}/permutation_summary.csv — observed r², p-value, CI

Usage:
  python Scripts/lib/permutation_test.py
  python Scripts/lib/permutation_test.py --slug znhoughton_opt-babylm-350m-20eps-seed964
  python Scripts/lib/permutation_test.py --B 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pls_utils import nipals_pls, pearsonr, compute_scale, apply_scale, load_device

BASE = Path(__file__).resolve().parents[2]
K    = 15

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--B",     type=int, default=1000)
parser.add_argument("--seed",  type=int, default=964)
parser.add_argument("--layer", default="last")
args = parser.parse_args()

device  = load_device(args.gpu)
out_dir = BASE / "Results" / args.slug / f"layer_{args.layer}"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Slug: {args.slug}  layer: {args.layer}  B={args.B}  device: {device}")

# -- Load data ----------------------------------------------------------------
corpus_npz = np.load(BASE / "Data/embeddings" / args.slug / f"layer_{args.layer}.npz",
                     allow_pickle=True)
X_corpus = torch.from_numpy(corpus_npz["diff_vecs"].astype(np.float32))
y_corpus = torch.from_numpy(corpus_npz["preference"].astype(np.float32))

novel_npz = np.load(BASE / "Data/novel_embeddings" / args.slug / f"layer_{args.layer}.npz",
                    allow_pickle=True)
X_novel = torch.from_numpy(novel_npz["diff_vecs"].astype(np.float32))
y_novel = torch.from_numpy(novel_npz["preference"].astype(np.float32))

print(f"Corpus: {len(y_corpus):,}  Novel: {len(y_novel):,}  dim={X_corpus.shape[1]}")

# -- Observed r² (fit on real y) ----------------------------------------------
X_c_sc, mean_c, std_c = compute_scale(X_corpus)
X_n_sc = apply_scale(X_novel, mean_c, std_c)
_, W_star_obs, b_obs = nipals_pls(X_c_sc, y_corpus, K, device)
y_pred_obs = (X_n_sc.to(device) @ W_star_obs.to(device)).cpu() @ b_obs
r_obs  = pearsonr(y_novel, y_pred_obs)
r2_obs = r_obs ** 2
print(f"\nObserved r²={r2_obs:.4f}")

# -- Permutation distribution -------------------------------------------------
rng = np.random.default_rng(args.seed)
perm_r2 = np.zeros(args.B)

for b in range(args.B):
    if (b + 1) % 100 == 0:
        print(f"  Permutation {b+1}/{args.B}")
    idx    = rng.permutation(len(y_corpus))
    y_perm = y_corpus[torch.from_numpy(idx)]
    X_c_sc_p, mean_p, std_p = compute_scale(X_corpus)
    X_n_sc_p = apply_scale(X_novel, mean_p, std_p)
    _, W_star_p, b_p = nipals_pls(X_c_sc_p, y_perm, K, device)
    y_pred_p = (X_n_sc_p.to(device) @ W_star_p.to(device)).cpu() @ b_p
    perm_r2[b] = pearsonr(y_novel, y_pred_p) ** 2

p_value = float((perm_r2 >= r2_obs).mean())
ci_lo, ci_hi = np.percentile(perm_r2, [2.5, 97.5])

print(f"\nPermutation null: mean={perm_r2.mean():.4f}  "
      f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"Observed r²={r2_obs:.4f}  p={p_value:.4f}")

# -- Save ---------------------------------------------------------------------
pd.DataFrame({"perm_r2": perm_r2}).to_csv(
    out_dir / "permutation_test.csv", index=False)

pd.DataFrame([{
    "observed_r2":    round(r2_obs, 6),
    "perm_mean_r2":   round(perm_r2.mean(), 6),
    "perm_ci_lo":     round(ci_lo, 6),
    "perm_ci_hi":     round(ci_hi, 6),
    "p_value":        round(p_value, 4),
    "B":              args.B,
}]).to_csv(out_dir / "permutation_summary.csv", index=False)

print(f"Saved permutation outputs to {out_dir}")
