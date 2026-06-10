"""
pls_analysis.py
------------------------------
Fit PLS (K=15) on corpus diff_vecs, project novel pairs with frozen W*/β.

Output (per model slug):
  Results/{slug}/corpus_pls_scores.csv
  Results/{slug}/novel_pls_scores.csv

Usage:
  python Scripts/lib/pls_analysis.py
  python Scripts/lib/pls_analysis.py --slug znhoughton_opt-babylm-350m-20eps-seed964
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pls_utils import nipals_pls, pearsonr, spearmanr, compute_scale, apply_scale, load_device

BASE = Path(__file__).resolve().parents[2]
K    = 15

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--layer", default="last", choices=["last", "second_to_last"])
args = parser.parse_args()

SLUG      = args.slug
LAYER     = args.layer
NPZ_FILE  = f"layer_{LAYER}.npz"
COMP_COLS = [f"C{k+1}" for k in range(K)]
device    = load_device(args.gpu)
out_dir   = BASE / "Results" / SLUG / f"layer_{LAYER}"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Slug: {SLUG}  layer: {LAYER}  device: {device}")

corpus_npz = np.load(BASE / "Data/embeddings" / SLUG / NPZ_FILE, allow_pickle=True)
X_corpus   = torch.from_numpy(corpus_npz["diff_vecs"].astype(np.float32))
y_corpus   = torch.from_numpy(corpus_npz["preference"].astype(np.float32))
w1_corpus  = corpus_npz["word1"].astype(str)
w2_corpus  = corpus_npz["word2"].astype(str)

novel_npz  = np.load(BASE / "Data/novel_embeddings" / SLUG / NPZ_FILE, allow_pickle=True)
X_novel    = torch.from_numpy(novel_npz["diff_vecs"].astype(np.float32))
y_novel    = torch.from_numpy(novel_npz["preference"].astype(np.float32))
w1_novel   = novel_npz["word1"].astype(str)
w2_novel   = novel_npz["word2"].astype(str)

print(f"Corpus: {len(y_corpus):,} pairs  dim={X_corpus.shape[1]}")
print(f"Novel : {len(y_novel):,} pairs")

X_c_sc, mean_c, std_c = compute_scale(X_corpus.to(device))
X_n_sc = apply_scale(X_novel.to(device), mean_c, std_c)

T_corpus, W_star, b = nipals_pls(X_c_sc, y_corpus, K, device)

T_novel = (X_n_sc @ W_star.to(device)).cpu()

y_pred_corpus = T_corpus @ b
y_pred_novel  = T_novel  @ b

corpus_r2 = 1.0 - float((y_corpus - y_pred_corpus).var()) / (float(y_corpus.var()) + 1e-14)
r_novel   = pearsonr(y_novel, y_pred_novel)
rho_novel = spearmanr(y_novel, y_pred_novel)

print(f"\nCorpus R²  : {corpus_r2:.4f}")
print(f"Novel r    : {r_novel:.4f}  r²={r_novel**2:.4f}")
print(f"Novel rho  : {rho_novel:.4f}")
print(f"Transfer   : {r_novel**2 / (corpus_r2 + 1e-14) * 100:.1f}% of corpus R²")

df_c = pd.DataFrame(T_corpus.numpy(), columns=COMP_COLS)
df_c["preference"] = y_corpus.numpy()
df_c["word1"] = w1_corpus; df_c["word2"] = w2_corpus
df_c.to_csv(out_dir / "corpus_pls_scores.csv", index=False)

df_n = pd.DataFrame(T_novel.numpy(), columns=COMP_COLS)
df_n["preference"] = y_novel.numpy()
df_n["word1"] = w1_novel; df_n["word2"] = w2_novel
df_n.to_csv(out_dir / "novel_pls_scores.csv", index=False)

print(f"\nSaved corpus_pls_scores.csv and novel_pls_scores.csv to {out_dir}")
