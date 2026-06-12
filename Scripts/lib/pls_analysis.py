"""
pls_analysis.py
---------------
Fit PLS (K=15) on corpus diff_vecs; apply frozen projection to novel pairs.

New args
--------
  --embed-dir-corpus   path to dir containing layer_*.npz for corpus
                       (default: Data/embeddings/{slug})
  --embed-dir-novel    path to dir containing layer_*.npz for novel
                       (default: Data/novel_embeddings/{slug})
  --out-dir            output directory
                       (default: Results/{slug}/layer_{layer})
  --control            Hewitt & Liang control: permute preference labels
                       before fitting; outputs prefixed with 'control_'

Outputs
-------
  {out_dir}/corpus_pls_scores.csv
  {out_dir}/novel_pls_scores.csv
  (control run: control_corpus_pls_scores.csv, control_novel_pls_scores.csv)

Usage
-----
  python Scripts/lib/pls_analysis.py --slug znhoughton_opt-babylm-350m-20eps-seed964
  python Scripts/lib/pls_analysis.py --slug ... --control
  python Scripts/lib/pls_analysis.py --slug ... \\
    --embed-dir-corpus Data/embeddings_isolated/{slug} \\
    --embed-dir-novel  Data/novel_embeddings_isolated/{slug} \\
    --out-dir          Results/{slug}/layer_last_isolated
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
SEED = 964

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--layer", default="last")
parser.add_argument("--embed-dir-corpus", dest="embed_dir_corpus", default=None,
                    help="Dir containing corpus layer_*.npz")
parser.add_argument("--embed-dir-novel",  dest="embed_dir_novel",  default=None,
                    help="Dir containing novel layer_*.npz")
parser.add_argument("--out-dir", dest="out_dir", default=None,
                    help="Output directory")
parser.add_argument("--control", action="store_true",
                    help="Hewitt & Liang control: shuffle labels before fitting")
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
COMP_COLS = [f"C{k+1}" for k in range(K)]

print(f"Slug: {SLUG}  layer: {LAYER}  control: {args.control}  device: {device}")
print(f"Corpus dir : {corpus_dir}")
print(f"Novel dir  : {novel_dir}")
print(f"Output dir : {out_dir}")

corpus_npz = np.load(corpus_dir / NPZ, allow_pickle=True)
X_corpus   = torch.from_numpy(corpus_npz["diff_vecs"].astype(np.float32))
y_corpus   = torch.from_numpy(corpus_npz["preference"].astype(np.float32))
w1_corpus  = corpus_npz["word1"].astype(str)
w2_corpus  = corpus_npz["word2"].astype(str)

novel_npz  = np.load(novel_dir / NPZ, allow_pickle=True)
X_novel    = torch.from_numpy(novel_npz["diff_vecs"].astype(np.float32))
y_novel    = torch.from_numpy(novel_npz["preference"].astype(np.float32))
w1_novel   = novel_npz["word1"].astype(str)
w2_novel   = novel_npz["word2"].astype(str)

print(f"Corpus: {len(y_corpus):,} pairs  dim={X_corpus.shape[1]}")
print(f"Novel : {len(y_novel):,} pairs")

if args.control:
    # Corpus and novel use independent seeds so the shuffles are not correlated.
    # seed_offset=1 for novel matches mlp_comparison.py's convention.
    y_corpus = y_corpus[torch.from_numpy(
        np.random.default_rng(SEED).permutation(len(y_corpus)))]
    y_novel  = y_novel[torch.from_numpy(
        np.random.default_rng(SEED + 1).permutation(len(y_novel)))]

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

df_c = pd.DataFrame(T_corpus.numpy(), columns=COMP_COLS)
df_c["preference"] = y_corpus.numpy()
df_c["word1"] = w1_corpus; df_c["word2"] = w2_corpus
df_c.to_csv(out_dir / f"{prefix}corpus_pls_scores.csv", index=False)

df_n = pd.DataFrame(T_novel.numpy(), columns=COMP_COLS)
df_n["preference"] = y_novel.numpy()
df_n["word1"] = w1_novel; df_n["word2"] = w2_novel
df_n.to_csv(out_dir / f"{prefix}novel_pls_scores.csv", index=False)

print(f"\nSaved {prefix}corpus_pls_scores.csv and {prefix}novel_pls_scores.csv -> {out_dir}")
