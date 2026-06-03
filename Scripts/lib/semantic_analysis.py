"""
semantic_analysis.py
---------------------------------------
Semantic analyses of PLS components using spaCy en_core_web_md vectors.
Runs both analyses in one invocation to avoid loading spaCy twice.

Analysis 1 — direction (semantic_direction):
  For each component Ck, compute the mean W1→W2 direction in top/bottom
  quartile pairs, project all pairs onto that axis, and report Pearson r
  with component score. Also adds pair_cosine_sim to delta_features.csv.

Analysis 2 — clustering (semantic_clustering):
  Per-word W1-bias score per component (mean position difference).
  Tests whether W1-biased and W2-biased words form semantically coherent
  clusters vs. chance (bootstrap z-score against random groups of size N).

Output (per slug):
  semantic_direction.csv
  semantic_clustering.csv
  word_bias_scores.csv
  delta_features.csv (pair_cosine_sim column appended)

Usage:
  python Scripts/lib/semantic_analysis.py
  python Scripts/lib/semantic_analysis.py --slug znhoughton_opt-babylm-350m-20eps-seed964
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from scipy.stats import pearsonr as scipy_pearsonr

BASE = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls")

parser = argparse.ArgumentParser()
parser.add_argument("--slug", default="znhoughton_opt-babylm-125m-20eps-seed964")
args   = parser.parse_args()

out_dir   = BASE / "Results" / args.slug
out_dir.mkdir(parents=True, exist_ok=True)
SEED      = 964
TOP_Q     = 0.25
N_TOP     = 100
MIN_APPEAR = 5
N_BOOTSTRAP = 1000

rng = np.random.default_rng(SEED)
random.seed(SEED)

print(f"Slug: {args.slug}")
print("Loading spaCy en_core_web_md...")
nlp = spacy.load("en_core_web_md")

print("Loading novel PLS scores...")
df = pd.read_csv(out_dir / "novel_pls_scores.csv")
components = [f"C{i}" for i in range(1, 16)]
print(f"  {len(df):,} pairs")

words = list(pd.unique(df[["word1", "word2"]].values.ravel()))
vecs  = {}
for w in words:
    tok = nlp.vocab[w]
    if tok.has_vector:
        v = tok.vector.astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            vecs[w] = v / n
print(f"  {len(vecs):,}/{len(words):,} words have vectors")

# ── Analysis 1: direction ────────────────────────────────────────────────────
print("\n=== Analysis 1: Pair cosine similarity + W1→W2 direction ===")

w1_vecs  = np.stack([vecs.get(w, np.zeros(300, np.float32)) for w in df["word1"]])
w2_vecs  = np.stack([vecs.get(w, np.zeros(300, np.float32)) for w in df["word2"]])
has_both = df["word1"].isin(vecs).values & df["word2"].isin(vecs).values

pair_cos = np.sum(w1_vecs * w2_vecs, axis=1).astype(float)
pair_cos[~has_both] = np.nan
print(f"  Valid pairs: {has_both.sum():,} ({has_both.mean():.1%})")

delta = pd.read_csv(out_dir / "delta_features.csv")
delta["pair_cosine_sim"] = pair_cos
delta.to_csv(out_dir / "delta_features.csv", index=False)

def np_pearsonr(X, y):
    Xc = X - X.mean(0); yc = y - y.mean()
    return (Xc.T @ yc) / (np.sqrt((Xc**2).sum(0)) * np.sqrt((yc**2).sum()) + 1e-12)

mask  = has_both
df_v  = df[mask].reset_index(drop=True)
diff  = w1_vecs[mask] - w2_vecs[mask]

word_list = list(vecs.keys())
word_mat  = np.stack([vecs[w] for w in word_list])

dir_results = []
for c in components:
    scores = df_v[c].values.astype(np.float32)
    k      = int(len(scores) * TOP_Q)
    hi, lo = np.argsort(scores)[-k:], np.argsort(scores)[:k]

    axis = diff[hi].mean(0) - diff[lo].mean(0)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        continue
    axis /= axis_norm

    proj  = diff @ axis
    r     = float(np_pearsonr(proj.reshape(-1,1), scores)[0])
    align = word_mat @ axis
    w1_side = [word_list[i] for i in np.argsort(align)[-10:][::-1]]
    w2_side = [word_list[i] for i in np.argsort(align)[:10]]

    dir_results.append({"component": c, "r_proj_score": r,
                        "w1_pole": ", ".join(w1_side),
                        "w2_pole": ", ".join(w2_side)})
    print(f"  {c}: r={r:.4f}  W1: {', '.join(w1_side[:4])}  ||  W2: {', '.join(w2_side[:4])}")

pd.DataFrame(dir_results).to_csv(out_dir / "semantic_direction.csv", index=False)
print(f"Saved semantic_direction.csv")

# ── Analysis 2: clustering ───────────────────────────────────────────────────
print("\n=== Analysis 2: Per-word semantic clustering ===")

w1_counts = df["word1"].value_counts()
w2_counts = df["word2"].value_counts()
qualified = {w for w in (set(df["word1"]) | set(df["word2"]))
             if w1_counts.get(w,0) + w2_counts.get(w,0) >= MIN_APPEAR}
print(f"  Words with >={MIN_APPEAR} appearances: {len(qualified):,}")

w1_means = df.groupby("word1")[components].mean()
w2_means = df.groupby("word2")[components].mean()

bias_records = []
for w in qualified:
    row = {"word": w}
    for c in components:
        m1 = w1_means.loc[w, c] if w in w1_means.index else np.nan
        m2 = w2_means.loc[w, c] if w in w2_means.index else np.nan
        row[f"bias_{c}"] = (m1 if np.isnan(m2) else
                            -m2 if np.isnan(m1) else m1 - m2)
    bias_records.append(row)

bias_df = pd.DataFrame(bias_records).set_index("word")
bias_df  = bias_df.loc[bias_df.index.isin(vecs)]
words_wv = list(bias_df.index)
V        = np.stack([vecs[w] for w in words_wv])
sim_mat  = V @ V.T
print(f"  {len(words_wv):,} words with vectors and bias scores")

def group_sim(indices):
    n = len(indices)
    if n < 2:
        return np.nan
    sub = sim_mat[np.ix_(indices, indices)]
    return sub[np.triu_indices(n, k=1)].mean()

all_idx       = np.arange(len(words_wv))
baseline_sims = [group_sim(rng.choice(all_idx, N_TOP, replace=False)) for _ in range(N_BOOTSTRAP)]
b_mean, b_sd  = np.mean(baseline_sims), np.std(baseline_sims)
print(f"  Baseline sim (N={N_TOP}): {b_mean:.4f} ± {b_sd:.4f}")

word_idx   = {w: i for i, w in enumerate(words_wv)}
clust_results = []
for c in components:
    col = f"bias_{c}"
    sub = bias_df[col].dropna().sort_values(ascending=False)
    if len(sub) < 2 * N_TOP:
        continue

    w1_words = [w for w in sub.head(N_TOP).index if w in word_idx]
    w2_words = [w for w in sub.tail(N_TOP).index if w in word_idx]
    w1_idx   = [word_idx[w] for w in w1_words]
    w2_idx   = [word_idx[w] for w in w2_words]

    sim_w1 = group_sim(w1_idx);  sim_w2 = group_sim(w2_idx)
    z_w1   = (sim_w1 - b_mean) / (b_sd + 1e-12)
    z_w2   = (sim_w2 - b_mean) / (b_sd + 1e-12)

    bias_vals = bias_df.loc[words_wv, col].values
    w1_cent   = V[w1_idx].mean(0); w1_cent /= np.linalg.norm(w1_cent) + 1e-9
    w1_align  = V @ w1_cent
    finite    = np.isfinite(bias_vals)
    r_bias, _ = scipy_pearsonr(bias_vals[finite], w1_align[finite])

    clust_results.append({
        "component": c, "W1_sim": sim_w1, "W2_sim": sim_w2,
        "baseline_sim": b_mean, "W1_z": z_w1, "W2_z": z_w2,
        "r_bias_v_align": r_bias,
        "W1_words": ", ".join(w1_words[:15]),
        "W2_words": ", ".join(w2_words[:15]),
    })
    print(f"  {c}: W1-sim={sim_w1:.4f}(z={z_w1:+.2f})  W2-sim={sim_w2:.4f}(z={z_w2:+.2f})")

pd.DataFrame(clust_results).to_csv(out_dir / "semantic_clustering.csv", index=False)
bias_df.to_csv(out_dir / "word_bias_scores.csv")
print(f"Saved semantic_clustering.csv and word_bias_scores.csv")
