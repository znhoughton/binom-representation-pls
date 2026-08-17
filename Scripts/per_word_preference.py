"""
per_word_preference.py
----------------------
Is the model's per-word ordering preference derived from word properties, or is
it an arbitrary stored scalar?

Under the attention-zeroed condition the model's preference is exactly
    y_true = g(W1) - g(W2)
an additive function of the two words evaluated separately (see writeup, Exp 1
Methods). So g(w) can be recovered from the model's OWN OUTPUT PROBABILITIES by
fitting an additive per-word model. No probe is involved at any step.

We then ask how much of g-hat is predicted by:
  (a) hand-coded word properties  -- length, syllables, frequency, stress.
      These are NOT derived from ordering data, so they carry the argument.
  (b) distributional vectors       -- word2vec. CONTAMINATED for this purpose:
      a word that tends to precede "and" has that tendency encoded in its
      vector already. Read (b) as a CEILING on how much of g-hat is systematic
      at all, not as evidence of linguistic content.

Signs matter as much as R2. If shorter and more frequent words get higher g-hat,
that reproduces the short-before-long and frequent-first constraints from the
human binomial literature, which arbitrary storage has no reason to do.

Reliability: g-hat is noisy for words appearing in few pairs. Split-half
reliability gives the ceiling that measurement noise imposes, so R2 can be read
against what is explainable rather than against 1.0.

Usage:
    python Scripts/per_word_preference.py
    python Scripts/per_word_preference.py --models znhoughton_opt-babylm-125m-20eps-seed964
    python Scripts/per_word_preference.py --vectors glove-wiki-gigaword-300
    python Scripts/per_word_preference.py --skip-vectors
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

SEED = 964
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "Results"

# Enough pairs per word for g-hat to mean anything.
MIN_APPEARANCES = 5

DEFAULT_MODELS = [
    "znhoughton_opt-babylm-125m-20eps-seed964",
    "znhoughton_opt-babylm-350m-20eps-seed964",
    "znhoughton_opt-babylm-1_3b-20eps-seed964",
    "EleutherAI_pythia-1b",
    "gpt2",
    "meta-llama_Llama-3.1-8B",
]

RIDGE_GRID = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


# ── word properties ────────────────────────────────────────────────────────
def build_property_table(words):
    """Hand-coded per-word properties. None are derived from ordering data."""
    from nltk.corpus import cmudict
    from wordfreq import zipf_frequency

    cmu = cmudict.dict()
    vowel_run = re.compile(r"[aeiouy]+")

    rows = []
    for w in words:
        prons = cmu.get(w.lower())
        if prons:
            ph = prons[0]
            n_syl = sum(1 for p in ph if p[-1].isdigit())
            # primary stress on the first syllable?
            stresses = [p[-1] for p in ph if p[-1].isdigit()]
            initial_stress = int(bool(stresses) and stresses[0] == "1")
            in_cmu = 1
        else:
            # heuristic fallback: count vowel runs
            n_syl = max(1, len(vowel_run.findall(w.lower())))
            initial_stress = 0
            in_cmu = 0
        rows.append(dict(
            word=w,
            n_letters=len(w),
            n_syllables=n_syl,
            initial_stress=initial_stress,
            zipf=zipf_frequency(w.lower(), "en"),
            in_cmu=in_cmu,
        ))
    return pd.DataFrame(rows)


# ── recovering g-hat ───────────────────────────────────────────────────────
def design_matrix(w1_idx, w2_idx, n_words):
    """Antisymmetric design: +1 for the first word, -1 for the second."""
    n = len(w1_idx)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2] = w1_idx
    cols[1::2] = w2_idx
    vals = np.empty(2 * n)
    vals[0::2] = 1.0
    vals[1::2] = -1.0
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n_words))


def fit_g(X, y, lam):
    """Ridge on the additive model. g is identified up to a constant; the
    penalty picks the minimum-norm (centred) solution."""
    n_words = X.shape[1]
    aug = sparse.vstack([X, np.sqrt(lam) * sparse.eye(n_words, format="csr")],
                        format="csr")
    rhs = np.concatenate([y, np.zeros(n_words)])
    return lsqr(aug, rhs, atol=1e-8, btol=1e-8, iter_lim=3000)[0]


def r2(pred, true):
    if np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return np.nan
    return float(np.corrcoef(pred, true)[0, 1] ** 2)


def choose_lambda(X, y, rng, n_folds=5):
    """Pick lambda by held-out-pair CV. A held-out pair is only scorable if
    both its words appear in the training fold."""
    n = X.shape[0]
    fold = rng.integers(0, n_folds, n)
    best, best_lam = -np.inf, RIDGE_GRID[0]
    for lam in RIDGE_GRID:
        scores = []
        for f in range(n_folds):
            tr, te = fold != f, fold == f
            g = fit_g(X[tr], y[tr], lam)
            seen = np.asarray(np.abs(X[tr]).sum(axis=0)).ravel() > 0
            Xte = X[te]
            usable = np.asarray(
                (np.abs(Xte)[:, ~seen]).sum(axis=1)).ravel() == 0
            if usable.sum() < 50:
                continue
            scores.append(r2((Xte[usable] @ g), y[te][usable]))
        if scores and np.nanmean(scores) > best:
            best, best_lam = np.nanmean(scores), lam
    return best_lam, best


def split_half_reliability(X, y, lam, words_keep_mask, rng, n_rep=5):
    """Fit g on two disjoint halves of the pairs; correlate. Squared correlation
    is the share of g-hat variance that is signal rather than estimation noise."""
    n = X.shape[0]
    out = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        ga = fit_g(X[a], y[a], lam)
        gb = fit_g(X[b], y[b], lam)
        m = words_keep_mask & (ga != 0) & (gb != 0)
        if m.sum() > 50:
            out.append(np.corrcoef(ga[m], gb[m])[0, 1])
    if not out:
        return np.nan, np.nan
    r = float(np.mean(out))
    # Spearman-Brown: reliability of the full-data estimate
    full = 2 * r / (1 + r) if r > -1 else np.nan
    return r, full


# ── property / vector regressions ──────────────────────────────────────────
def cv_ridge_r2(Z, y, rng, lam_grid=(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3),
                n_folds=5):
    """Out-of-fold R2 of a ridge regression, lambda chosen inside each fold."""
    n = len(y)
    fold = rng.integers(0, n_folds, n)
    preds = np.full(n, np.nan)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        Ztr, Zte = (Z[tr] - mu) / sd, (Z[te] - mu) / sd
        ybar = y[tr].mean()
        G, c = Ztr.T @ Ztr, Ztr.T @ (y[tr] - ybar)
        eye = np.eye(Z.shape[1])
        # inner split to pick lambda
        inner = rng.random(tr.sum()) < 0.8
        best, best_w = -np.inf, None
        for lam in lam_grid:
            w = np.linalg.solve(Ztr[inner].T @ Ztr[inner] + lam * eye,
                                Ztr[inner].T @ (y[tr][inner] - y[tr][inner].mean()))
            s = r2(Ztr[~inner] @ w, y[tr][~inner])
            if not np.isnan(s) and s > best:
                best, best_w = s, None
                best_lam = lam
        best_lam = best_lam if best_w is None else best_lam
        w = np.linalg.solve(G + best_lam * eye, c)
        preds[te] = Zte @ w + ybar
    return r2(preds[~np.isnan(preds)], y[~np.isnan(preds)])


def ols_report(Z, y, names):
    """Standardised coefficients, so the SIGN of each constraint is readable."""
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    ys = (y - y.mean()) / (y.std() + 1e-9)
    A = np.column_stack([np.ones(len(ys)), Zs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return dict(zip(names, beta[1:])), r2(A @ beta, ys)


# ── per-model pipeline ─────────────────────────────────────────────────────
def load_attested(slug, condition):
    path = RES / slug / "by_layer_corpus_pred.csv"
    if not path.exists():
        path = path.with_suffix(".csv.xz")
        if not path.exists():
            return None
    df = pd.read_csv(path)
    df = df[(df["condition"] == condition) & (df["mode"] == "mean_pooled")]
    if df.empty:
        return None
    df = df[df["layer"] == df["layer"].max()]
    return df.dropna(subset=["word1", "word2", "y_true"])


def load_novel(slug, condition):
    """The novel set, from the probe's CV prediction dumps. y_true here is the
    model's own preference for pairs absent from BabyLM's training corpus, so no
    pair-specific memory can be averaged into g-hat."""
    d = RES / slug / "cv_preds"
    if not d.is_dir():
        return None
    pat = re.compile(rf"^{re.escape(condition)}_layer(\d+)_mean_pooled_pair_novel\.npz$")
    best_layer, best_path = -1, None
    for f in d.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best_layer:
            best_layer, best_path = int(m.group(1)), f
    if best_path is None:
        return None
    z = np.load(best_path, allow_pickle=True)
    return pd.DataFrame(dict(word1=z["word1"], word2=z["word2"],
                             y_true=z["y_true"].astype(float))).dropna()


def run_model(slug, condition, kv, args, rng):
    df = (load_novel(slug, condition) if args.set == "novel"
          else load_attested(slug, condition))
    if df is None or df.empty:
        print(f"  [skip] {slug}: no {args.set} data for {condition}", flush=True)
        return None

    words = pd.unique(pd.concat([df["word1"], df["word2"]], ignore_index=True))
    widx = {w: i for i, w in enumerate(words)}
    X = design_matrix(df["word1"].map(widx).to_numpy(),
                      df["word2"].map(widx).to_numpy(), len(words))
    y = df["y_true"].to_numpy(float)

    counts = np.asarray(np.abs(X).sum(axis=0)).ravel()
    keep = counts >= MIN_APPEARANCES

    lam, cv_pair_r2 = choose_lambda(X, y, rng)
    g = fit_g(X, y, lam)
    r_half, rel_full = split_half_reliability(X, y, lam, keep, rng)

    print(f"  pairs={len(y)}  words={len(words)}  kept={keep.sum()}"
          f"  lambda={lam}  additive-CV R2={cv_pair_r2:.3f}"
          f"  split-half r={r_half:.3f} (reliability={rel_full:.3f})", flush=True)

    props = build_property_table(words[keep]).set_index("word")
    gk = g[keep]
    wk = words[keep]
    props = props.loc[wk]

    prop_names = ["n_letters", "n_syllables", "initial_stress", "zipf"]
    Zp = props[prop_names].to_numpy(float)

    betas, r2_in = ols_report(Zp, gk, prop_names)
    r2_prop_cv = cv_ridge_r2(Zp, gk, rng)

    row = dict(model=slug, item_set=args.set, condition=condition, n_pairs=len(y),
               n_words=len(words), n_words_kept=int(keep.sum()),
               lam=lam, additive_cv_r2=cv_pair_r2,
               split_half_r=r_half, reliability=rel_full,
               r2_properties_in=r2_in, r2_properties_cv=r2_prop_cv,
               **{f"beta_{k}": v for k, v in betas.items()})

    if kv is not None:
        have = np.array([w.lower() in kv.key_to_index for w in wk])
        if have.sum() > 200:
            V = np.vstack([kv[w.lower()] for w in wk[have]])
            row["n_words_with_vector"] = int(have.sum())
            row["r2_vectors_cv"] = cv_ridge_r2(V, gk[have], rng)
            row["r2_combined_cv"] = cv_ridge_r2(
                np.hstack([Zp[have], V]), gk[have], rng)
            # properties alone on the SAME subset, for a fair comparison
            row["r2_properties_cv_vecsubset"] = cv_ridge_r2(Zp[have], gk[have], rng)

    per_word = props.reset_index().assign(g_hat=gk,
                                          n_appearances=counts[keep],
                                          model=slug, condition=condition,
                                          item_set=args.set)
    return row, per_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="*",
                    default=["attn_zeroed", "default"])
    ap.add_argument("--vectors", default="word2vec-google-news-300")
    ap.add_argument("--skip-vectors", action="store_true")
    ap.add_argument("--set", choices=["attested", "novel"], default="attested",
                    help="novel = pairs absent from BabyLM training, so g-hat "
                         "cannot be an average of memorised pair preferences")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    kv = None
    if not args.skip_vectors:
        try:
            import gensim.downloader as api
            print(f"loading vectors: {args.vectors} ...", flush=True)
            kv = api.load(args.vectors)
            print(f"  {len(kv.key_to_index)} vectors, dim {kv.vector_size}",
                  flush=True)
        except Exception as e:
            print(f"  vector load failed ({e}); continuing without", flush=True)

    rows, per_words = [], []
    for cond in args.conditions:
        print(f"\n=== condition: {cond} ===", flush=True)
        for slug in args.models:
            print(f"{slug}", flush=True)
            out = run_model(slug, cond, kv, args, rng)
            if out is None:
                continue
            row, pw = out
            rows.append(row)
            per_words.append(pw)

    if not rows:
        sys.exit("no models produced results")

    res = pd.DataFrame(rows)
    sfx = args.out_suffix or f"_{args.set}"
    res.to_csv(RES / f"per_word_preference{sfx}.csv", index=False)
    pd.concat(per_words, ignore_index=True).to_csv(
        RES / f"per_word_g_hat{sfx}.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n" + "=" * 78)
    print("HOW MUCH OF g-hat DO WORD PROPERTIES EXPLAIN?")
    print("=" * 78)
    cols = ["model", "item_set", "condition", "n_words_kept", "reliability",
            "r2_properties_cv", "r2_vectors_cv", "r2_combined_cv"]
    print(res[[c for c in cols if c in res.columns]].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("STANDARDISED COEFFICIENTS  (sign is the substantive result)")
    print("  positive beta = higher g-hat = word prefers FIRST position")
    print("  human literature predicts: shorter first (beta_n_letters < 0),")
    print("                             more frequent first (beta_zipf > 0)")
    print("=" * 78)
    bcols = [c for c in res.columns if c.startswith("beta_")]
    print(res[["model", "item_set", "condition", "r2_properties_in"] + bcols]
          .round(3).to_string(index=False))

    print("\nreliability = Spearman-Brown corrected split-half; R2 should be")
    print("read against it, not against 1.0.")
    print(f"\nwrote {RES / 'per_word_preference.csv'}")
    print(f"wrote {RES / 'per_word_g_hat.csv'}")


if __name__ == "__main__":
    main()
