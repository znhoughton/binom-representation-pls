"""
pilot_behavioral_ushape.py
--------------------------
Behavioral analogue of the frequency-residual (U-shape) analysis.

THE PROBE VERSION (existing, Exp 1-3)
    1. train MLP on NOVEL pairs' representations  -> a model of pure abstraction
    2. apply it to ATTESTED pairs                 -> y_pred
    3. residual = |y_true - y_pred|
    4. regress residual on log freq + log freq^2  -> beta2 > 0 means U-shaped,
       i.e. the abstraction model fails hardest at the high-frequency tail,
       which is where pair-specific memorization lives.

THE BEHAVIORAL ANALOGUE (this script)
    Identical logic, identical regression, but the abstraction model is built
    from the model's OWN PREFERENCES on novel pairs rather than from its
    representations. No probe, no hidden states.

    1. fit an additive per-word model on NOVEL pairs:  y_true ~ b_A - b_B
    2. apply it to ATTESTED pairs                   -> y_hat
    3. residual = |y_true - y_hat|
    4. same regression                              -> beta2_behavioral

Both beta2 values are computed here with the same estimator on the same pairs,
so they are directly comparable. Convergence across models is the payoff:
if the probe says a model memorizes heavily and the behavioral measure says the
same, the two instruments are tracking one underlying quantity.

Usage:
    python Scripts/pilot_behavioral_ushape.py --models EleutherAI_pythia-1b ...
    python Scripts/pilot_behavioral_ushape.py --all
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

BASE = Path(__file__).resolve().parents[1]
SEED = 964
DAMP = 3.0          # ridge damping for the additive fit (tuned in earlier pilot)
MIN_FREQ = 1        # attested pairs need freq >= this for log()


# ── data ─────────────────────────────────────────────────────────────────────

def load_attested(model):
    """Attested pairs: y_true (behavior) and y_pred (novel-trained probe)."""
    p = BASE / "Results" / model / "by_layer_corpus_pred.csv.xz"
    if not p.exists():
        return None
    d = pd.read_csv(p, usecols=["condition", "layer", "mode", "word1", "word2",
                                "y_true", "y_pred"])
    d = d[d["mode"] == "mean_pooled"]
    if d.empty:
        return None
    d = d[d.layer.astype(int) == d.layer.astype(int).max()]
    d["word1"] = d.word1.astype(str)
    d["word2"] = d.word2.astype(str)
    return d[["condition", "word1", "word2", "y_true", "y_pred"]]


def load_novel(model, condition):
    """Novel-pair y_true. Lives in cv_preds NPZs; y_true is constant across
    layer/mode/split, so any matching file will do."""
    pat = str(BASE / "Results" / model / "cv_preds" /
              f"{condition}_layer*_mean_pooled_pair_novel.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    z = np.load(files[-1], allow_pickle=True)
    return pd.DataFrame({"word1": z["word1"].astype(str),
                         "word2": z["word2"].astype(str),
                         "y_true": z["y_true"].astype(float)}).dropna()


def load_freq():
    """Binomial counts in the BabyLM corpus (matches the writeup's regression)."""
    f = pd.read_csv(BASE / "Data" / "corpus_binomials.csv",
                    usecols=["word1", "word2", "freq_w1_w2", "freq_w2_w1"])
    f["word1"] = f.word1.astype(str)
    f["word2"] = f.word2.astype(str)
    f["freq"] = f.freq_w1_w2 + f.freq_w2_w1
    return f[["word1", "word2", "freq"]]


# ── the behavioral abstraction model ─────────────────────────────────────────

def fit_additive(novel, damp=DAMP):
    """Fit y_true ~ b_A - b_B on novel pairs. Returns {word: b}.

    This is the behavioral counterpart of training the probe on the novel set:
    a model of what ordering preference is predictable WITHOUT any pair-specific
    information, since these pairs are absent from BabyLM's training corpus.
    """
    words = pd.Index(sorted(set(novel.word1) | set(novel.word2)))
    i1 = words.get_indexer(novel.word1)
    i2 = words.get_indexer(novel.word2)
    n, W = len(novel), len(words)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=int)
    cols[0::2], cols[1::2] = i1, i2
    vals = np.tile([1.0, -1.0], n)
    A = csr_matrix((vals, (rows, cols)), shape=(n, W))
    b = lsqr(A, novel.y_true.values, damp=damp, atol=1e-8, btol=1e-8,
             iter_lim=600)[0]
    return dict(zip(words, b))


# ── the regression both methods share ────────────────────────────────────────

def ushape(resid, freq):
    """|residual| ~ b0 + b1*log(freq) + b2*log(freq)^2.  Returns b2 with CI."""
    lf = np.log(freq)
    X = np.column_stack([np.ones(len(lf)), lf, lf ** 2])
    y = np.asarray(resid, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_v = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid_v @ resid_v) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))[2]
    return beta[2], beta[2] - 1.96 * se, beta[2] + 1.96 * se


def r2(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(np.asarray(a)[m], np.asarray(b)[m])[0, 1] ** 2)


# ── per-model analysis ───────────────────────────────────────────────────────

def run_model(model, freq, verbose=True):
    att = load_attested(model)
    if att is None:
        return []
    out = []
    for cond in sorted(att.condition.unique()):
        a = att[att.condition == cond].merge(freq, on=["word1", "word2"], how="inner")
        a = a[a.freq >= MIN_FREQ].dropna(subset=["y_true", "y_pred", "freq"])
        nov = load_novel(model, cond)
        if nov is None or len(a) < 500:
            continue

        b = fit_additive(nov)
        a = a.copy()
        a["y_behav"] = [b.get(w1, 0.0) - b.get(w2, 0.0)
                        for w1, w2 in zip(a.word1, a.word2)]

        a["res_probe"] = (a.y_true - a.y_pred).abs()
        a["res_behav"] = (a.y_true - a.y_behav).abs()

        bp, bp_lo, bp_hi = ushape(a.res_probe, a.freq)
        bb, bb_lo, bb_hi = ushape(a.res_behav, a.freq)

        row = dict(model=model, condition=cond, n=len(a),
                   r2_probe=r2(a.y_pred, a.y_true),
                   r2_behav=r2(a.y_behav, a.y_true),
                   beta2_probe=bp, beta2_probe_lo=bp_lo, beta2_probe_hi=bp_hi,
                   beta2_behav=bb, beta2_behav_lo=bb_lo, beta2_behav_hi=bb_hi)
        out.append(row)
        if verbose:
            print(f"  {model:<46s} {cond:<12s} n={len(a):>6,}  "
                  f"R2 probe={row['r2_probe']:.3f} behav={row['r2_behav']:.3f}  |  "
                  f"b2 probe={bp:+.4f} [{bp_lo:+.3f},{bp_hi:+.3f}]  "
                  f"behav={bb:+.4f} [{bb_lo:+.3f},{bb_hi:+.3f}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="Results/behavioral_ushape.csv")
    args = ap.parse_args()

    if args.all:
        models = sorted(p.parent.name for p in
                        (BASE / "Results").glob("*/by_layer_corpus_pred.csv.xz"))
    elif args.models:
        models = args.models
    else:
        models = ["znhoughton_opt-babylm-125m-20eps-seed964",
                  "znhoughton_opt-babylm-350m-20eps-seed964",
                  "znhoughton_opt-babylm-1_3b-20eps-seed964",
                  "EleutherAI_pythia-160m", "EleutherAI_pythia-410m",
                  "EleutherAI_pythia-1b"]

    print("=" * 100)
    print("BEHAVIORAL ANALOGUE OF THE U-SHAPE (frequency-residual) ANALYSIS")
    print("=" * 100)
    print("  probe    : abstraction model = MLP trained on novel-pair REPRESENTATIONS")
    print("  behav    : abstraction model = additive per-word fit on novel-pair PREFERENCES")
    print("  both     : residual vs attested pairs, regressed on log freq + log freq^2")
    print("  beta2 > 0 means U-shaped error, i.e. memorization at the high-frequency tail\n")

    rows = []
    freq = load_freq()
    for m in models:
        rows += run_model(m, freq)

    if not rows:
        print("no results")
        return
    df = pd.DataFrame(rows)
    (BASE / args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE / args.out, index=False)
    print(f"\nsaved -> {args.out}")

    print("\n" + "=" * 100)
    print("CONVERGENCE: does the behavioral beta2 track the probe beta2 across models?")
    print("=" * 100)
    print("  (each point is a model x condition, not a pair)")
    for cond in sorted(df.condition.unique()):
        s = df[df.condition == cond].dropna(subset=["beta2_probe", "beta2_behav"])
        if len(s) < 3:
            continue
        r = np.corrcoef(s.beta2_probe, s.beta2_behav)[0, 1]
        print(f"  {cond:<14s} n_models={len(s):>3d}   r(beta2_probe, beta2_behav) = {r:+.4f}")
    s = df.dropna(subset=["beta2_probe", "beta2_behav"])
    if len(s) >= 3:
        r = np.corrcoef(s.beta2_probe, s.beta2_behav)[0, 1]
        print(f"  {'pooled':<14s} n={len(s):>3d}   r = {r:+.4f}")

    print("\n  sign agreement: "
          f"{(np.sign(s.beta2_probe) == np.sign(s.beta2_behav)).mean():.1%} "
          f"of model x condition cells agree on the sign of beta2")


if __name__ == "__main__":
    main()
