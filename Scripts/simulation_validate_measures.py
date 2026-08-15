"""
simulation_validate_measures.py
-------------------------------
Method validation: do our measures recover what we claim, when ground truth is known?

THREE COMPONENTS, because the two CV splits treat them differently
  alpha  FEATURE-BASED ABSTRACTION.  Ordering derived from word properties
         (length, frequency, phonology...). Transfers to words never seen,
         because the features are what carry it.
  lam    LEXICAL BIAS.  A per-word "goes first" scalar NOT derivable from
         features. Transfers across partners but NOT to novel words: a word
         the probe never saw has no learnable bias.
  mu     PAIR-SPECIFIC MEMORY.  An offset attached to a particular combination,
         present only for pairs the model was exposed to, scaled by exposure.
         Transfers to nothing.

PREDICTED BEHAVIOUR OF EACH MEASURE  (this is what the simulation tests)
  word-held-out R2   should track alpha ONLY. Held-out words have no learnable
                     lexical bias and their pairs have no learnable memory.
  pair-vs-word gap   should track lam. Under pair-held-out the words appeared in
                     training, so their biases are learnable; under word-held-out
                     they are not. The gap is exactly that difference.
  beta2              should track mu. Neither CV split can learn a test pair's
                     memory, so mu is invisible to R2 and shows up instead as
                     exposure-dependent residual error.

If those hold, "word-held-out R2 > 0 means abstraction" stops being an
interpretive assumption and becomes a validated property of the measure. The
sweep additionally gives sensitivity: how large each component must be before
its measure registers it.

Usage:
    python Scripts/simulation_validate_measures.py
    python Scripts/simulation_validate_measures.py --quick
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parents[1]
SEED = 964

# generative sizes
N_WORDS = 600
N_ATTESTED = 6000
N_NOVEL = 6000
D_FEAT = 8         # ordering-relevant word properties
D_REP = 96         # hidden-state dimensionality
NOISE_Y = 0.5
NOISE_H = 0.6


def make_world(rng):
    """Words, their features, and their non-feature lexical biases."""
    feats = rng.normal(size=(N_WORDS, D_FEAT))
    lex = rng.normal(size=N_WORDS)                   # not derivable from feats
    w_abs = rng.normal(size=D_FEAT)                  # the abstract ordering rule
    w_abs /= np.linalg.norm(w_abs)
    # random projections: how the representation encodes first/second position
    P1 = rng.normal(size=(D_FEAT, D_REP)) / np.sqrt(D_FEAT)
    P2 = rng.normal(size=(D_FEAT, D_REP)) / np.sqrt(D_FEAT)
    Q = rng.normal(size=D_REP) / np.sqrt(D_REP)      # SHARED lexical-bias axis
    V = rng.normal(size=D_REP) / np.sqrt(D_REP)      # where pair memory lands
    # word-specific embeddings: arbitrary, unrelated to features. A probe can only
    # map E[w] -> lex[w] for words it has actually trained on.
    E = rng.normal(size=(N_WORDS, D_REP)) / np.sqrt(D_REP)
    return dict(feats=feats, lex=lex, w_abs=w_abs, P1=P1, P2=P2, Q=Q, V=V, E=E)


def make_pairs(world, n, rng, exposed):
    """Draw pairs; assign exposure (0 for the novel set, lognormal otherwise)."""
    i1 = rng.integers(0, N_WORDS, n)
    i2 = rng.integers(0, N_WORDS, n)
    keep = i1 != i2
    i1, i2 = i1[keep], i2[keep]
    expo = (np.exp(rng.normal(4.0, 2.5, len(i1))).astype(int) if exposed
            else np.zeros(len(i1), dtype=int))
    mem = rng.normal(size=len(i1))                   # each pair's idiosyncratic offset
    return i1, i2, expo, mem


def synthesise(world, pairs, alpha, lam, mu, rng):
    """Build probe input X and target y from the three components."""
    i1, i2, expo, mem = pairs
    f1, f2 = world["feats"][i1], world["feats"][i2]

    # memory is only available in proportion to exposure
    mem_w = mem * np.log1p(expo) / np.log1p(expo).max() if expo.max() > 0 else mem * 0.0

    y = (alpha * (f1 - f2) @ world["w_abs"]
         + lam * (world["lex"][i1] - world["lex"][i2])
         + mu * mem_w
         + rng.normal(scale=NOISE_Y, size=len(i1)))

    # representations: position-dependent encoding of features, plus lexical bias
    # and pair memory, both of which flip sign with the ordering
    lex_term = lam * (world["lex"][i1] - world["lex"][i2])[:, None] * world["Q"]
    mem_term = mu * mem_w[:, None] * world["V"]
    h_a = f1 @ world["P1"] + f2 @ world["P2"] + lex_term + mem_term \
        + rng.normal(scale=NOISE_H, size=(len(i1), D_REP))
    h_n = f2 @ world["P1"] + f1 @ world["P2"] - lex_term - mem_term \
        + rng.normal(scale=NOISE_H, size=(len(i1), D_REP))
    X = h_a - h_n                                    # antisymmetric feature
    return X, y, i1, i2, expo


def ridge_cv_r2(X, y, folds, alphas=(1e0, 1e1, 1e2, 1e3, 1e4)):
    """Out-of-fold R2 using the same antisymmetric ridge as the real linear probe."""
    r2s = []
    for tr, te in folds:
        if len(te) < 40 or len(tr) < 100:
            continue
        m, s = X[tr].mean(0), X[tr].std(0) + 1e-8
        Ztr, Zte = (X[tr] - m) / s, (X[te] - m) / s
        ZtZ, Zty = Ztr.T @ Ztr, Ztr.T @ y[tr]
        eye = np.eye(X.shape[1])
        best, best_w = np.inf, None
        for a in alphas:                             # pick alpha on a held-in split
            w = np.linalg.solve(ZtZ + a * eye, Zty)
            err = ((Ztr @ w) - y[tr]).var()
            if err < best:
                best, best_w = err, w
        pred = Zte @ best_w
        if pred.std() > 0:
            r2s.append(np.corrcoef(pred, y[te])[0, 1] ** 2)
    return float(np.mean(r2s)) if r2s else np.nan


def pair_folds(n, seed=SEED, k=5):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


def word_folds(i1, i2, seed=SEED, k=5):
    """Hold out words: test pairs must have BOTH words held out."""
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, k, N_WORDS)
    out = []
    for f in range(k):
        te = np.where((assign[i1] == f) & (assign[i2] == f))[0]
        tr = np.where((assign[i1] != f) & (assign[i2] != f))[0]
        out.append((tr, te))
    return out


def beta2(resid, expo):
    """|residual| ~ b0 + b1 log(freq) + b2 log(freq)^2, as in the real pipeline."""
    m = expo > 0
    lf = np.log(expo[m])
    A = np.column_stack([np.ones(m.sum()), lf, lf ** 2])
    b, *_ = np.linalg.lstsq(A, np.abs(resid[m]), rcond=None)
    return b[2]


def one_run(alpha, lam, mu, seed):
    rng = np.random.default_rng(seed)
    world = make_world(rng)
    nov = make_pairs(world, N_NOVEL, rng, exposed=False)
    att = make_pairs(world, N_ATTESTED, rng, exposed=True)
    Xn, yn, n1, n2, _ = synthesise(world, nov, alpha, lam, mu, rng)
    Xa, ya, _, _, ea = synthesise(world, att, alpha, lam, mu, rng)

    r2_pair = ridge_cv_r2(Xn, yn, pair_folds(len(yn)))
    r2_word = ridge_cv_r2(Xn, yn, word_folds(n1, n2))

    # train on novel (no memory available there), predict attested, regress residual
    m, s = Xn.mean(0), Xn.std(0) + 1e-8
    Z = (Xn - m) / s
    w = np.linalg.solve(Z.T @ Z + 1e2 * np.eye(Xn.shape[1]), Z.T @ yn)
    resid = ya - ((Xa - m) / s) @ w
    return dict(alpha=alpha, lam=lam, mu=mu,
                r2_pair=r2_pair, r2_word=r2_word,
                gap=r2_pair - r2_word, beta2=beta2(resid, ea))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    grid = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 1.0, 2.0]
    rows = []
    print("=" * 88)
    print("SIMULATION: do the measures recover the components they claim to?")
    print("=" * 88)
    print("  alpha = feature abstraction   lam = lexical bias   mu = pair memory\n")
    for a, l, u in itertools.product(grid, repeat=3):
        for r in range(args.reps):
            rows.append(one_run(a, l, u, SEED + r))
    df = pd.DataFrame(rows)
    out = BASE / "Results" / "simulation_validate_measures.csv"
    df.to_csv(out, index=False)

    g = df.groupby(["alpha", "lam", "mu"], as_index=False).mean()

    def sens(target, measure):
        """Correlation of a measure with each ground-truth component, across the grid."""
        return {c: np.corrcoef(g[c], g[measure])[0, 1] for c in ("alpha", "lam", "mu")}

    print("  SELECTIVITY: correlation of each measure with each true component")
    print(f"  {'measure':>12s} {'alpha':>10s} {'lam':>10s} {'mu':>10s}")
    for meas in ("r2_word", "gap", "beta2"):
        s = sens(None, meas)
        print(f"  {meas:>12s} {s['alpha']:>+10.3f} {s['lam']:>+10.3f} {s['mu']:>+10.3f}")

    print("\n  word-held-out R2 as alpha varies (lam=mu=0):")
    for _, r in g[(g.lam == 0) & (g.mu == 0)].iterrows():
        print(f"    alpha={r.alpha:>4.2f}  R2_word={r.r2_word:.4f}")
    print("\n  pair-vs-word GAP as lam varies (alpha=1, mu=0):")
    for _, r in g[(g.alpha == 1.0) & (g.mu == 0)].iterrows():
        print(f"    lam={r.lam:>4.2f}    gap={r.gap:+.4f}   R2_word={r.r2_word:.4f}")
    print("\n  beta2 as mu varies (alpha=1, lam=0):")
    for _, r in g[(g.alpha == 1.0) & (g.lam == 0)].iterrows():
        print(f"    mu={r.mu:>4.2f}     beta2={r.beta2:+.5f}  R2_word={r.r2_word:.4f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
