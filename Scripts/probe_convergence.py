"""
probe_convergence.py
--------------------
Do probes trained on DIFFERENT models' representations converge on the same
predictions?

Cross-model agreement on y_true established that the preferences themselves are
not arbitrary. A remaining steelman: each model might encode that (shared)
preference somewhere idiosyncratic, so the probe works per model without there
being any shared representational structure.

If that were so, two probes trained independently on two different
representation spaces would extract different things. Convergence between them
implies the encodings share structure.

CONFOUND: a probe is a smoothing operator, so y-hat is less pair-specific than
y_true and cross-model correlations rise mechanically. The control is a matched
smoother with no representational content at all: the additive per-word model
    y = g(W1) - g(W2)
fit on the SAME CV folds as the probe, using only word identity. Comparisons:

    corr(y_A,    y_B)     raw preference agreement            (the baseline)
    corr(g_A,    g_B)     agreement after per-word smoothing  (the control)
    corr(yhat_A, yhat_B)  agreement between two probes        (the test)
    corr(yhat_A, y_B)     does A's probe predict B's preference

The test is whether probe agreement EXCEEDS the additive control. If it only
matches it, the probe adds nothing beyond word identity and the representational
claim is unsupported.

Usage:
    python Scripts/probe_convergence.py
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import pearsonr

SEED = 964
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "Results"
RIDGE_LAMBDA = 3.0          # chosen by CV in per_word_preference.py

BABYLM = [
    "znhoughton_opt-babylm-125m-20eps-seed964",
    "znhoughton_opt-babylm-350m-20eps-seed964",
    "znhoughton_opt-babylm-1_3b-20eps-seed964",
]
SHORT = {s: s.split("babylm-")[1].split("-20eps")[0] for s in BABYLM}


def load(slug, condition, split):
    d = RES / slug / "cv_preds"
    pat = re.compile(rf"^{re.escape(condition)}_layer(\d+)_mean_pooled_{split}\.npz$")
    best, path = -1, None
    for f in d.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best:
            best, path = int(m.group(1)), f
    z = np.load(path, allow_pickle=True)
    key = np.char.add(np.char.add(z["word1"].astype(str), "\t"),
                      z["word2"].astype(str))
    return pd.DataFrame({"key": key, "w1": z["word1"].astype(str),
                         "w2": z["word2"].astype(str),
                         "y": z["y_true"].astype(float),
                         "yhat": z["y_pred"].astype(float),
                         "fold": z["fold"].astype(int)}) \
             .drop_duplicates("key").set_index("key")


def additive_oof(w1, w2, y, fold):
    """Out-of-fold predictions from the additive per-word model, on the probe's
    own folds. Pairs whose words are unseen in the training folds get 0, which
    is what a per-word model can say about them."""
    words = pd.unique(np.concatenate([w1, w2]))
    idx = {w: i for i, w in enumerate(words)}
    i1 = np.array([idx[w] for w in w1])
    i2 = np.array([idx[w] for w in w2])
    n, p = len(y), len(words)

    def design(rows):
        m = len(rows)
        r = np.repeat(np.arange(m), 2)
        c = np.empty(2 * m, dtype=np.int64)
        c[0::2], c[1::2] = i1[rows], i2[rows]
        v = np.empty(2 * m)
        v[0::2], v[1::2] = 1.0, -1.0
        return sparse.csr_matrix((v, (r, c)), shape=(m, p))

    out = np.zeros(n)
    for f in np.unique(fold):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        Xtr = design(tr)
        aug = sparse.vstack([Xtr, np.sqrt(RIDGE_LAMBDA) *
                             sparse.eye(p, format="csr")], format="csr")
        rhs = np.concatenate([y[tr], np.zeros(p)])
        g = lsqr(aug, rhs, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
        out[te] = design(te) @ g
    return out


def safe_r(x, y):
    """Pearson r that returns NaN instead of raising when a series is constant.
    On word_novel the additive control predicts zero for every test pair, by
    construction, so zero variance is an expected and informative outcome."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100 or x[m].std() < 1e-9 or y[m].std() < 1e-9:
        return float("nan")
    return float(pearsonr(x[m], y[m])[0])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pair_novel", "word_novel"],
                    default="word_novel",
                    help="word_novel is the diagnostic split: both words of a "
                         "test pair are held out, so the additive control "
                         "predicts zero and cannot produce any agreement")
    args = ap.parse_args()

    rows = []
    for cond in ("default", "attn_zeroed"):
        frames = {s: load(s, cond, args.split) for s in BABYLM}
        common = None
        for f in frames.values():
            common = f.index if common is None else common.intersection(f.index)
        frames = {s: f.loc[common] for s, f in frames.items()}
        # On word_novel a pair with exactly one word held out falls in neither
        # the train nor the test fold, so it is never predicted. Keep only pairs
        # predicted by every model, or the models are compared on different sets.
        finite = np.ones(len(common), bool)
        for s in BABYLM:
            finite &= np.isfinite(frames[s]["yhat"].to_numpy())
        frames = {s: f.loc[finite] for s, f in frames.items()}
        n = int(finite.sum())
        print(f"\n=== {cond}: {n} pairs predicted by all models "
              f"(of {len(common)} aligned) ===", flush=True)

        add = {}
        for s in BABYLM:
            f = frames[s]
            add[s] = additive_oof(f["w1"].to_numpy(), f["w2"].to_numpy(),
                                  f["y"].to_numpy(), f["fold"].to_numpy())
            print(f"  {SHORT[s]}: additive OOF r with y = "
                  f"{safe_r(add[s], f['y'].to_numpy()):.3f} "
                  f"(sd {add[s].std():.4f})   probe r with y = "
                  f"{safe_r(f['yhat'].to_numpy(), f['y'].to_numpy()):.3f}",
                  flush=True)

        for a, b in itertools.combinations(BABYLM, 2):
            ya, yb = frames[a]["y"].to_numpy(), frames[b]["y"].to_numpy()
            pa, pb = frames[a]["yhat"].to_numpy(), frames[b]["yhat"].to_numpy()
            rows.append(dict(
                condition=cond, model_a=SHORT[a], model_b=SHORT[b], n=n,
                r_y_y=safe_r(ya, yb),
                r_add_add=safe_r(add[a], add[b]),
                r_probe_probe=safe_r(pa, pb),
                r_probeA_yB=safe_r(pa, yb),
                r_probeB_yA=safe_r(pb, ya)))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "probe_convergence.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("DO TWO PROBES ON TWO DIFFERENT REPRESENTATION SPACES CONVERGE?")
    print("  r_y_y         raw preference agreement          (baseline)")
    print("  r_add_add     per-word smoothing agreement      (CONTROL)")
    print("  r_probe_probe agreement between the two probes  (TEST)")
    print("  test is meaningful only if r_probe_probe > r_add_add")
    print("=" * 78)
    print(df.round(3).to_string(index=False))

    print("\n" + "-" * 78)
    print("means by condition")
    print(df.groupby("condition")[["r_y_y", "r_add_add", "r_probe_probe",
                                   "r_probeA_yB", "r_probeB_yA"]]
          .mean().round(3).to_string())
    print(f"\nwrote {RES / 'probe_convergence.csv'}")


if __name__ == "__main__":
    main()
