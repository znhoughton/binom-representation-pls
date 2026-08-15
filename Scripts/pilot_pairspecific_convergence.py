"""
pilot_pairspecific_convergence.py
---------------------------------
Behavioral evidence for memorization, entirely within-model.

THE LOGIC
  Decompose the model's FINAL preference into
      per-word component   everything expressible as a comparison of the two
                           words' individual ordering biases (fit out-of-fold with
                           an additive model, one free parameter per word)
      residual             what remains: interactions, i.e. anything that depends
                           on WHICH TWO words are combined
  Then, at each checkpoint, ask how much of the final residual is already in place:
      cor( y_true(t), residual_final )   within exposure strata.

WHY A GRADIENT IN THAT CORRELATION MEANS MEMORIZATION
  The per-word decomposition has already removed every lexical-frequency effect,
  because those are per-word comparisons. And an abstract constraint learned
  across many pairs applies uniformly, regardless of how often any individual
  pair occurred, so it cannot produce exposure-dependence. If the residual
  converges faster for pairs the model saw more often, the model learned
  something about those specific pairs.

WHAT IT IS NOT
  The residual LEVEL is not memorization. It is non-zero even at zero exposure,
  where nothing can have been memorized, because the residual also contains
  compositional abstraction and sentence-context effects. Only the GRADIENT
  across exposure is the memorization signal.

WHY IT AVOIDS THE CONFOUNDS THAT SANK EARLIER ATTEMPTS
  The target is the model's own final state, so there is no external criterion
  (no sparse-count measurement error) and no prefix priming: the example sentence
  contributes identically at every checkpoint and to the final state, so it
  cannot create a difference between them.

Usage:
    python Scripts/pilot_pairspecific_convergence.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parents[1]
SEED = 964
DAMP = 3.0
N_BOOT = 2000

FAMILIES = [
    # (final slug, checkpoint glob prefix, exposure source)
    ("EleutherAI_pythia-160m", "EleutherAI_pythia-160m", "pile"),
    ("EleutherAI_pythia-410m", "EleutherAI_pythia-410m", "pile"),
    ("EleutherAI_pythia-1b",   "EleutherAI_pythia-1b",   "pile"),
    ("znhoughton_opt-babylm-125m-20eps-seed964",
     "znhoughton_opt-babylm-125m-20eps-seed964", "babylm"),
    ("znhoughton_opt-babylm-350m-20eps-seed964",
     "znhoughton_opt-babylm-350m-20eps-seed964", "babylm"),
    # 1.3B final slug uses "1_3b" while its checkpoints use "1.3b"
    ("znhoughton_opt-babylm-1_3b-20eps-seed964",
     "znhoughton_opt-babylm-1.3b-20eps-seed964", "babylm"),
]


def load_exposure(kind):
    """Counts of the binomial in the model's OWN training corpus."""
    if kind == "pile":
        p = pd.read_csv(BASE / "Results" / "corpus_binomials_infinigram_piletrain.csv")
        p = p[["word1", "word2", "freq_total"]].rename(columns={"freq_total": "exposure"})
    else:
        p = pd.read_csv(BASE / "Data" / "corpus_binomials.csv",
                        usecols=["word1", "word2", "freq_w1_w2", "freq_w2_w1"])
        p["exposure"] = p.freq_w1_w2 + p.freq_w2_w1
        p = p[["word1", "word2", "exposure"]]
    p["word1"] = p.word1.astype(str)
    p["word2"] = p.word2.astype(str)
    return p


def load_pref(slug):
    path = BASE / "Results" / slug / "by_layer_corpus_pred.csv.xz"
    if not path.exists():
        return None
    d = pd.read_csv(path, usecols=["condition", "layer", "mode",
                                   "word1", "word2", "y_true"])
    d["word1"] = d.word1.astype(str)
    d["word2"] = d.word2.astype(str)
    d = d[d.condition == "default"].drop_duplicates(["word1", "word2"])
    return d[["word1", "word2", "y_true"]].rename(columns={"y_true": "y"})


def oof_additive(y, w1, w2, damp=DAMP, seed=SEED):
    """Out-of-fold prediction from y ~ b_W1 - b_W2 (one parameter per word type)."""
    words = pd.Index(sorted(set(w1) | set(w2)))
    i1, i2 = words.get_indexer(w1), words.get_indexer(w2)
    n, W = len(y), len(words)
    oof = np.zeros(n)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        rows = np.repeat(np.arange(len(tr)), 2)
        cols = np.empty(2 * len(tr), dtype=int)
        cols[0::2], cols[1::2] = i1[tr], i2[tr]
        A = csr_matrix((np.tile([1., -1.], len(tr)), (rows, cols)),
                       shape=(len(tr), W))
        b = lsqr(A, y[tr], damp=damp, atol=1e-8, btol=1e-8, iter_lim=600)[0]
        oof[te] = b[i1[te]] - b[i2[te]]
    return oof


def boot_gradient(x_hi, y_hi, x_lo, y_lo, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        a = rng.integers(0, len(x_hi), len(x_hi))
        b = rng.integers(0, len(x_lo), len(x_lo))
        out[i] = (np.corrcoef(x_hi[a], y_hi[a])[0, 1]
                  - np.corrcoef(x_lo[b], y_lo[b])[0, 1])
    return np.percentile(out, [2.5, 97.5])


def strata_for(exposure):
    """Family-specific bins: Pile spans 0-5.6M, BabyLM 1-1095."""
    if exposure.max() > 10000:
        return [("0", 0, 0), ("1-500", 1, 500), ("501-5k", 501, 5000),
                ("5k-50k", 5001, 50000), (">50k", 50001, np.inf)]
    # BabyLM counts top out around 1,000 and are 70% singletons, so a ">100"
    # bin is too thin to estimate; the top bin is ">20".
    return [("1", 1, 1), ("2-5", 2, 5), ("6-20", 6, 20), (">20", 21, np.inf)]


def run(final_slug, ckpt_prefix, expo_kind):
    fin = load_pref(final_slug)
    if fin is None:
        print(f"  {final_slug}: no final results, skipping")
        return None
    expo = load_exposure(expo_kind)
    fin = fin.merge(expo, on=["word1", "word2"]).dropna().reset_index(drop=True)
    fin["pw"] = oof_additive(fin.y.values, fin.word1.values, fin.word2.values)
    fin["resid"] = fin.y - fin.pw
    r2 = np.corrcoef(fin.pw, fin.y)[0, 1] ** 2

    STR = strata_for(fin.exposure)
    ckpts = sorted((int(re.search(r"_step(\d+)$", p.name).group(1)), p.name)
                   for p in (BASE / "Results").glob(f"{ckpt_prefix}_step*"))

    print(f"\n=== {final_slug} ===")
    print(f"  n={len(fin):,}   per-word R2 of final preference = {r2:.3f}   "
          f"residual R2 = {1 - r2:.3f}")
    print(f"  {'step':>7s}" + "".join(f"{n:>10s}" for n, _, _ in STR)
          + f"{'gradient':>11s}{'95% CI':>18s}")

    rows = []
    for step, slug in ckpts + [(10 ** 9, final_slug)]:
        x = load_pref(slug)
        if x is None:
            continue
        m = fin.merge(x, on=["word1", "word2"], suffixes=("", "_t")).dropna()
        vals, cells = [], {}
        for n, lo, hi in STR:
            s = m[(m.exposure >= lo) & (m.exposure <= hi)]
            if len(s) < 80:
                vals.append(None); continue
            vals.append(np.corrcoef(s.y_t, s.resid)[0, 1])
            cells[n] = s
        lbl = "final" if step == 10 ** 9 else str(step)
        line = f"  {lbl:>7s}" + "".join(
            f"{v:>+10.4f}" if v is not None else f"{'-':>10s}" for v in vals)
        lo_name, hi_name = STR[0][0], STR[-1][0]
        if lo_name in cells and hi_name in cells:
            hi_s, lo_s = cells[hi_name], cells[lo_name]
            grad = (np.corrcoef(hi_s.y_t, hi_s.resid)[0, 1]
                    - np.corrcoef(lo_s.y_t, lo_s.resid)[0, 1])
            ci = boot_gradient(hi_s.y_t.values, hi_s.resid.values,
                               lo_s.y_t.values, lo_s.resid.values)
            line += f"{grad:>+11.4f}  [{ci[0]:+.3f},{ci[1]:+.3f}]"
            rows.append(dict(model=final_slug, step=step, gradient=grad,
                             ci_lo=ci[0], ci_hi=ci[1]))
        print(line)
    return rows


def main():
    print("=" * 100)
    print("PAIR-SPECIFIC CONVERGENCE  (within-model; no external criterion)")
    print("=" * 100)
    print("  cor( y_true at step t , pair-specific residual of the FINAL preference )")
    print("  gradient = highest exposure stratum minus lowest")
    print("  A positive gradient means the pair-specific component is established")
    print("  faster for pairs the model saw more often, with per-word effects removed.")
    allrows = []
    for f, c, k in FAMILIES:
        r = run(f, c, k)
        if r:
            allrows += r
    if allrows:
        out = BASE / "Results" / "pairspecific_convergence.csv"
        pd.DataFrame(allrows).to_csv(out, index=False)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
