"""
pilot_behavioral_trajectory.py
------------------------------
Abstraction and memorization across training, on an ABSOLUTE scale.

Why not beta2. The U-shape measures where an abstraction model fails, and that
abstraction model is fit on novel pairs. Early in training the model has little
generalizable ordering knowledge, so the abstraction model captures almost
nothing, its residual is undifferentiated, and no curvature is detectable.
beta2 ~ 0 at step 16 is therefore consistent BOTH with "no memorization" and
with "memorization but no abstraction baseline to contrast against". It cannot
answer the timing question.

This measure can, because neither component is defined as a residual from the
other. Everything is a correlation between two named quantities:

    x = the model's own preference          y_true = log P(W1 and W2) - log P(W2 and W1)
    y = ordering log-odds in a corpus       log((count_W1W2 + .5)/(count_W2W1 + .5))
        the model never trained on

  ABSTRACTION(t)  = cor(x, y) among pairs with ZERO occurrences in training.
                    Nothing to retrieve, so any agreement is generalization.

  MEMORIZATION(t) = cor(x, y) among HIGH-exposure pairs
                    minus cor(x, y) among zero-exposure pairs.
                    The extra accuracy that having seen the pair buys.

Both are in correlation units at every checkpoint, so a value at step 16 means
the same thing as a value at the final checkpoint.

Piloted on Pythia, where exposure and criterion live in one dataset:
    exposure  = Pile directional counts (infinigram)  -> what Pythia saw
    criterion = BabyLM corpus ordering                -> never trained on
For BabyLM the roles reverse, but the zero-exposure stratum then lives in the
novel set while the exposed strata live in the attested set, and we currently
have Pile counts for one and Dolma counts for the other. Using two different
criteria across strata would not be comparable, so that arm needs one extra
infinigram run before it can be built.

Usage:
    python Scripts/pilot_behavioral_trajectory.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
SEED = 964
N_BOOT = 2000

# Pythia steps -> cumulative training tokens (1024 seqs x 2048 tokens per step)
TOKENS_PER_STEP = 2_097_152
FAMILIES = ["EleutherAI_pythia-160m", "EleutherAI_pythia-410m", "EleutherAI_pythia-1b"]
HIGH_EXPOSURE = 500      # Pile count above which we call a pair "high exposure"


def load_criterion():
    """BabyLM-corpus ordering log-odds: the correctness criterion for Pythia."""
    b = pd.read_csv(BASE / "Data" / "corpus_binomials.csv",
                    usecols=["word1", "word2", "freq_w1_w2", "freq_w2_w1"])
    b["word1"] = b.word1.astype(str)
    b["word2"] = b.word2.astype(str)
    b["crit"] = np.log((b.freq_w1_w2 + 0.5) / (b.freq_w2_w1 + 0.5))
    return b[["word1", "word2", "crit"]]


def load_exposure():
    """Pile counts: total = exposure; directional log-odds = the OWN-corpus target.

    Memorisation means reproducing the asymmetry the model actually observed, so
    the own-corpus criterion is the Pile's ordering, not BabyLM's.
    """
    p = pd.read_csv(BASE / "Results" / "corpus_binomials_infinigram_piletrain.csv")
    p["word1"] = p.word1.astype(str)
    p["word2"] = p.word2.astype(str)
    p["pile_lo"] = np.log((p.freq_w1w2 + 0.5) / (p.freq_w2w1 + 0.5))
    return (p[["word1", "word2", "freq_total", "pile_lo"]]
            .rename(columns={"freq_total": "exposure"}))


def load_wordfeats(pairs):
    """Delta unigram frequency and length, for partialling out of the abstraction index."""
    from wordfreq import zipf_frequency
    cache = {}
    def z(w):
        if w not in cache:
            cache[w] = zipf_frequency(w, "en")
        return cache[w]
    return pd.DataFrame({
        "word1": pairs.word1, "word2": pairs.word2,
        "d_zipf": [z(a) - z(b) for a, b in zip(pairs.word1, pairs.word2)],
        "d_len": pairs.word1.str.len() - pairs.word2.str.len(),
    })


def load_pref(slug):
    """One y_true per (condition, pair). Returns None if the cell is absent."""
    path = BASE / "Results" / slug / "by_layer_corpus_pred.csv.xz"
    if not path.exists():
        return None
    d = pd.read_csv(path, usecols=["condition", "layer", "mode",
                                   "word1", "word2", "y_true"])
    d["word1"] = d.word1.astype(str)
    d["word2"] = d.word2.astype(str)
    return (d.drop_duplicates(["condition", "word1", "word2"])
             .pivot(index=["word1", "word2"], columns="condition", values="y_true")
             .reset_index())


def boot_gap(x_hi, y_hi, x_lo, y_lo, seed=SEED, n=N_BOOT):
    """Bootstrap CI for cor(x_hi,y_hi) - cor(x_lo,y_lo); strata resampled separately."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        a = rng.integers(0, len(x_hi), len(x_hi))
        b = rng.integers(0, len(x_lo), len(x_lo))
        out[i] = (np.corrcoef(x_hi[a], y_hi[a])[0, 1]
                  - np.corrcoef(x_lo[b], y_lo[b])[0, 1])
    return np.percentile(out, [2.5, 97.5])


def partial_r(x, y, controls):
    """Correlation of x and y after residualising both on `controls`."""
    C = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in controls])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def measure(df, cond):
    """All four indices for one condition.

    abstraction      cor(y_true, external criterion) on ZERO-exposure pairs.
    abstraction_pf   same, after partialling out d_zipf and d_len from both.
                     Unigram frequency is learned almost immediately and itself
                     correlates with the criterion at r = .133, so the raw value
                     can rise early without any binomial-specific knowledge.
    memorization     raw exposure gain in predicting the external criterion.
                     Confounded: the two strata differ in composition, not just
                     exposure. Use the DiD against attn_zeroed to remove that.
    memorization_own cor(y_true, PILE log-odds) on high-exposure pairs, partialling
                     out the external criterion. Memorising means reproducing the
                     asymmetry the model actually saw, which is the Pile's, not
                     BabyLM's; controlling for the external criterion isolates
                     Pile-specific idiosyncrasy from general English ordering.
    """
    d = df.dropna(subset=[cond, "crit", "exposure", "pile_lo", "d_zipf", "d_len"])
    lo = d[d.exposure == 0]
    hi = d[d.exposure > HIGH_EXPOSURE]
    nan5 = (np.nan,) * 4 + (len(lo), len(hi), (np.nan, np.nan))
    if len(lo) < 100 or len(hi) < 100:
        return nan5

    r_lo = np.corrcoef(lo[cond], lo.crit)[0, 1]
    r_hi = np.corrcoef(hi[cond], hi.crit)[0, 1]
    r_lo_pf = partial_r(lo[cond].values, lo.crit.values,
                        [lo.d_zipf.values, lo.d_len.values])
    # Same controls as the abstraction index, so the two are comparably stringent:
    # partial out the external criterion (general English ordering) AND the
    # word-level channels the model learns almost immediately.
    r_own = partial_r(hi[cond].values, hi.pile_lo.values,
                      [hi.crit.values, hi.d_zipf.values, hi.d_len.values])
    ci = boot_gap(hi[cond].values, hi.crit.values, lo[cond].values, lo.crit.values)
    return r_lo, r_lo_pf, r_hi - r_lo, r_own, len(lo), len(hi), ci


def main():
    crit, expo = load_criterion(), load_exposure()

    print("=" * 96)
    print("ABSTRACTION AND MEMORIZATION ACROSS TRAINING (behavioral, no probe)")
    print("=" * 96)
    print("  x = model's own preference y_true")
    print("  y = BabyLM-corpus ordering log-odds  (Pythia never trained on it)")
    print("  ABSTRACTION  = cor(x, y) on pairs with ZERO Pile occurrences")
    print(f"  MEMORIZATION = cor(x, y) on pairs with >{HIGH_EXPOSURE} Pile occurrences,"
          f" minus the above\n")

    rows = []
    for fam in FAMILIES:
        slugs = [(0, fam)] + sorted(
            ((int(re.search(r"_step(\d+)$", p.name).group(1)), p.name)
             for p in (BASE / "Results").glob(f"{fam}_step*")),
            key=lambda t: t[0])
        # step 0 sentinel marks the final checkpoint; sort it last
        slugs = sorted(slugs, key=lambda t: (t[0] == 0, t[0]))

        print(f"--- {fam} ---")
        print(f"  {'step':>7s} {'tokens':>8s} {'ABSTR':>8s} {'ABSTR|freq':>11s} "
              f"{'rawgap':>8s} {'ablgap':>8s} {'MEM(DiD)':>9s} {'MEM(own)':>9s}")
        for step, slug in slugs:
            df = load_pref(slug)
            if df is None:
                continue
            df = (df.merge(expo, on=["word1", "word2"])
                    .merge(crit, on=["word1", "word2"]))
            df = df.merge(load_wordfeats(df[["word1", "word2"]]),
                          on=["word1", "word2"])
            res = {}
            for cond in ("default", "attn_zeroed"):
                if cond in df:
                    res[cond] = measure(df, cond)
            if "default" not in res or np.isnan(res["default"][0]):
                continue
            a, a_pf, gap, own, n0, nhi, ci = res["default"]
            abl = res.get("attn_zeroed", (np.nan,) * 7)[2]
            did = gap - abl if not np.isnan(abl) else np.nan

            tok = "final" if step == 0 else f"{step*TOKENS_PER_STEP/1e6:.0f}M"
            lbl = "final" if step == 0 else str(step)
            print(f"  {lbl:>7s} {tok:>8s} {a:>+8.4f} {a_pf:>+11.4f} "
                  f"{gap:>+8.4f} {abl:>+8.4f} {did:>+9.4f} {own:>+9.4f}")
            rows.append(dict(family=fam, step=step,
                             abstraction=a, abstraction_partial_freq=a_pf,
                             raw_gap=gap, ablated_gap=abl,
                             memorization_did=did, memorization_own=own,
                             ci_lo=ci[0], ci_hi=ci[1], n_unseen=n0, n_high=nhi))
        print()

    out = BASE / "Results" / "behavioral_trajectory.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved -> {out}")

    print("\nREADING IT")
    print("  ABSTRACTION > 0 at the earliest checkpoints  -> generalization is present early")
    print("  MEMORIZATION ~ 0 early, rising later         -> the two are STAGED")
    print("  both > 0 from the start                      -> PARALLEL from onset")
    print("  memorization CI excluding 0                  -> the exposure gain is real")


if __name__ == "__main__":
    main()
