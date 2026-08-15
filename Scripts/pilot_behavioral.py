"""
pilot_behavioral.py
-------------------
Behavioral pilot: measure memorization and abstraction from the model's OWN
ordering preferences (y_true). No probe is used anywhere in this script.

y_true = log P("W1 and W2") - log P("W2 and W1"), read off the model's output.
It is already stored in Results/{model}/by_layer_corpus_pred.csv.xz, and it does
not vary by layer or mode -- those columns only matter for the probe.

Design (piloted on Pythia, trained on the Pile):
  exposure          = Pile directional counts via infinigram   -> what Pythia SAW
  correctness       = BabyLM corpus ordering counts            -> Pythia never trained on it

THE CORE TABLE is accuracy-against-an-independent-corpus, broken down by
training exposure and by attention condition. Both claims read off it:

  ABSTRACTION   the unseen-exposure row under `default`.
                Pairs absent from training, scored against a corpus the model
                never saw. Cannot be memorization (never encountered) and cannot
                be noise (noise does not track an external criterion).

  MEMORIZATION  the slope across exposure rows, and how it changes under ablation.
                Attention-zeroing stops W2 and "and" from attending to W1, so the
                model must predict W2 without knowing what preceded it. If the
                benefit of having seen a pair disappears under ablation, that
                benefit was retrieved pair-specific knowledge.

Usage:
    python Scripts/pilot_behavioral.py --model EleutherAI_pythia-1b
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
SEED = 964
N_BOOT = 2000

STRATA = [("unseen (0)", 0, 0), ("1-5", 1, 5), ("6-50", 6, 50),
          ("51-500", 51, 500), (">500", 501, np.inf)]


# ── loading ──────────────────────────────────────────────────────────────────

def load_behavioral(model):
    """One row per (word1, word2) with y_true under each attention condition."""
    path = BASE / "Results" / model / "by_layer_corpus_pred.csv.xz"
    df = pd.read_csv(path, usecols=["condition", "layer", "mode",
                                    "word1", "word2", "y_true"])
    df["word1"] = df.word1.astype(str)
    df["word2"] = df.word2.astype(str)

    # y_true is a property of the forward pass; verify rather than assume.
    spread = (df.groupby(["condition", "word1", "word2"])["y_true"]
                .agg(lambda s: s.max() - s.min()).max())
    print(f"  [check] max within-pair spread of y_true across layer/mode: {spread:.2e}")
    if spread > 1e-6:
        print("  [WARN] y_true varies across layer/mode -- investigate")

    wide = (df.drop_duplicates(["condition", "word1", "word2"])
              .pivot(index=["word1", "word2"], columns="condition", values="y_true")
              .reset_index())
    wide.columns.name = None
    return wide


def attach_counts(df):
    """Pile directional counts (exposure) + BabyLM ordering counts (criterion)."""
    pile = (pd.read_csv(BASE / "Results" / "corpus_binomials_infinigram_piletrain.csv")
              .rename(columns={"freq_w1w2": "pile_w1w2", "freq_w2w1": "pile_w2w1",
                               "freq_total": "pile_total"}))
    baby = (pd.read_csv(BASE / "Data" / "corpus_binomials.csv",
                        usecols=["word1", "word2", "freq_w1_w2", "freq_w2_w1"])
              .rename(columns={"freq_w1_w2": "baby_w1w2", "freq_w2_w1": "baby_w2w1"}))
    for d in (pile, baby):
        d["word1"] = d.word1.astype(str)
        d["word2"] = d.word2.astype(str)

    out = (df.merge(pile, on=["word1", "word2"], how="left")
             .merge(baby, on=["word1", "word2"], how="left"))
    out["baby_total"] = out.baby_w1w2 + out.baby_w2w1
    out["baby_lo"] = np.log((out.baby_w1w2 + 0.5) / (out.baby_w2w1 + 0.5))
    return out


# ── stats helpers ────────────────────────────────────────────────────────────

def r_ci(x, y):
    """Pearson r with a normal-approximation CI."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 50 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, np.nan, len(x)
    r = np.corrcoef(x, y)[0, 1]
    se = 1 / np.sqrt(len(x) - 3)
    z = np.arctanh(r)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se), len(x)


def boot_diff(x1, y1, x2, y2, seed=SEED, n=N_BOOT):
    """Bootstrap CI for cor(x1,y1) - cor(x2,y2).

    The two groups are independent strata of different sizes, so each is
    resampled separately at its own n.
    """
    rng = np.random.default_rng(seed)
    k1, k2 = len(x1), len(x2)
    out = np.empty(n)
    for i in range(n):
        i1 = rng.integers(0, k1, k1)
        i2 = rng.integers(0, k2, k2)
        out[i] = (np.corrcoef(x1[i1], y1[i1])[0, 1]
                  - np.corrcoef(x2[i2], y2[i2])[0, 1])
    return np.percentile(out, [2.5, 97.5])


def stratify(d, col="pile_total"):
    for name, lo, hi in STRATA:
        yield name, d[(d[col] >= lo) & (d[col] <= hi)]


# ── analyses ─────────────────────────────────────────────────────────────────

def sanity(df):
    print("\n" + "=" * 82)
    print("STEP 1  Sanity: does each condition retain analysable signal?")
    print("=" * 82)
    for c in ("default", "attn_zeroed"):
        v = df[c].dropna()
        print(f"  {c:<14s} n={len(v):>6,}  mean={v.mean():+.4f}  sd={v.std():.4f}  "
              f"median|y|={v.abs().median():.4f}")
    r, lo, hi, n = r_ci(df.default, df.attn_zeroed)
    print(f"  r(default, attn_zeroed) = {r:+.4f} [{lo:+.3f},{hi:+.3f}]  n={n:,}")
    print(f"\n  Pile coverage: {df.pile_total.notna().mean():.1%}   "
          f"pairs unseen by Pythia (Pile==0): {(df.pile_total == 0).sum():,}")


def core_table(df):
    print("\n" + "=" * 82)
    print("STEP 2  CORE TABLE: agreement with an independent corpus,")
    print("        by training exposure x attention condition")
    print("=" * 82)
    print("  Criterion: BabyLM-corpus ordering log-odds. Pythia never trained on it.")
    print("  Read ABSTRACTION off the 'unseen' row; read MEMORIZATION off the slope.\n")
    d = df.dropna(subset=["default", "attn_zeroed", "pile_total", "baby_lo"])

    print(f"  {'Pile exposure':<15s} {'n':>7s} {'default':>20s} {'attn_zeroed':>20s} {'gap':>8s}")
    print("  " + "-" * 76)
    rows = {}
    for name, s in stratify(d):
        if len(s) < 50:
            print(f"  {name:<15s} {len(s):>7,}   (too few)")
            continue
        rd, ld, hd, _ = r_ci(s.default, s.baby_lo)
        ra, la, ha, _ = r_ci(s.attn_zeroed, s.baby_lo)
        rows[name] = (s, rd, ra)
        print(f"  {name:<15s} {len(s):>7,} "
              f"{rd:>+8.4f} [{ld:+.2f},{hd:+.2f}] "
              f"{ra:>+8.4f} [{la:+.2f},{ha:+.2f}] {rd - ra:>+8.4f}")

    if "unseen (0)" in rows and ">500" in rows:
        su, ru_d, ru_a = rows["unseen (0)"]
        st, rt_d, rt_a = rows[">500"]
        print(f"\n  EXPOSURE EFFECT (>500 minus unseen)")
        print(f"    default      {rt_d - ru_d:+.4f}   <- benefit of having seen the pair")
        print(f"    attn_zeroed  {rt_a - ru_a:+.4f}   <- same benefit with cross-word attention blocked")
        lo, hi = boot_diff(st.default.values, st.baby_lo.values,
                           su.default.values, su.baby_lo.values)
        print(f"    default exposure effect 95% CI [{lo:+.4f}, {hi:+.4f}]")
        lo, hi = boot_diff(st.attn_zeroed.values, st.baby_lo.values,
                           su.attn_zeroed.values, su.baby_lo.values)
        print(f"    attn_zeroed exposure effect 95% CI [{lo:+.4f}, {hi:+.4f}]")
        print("\n    If the default effect is positive and the ablated one is not,")
        print("    the benefit of training exposure is delivered via cross-word attention,")
        print("    i.e. it is retrieved pair-specific knowledge.")


def magnitude(df):
    print("\n" + "=" * 82)
    print("STEP 3  Preference magnitude under ablation")
    print("=" * 82)
    print("  Secondary view: does ablation shrink the preference more for seen pairs?\n")
    d = df.dropna(subset=["default", "attn_zeroed", "pile_total"]).copy()
    d["delta"] = d.default - d.attn_zeroed

    print(f"  {'Pile exposure':<15s} {'n':>7s} {'mean|default|':>14s} {'mean|attn0|':>12s} "
          f"{'shrinkage':>10s} {'mean|Delta|':>12s}")
    print("  " + "-" * 76)
    for name, s in stratify(d):
        if len(s) < 50:
            continue
        md, ma = s.default.abs().mean(), s.attn_zeroed.abs().mean()
        print(f"  {name:<15s} {len(s):>7,} {md:>14.4f} {ma:>12.4f} "
              f"{md - ma:>+10.4f} {s.delta.abs().mean():>12.4f}")

    seen = d[d.pile_total > 0]
    r, lo, hi, n = r_ci(np.log(seen.pile_total), seen.delta.abs())
    print(f"\n  |Delta| ~ log Pile count: r={r:+.4f} [{lo:+.3f},{hi:+.3f}]  n={n:,}")
    print("  NOTE |Delta| is a poor index: it counts change in either direction and is")
    print("  dominated by noise. The signed shrinkage column and STEP 2 are the real tests.")


def confounds(df):
    print("\n" + "=" * 82)
    print("STEP 4  Is the abstraction result reducible to length and frequency?")
    print("=" * 82)
    from wordfreq import zipf_frequency
    d = df[df.pile_total == 0].dropna(subset=["default", "attn_zeroed", "baby_lo"]).copy()
    cache = {}
    def z(w):
        if w not in cache:
            cache[w] = zipf_frequency(w, "en")
        return cache[w]
    d["d_zipf"] = [z(a) - z(b) for a, b in zip(d.word1, d.word2)]
    d["d_len"] = d.word1.str.len() - d.word2.str.len()

    for lbl, col in (("d_zipf", "d_zipf"), ("d_len", "d_len")):
        r, lo, hi, n = r_ci(d[col], d.baby_lo)
        print(f"  {lbl + ' vs criterion':<32s} r={r:+.4f} [{lo:+.3f},{hi:+.3f}]  n={n:,}")

    X = np.column_stack([np.ones(len(d)), d.d_zipf, d.d_len])
    for cond in ("default", "attn_zeroed"):
        ry = d[cond] - X @ np.linalg.lstsq(X, d[cond], rcond=None)[0]
        rc = d.baby_lo - X @ np.linalg.lstsq(X, d.baby_lo, rcond=None)[0]
        r0, lo0, hi0, _ = r_ci(d[cond], d.baby_lo)
        r1, lo1, hi1, n = r_ci(ry, rc)
        print(f"  {cond:<14s} raw r={r0:+.4f}   partial r={r1:+.4f} "
              f"[{lo1:+.3f},{hi1:+.3f}]  n={n:,}")
    print("\n  A surviving partial correlation means the model's generalization to")
    print("  unseen pairs is not reducible to word length and unigram frequency.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI_pythia-1b")
    args = ap.parse_args()

    print("=" * 82)
    print(f"BEHAVIORAL PILOT  |  model: {args.model}")
    print("=" * 82)
    print("  Every number below comes from y_true, the model's own output preference.")

    df = attach_counts(load_behavioral(args.model))
    print(f"  pairs loaded: {len(df):,}")

    sanity(df)
    core_table(df)
    magnitude(df)
    confounds(df)
    print("\n" + "=" * 82)


if __name__ == "__main__":
    main()
