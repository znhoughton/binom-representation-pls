#!/usr/bin/env python3
"""
assess_harvest.py
-----------------
Does the harvested item set still support Experiment 1-3?

Run after harvest_carrier_sentences.py. Answers three questions, in raw counts
rather than percentages:

  COVERAGE  how many pairs survive, per set
  SPREAD    does Eq. 1 still have its predictor range, especially at the low
            frequency end
  BIAS      do retained pairs differ systematically from dropped ones

Why the low-frequency end specifically
--------------------------------------
Eq. 1 uses log_freq = log(n_w1_w2 + n_w2_w1) and rel_freq = n_w1_w2/total - 0.5.
For a singleton, total = 1, so log_freq sits at its floor and rel_freq is
exactly +0.5 or -0.5. Every singleton is pinned to the corner of the predictor
space, and the rel_freq x log_freq interaction draws much of its leverage from
the contrast between that corner and the frequent items in the interior.
Losing singletons therefore costs more than losing the same number of items
from the middle, which is why retention is reported by frequency stratum.

Run with base conda, not PRenv (see the spearmanr DLL crash).

Usage
-----
  python Scripts/corpus/assess_harvest.py
  python Scripts/corpus/assess_harvest.py --harvest Data/carrier_sentences.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DATA = BASE / "Data"

STRATA = [(1, 2, "1 (singleton)"), (2, 3, "2"), (3, 6, "3-5"),
          (6, 21, "6-20"), (21, 101, "21-100"), (101, 10 ** 9, "101+")]


def strata_table(all_n, kept_mask, label):
    print(f"  {label:>16s} {'all':>10s} {'retained':>10s} {'retention':>10s}")
    for lo, hi, name in STRATA:
        m = (all_n >= lo) & (all_n < hi)
        if m.sum() == 0:
            continue
        k = int((m & kept_mask).sum())
        print(f"  {name:>16s} {int(m.sum()):10,} {k:10,} "
              f"{k / m.sum():9.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default=str(DATA / "carrier_sentences.csv"))
    p.add_argument("--per-pair", type=int, default=2, dest="per_pair")
    args = p.parse_args()

    hv = pd.read_csv(args.harvest)
    need = [f"sent_{i+1}" for i in range(args.per_pair)]
    hv["ok"] = hv[need].notna().all(axis=1)
    got = set(zip(hv.loc[hv["ok"], "word1"], hv.loc[hv["ok"], "word2"]))
    print(f"harvest: {len(hv):,} pairs with >=1 sentence, "
          f"{len(got):,} with >={args.per_pair}\n")

    # ── Corpus / attested set: the items Eq. 1 is fit on ─────────────────────
    cor = pd.read_csv(DATA / "corpus_binomials.csv")
    cor = cor[cor["word1"] != cor["word2"]]          # reduplicatives, as Eq. 1 does
    cor["total"] = cor["freq_w1_w2"] + cor["freq_w2_w1"]
    cor = cor[cor["total"] > 0]
    cor["rel_freq"] = cor["freq_w1_w2"] / cor["total"] - 0.5
    cor["log_freq"] = np.log(cor["total"])
    cor["kept"] = [(a, b) in got for a, b in zip(cor["word1"], cor["word2"])]

    print("=== CORPUS / attested (Eq. 1 items) ===")
    print(f"  eligible after reduplicative filter: {len(cor):,}")
    print(f"  RETAINED: {int(cor['kept'].sum()):,}\n")
    strata_table(cor["total"].values, cor["kept"].values, "BabyLM count")

    k = cor[cor["kept"]]
    print(f"\n  log_freq range : all {cor.log_freq.min():.2f}-{cor.log_freq.max():.2f}"
          f"   retained {k.log_freq.min():.2f}-{k.log_freq.max():.2f}")
    print(f"  rel_freq range : all {cor.rel_freq.min():+.2f}..{cor.rel_freq.max():+.2f}"
          f"   retained {k.rel_freq.min():+.2f}..{k.rel_freq.max():+.2f}")
    print(f"  sd(log_freq)   : all {cor.log_freq.std():.3f}   "
          f"retained {k.log_freq.std():.3f}")
    print(f"  sd(rel_freq)   : all {cor.rel_freq.std():.3f}   "
          f"retained {k.rel_freq.std():.3f}")

    # The interaction needs rel_freq to vary WITHIN frequency levels, not just
    # across them. A stratum where every item sits at +-0.5 contributes nothing
    # to the interaction beyond its own corner.
    print("\n  rel_freq spread within each frequency stratum (retained only)")
    print(f"  {'stratum':>16s} {'n':>8s} {'sd':>8s} {'|rel|<0.4':>10s}")
    for lo, hi, name in STRATA:
        m = (k["total"] >= lo) & (k["total"] < hi)
        if m.sum() == 0:
            continue
        r = k.loc[m, "rel_freq"]
        print(f"  {name:>16s} {int(m.sum()):8,} {r.std():8.3f} "
              f"{int((r.abs() < 0.4).sum()):10,}")

    # ── Selection bias among singletons ──────────────────────────────────────
    sing = cor[cor["total"] == 1]
    if len(sing):
        a, b = sing[sing["kept"]], sing[~sing["kept"]]
        print(f"\n  singletons: retained {len(a):,}  dropped {len(b):,}")
        if len(a) and len(b):
            print(f"    mean rel_freq  retained {a.rel_freq.mean():+.4f}   "
                  f"dropped {b.rel_freq.mean():+.4f}")
            print(f"    share at +0.5  retained {(a.rel_freq > 0).mean():.1%}   "
                  f"dropped {(b.rel_freq > 0).mean():.1%}")

    # ── Novel set: probe items ───────────────────────────────────────────────
    nov = pd.read_csv(DATA / "wikipedia_novel_binomials.csv")
    nov = nov[nov["word1"] != nov["word2"]]
    nov["kept"] = [(a, b) in got for a, b in zip(nov["word1"], nov["word2"])]
    kn = nov[nov["kept"]]
    words = set(kn["word1"]) | set(kn["word2"])
    print(f"\n=== NOVEL (probe items) ===")
    print(f"  eligible: {len(nov):,}   RETAINED: {len(kn):,}")
    print(f"  unique word types retained: {len(words):,}")
    # word_novel holds out both words of a pair, so the expected test-set size
    # per fold is roughly n / FOLDS^2.
    print(f"  expected word_novel test items per fold (~n/100): {len(kn)//100:,}")
    print(f"  pilot ran on 40,000 pairs with fold sd 0.01-0.04")


if __name__ == "__main__":
    main()
