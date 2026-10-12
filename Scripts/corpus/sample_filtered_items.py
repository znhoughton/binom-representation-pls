#!/usr/bin/env python3
"""
sample_filtered_items.py
------------------------
Eyeball check on the harvested carrier sentences.

Both sentences per pair are real corpus occurrences with genuine left context.
An earlier version of this script sampled spliced donor contexts instead, where
the second "sentence" was another item's prefix with this binomial pasted on.
That produced strings like "Imagine Research: develops geologist and
ornithologist" and was abandoned once the harvest showed real second
occurrences are findable for 246,889 pairs.

What to check
-------------
  - the binomial really occurs in both sentences, not as a fragment or a
    cross-clause false match ("he ate bread and butter melted"), which the
    regex harvest cannot rule out on its own
  - both prefixes are genuine left context, not list markers or stray headings
  - the two contexts are substantively different, not near-duplicate boilerplate
    from mirrored Wikipedia text, which would defeat the point of measuring the
    preference in a second context
  - the binomial is the same construction in both, not two different senses

Run with base conda, not PRenv (see the spearmanr DLL crash).

Usage
-----
  python Scripts/corpus/sample_filtered_items.py
  python Scripts/corpus/sample_filtered_items.py --set corpus --n 100
  python Scripts/corpus/sample_filtered_items.py --out check.csv
"""

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DATA = BASE / "Data"
SEED = 964


def span(sent, w1, w2):
    for a, b in ((w1, w2), (w2, w1)):
        m = re.search(r"\b" + re.escape(a) + r"\s+and\s+" + re.escape(b) + r"\b",
                      sent, re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harvest", default=str(DATA / "carrier_sentences.csv"))
    p.add_argument("--set", default="all", choices=["all", "corpus", "novel"],
                   dest="which")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.harvest)
    df = df[df["sent_1"].notna() & df["sent_2"].notna()]
    if args.which != "all":
        df = df[df["set"] == args.which]
    print(f"{len(df):,} pairs with two sentences "
          f"({args.which})\n", flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)

    rows, n_bad_span, n_dupe = [], 0, 0
    for j, r in sub.iterrows():
        w1, w2 = str(r["word1"]), str(r["word2"])
        s1, s2 = str(r["sent_1"]), str(r["sent_2"])
        sp1, sp2 = span(s1, w1, w2), span(s2, w1, w2)
        ok = sp1 is not None and sp2 is not None
        if not ok:
            n_bad_span += 1
        sim = difflib.SequenceMatcher(None, s1, s2).ratio()
        if sim > 0.8:
            n_dupe += 1
        rows.append(dict(word1=w1, word2=w2, set=r["set"],
                         span_ok=ok, similarity=round(sim, 3),
                         prefix1=len(s1[:sp1[0]].strip()) if sp1 else -1,
                         prefix2=len(s2[:sp2[0]].strip()) if sp2 else -1,
                         sent_1=s1, sent_2=s2))
        flag = "" if ok else "   [SPAN NOT FOUND]"
        dup = "   [NEAR-DUPLICATE]" if sim > 0.8 else ""
        print(f"[{j+1:3d}] {w1} and {w2}  ({r['set']}){flag}{dup}")
        # Window on the span. Truncating from the left hid the binomial on
        # long sentences, which defeats the point of a visual check.
        def win(sent, sp):
            if sp is None:
                return sent[:170]
            a = max(0, sp[0] - 90)
            b = min(len(sent), sp[1] + 60)
            return ("..." if a else "") + sent[a:b] + ("..." if b < len(sent) else "")
        print(f"      1: {win(s1, sp1)}")
        print(f"      2: {win(s2, sp2)}")
        print()

    out = pd.DataFrame(rows)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    print("=" * 60)
    print(f"span found in both sentences : {int(out.span_ok.sum())}/{len(out)}")
    print(f"near-duplicate pairs (>0.8)  : {n_dupe}/{len(out)}")
    g = out[out.span_ok]
    if len(g):
        print(f"prefix chars, sentence 1     : median {int(g.prefix1.median())}, "
              f"min {int(g.prefix1.min())}")
        print(f"prefix chars, sentence 2     : median {int(g.prefix2.median())}, "
              f"min {int(g.prefix2.min())}")
        print(f"sentence similarity          : median {g.similarity.median():.3f}, "
              f"max {g.similarity.max():.3f}")


if __name__ == "__main__":
    main()
