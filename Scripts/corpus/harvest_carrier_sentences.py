#!/usr/bin/env python3
"""
harvest_carrier_sentences.py
----------------------------
Collect TWO carrier sentences per binomial, each with real left context.

Why
---
The probe's step-0 failure traces to two properties of the current item set.
29% of novel items have the binomial at sentence start, so they share one
constant context and the probe's task there is degenerate. And the target
preference is read off the same forward pass the probe's input comes from, so
the probe can recompute it. Two prefixed sentences per pair fix both: filter on
the prefix, and measure the preference in a different sentence from the one
probed.

Why this is cheap
-----------------
The original extraction ran benepar on every candidate to confirm each pair is
a minimal coordinate phrase. That verification has already happened: every pair
in the two CSVs passed it. Finding further occurrences of an already-validated
pair needs only a regex and a sentence boundary, so this is a streaming scan
rather than a parse, and runs in a fraction of the original cost.

Availability is not the constraint people expect. BabyLM counts say 71% of
attested pairs occur once, but that is the wrong corpus: those same pairs have
a median Pile count of 328, and 94.6% of the BabyLM singletons occur at least
twice. Wikipedia is tried first here because it is already wired into the
pipeline and needs no new corpus access.

Caveat to decide on later
-------------------------
Harvested sentences are regex matches, not benepar-verified minimal coordinate
phrases, so a match could in principle span a clause boundary ("he ate bread
and butter melted"). The pair itself is still validated; only this occurrence
is not. Verifying just the harvested candidates with benepar is a cheap
follow-up if the counts come out high enough to be worth it.

Usage
-----
  # Sample first to project coverage, which is what --max-articles is for
  python Scripts/corpus/harvest_carrier_sentences.py --max-articles 50000

  # Full pass (server job)
  python Scripts/corpus/harvest_carrier_sentences.py

  # Shard it
  python Scripts/corpus/harvest_carrier_sentences.py --num-shards 8 --shard-index 0
"""

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DATA = BASE / "Data"

SEED = 964
# Matches "<word> and <word>". One regex over the sentence beats 389k per-pair
# regexes by several orders of magnitude.
PAIR_RE = re.compile(r"\b([A-Za-z]+)\s+and\s+([A-Za-z]+)\b", re.IGNORECASE)
# Rule-based sentence split. Good enough to locate a prefix, and far cheaper
# than loading a spaCy model; the original extractor's parse is not needed here.
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

MAX_SENT_TOKENS = 40          # matches the original extraction
MAX_LINE_CHARS = MAX_SENT_TOKENS * 7


def load_targets():
    """All pairs from both sets, keyed by the sorted (w1, w2) tuple."""
    out = {}
    for name, fn in [("corpus", "corpus_binomials.csv"),
                     ("novel", "wikipedia_novel_binomials.csv")]:
        df = pd.read_csv(DATA / fn)
        for w1, w2 in zip(df["word1"].astype(str), df["word2"].astype(str)):
            key = tuple(sorted((w1.lower().strip(), w2.lower().strip())))
            out.setdefault(key, name)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-prefix", type=int, default=10, dest="min_prefix",
                   help="Minimum characters of left context (default 10, the "
                        "threshold validated in the step-0 pilot)")
    p.add_argument("--per-pair", type=int, default=2, dest="per_pair",
                   help="Sentences to keep per pair (default 2)")
    p.add_argument("--max-similarity", type=float, default=0.8,
                   dest="max_similarity",
                   help="Reject a further sentence this similar to one already "
                        "kept for the pair (default 0.8)")
    p.add_argument("--max-articles", type=int, default=None, dest="max_articles",
                   help="Stop after N articles. Use for a coverage projection.")
    p.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    p.add_argument("--shard-index", type=int, default=0, dest="shard_index")
    p.add_argument("--report-every", type=int, default=50000, dest="report_every",
                   help="Articles between coverage reports")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    targets = load_targets()
    print(f"targets: {len(targets):,} unique pairs", flush=True)

    out_path = Path(args.out) if args.out else (
        DATA / (f"carrier_sentences_shard{args.shard_index:02d}"
                f"of{args.num_shards:02d}.csv" if args.num_shards > 1
                else "carrier_sentences.csv"))

    found = {}          # key -> list of sentences
    seen = {}           # key -> set of sentences already taken, for dedup

    print("Streaming Wikipedia...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)
    if args.num_shards > 1:
        ds = ds.shuffle(seed=SEED, buffer_size=10_000)
        ds = ds.shard(num_shards=args.num_shards, index=args.shard_index)

    t0 = time.perf_counter()
    n_art = 0
    for article in tqdm(ds, unit="article"):
        n_art += 1
        if args.max_articles and n_art > args.max_articles:
            break

        for line in article.get("text", "").splitlines():
            if " and " not in line.lower():
                continue
            for sent in SENT_RE.split(line.strip()):
                if not sent or len(sent) > MAX_LINE_CHARS:
                    continue
                for m in PAIR_RE.finditer(sent):
                    w1, w2 = m.group(1).lower(), m.group(2).lower()
                    key = (w1, w2) if w1 < w2 else (w2, w1)
                    if key not in targets:
                        continue
                    got = found.get(key)
                    if got is not None and len(got) >= args.per_pair:
                        continue
                    if len(sent[:m.start()].strip()) < args.min_prefix:
                        continue
                    bucket = seen.setdefault(key, set())
                    if sent in bucket:
                        continue
                    # Wikipedia carries a lot of templated and mirrored text, so
                    # exact dedup is not enough: "Genomes are circular and
                    # segmented, around 2.0-31kb" and the same line with
                    # different numbers are different strings but the same
                    # context, which makes a cross-context measure vacuous for
                    # that pair. Runs at most once per pair reaching its second
                    # sentence, so the cost is negligible.
                    prior = found.get(key) or []
                    if any(difflib.SequenceMatcher(None, sent, q).ratio()
                           > args.max_similarity for q in prior):
                        continue
                    bucket.add(sent)
                    found.setdefault(key, []).append(sent)

        if n_art % args.report_every == 0:
            n1 = len(found)
            n2 = sum(1 for v in found.values() if len(v) >= args.per_pair)
            el = time.perf_counter() - t0
            print(f"\n  {n_art:,} articles  {el/60:.1f}m  "
                  f">=1 sentence: {n1:,} ({n1/len(targets):.1%})  "
                  f">={args.per_pair} sentences: {n2:,} ({n2/len(targets):.1%})",
                  flush=True)

    rows = []
    for key, sents in found.items():
        r = {"word1": key[0], "word2": key[1], "set": targets[key],
             "n_found": len(sents)}
        for i in range(args.per_pair):
            r[f"sent_{i+1}"] = sents[i] if i < len(sents) else None
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    n1 = len(found)
    n2 = sum(1 for v in found.values() if len(v) >= args.per_pair)
    print(f"\nscanned {n_art:,} articles in "
          f"{(time.perf_counter()-t0)/60:.1f}m", flush=True)
    print(f"  >=1 sentence  : {n1:,} / {len(targets):,} ({n1/len(targets):.1%})")
    print(f"  >={args.per_pair} sentences : {n2:,} / {len(targets):,} "
          f"({n2/len(targets):.1%})")
    if not df.empty:
        for s in ["corpus", "novel"]:
            sub = df[df["set"] == s]
            tot = sum(1 for v in targets.values() if v == s)
            full = (sub["n_found"] >= args.per_pair).sum()
            print(f"  {s:7s}: >=1 {len(sub):,}/{tot:,} ({len(sub)/tot:.1%})   "
                  f">={args.per_pair} {full:,}/{tot:,} ({full/tot:.1%})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
