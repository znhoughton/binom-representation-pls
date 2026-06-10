"""
merge_extraction_shards.py
--------------------------
Merge per-shard CSV files produced by the parallel extraction scripts into
final corpus_binomials.csv and wikipedia_novel_binomials.csv.

For corpus shards:   sums freq_w1_w2 / freq_w2_w1 across shards for the same pair
For wikipedia shards: sums wiki_count across shards for the same pair
In both cases: keeps the first example_sentence and pos tags encountered.

Usage:
  python merge_extraction_shards.py --mode corpus --num-shards 12
  python merge_extraction_shards.py --mode novel  --num-shards 12
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT  = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / "Data"


def merge_corpus(num_shards):
    pattern = DATA_DIR / f"corpus_binomials_shard*of{num_shards:02d}.csv"
    files   = sorted(DATA_DIR.glob(f"corpus_binomials_shard*of{num_shards:02d}.csv"))
    if not files:
        raise FileNotFoundError(f"No shard files found matching {pattern}")

    print(f"Merging {len(files)} corpus shard files...")
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    merged = (combined
              .groupby(["word1", "word2"], sort=False)
              .agg(
                  freq_w1_w2      = ("freq_w1_w2",      "sum"),
                  freq_w2_w1      = ("freq_w2_w1",      "sum"),
                  pos1            = ("pos1",             "first"),
                  pos2            = ("pos2",             "first"),
                  example_sentence= ("example_sentence", "first"),
              )
              .reset_index()
              .sort_values(["word1", "word2"])
              .reset_index(drop=True))

    out = DATA_DIR / "corpus_binomials.csv"
    merged.to_csv(out, index=False)
    print(f"  {len(merged):,} unique pairs -> {out}")

    # Clean up shard files
    for f in files:
        f.unlink()
    print(f"  Deleted {len(files)} shard files.")
    return merged


def merge_novel(num_shards):
    files = sorted(DATA_DIR.glob(f"wikipedia_novel_binomials_shard*of{num_shards:02d}.csv"))
    if not files:
        raise FileNotFoundError("No wikipedia shard files found.")

    print(f"Merging {len(files)} wikipedia shard files...")
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    merged = (combined
              .groupby(["word1", "word2"], sort=False)
              .agg(
                  wiki_count      = ("wiki_count",       "sum"),
                  pos1            = ("pos1",             "first"),
                  pos2            = ("pos2",             "first"),
                  example_sentence= ("example_sentence", "first"),
              )
              .reset_index()
              .sort_values(["word1", "word2"])
              .reset_index(drop=True))

    out = DATA_DIR / "wikipedia_novel_binomials.csv"
    merged.to_csv(out, index=False)
    print(f"  {len(merged):,} unique pairs -> {out}")

    for f in files:
        f.unlink()
    print(f"  Deleted {len(files)} shard files.")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["corpus", "novel"], required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    if args.mode == "corpus":
        merge_corpus(args.num_shards)
    else:
        merge_novel(args.num_shards)


if __name__ == "__main__":
    main()
