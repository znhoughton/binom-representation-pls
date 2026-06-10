"""
Compute feature differences for novel pairs and save to CSV.
R script (feature_correlations.R) handles the actual correlation analysis.

Features:
  - babylm_log_freq : log(BabyLM corpus count + 1)
  - word_length     : character count
  - n_syllables     : syllable count from CMU pronouncing dict
  - animacy_binary  : 1=animate, 0=inanimate (animacy_word_list.csv)

Usage:
  python Scripts/compute_delta_features.py
  python Scripts/compute_delta_features.py --slug znhoughton_opt-babylm-350m-20eps-seed964
"""

import argparse
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import cmudict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--layer", default="last", choices=["last", "second_to_last"])
parser.add_argument("--brysbaert", default=None)
parser.add_argument("--kuperman",  default=None)
args = parser.parse_args()

BASE           = Path(__file__).resolve().parents[2]
out_dir        = BASE / "Results" / args.slug / f"layer_{args.layer}"
out_dir.mkdir(parents=True, exist_ok=True)

NOVEL_PLS      = str(out_dir / "novel_pls_scores.csv")
BABYLM_FREQS   = str(BASE / "Data/babylm_word_freqs.csv")
ANIMACY_NORMS  = str(BASE / "Data/animacy_word_list.csv")
OUT_PATH       = str(out_dir / "delta_features.csv")

BRYSBAERT_PATH = args.brysbaert
KUPERMAN_PATH  = args.kuperman
print(f"Slug: {args.slug}")

nltk.download("cmudict", quiet=True)
cmu = cmudict.dict()

def n_syllables(word):
    entries = cmu.get(word.lower())
    if not entries:
        return None
    return sum(1 for ph in entries[0] if ph[-1].isdigit())

print("Loading novel PLS scores...")
df = pd.read_csv(NOVEL_PLS)
print(f"  {len(df):,} pairs")

words = pd.unique(df[["word1", "word2"]].values.ravel())
wf = pd.DataFrame({"word": words}).set_index("word")

# BabyLM log frequency
bl = pd.read_csv(BABYLM_FREQS).set_index("word")
wf = wf.join(bl[["babylm_log_freq"]], how="left")
print(f"  BabyLM freq coverage: {wf['babylm_log_freq'].notna().mean():.1%}")

# word length
wf["word_length"] = [len(w) for w in words]

# syllable count
wf["n_syllables"] = [n_syllables(w) for w in words]
print(f"  Syllable coverage: {wf['n_syllables'].notna().mean():.1%}")

# animacy
import os
if os.path.exists(ANIMACY_NORMS):
    an = pd.read_csv(ANIMACY_NORMS)[["word", "animacy"]]
    an["word"] = an["word"].str.lower()
    an = an.drop_duplicates("word").set_index("word")
    an["animacy_binary"] = an["animacy"].map({"animate": 1, "inanimate": 0})
    wf = wf.join(an[["animacy_binary"]], how="left")
    print(f"  Animacy coverage: {wf['animacy_binary'].notna().mean():.1%}")
else:
    wf["animacy_binary"] = np.nan

# optional Brysbaert
if BRYSBAERT_PATH and os.path.exists(BRYSBAERT_PATH):
    bn = pd.read_csv(BRYSBAERT_PATH)
    bn.columns = [c.strip() for c in bn.columns]
    word_col = [c for c in bn.columns if c.lower() in ("word", "words")][0]
    conc_col = [c for c in bn.columns if "conc" in c.lower() and "m" in c.lower()][0]
    bn = bn[[word_col, conc_col]].rename(columns={word_col: "word", conc_col: "concreteness"})
    bn["word"] = bn["word"].str.lower()
    bn = bn.drop_duplicates("word").set_index("word")
    wf = wf.join(bn, how="left")
    print(f"  Concreteness coverage: {wf['concreteness'].notna().mean():.1%}")
else:
    wf["concreteness"] = np.nan

# optional Kuperman AoA
if KUPERMAN_PATH and os.path.exists(KUPERMAN_PATH):
    kup = pd.read_csv(KUPERMAN_PATH)
    kup.columns = [c.strip() for c in kup.columns]
    word_col = [c for c in kup.columns if c.lower() in ("word", "words")][0]
    aoa_col  = [c for c in kup.columns if "aoa" in c.lower() or "rating" in c.lower()][0]
    kup = kup[[word_col, aoa_col]].rename(columns={word_col: "word", aoa_col: "aoa"})
    kup["word"] = kup["word"].str.lower()
    kup = kup.drop_duplicates("word").set_index("word")
    wf = wf.join(kup, how="left")
    print(f"  AoA coverage: {wf['aoa'].notna().mean():.1%}")
else:
    wf["aoa"] = np.nan

# compute delta features
feat_cols = [c for c in wf.columns if wf[c].notna().any()]
print(f"\nFeatures: {feat_cols}")

out = df[["word1", "word2", "preference"] + [f"C{i}" for i in range(1, 16)]].copy()
for col in feat_cols:
    v1 = df["word1"].map(wf[col])
    v2 = df["word2"].map(wf[col])
    out[f"delta_{col}"] = v1 - v2

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(out):,} rows with {len(feat_cols)} delta features to {OUT_PATH}")
print("Done.")