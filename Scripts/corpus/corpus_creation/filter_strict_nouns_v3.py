"""
filter_strict_nouns_v3.py
--------------------------
Exclude any word that WordNet can morphologically analyse as ANY verb form
(wn.morphy(word, VERB) is not None). Also excludes contractions.
Restores from the always-NOUN backup first.
"""

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from pathlib import Path
import shutil, random

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

BASE        = Path(__file__).resolve().parents[2]
SLUG        = "znhoughton_opt-babylm-125m-20eps-seed964"
NPZ_CURRENT = BASE / "Data/novel_embeddings" / SLUG / "layer_last.npz"
NPZ_BACKUP  = BASE / "Data/novel_embeddings" / SLUG / "layer_last_always_noun.npz"

shutil.copy2(NPZ_BACKUP, NPZ_CURRENT)
print(f"Restored from backup: {NPZ_BACKUP.name}")

npz       = np.load(NPZ_CURRENT, allow_pickle=True)
w1        = npz["word1"].astype(str)
w2        = npz["word2"].astype(str)
all_words = sorted(set(w1) | set(w2))
print(f"Pairs: {len(w1):,}   Unique words: {len(all_words):,}")

CONTRACTIONS = {
    "wasn", "didn", "doesn", "couldn", "wouldn", "shouldn",
    "isn", "aren", "weren", "haven", "hasn", "hadn", "won",
    "mustn", "needn", "daren", "shan",
}

verb_flagged = set()
for word in all_words:
    if word in CONTRACTIONS:
        verb_flagged.add(word)
    elif wn.morphy(word, wn.VERB) is not None:
        verb_flagged.add(word)

print(f"Flagged (any verb morphy or contraction): {len(verb_flagged):,}")
print(f"Retained words:                           {len(all_words) - len(verb_flagged):,}")

# Sample: flagged words that look like legitimate nouns (false positives check)
random.seed(964)
sample_flagged = random.sample(sorted(verb_flagged), min(30, len(verb_flagged)))
print(f"\nSample of flagged words (may include legit nouns):\n  {sorted(sample_flagged)}")

# Sample: retained words
retained_words = [w for w in all_words if w not in verb_flagged]
sample_retained = random.sample(retained_words, min(30, len(retained_words)))
print(f"\nSample of retained words:\n  {sorted(sample_retained)}")

# Filter pairs
mask = np.array([(a not in verb_flagged) and (b not in verb_flagged)
                 for a, b in zip(w1, w2)])
print(f"\nPairs retained: {mask.sum():,}")
print(f"Pairs dropped:  {(~mask).sum():,}")

# Spot check
rng  = np.random.default_rng(964)
kw1  = w1[mask]; kw2 = w2[mask]; kp = npz["preference"][mask]
idx  = rng.choice(mask.sum(), size=25, replace=False)
print("\n--- 25 random retained pairs ---")
for a, b, p in zip(kw1[idx], kw2[idx], kp[idx]):
    print(f"  {a:24s} + {b:24s}   pref={p:+.2f}")

np.savez_compressed(
    NPZ_CURRENT,
    diff_vecs  = npz["diff_vecs"][mask],
    word1      = kw1,
    word2      = kw2,
    preference = kp,
)
print(f"\nSaved: {mask.sum():,} pairs to {NPZ_CURRENT}")