"""
extract_corpus_binomials.py
---------------------------
Extract all N+N and V+V binomials from the BabyLM corpus.

Steps:
  1. Stream the HuggingFace dataset znhoughton/babylm-150m-v3
  2. Regex-extract all "word and word" candidates
  3. POS-tag each unique pair phrase ("w1 and w2") via spaCy
  4. Keep only NOUN+NOUN (incl. PROPN) and VERB+VERB pairs
  5. Store in alphabetical order (word1 < word2)
  6. Save to Data/corpus_binomials.csv with corpus frequency counts

Output columns:
  word1, word2             -- alphabetically ordered
  freq_w1_w2               -- count of "word1 and word2" in corpus
  freq_w2_w1               -- count of "word2 and word1" in corpus
  pos1, pos2               -- spaCy POS tags
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm

OUT_PATH = Path(__file__).resolve().parent.parent / "Data" / "corpus_binomials.csv"
PATTERN  = re.compile(r'\b([a-z]{2,})\s+and\s+([a-z]{2,})\b')
BATCH_SIZE = 512

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# ── Step 1: stream corpus, collect all "w1 and w2" counts ────────────────────
print("Streaming corpus...")
pair_counts = Counter()

dataset = load_dataset("znhoughton/babylm-150m-v3", split="train+dev",
                       streaming=False)

for row in tqdm(dataset, desc="Extracting candidates", unit="rows"):
    text = row["text"].lower()
    for w1, w2 in PATTERN.findall(text):
        if w1 != w2:
            pair_counts[(w1, w2)] += 1

print(f"Raw ordered pairs found: {len(pair_counts)}")

# ── Step 2: unique unordered pairs ───────────────────────────────────────────
# For each unordered pair {w1, w2}, store both orderings' counts
unordered = {}
for (w1, w2), cnt in pair_counts.items():
    key = tuple(sorted([w1, w2]))
    if key not in unordered:
        unordered[key] = {"freq_w1_w2": 0, "freq_w2_w1": 0}
    alpha1, alpha2 = key
    if (w1, w2) == (alpha1, alpha2):
        unordered[key]["freq_w1_w2"] += cnt
    else:
        unordered[key]["freq_w2_w1"] += cnt

unique_pairs = list(unordered.keys())
print(f"Unique unordered pairs: {len(unique_pairs)}")

# ── Step 3: POS-tag each pair phrase ─────────────────────────────────────────
print("POS tagging pair phrases...")
pair_pos = {}
phrases   = [f"{w1} and {w2}" for w1, w2 in unique_pairs]

for i in tqdm(range(0, len(phrases), BATCH_SIZE), desc="POS tagging"):
    batch_pairs   = unique_pairs[i : i + BATCH_SIZE]
    batch_phrases = phrases[i : i + BATCH_SIZE]
    for pair, doc in zip(batch_pairs, nlp.pipe(batch_phrases)):
        tokens = [t for t in doc if t.text != "and"]
        if len(tokens) >= 2:
            pair_pos[pair] = (tokens[0].pos_, tokens[1].pos_)

# ── Step 4: filter to N+N or V+V ─────────────────────────────────────────────
rows = []
for pair, (pos1, pos2) in pair_pos.items():
    is_nn = pos1 in ("NOUN", "PROPN") and pos2 in ("NOUN", "PROPN")
    is_vv = pos1 == "VERB" and pos2 == "VERB"
    if is_nn or is_vv:
        w1, w2 = pair
        rows.append({
            "word1":      w1,
            "word2":      w2,
            "freq_w1_w2": unordered[pair]["freq_w1_w2"],
            "freq_w2_w1": unordered[pair]["freq_w2_w1"],
            "pos1":       pos1,
            "pos2":       pos2,
        })

df = pd.DataFrame(rows).sort_values(["word1", "word2"]).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(df)} pairs to {OUT_PATH}")
print(df["pos1"].value_counts().to_string())