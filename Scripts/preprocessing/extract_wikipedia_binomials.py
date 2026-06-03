"""
extract_wikipedia_binomials.py
-------------------------------
Extract N+N and V+V binomials from English Wikipedia, then filter to only
keep pairs that are:
  1. Both words in BabyLM vocab  (model has seen both words during training)
  2. Zero occurrences of "A and B" or "B and A" in the BabyLM corpus
     (i.e., not already attested in corpus_binomials.csv)

Process:
  - Stream Wikipedia (wikimedia/wikipedia 20231101.en) to avoid full download
  - Regex first pass: collect all unique (w1, w2) candidate pairs
  - spaCy POS filter: keep only NOUN+NOUN (incl. PROPN) and VERB+VERB
  - Apply vocab and novelty filters
  - Save: Data/wikipedia_novel_binomials.csv

Columns saved: word1, word2, wiki_count  (number of Wikipedia occurrences)
"""

import re
import spacy
import pandas as pd
from pathlib import Path
from collections import Counter
from datasets import load_dataset

PROJECT   = Path(r"D:\PhD Stuff\Linguistics Stuff\binom-corpus-pls")
WIKI_OUT  = PROJECT / "Data" / "wikipedia_novel_binomials.csv"
VOCAB_F   = PROJECT / "Data" / "babylm_vocab.txt"
CORPUS_F  = PROJECT / "Data" / "corpus_binomials.csv"
CHECKPOINT = PROJECT / "Data" / "wiki_candidate_counts.pkl"

CAND_RE  = re.compile(r'\b([a-z]{2,})\s+and\s+([a-z]{2,})\b')
MIN_WORD_LEN = 2

# â”€â”€ Load filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("Loading BabyLM vocab...")
babylm_vocab = set(VOCAB_F.read_text(encoding="utf-8").splitlines())
print(f"  {len(babylm_vocab):,} words")

print("Loading attested BabyLM corpus pairs...")
corpus_df = pd.read_csv(CORPUS_F)
attested  = set(zip(corpus_df["word1"], corpus_df["word2"]))  # already alphabetical
print(f"  {len(attested):,} attested pairs")

# â”€â”€ Step 1: Stream Wikipedia, collect candidate pair counts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if CHECKPOINT.exists():
    import pickle
    print(f"\nCheckpoint found â€” loading candidate counts from {CHECKPOINT}")
    with open(CHECKPOINT, "rb") as f:
        pair_counts = pickle.load(f)
    print(f"  {len(pair_counts):,} unique candidates loaded")
else:
    print("\nStreaming Wikipedia (this will take 60â€“90 min)...")
    print("Regex pass only â€” spaCy POS tagging comes after.\n")

    pair_counts = Counter()
    ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                      split="train", streaming=True)

    for art_i, article in enumerate(ds):
        text = article.get("text", "").lower()
        for m in CAND_RE.finditer(text):
            w1, w2 = m.group(1), m.group(2)
            if w1 == w2:
                continue
            key = (w1, w2) if w1 < w2 else (w2, w1)
            pair_counts[key] += 1

        if (art_i + 1) % 100_000 == 0:
            print(f"  {art_i+1:,} articles  |  {len(pair_counts):,} unique candidates")

    # Save checkpoint
    import pickle
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(pair_counts, f)
    print(f"\nCandidate extraction done. {len(pair_counts):,} unique pairs.")
    print(f"Checkpoint saved â†’ {CHECKPOINT}")

# â”€â”€ Step 2: Vocab + novelty pre-filter (before spaCy, to reduce load) â”€â”€â”€â”€â”€â”€â”€â”€
print("\nPre-filtering candidates (vocab + novelty)...")
prefilt = {
    (w1, w2): cnt
    for (w1, w2), cnt in pair_counts.items()
    if w1 in babylm_vocab
    and w2 in babylm_vocab
    and (w1, w2) not in attested
}
print(f"  After vocab + novelty filter: {len(prefilt):,} unique pairs")

# â”€â”€ Step 3: spaCy POS filter (word-level caching) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tag each UNIQUE WORD once instead of each pair â€” reduces spaCy calls from
# 13M to ~200-400K unique words, cutting runtime from ~44h to ~5 min.
print("\nLoading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])

NOUN_TAGS = {"NOUN", "PROPN"}

# Collect unique words from the pre-filtered pairs
unique_words = sorted(set(w for pair in prefilt for w in pair))
print(f"Unique words to tag: {len(unique_words):,}")

# Tag each word in context "the {word}" to help disambiguate noun vs verb
word_pos = {}
BATCH_SIZE = 4096
phrases = [f"the {w}" for w in unique_words]
for i in range(0, len(phrases), BATCH_SIZE):
    batch_words  = unique_words[i : i + BATCH_SIZE]
    batch_phrases = phrases[i : i + BATCH_SIZE]
    for word, doc in zip(batch_words, nlp.pipe(batch_phrases)):
        # second token is the word itself (first is "the")
        toks = [t for t in doc if t.text != "the"]
        word_pos[word] = toks[0].pos_ if toks else "X"
    if (i // BATCH_SIZE + 1) % 10 == 0:
        print(f"  {min(i+BATCH_SIZE, len(unique_words)):,} / {len(unique_words):,} words tagged")

print(f"Word POS tagging done.")

# Filter pairs using cached POS
pairs_list = list(prefilt.items())
kept = []
for (w1, w2), cnt in pairs_list:
    p1 = word_pos.get(w1, "X")
    p2 = word_pos.get(w2, "X")
    if (p1 in NOUN_TAGS and p2 in NOUN_TAGS) or (p1 == "VERB" and p2 == "VERB"):
        kept.append({"word1": w1, "word2": w2, "pos1": p1, "pos2": p2,
                     "wiki_count": cnt})

print(f"\nAfter POS filter: {len(kept):,} novel binomials")

# â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
out_df = pd.DataFrame(kept).sort_values(["word1", "word2"]).reset_index(drop=True)
out_df.to_csv(WIKI_OUT, index=False)
print(f"Saved â†’ {WIKI_OUT}")

# Summary
print(f"\n=== Summary ===")
print(f"Wikipedia candidates (regex):         {len(pair_counts):,}")
print(f"After vocab + novelty filter:         {len(prefilt):,}")
print(f"After POS filter (N+N or V+V):        {len(kept):,}")
noun_n = sum(1 for r in kept if r["pos1"] in NOUN_TAGS)
verb_n = sum(1 for r in kept if r["pos1"] == "VERB")
print(f"  NOUN+NOUN (incl. PROPN): {noun_n:,}")
print(f"  VERB+VERB:               {verb_n:,}")
