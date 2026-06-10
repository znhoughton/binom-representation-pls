"""
benchmark_prefilter.py
----------------------
Measures how many sentences would reach benepar under the two-stage approach.
Runs spaCy WITHOUT benepar on a sample of the corpus and counts:
  - total sentences seen
  - sentences with " and " in them
  - sentences with a valid WORD_RE + OPEN_CLASS + "and" + OPEN_CLASS + WORD_RE triplet

This tells us the benepar savings if we add a fast pre-filter stage.
No benepar involved — should complete in a minute or two.
"""

import re
import time
from pathlib import Path

import spacy
from datasets import load_dataset
from tqdm import tqdm

PROJECT       = Path(__file__).resolve().parent.parent.parent
WORD_RE       = re.compile(r'^[a-z]{2,}$')
OPEN_CLASS    = {"NOUN", "VERB", "ADJ", "ADV"}
MAX_SENT_TOKENS = 40
SAMPLE_DOCS   = 200_000  # docs to sample (fast since no benepar)
BATCH_SIZE    = 1024

print("Loading spaCy (no benepar)...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "senter"])
print("Loaded.")

print("Loading corpus sample...")
dataset = load_dataset("znhoughton/babylm-150m-v3", split="train+dev", streaming=False)
print(f"  {len(dataset):,} total docs")

# ── Counters ────────────────────────────────────────────────────────────────
docs_seen        = 0
docs_with_and    = 0
total_sents      = 0
sents_with_and   = 0
sents_candidate  = 0   # pass WORD_RE + OPEN_CLASS check — would go to benepar

buffer = []
t0 = time.time()

for row in tqdm(dataset, total=SAMPLE_DOCS, desc="Sampling", unit="doc"):
    if docs_seen >= SAMPLE_DOCS:
        break
    docs_seen += 1
    text = row["text"]
    if " and " not in text.lower():
        continue
    docs_with_and += 1
    buffer.append(text)

    if len(buffer) >= BATCH_SIZE:
        for doc in nlp.pipe(buffer, batch_size=BATCH_SIZE):
            for sent in doc.sents:
                toks = list(sent)
                n = len(toks)
                if n < 3 or n > MAX_SENT_TOKENS:
                    continue
                total_sents += 1
                has_and = any(t.text.lower() == "and" for t in toks)
                if has_and:
                    sents_with_and += 1
                # Check for valid triplet
                for i in range(n - 2):
                    if toks[i+1].text.lower() != "and":
                        continue
                    if (WORD_RE.match(toks[i].text.lower()) and
                        WORD_RE.match(toks[i+2].text.lower()) and
                        toks[i].pos_ in OPEN_CLASS and
                        toks[i+2].pos_ in OPEN_CLASS):
                        sents_candidate += 1
                        break  # count sentence once even if multiple triplets
        buffer.clear()

# flush remainder
if buffer:
    for doc in nlp.pipe(buffer, batch_size=BATCH_SIZE):
        for sent in doc.sents:
            toks = list(sent)
            n = len(toks)
            if n < 3 or n > MAX_SENT_TOKENS:
                continue
            total_sents += 1
            has_and = any(t.text.lower() == "and" for t in toks)
            if has_and:
                sents_with_and += 1
            for i in range(n - 2):
                if toks[i+1].text.lower() != "and":
                    continue
                if (WORD_RE.match(toks[i].text.lower()) and
                    WORD_RE.match(toks[i+2].text.lower()) and
                    toks[i].pos_ in OPEN_CLASS and
                    toks[i+2].pos_ in OPEN_CLASS):
                    sents_candidate += 1
                    break

elapsed = time.time() - t0

print(f"\n{'='*55}")
print(f"Docs sampled:            {docs_seen:>10,}")
print(f"Docs with ' and ':       {docs_with_and:>10,}  ({docs_with_and/docs_seen*100:.1f}%)")
print(f"{'='*55}")
print(f"Total sentences (benepar-eligible length):")
print(f"  All:                   {total_sents:>10,}")
print(f"  Have 'and':            {sents_with_and:>10,}  ({sents_with_and/total_sents*100:.1f}%)")
print(f"  Valid triplet (->benepar): {sents_candidate:>7,}  ({sents_candidate/total_sents*100:.1f}%)")
print(f"{'='*55}")
if sents_candidate > 0:
    print(f"Benepar work reduction:  {(1 - sents_candidate/total_sents)*100:.1f}% fewer sentences")
    print(f"Estimated speedup:       {total_sents/sents_candidate:.1f}x")
print(f"Elapsed:                 {elapsed:.0f}s")
print(f"Throughput:              {docs_seen/elapsed:.0f} docs/sec (no benepar)")