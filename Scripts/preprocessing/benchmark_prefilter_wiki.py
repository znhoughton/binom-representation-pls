"""
benchmark_prefilter_wiki.py
---------------------------
Same as benchmark_prefilter.py but for the Wikipedia corpus.
Wikipedia already does line-level pre-filtering before benepar, so
we measure: of the lines that reach benepar, what fraction have a
valid WORD_RE + OPEN_CLASS + "and" triplet (+ vocab + novelty filters)?

No benepar involved — should complete in a few minutes.
"""

import re
import time
from pathlib import Path

import spacy
from datasets import load_dataset
from tqdm import tqdm

PROJECT        = Path(__file__).resolve().parent.parent.parent
DATA_DIR       = PROJECT / "Data"
VOCAB_F        = DATA_DIR / "babylm_vocab.txt"
CORPUS_F       = DATA_DIR / "corpus_binomials.csv"

WORD_RE        = re.compile(r'^[a-z]{2,}$')
OPEN_CLASS     = {"NOUN", "VERB", "ADJ", "ADV"}
MAX_SENT_TOKENS = 40
MAX_LINE_CHARS  = MAX_SENT_TOKENS * 7
BATCH_SIZE      = 1024
SAMPLE_ARTICLES = 50_000

print("Loading BabyLM vocab...")
babylm_vocab = set(VOCAB_F.read_text(encoding="utf-8").splitlines())
print(f"  {len(babylm_vocab):,} words")

print("Loading attested corpus pairs...")
if CORPUS_F.exists():
    import pandas as pd
    corpus_df = pd.read_csv(CORPUS_F)
    attested  = set(zip(corpus_df["word1"], corpus_df["word2"]))
    print(f"  {len(attested):,} attested pairs (will be excluded)")
else:
    attested = set()
    print("  corpus_binomials.csv not found — novelty filter disabled")

print("Loading spaCy (no benepar)...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "senter"])
print("Loaded.")

print(f"Streaming Wikipedia (sampling {SAMPLE_ARTICLES:,} articles)...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en",
                       split="train", streaming=True)

articles_seen  = 0
lines_to_benepar = 0
total_sents    = 0
sents_with_and = 0
sents_candidate = 0

buffer = []
t0 = time.time()

for article in tqdm(dataset, total=SAMPLE_ARTICLES, desc="Articles", unit="article"):
    if articles_seen >= SAMPLE_ARTICLES:
        break
    articles_seen += 1

    lines = [l.strip() for l in article.get("text", "").splitlines()
             if " and " in l.lower() and len(l) <= MAX_LINE_CHARS]
    lines_to_benepar += len(lines)
    buffer.extend(lines)

    while len(buffer) >= BATCH_SIZE:
        batch = buffer[:BATCH_SIZE]
        buffer = buffer[BATCH_SIZE:]
        for doc in nlp.pipe(batch, batch_size=BATCH_SIZE):
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
                    w1, w2 = toks[i].text.lower(), toks[i+2].text.lower()
                    if not (WORD_RE.match(w1) and WORD_RE.match(w2)):
                        continue
                    if w1 not in babylm_vocab or w2 not in babylm_vocab:
                        continue
                    if toks[i].pos_ not in OPEN_CLASS or toks[i+2].pos_ not in OPEN_CLASS:
                        continue
                    key = tuple(sorted([w1, w2]))
                    if key in attested:
                        continue
                    sents_candidate += 1
                    break

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
                w1, w2 = toks[i].text.lower(), toks[i+2].text.lower()
                if not (WORD_RE.match(w1) and WORD_RE.match(w2)):
                    continue
                if w1 not in babylm_vocab or w2 not in babylm_vocab:
                    continue
                if toks[i].pos_ not in OPEN_CLASS or toks[i+2].pos_ not in OPEN_CLASS:
                    continue
                key = tuple(sorted([w1, w2]))
                if key in attested:
                    continue
                sents_candidate += 1
                break

elapsed = time.time() - t0

print(f"\n{'='*55}")
print(f"Articles sampled:        {articles_seen:>10,}")
print(f"Lines sent to benepar:   {lines_to_benepar:>10,}")
print(f"{'='*55}")
print(f"Total sentences (benepar-eligible length):")
print(f"  All:                   {total_sents:>10,}")
print(f"  Have 'and':            {sents_with_and:>10,}  ({sents_with_and/max(total_sents,1)*100:.1f}%)")
print(f"  Valid triplet (->benepar): {sents_candidate:>7,}  ({sents_candidate/max(total_sents,1)*100:.1f}%)")
print(f"{'='*55}")
if sents_candidate > 0:
    print(f"Benepar work reduction:  {(1 - sents_candidate/total_sents)*100:.1f}% fewer sentences")
    print(f"Estimated speedup:       {total_sents/sents_candidate:.1f}x")
print(f"Elapsed:                 {elapsed:.0f}s")
print(f"Throughput:              {articles_seen/elapsed:.0f} articles/sec (no benepar)")