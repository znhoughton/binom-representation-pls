"""
extract_wikipedia_binomials.py
-------------------------------
Extract novel open-class binomials from English Wikipedia using the same
constituency-based scheme as extract_corpus_binomials.py. Designed to be
run in parallel shards via run_parallel_extraction.py.

A pair is kept only when both words are:
  1. Lowercase [a-z]{2+}, open-class (NOUN, VERB, ADJ, ADV)
  2. In the BabyLM vocabulary
  3. NOT already attested in corpus_binomials.csv
  4. Genuinely coordinate: LCA spans EXACTLY [W1, and, W2]

Output (per shard): Data/wikipedia_novel_binomials_shard{I:02d}of{N:02d}.csv
Final merged output: Data/wikipedia_novel_binomials.csv (written by merge_extraction_shards.py)

Usage:
  python extract_wikipedia_binomials.py                        # single process
  python extract_wikipedia_binomials.py --num-shards 12 --shard-index 3
"""

import argparse
import re
from pathlib import Path

import spacy
import benepar
import pandas as pd
from nltk import Tree
from datasets import load_dataset
from tqdm import tqdm

PROJECT        = Path(__file__).resolve().parent.parent.parent
DATA_DIR       = PROJECT / "Data"
VOCAB_F        = DATA_DIR / "babylm_vocab.txt"
CORPUS_F       = DATA_DIR / "corpus_binomials.csv"

WORD_RE        = re.compile(r'^[a-z]{2,}$')
PHRASE_LABELS  = {"NP", "VP", "ADJP", "ADVP", "PP", "NX", "QP", "UCP", "CONJP"}
OPEN_CLASS     = {"NOUN", "VERB", "ADJ", "ADV"}
MAX_SENT_TOKENS = 40
MAX_LINE_CHARS  = MAX_SENT_TOKENS * 7
BATCH_SIZE      = 1024


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-shards",  type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
args = parser.parse_args()

num_shards  = args.num_shards
shard_index = args.shard_index

out_path = (DATA_DIR / f"wikipedia_novel_binomials_shard{shard_index:02d}of{num_shards:02d}.csv"
            if num_shards > 1 else DATA_DIR / "wikipedia_novel_binomials.csv")

print(f"Shard {shard_index + 1}/{num_shards} -> {out_path.name}")


# ── Filters ───────────────────────────────────────────────────────────────────
print("Loading BabyLM vocab...")
babylm_vocab = set(VOCAB_F.read_text(encoding="utf-8").splitlines())
print(f"  {len(babylm_vocab):,} words")

print("Loading attested corpus pairs...")
if CORPUS_F.exists():
    corpus_df = pd.read_csv(CORPUS_F)
    attested  = set(zip(corpus_df["word1"], corpus_df["word2"]))
    print(f"  {len(attested):,} attested pairs (will be excluded)")
else:
    attested = set()
    print("  corpus_binomials.csv not found — novelty filter disabled")


# ── Models ────────────────────────────────────────────────────────────────────
print("Loading spaCy (fast, no benepar)...")
nlp_fast = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "senter"])
print("Loading spaCy + benepar...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "senter"])
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
print("Models loaded.")


# ── Constituency helper ───────────────────────────────────────────────────────
def find_lca_info(parse_string, idx1, idx2):
    try:
        tree      = Tree.fromstring(parse_string)
        positions = tree.treepositions("leaves")
        if max(idx1, idx2) >= len(positions):
            return None, None, None
        p1, p2 = positions[idx1], positions[idx2]
        depth = 0
        for a, b in zip(p1, p2):
            if a == b: depth += 1
            else:      break
        path = p1[:depth]
        node = tree
        for i in path: node = node[i]
        leaves = [i for i, pos in enumerate(positions) if pos[:depth] == path]
        return node.label(), leaves[0], leaves[-1]
    except Exception:
        return None, None, None


# ── Doc processing ────────────────────────────────────────────────────────────
def process_doc(doc, pairs):
    for sent in doc.sents:
        toks = list(sent)
        n    = len(toks)
        if n < 3 or n > MAX_SENT_TOKENS:
            continue

        parse_str = sent._.parse_string

        for i in range(n - 2):
            t_w1, t_and, t_w2 = toks[i], toks[i+1], toks[i+2]
            if t_and.text.lower() != "and":
                continue
            w1, w2 = t_w1.text.lower(), t_w2.text.lower()
            if not (WORD_RE.match(w1) and WORD_RE.match(w2)):
                continue
            if w1 not in babylm_vocab or w2 not in babylm_vocab:
                continue
            key = tuple(sorted([w1, w2]))
            if key in attested:
                continue
            pos1, pos2 = t_w1.pos_, t_w2.pos_
            if pos1 not in OPEN_CLASS or pos2 not in OPEN_CLASS:
                continue
            lca, lca_first, lca_last = find_lca_info(parse_str, i, i+2)
            if lca not in PHRASE_LABELS or lca_first != i or lca_last != i+2:
                continue

            if key not in pairs:
                pairs[key] = {"wiki_count": 0, "pos1": pos1, "pos2": pos2,
                              "example_sentence": sent.text.strip()}
            pairs[key]["wiki_count"] += 1


def get_candidate_sents(texts):
    """Stage 1: fast spaCy (no benepar) — return sentence texts with a valid triplet."""
    candidates = []
    for doc in nlp_fast.pipe(texts, batch_size=len(texts)):
        for sent in doc.sents:
            toks = list(sent)
            n = len(toks)
            if n < 3 or n > MAX_SENT_TOKENS:
                continue
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
                candidates.append(sent.text)
                break
    return candidates


def flush_batch(buffer, pairs):
    if not buffer:
        return
    try:
        candidates = get_candidate_sents(buffer)
        if not candidates:
            return
        for doc in nlp.pipe(candidates, batch_size=len(candidates)):
            process_doc(doc, pairs)
    except Exception:
        for text in buffer:
            try:
                for sent_text in get_candidate_sents([text]):
                    try:
                        process_doc(nlp(sent_text), pairs)
                    except Exception:
                        pass
            except Exception:
                pass


# ── Extraction ────────────────────────────────────────────────────────────────
print("Streaming Wikipedia...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.en",
                       split="train", streaming=True)
if num_shards > 1:
    dataset = dataset.shuffle(seed=964, buffer_size=10_000)
    dataset = dataset.shard(num_shards=num_shards, index=shard_index)

pairs  = {}
buffer = []

for article in tqdm(dataset, desc=f"Shard {shard_index}", unit="article"):
    lines = [l.strip() for l in article.get("text", "").splitlines()
             if " and " in l.lower() and len(l) <= MAX_LINE_CHARS]
    buffer.extend(lines)
    if len(buffer) >= BATCH_SIZE:
        flush_batch(buffer[:BATCH_SIZE], pairs)
        buffer = buffer[BATCH_SIZE:]

flush_batch(buffer, pairs)

print(f"\nFound {len(pairs):,} unique novel pairs.")

# ── Save ──────────────────────────────────────────────────────────────────────
rows = [{"word1": k[0], "word2": k[1],
         "wiki_count": v["wiki_count"],
         "pos1": v["pos1"], "pos2": v["pos2"],
         "example_sentence": v["example_sentence"]}
        for k, v in pairs.items()]

pd.DataFrame(rows).sort_values(["word1", "word2"]).reset_index(drop=True)\
  .to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
