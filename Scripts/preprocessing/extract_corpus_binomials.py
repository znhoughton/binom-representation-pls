"""
extract_corpus_binomials.py
---------------------------
Extract open-class binomials from the BabyLM corpus with benepar constituency
checking. Designed to be run in parallel shards via run_parallel_extraction.py.

A pair "W1 and W2" is kept only when:
  1. Both words are lowercase [a-z]{2+} and open-class (NOUN, VERB, ADJ, ADV)
  2. The sentence contains "and" (fast pre-filter)
  3. The sentence is <= MAX_SENT_TOKENS tokens
  4. Their LCA in the parse tree is a phrase-level node (NP, VP, ADJP, ...)
  5. The LCA spans EXACTLY [W1, and, W2] — no extra words on either side

Output (per shard): Data/corpus_binomials_shard{I:02d}of{N:02d}.csv
Final merged output: Data/corpus_binomials.csv (written by merge_extraction_shards.py)

Usage:
  python extract_corpus_binomials.py                        # single process
  python extract_corpus_binomials.py --num-shards 12 --shard-index 3
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
WORD_RE        = re.compile(r'^[a-z]{2,}$')
PHRASE_LABELS  = {"NP", "VP", "ADJP", "ADVP", "PP", "NX", "QP", "UCP", "CONJP"}
OPEN_CLASS     = {"NOUN", "VERB", "ADJ", "ADV"}
MAX_SENT_TOKENS = 40   # benepar is O(n^3); cap tightly — valid binomials are short
BATCH_SIZE      = 1024


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-shards",  type=int, default=1)
parser.add_argument("--shard-index", type=int, default=0)
args = parser.parse_args()

num_shards  = args.num_shards
shard_index = args.shard_index

out_path = (DATA_DIR / f"corpus_binomials_shard{shard_index:02d}of{num_shards:02d}.csv"
            if num_shards > 1 else DATA_DIR / "corpus_binomials.csv")

print(f"Shard {shard_index + 1}/{num_shards} -> {out_path.name}")


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
            pos1, pos2 = t_w1.pos_, t_w2.pos_
            if pos1 not in OPEN_CLASS or pos2 not in OPEN_CLASS:
                continue
            lca, lca_first, lca_last = find_lca_info(parse_str, i, i+2)
            if lca not in PHRASE_LABELS or lca_first != i or lca_last != i+2:
                continue

            key = tuple(sorted([w1, w2]))
            alpha1, alpha2 = key
            if key not in pairs:
                pairs[key] = {"freq_w1_w2": 0, "freq_w2_w1": 0,
                              "pos1": pos1, "pos2": pos2,
                              "example_sentence": sent.text.strip()}
            if (w1, w2) == (alpha1, alpha2):
                pairs[key]["freq_w1_w2"] += 1
            else:
                pairs[key]["freq_w2_w1"] += 1


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
                if toks[i].pos_ not in OPEN_CLASS or toks[i+2].pos_ not in OPEN_CLASS:
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
print("Loading corpus...")
dataset = load_dataset("znhoughton/babylm-150m-v3", split="train+dev",
                       streaming=False)
if num_shards > 1:
    dataset = dataset.shuffle(seed=964)
    dataset = dataset.shard(num_shards=num_shards, index=shard_index,
                            contiguous=True)
print(f"  {len(dataset):,} docs in this shard")

pairs  = {}
buffer = []

for row in tqdm(dataset, desc=f"Shard {shard_index}", unit="doc"):
    text = row["text"]
    if " and " not in text.lower():
        continue
    buffer.append(text)
    if len(buffer) >= BATCH_SIZE:
        flush_batch(buffer, pairs)
        buffer.clear()

flush_batch(buffer, pairs)

print(f"\nFound {len(pairs):,} unique pairs.")

# ── Save ──────────────────────────────────────────────────────────────────────
rows = [{"word1": k[0], "word2": k[1],
         "freq_w1_w2": v["freq_w1_w2"], "freq_w2_w1": v["freq_w2_w1"],
         "pos1": v["pos1"], "pos2": v["pos2"],
         "example_sentence": v["example_sentence"]}
        for k, v in pairs.items()]

pd.DataFrame(rows).sort_values(["word1", "word2"]).reset_index(drop=True)\
  .to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
