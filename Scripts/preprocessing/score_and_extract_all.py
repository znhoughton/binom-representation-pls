#!/usr/bin/env python3
"""
score_and_extract_all.py
------------------------
Unified scoring and embedding extraction for corpus and novel binomials.

For each pair (word1, word2, example_sentence):
  - Constructs alpha sentence:     ...word1 and word2...  (word1 < word2 alphabetically)
  - Constructs non-alpha sentence: ...word2 and word1...
  - Scores ONLY the span tokens [word1, and, word2] conditioned on the preceding
    sentence context (tokens before word1 cancel in the difference, so this is
    equivalent to full-sentence log-prob differencing)
  - Extracts last-layer hidden states at the span tokens, mean-pooled

Outputs per pair:
  preference    = logprob(alpha span) - logprob(non-alpha span)
  vec_alpha     = mean_pool(hidden_states at [w1, and, w2] in alpha sentence)
  vec_non_alpha = mean_pool(hidden_states at [w2, and, w1] in non-alpha sentence)
  diff_vec      = vec_alpha - vec_non_alpha

Usage:
  python Scripts/preprocessing/score_and_extract_all.py --mode corpus
  python Scripts/preprocessing/score_and_extract_all.py --mode novel
  python Scripts/preprocessing/score_and_extract_all.py --mode corpus --models znhoughton/opt-babylm-125m-20eps-seed964
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE_DIR = "D:/hf_model_cache"

MODELS = [
    "znhoughton/opt-babylm-125m-20eps-seed964",
    "znhoughton/opt-babylm-350m-20eps-seed964",
    "znhoughton/opt-babylm-1.3b-20eps-seed964",
]

CSV_PATHS = {
    "corpus": PROJECT_ROOT / "Data" / "corpus_binomials.csv",
    "novel":  PROJECT_ROOT / "Data" / "wikipedia_novel_binomials.csv",
}
OUT_DIRS = {
    "corpus": PROJECT_ROOT / "Data" / "embeddings",
    "novel":  PROJECT_ROOT / "Data" / "novel_embeddings",
}


# ── Span helpers ──────────────────────────────────────────────────────────────

def find_span(sentence, w1, w2):
    """
    Find 'w1 and w2' in sentence (case-insensitive, word-boundary-aware).
    Returns (match_start, match_end) or None.
    """
    pattern = re.compile(
        r'\b' + re.escape(w1) + r'\s+and\s+' + re.escape(w2) + r'\b',
        re.IGNORECASE
    )
    m = pattern.search(sentence)
    return (m.start(), m.end()) if m else None


def tok_indices_for_span(offsets, char_start, char_end):
    return [i for i, (s, e) in enumerate(offsets)
            if s < char_end and e > char_start and e > s]


# ── Per-pair scoring ──────────────────────────────────────────────────────────

@torch.no_grad()
def score_pair(model, tokenizer, word1, word2, sentence, device, last_layer_idx):
    """
    word1 < word2 alphabetically.

    Returns (preference, diff_vec, vec_alpha, vec_non_alpha) or None on failure.
    """
    # Find which order appears in the sentence
    span_a = find_span(sentence, word1, word2)
    span_b = find_span(sentence, word2, word1)

    if span_a is None and span_b is None:
        return None

    # Use whichever match was found; construct both orderings from the same span
    s, e = span_a if span_a is not None else span_b

    # Both sentences share the same prefix/suffix; only the span differs.
    # Using lowercase for the span ensures consistent tokenization across orderings.
    alpha_sent     = sentence[:s] + word1 + " and " + word2
    non_alpha_sent = sentence[:s] + word2 + " and " + word1

    def process(sent, w_first, w_second):
        # Compute char positions fresh from w_first length — do NOT reuse outer
        # word1/word2 lengths, which would be wrong when w_first == word2.
        _w1_cs  = s
        _w1_ce  = s + len(w_first)
        _and_cs = _w1_ce + 1        # space then "and"
        _and_ce = _w1_ce + 4        # "and" is 3 chars
        _w2_cs  = _w1_ce + 5        # space then w_second
        _w2_ce  = _w2_cs + len(w_second)

        enc     = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()

        w1_toks  = tok_indices_for_span(offsets, _w1_cs,  _w1_ce)
        and_toks = tok_indices_for_span(offsets, _and_cs, _and_ce)
        w2_toks  = tok_indices_for_span(offsets, _w2_cs,  _w2_ce)

        if not w1_toks or not and_toks or not w2_toks:
            return None

        span_toks = w1_toks + and_toks + w2_toks
        last_tok  = max(span_toks)

        # Truncate at the end of the span — suffix tokens are unused
        inputs = {k: v[:, :last_tok + 1].to(device)
                  for k, v in enc.items() if k != "offset_mapping"}

        out = model(**inputs, output_hidden_states=True)

        # Span log-prob: P(w1 | context) * P(and | ...) * P(w2 | ...)
        lp  = F.log_softmax(out.logits[0], dim=-1)
        ids = inputs["input_ids"][0]
        logprob = sum(lp[t - 1, ids[t]].item() for t in span_toks if t > 0)

        # Mean-pooled hidden states over span tokens — both last and second-to-last layers
        hs_last   = out.hidden_states[last_layer_idx][0].float().cpu()
        hs_second = out.hidden_states[last_layer_idx - 1][0].float().cpu()
        vec_last   = hs_last[span_toks].mean(0).numpy().astype(np.float32)
        vec_second = hs_second[span_toks].mean(0).numpy().astype(np.float32)

        return logprob, vec_last, vec_second

    res_alpha     = process(alpha_sent,     word1, word2)
    res_non_alpha = process(non_alpha_sent, word2, word1)

    if res_alpha is None or res_non_alpha is None:
        return None

    alpha_lp,    va_last,  va_second  = res_alpha
    non_alpha_lp, vna_last, vna_second = res_non_alpha

    preference = alpha_lp - non_alpha_lp

    return (preference,
            va_last - vna_last,   va_last,   vna_last,
            va_second - vna_second, va_second, vna_second)


# ── Per-model extraction ──────────────────────────────────────────────────────

def process_model(model_id, pairs_df, device, out_dir):
    slug     = model_id.replace("/", "_").replace(".", "_")
    out_path = out_dir / slug / "layer_last.npz"
    done_flag = out_dir / slug / "done.txt"

    if done_flag.exists():
        print(f"\n{model_id}: already done, skipping.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nLoading {model_id}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR,
                                              use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()

    last_layer_idx = model.config.num_hidden_layers  # hidden_states[0]=embed, [n]=last layer

    preferences         = []
    diff_vecs_last      = []; vec_alphas_last      = []; vec_non_alphas_last      = []
    diff_vecs_second    = []; vec_alphas_second    = []; vec_non_alphas_second    = []
    word1s, word2s = [], []
    skipped = 0

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"  {slug}"):
        w1   = str(row["word1"]).lower().strip()
        w2   = str(row["word2"]).lower().strip()
        sent = str(row["example_sentence"]).strip()

        try:
            result = score_pair(model, tokenizer, w1, w2, sent, device, last_layer_idx)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  WARNING {w1}/{w2}: {exc}")
            continue

        if result is None:
            skipped += 1
            continue

        pref, diff_l, va_l, vna_l, diff_s, va_s, vna_s = result
        preferences.append(pref)
        diff_vecs_last.append(diff_l);   vec_alphas_last.append(va_l);   vec_non_alphas_last.append(vna_l)
        diff_vecs_second.append(diff_s); vec_alphas_second.append(va_s); vec_non_alphas_second.append(vna_s)
        word1s.append(w1)
        word2s.append(w2)

    print(f"  Extracted {len(preferences):,} pairs  ({skipped} skipped)")

    shared = dict(
        word1      = np.array(word1s, dtype=object),
        word2      = np.array(word2s, dtype=object),
        preference = np.array(preferences, dtype=np.float32),
    )

    out_last = out_path.parent / "layer_last.npz"
    np.savez_compressed(out_last,
        diff_vecs     = np.stack(diff_vecs_last).astype(np.float32),
        vec_alpha     = np.stack(vec_alphas_last).astype(np.float32),
        vec_non_alpha = np.stack(vec_non_alphas_last).astype(np.float32),
        **shared)
    print(f"  Saved -> {out_last}")

    out_second = out_path.parent / "layer_second_to_last.npz"
    np.savez_compressed(out_second,
        diff_vecs     = np.stack(diff_vecs_second).astype(np.float32),
        vec_alpha     = np.stack(vec_alphas_second).astype(np.float32),
        vec_non_alpha = np.stack(vec_non_alphas_second).astype(np.float32),
        **shared)
    print(f"  Saved -> {out_second}")

    done_flag.write_text("done")

    del model
    torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["corpus", "novel"], required=True,
                        help="Which dataset to extract embeddings for")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--gpu",    type=int, default=0)
    args = parser.parse_args()

    device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    csv_path = CSV_PATHS[args.mode]
    out_dir  = OUT_DIRS[args.mode]

    print(f"Mode:       {args.mode}")
    print(f"Device:     {device}")
    print(f"Input CSV:  {csv_path}")
    print(f"Output dir: {out_dir}")

    pairs_df = pd.read_csv(csv_path)
    print(f"Pairs:      {len(pairs_df):,}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        process_model(model_id, pairs_df, device, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
