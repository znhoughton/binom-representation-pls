#!/usr/bin/env python3
"""
extract_embeddings.py
---------------------
Unified embedding extraction for all analysis conditions.
Replaces preprocessing/score_and_extract_all.py.

Output .npz always contains:
  word1, word2   : string pair identifiers (alphabetically ordered, word1 < word2)
  preference     : log-prob(alpha span) - log-prob(non-alpha span), from binomial context
  vec_alpha      : embedding for the alpha-order element
  vec_non_alpha  : embedding for the non-alpha-order element
  diff_vecs      : vec_alpha - vec_non_alpha

Extraction conditions
---------------------
--context binomial --extract word  (default / current behavior)
    Mean-pool span tokens [word1, and, word2] in the original binomial sentence.

--context binomial --extract last
    Take the final token of the truncated input (i.e. last subword of word2 in alpha
    order; last subword of word1 in non-alpha order).

--context isolated --extract word
    Each word is embedded in its own isolated context (e.g. "the bread").
    Preference is copied from an existing binomial .npz (auto-derived or via
    --pref-source). Useful for testing whether static word representations alone
    predict ordering.

Usage examples
--------------
  # Default condition (reproduces score_and_extract_all.py output):
  python Scripts/extract_embeddings.py \\
    --model znhoughton/opt-babylm-350m-20eps-seed964 \\
    --data corpus --layer last \\
    --out Data/embeddings/znhoughton_opt-babylm-350m-20eps-seed964

  # Last-token extraction:
  python Scripts/extract_embeddings.py \\
    --model znhoughton/opt-babylm-350m-20eps-seed964 \\
    --data corpus --layer last --extract last \\
    --out Data/embeddings_last/znhoughton_opt-babylm-350m-20eps-seed964

  # Isolated context (requires binomial extraction to already exist):
  python Scripts/extract_embeddings.py \\
    --model znhoughton/opt-babylm-350m-20eps-seed964 \\
    --data corpus --layer last --context isolated \\
    --out Data/embeddings_isolated/znhoughton_opt-babylm-350m-20eps-seed964
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = "D:/hf_model_cache"

CSV_PATHS = {
    "corpus": PROJECT_ROOT / "Data" / "corpus_binomials.csv",
    "novel":  PROJECT_ROOT / "Data" / "wikipedia_novel_binomials.csv",
}
BINOMIAL_OUT_DIRS = {
    "corpus": PROJECT_ROOT / "Data" / "embeddings",
    "novel":  PROJECT_ROOT / "Data" / "novel_embeddings",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, help="HuggingFace model ID")
    p.add_argument("--data",     choices=["corpus", "novel"], required=True)
    p.add_argument("--layer",    default="last",
                   help="Layer to extract: 'last', 'second_to_last', or integer index")
    p.add_argument("--context",  choices=["binomial", "isolated"], default="binomial")
    p.add_argument("--extract",  choices=["word", "last"], default="word",
                   help="word: pool over target token(s);  last: final token of input")
    p.add_argument("--pool",     choices=["mean", "first", "last"], default="mean",
                   help="How to aggregate subword tokens (used when --extract word)")
    p.add_argument("--template", default="the {word}",
                   help="Isolated-context template (used when --context isolated)")
    p.add_argument("--pref-source", dest="pref_source", default=None,
                   help="Path to existing binomial .npz to copy preference+word pairs from "
                        "(isolated context only; auto-derived from default paths if omitted)")
    p.add_argument("--out",      required=True,
                   help="Output directory; file saved as layer_{layer}.npz inside it")
    p.add_argument("--gpu",      type=int, default=0)
    p.add_argument("--force",    action="store_true",
                   help="Re-extract even if the output file already exists")
    return p.parse_args()


# ── Layer helpers ─────────────────────────────────────────────────────────────

def layer_tag(layer_arg: str) -> str:
    """Map CLI --layer value to canonical filename tag (e.g. 'layer_last')."""
    if layer_arg in ("last", "second_to_last"):
        return f"layer_{layer_arg}"
    return f"layer_{int(layer_arg)}"


def resolve_layer_idx(model, layer_arg: str) -> int:
    n = model.config.num_hidden_layers
    if layer_arg == "last":
        return n
    if layer_arg == "second_to_last":
        return n - 1
    idx = int(layer_arg)
    if not (0 <= idx <= n):
        raise ValueError(f"Layer index {idx} out of range [0, {n}]")
    return idx


# ── Token-span helpers ────────────────────────────────────────────────────────

def find_span(sentence: str, w1: str, w2: str):
    m = re.search(
        r'\b' + re.escape(w1) + r'\s+and\s+' + re.escape(w2) + r'\b',
        sentence, re.IGNORECASE
    )
    return (m.start(), m.end()) if m else None


def tok_indices_for_chars(offsets, char_start: int, char_end: int):
    return [i for i, (s, e) in enumerate(offsets)
            if s < char_end and e > char_start and e > s]


def pool_vecs(hs: torch.Tensor, indices: list, mode: str) -> torch.Tensor:
    if mode == "mean":
        return hs[indices].mean(0)
    if mode == "first":
        return hs[indices[0]]
    return hs[indices[-1]]  # last


# ── Binomial extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def _process_binomial_sent(model, tokenizer, sent: str, device,
                            w_first: str, w_second: str, span_start: int,
                            layer_idx: int, extract_mode: str, pool_mode: str):
    """
    Forward-pass one (ordered) binomial sentence.
    Returns (logprob, embedding) or None on tokenization failure.
    """
    cs_w1  = span_start
    ce_w1  = span_start + len(w_first)
    cs_and = ce_w1 + 1          # space
    ce_and = ce_w1 + 4          # "and" = 3 chars
    cs_w2  = ce_w1 + 5          # space + w_second
    ce_w2  = cs_w2 + len(w_second)

    enc     = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc["offset_mapping"][0].tolist()

    w1_toks  = tok_indices_for_chars(offsets, cs_w1,  ce_w1)
    and_toks = tok_indices_for_chars(offsets, cs_and, ce_and)
    w2_toks  = tok_indices_for_chars(offsets, cs_w2,  ce_w2)

    if not w1_toks or not and_toks or not w2_toks:
        return None

    span_toks = w1_toks + and_toks + w2_toks
    last_tok  = max(span_toks)

    inputs = {k: v[:, :last_tok + 1].to(device)
              for k, v in enc.items() if k != "offset_mapping"}

    out = model(**inputs, output_hidden_states=True)

    lp  = F.log_softmax(out.logits[0], dim=-1)
    ids = inputs["input_ids"][0]
    logprob = sum(lp[t - 1, ids[t]].item() for t in span_toks if t > 0)

    hs = out.hidden_states[layer_idx][0].float().cpu()

    if extract_mode == "word":
        vec = pool_vecs(hs, span_toks, pool_mode)
    else:  # extract_mode == "last"
        vec = hs[last_tok]

    return logprob, vec.numpy().astype(np.float32)


@torch.no_grad()
def extract_binomial_pair(model, tokenizer, word1, word2, sentence, device,
                          layer_idx, extract_mode, pool_mode):
    """
    Returns (preference, vec_alpha, vec_non_alpha) or None on failure.
    word1 < word2 alphabetically.
    """
    span_a = find_span(sentence, word1, word2)
    span_b = find_span(sentence, word2, word1)
    if span_a is None and span_b is None:
        return None

    s = (span_a if span_a is not None else span_b)[0]

    alpha_sent     = sentence[:s] + word1 + " and " + word2
    non_alpha_sent = sentence[:s] + word2 + " and " + word1

    res_a = _process_binomial_sent(model, tokenizer, alpha_sent, device,
                                   word1, word2, s, layer_idx, extract_mode, pool_mode)
    res_b = _process_binomial_sent(model, tokenizer, non_alpha_sent, device,
                                   word2, word1, s, layer_idx, extract_mode, pool_mode)

    if res_a is None or res_b is None:
        return None

    alpha_lp, va  = res_a
    non_lp,   vna = res_b

    return alpha_lp - non_lp, va, vna


# ── Isolated extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_isolated_pair(model, tokenizer, word1, word2, preference: float, device,
                          layer_idx, pool_mode, template: str):
    """
    Embed word1 and word2 each in their isolated context (e.g. "the bread").
    preference is copied from the binomial source — not re-computed here.
    Returns (preference, vec_alpha, vec_non_alpha) or None on failure.
    """
    word_offset = template.find("{word}")
    if word_offset == -1:
        raise ValueError("Template must contain the literal placeholder {word}")

    def embed_word(word: str):
        sent      = template.replace("{word}", word)
        wstart    = word_offset
        wend      = wstart + len(word)
        enc       = tokenizer(sent, return_offsets_mapping=True, return_tensors="pt")
        offsets   = enc["offset_mapping"][0].tolist()
        word_toks = tok_indices_for_chars(offsets, wstart, wend)
        if not word_toks:
            return None
        inputs = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}
        out = model(**inputs, output_hidden_states=True)
        hs  = out.hidden_states[layer_idx][0].float().cpu()
        return pool_vecs(hs, word_toks, pool_mode).numpy().astype(np.float32)

    va  = embed_word(word1)
    vna = embed_word(word2)
    if va is None or vna is None:
        return None

    return preference, va, vna


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    slug     = args.model.replace("/", "_").replace(".", "_")
    out_dir  = Path(args.out)
    out_file = out_dir / f"{layer_tag(args.layer)}.npz"

    if out_file.exists() and not args.force:
        print(f"Output already exists: {out_file}  (use --force to re-extract)")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}  Layer: {args.layer}  Device: {device}")
    print(f"Context:  {args.context}  Extract: {args.extract}  Pool: {args.pool}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()

    layer_idx = resolve_layer_idx(model, args.layer)
    print(f"Layer index: {layer_idx} / {model.config.num_hidden_layers}")

    preferences    = []
    vec_alphas     = []
    vec_non_alphas = []
    word1s, word2s = [], []
    skipped = 0

    # ── Binomial extraction ───────────────────────────────────────────────────
    if args.context == "binomial":
        pairs_df = pd.read_csv(CSV_PATHS[args.data])
        print(f"Input pairs: {len(pairs_df):,}")

        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
            w1   = str(row["word1"]).lower().strip()
            w2   = str(row["word2"]).lower().strip()
            sent = str(row["example_sentence"]).strip()
            try:
                result = extract_binomial_pair(
                    model, tokenizer, w1, w2, sent, device,
                    layer_idx, args.extract, args.pool
                )
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"  WARNING {w1}/{w2}: {exc}")
                continue
            if result is None:
                skipped += 1
                continue
            pref, va, vna = result
            preferences.append(pref)
            vec_alphas.append(va)
            vec_non_alphas.append(vna)
            word1s.append(w1)
            word2s.append(w2)

    # ── Isolated extraction ───────────────────────────────────────────────────
    else:
        if args.pref_source:
            src_path = Path(args.pref_source)
        else:
            src_path = BINOMIAL_OUT_DIRS[args.data] / slug / f"{layer_tag(args.layer)}.npz"

        if not src_path.exists():
            raise FileNotFoundError(
                f"Binomial source not found: {src_path}\n"
                f"Run binomial extraction first, or specify --pref-source."
            )

        print(f"Preference source: {src_path}")
        src       = np.load(src_path, allow_pickle=True)
        src_w1s   = src["word1"].astype(str)
        src_w2s   = src["word2"].astype(str)
        src_prefs = src["preference"].astype(np.float32)
        print(f"Input pairs: {len(src_w1s):,}")

        for w1, w2, pref in tqdm(zip(src_w1s, src_w2s, src_prefs), total=len(src_w1s)):
            try:
                result = extract_isolated_pair(
                    model, tokenizer, w1, w2, float(pref), device,
                    layer_idx, args.pool, args.template
                )
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"  WARNING {w1}/{w2}: {exc}")
                continue
            if result is None:
                skipped += 1
                continue
            pref_out, va, vna = result
            preferences.append(pref_out)
            vec_alphas.append(va)
            vec_non_alphas.append(vna)
            word1s.append(w1)
            word2s.append(w2)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Extracted {len(preferences):,} pairs  ({skipped} skipped)")
    va_arr  = np.stack(vec_alphas).astype(np.float32)
    vna_arr = np.stack(vec_non_alphas).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        word1         = np.array(word1s,      dtype=object),
        word2         = np.array(word2s,       dtype=object),
        preference    = np.array(preferences,  dtype=np.float32),
        vec_alpha     = va_arr,
        vec_non_alpha = vna_arr,
        diff_vecs     = (va_arr - vna_arr).astype(np.float32),
    )
    print(f"Saved -> {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
