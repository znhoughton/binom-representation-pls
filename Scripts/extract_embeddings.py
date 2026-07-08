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
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))

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
    p.add_argument("--layer",    default=None, nargs="+",
                   help="Layer(s) to extract: 'last', 'second_to_last', or integer index. "
                        "Multiple values extract all in one forward pass (default: last).")
    p.add_argument("--context",  choices=["binomial", "isolated", "attn_zeroed"], default="binomial")
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
    p.add_argument("--shard-index", type=int, default=None, dest="shard_index",
                   help="Shard index (0-based) for parallel extraction")
    p.add_argument("--num-shards",  type=int, default=None, dest="num_shards",
                   help="Total number of shards for parallel extraction")
    p.add_argument("--batch-size",  type=int, default=1, dest="batch_size",
                   help="Number of pairs to process per forward pass (default: 1)")
    p.add_argument("--no-merge",   action="store_true", dest="no_merge",
                   help="Skip chunk merging (exit after flushing; merge separately)")
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


def extract_layer_vecs(hidden_states, layer_indices: list[int], span_toks: list[int],
                       extract_mode: str, pool_mode: str) -> dict[int, np.ndarray]:
    """Extract vectors for multiple layers from already-computed hidden states."""
    result = {}
    last_tok = max(span_toks)
    for lidx in layer_indices:
        hs = hidden_states[lidx][0].float().cpu()
        if extract_mode == "word":
            vec = pool_vecs(hs, span_toks, pool_mode)
        else:
            vec = hs[last_tok]
        result[lidx] = vec.numpy().astype(np.float32)
    return result


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
                            layer_indices: list, extract_mode: str, pool_mode: str):
    """
    Forward-pass one (ordered) binomial sentence.
    Returns (logprob, {layer_idx: vec}) or None on tokenization failure.
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

    vecs = extract_layer_vecs(out.hidden_states, layer_indices,
                              span_toks, extract_mode, pool_mode)
    return logprob, vecs


@torch.no_grad()
def extract_binomial_pair(model, tokenizer, word1, word2, sentence, device,
                          layer_indices, extract_mode, pool_mode):
    """
    Returns (preference, {lidx: va}, {lidx: vna}) or None on failure.
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
                                   word1, word2, s, layer_indices, extract_mode, pool_mode)
    res_b = _process_binomial_sent(model, tokenizer, non_alpha_sent, device,
                                   word2, word1, s, layer_indices, extract_mode, pool_mode)

    if res_a is None or res_b is None:
        return None

    alpha_lp, vas  = res_a
    non_lp,   vnas = res_b

    return alpha_lp - non_lp, vas, vnas


@torch.no_grad()
def extract_binomial_batch(model, tokenizer, rows, device,
                           layer_indices, extract_mode, pool_mode,
                           zero_cross_attention=False):
    """
    Batched binomial extraction returning per-word vectors.
    Both orderings processed in one forward pass.
    If zero_cross_attention=True, w2 and "and" can't attend to w1.

    Returns list of (preference, {lidx: (w1_vec, and_vec, w2_vec)},
                                 {lidx: (w1_vec, and_vec, w2_vec)}) or None per pair.
    Each vec is mean-pooled across that word's subword tokens.
    """
    all_sents = []
    all_span_infos = []
    pair_indices = []

    MAX_LEFT_CHARS = 400  # ~80 tokens; prevents long sentences from blowing up VRAM
    for idx, (w1, w2, sent) in enumerate(rows):
        span_a = find_span(sent, w1, w2)
        span_b = find_span(sent, w2, w1)
        if span_a is None and span_b is None:
            pair_indices.append(None)
            continue
        s = (span_a if span_a is not None else span_b)[0]
        prefix = sent[max(0, s - MAX_LEFT_CHARS):s]
        new_s  = len(prefix)
        a_pos = len(all_sents)
        all_sents.append(prefix + w1 + " and " + w2)
        all_span_infos.append((w1, w2, new_s))
        na_pos = len(all_sents)
        all_sents.append(prefix + w2 + " and " + w1)
        all_span_infos.append((w2, w1, new_s))
        pair_indices.append((a_pos, na_pos))

    if not all_sents:
        return [None] * len(rows)

    B = len(all_sents)

    # Tokenize
    enc = tokenizer(all_sents, padding=True, return_offsets_mapping=True,
                    return_tensors="pt")
    offsets_batch = enc.pop("offset_mapping")
    seq_len = enc["input_ids"].shape[1]

    # Find per-word token indices
    word_toks_list = []  # (w1_toks, and_toks, w2_toks) per sentence
    valid_mask = []
    for bi in range(B):
        w_first, w_second, span_start = all_span_infos[bi]
        offs = offsets_batch[bi].tolist()
        cs_w1 = span_start
        ce_w1 = span_start + len(w_first)
        w1_toks = tok_indices_for_chars(offs, cs_w1, ce_w1)
        and_toks = tok_indices_for_chars(offs, ce_w1 + 1, ce_w1 + 4)
        w2_toks = tok_indices_for_chars(offs, ce_w1 + 5, ce_w1 + 5 + len(w_second))
        if not w1_toks or not and_toks or not w2_toks:
            word_toks_list.append(([], [], []))
            valid_mask.append(False)
        else:
            word_toks_list.append((w1_toks, and_toks, w2_toks))
            valid_mask.append(True)

    # Build padded index tensors for each word group (for vectorized pooling)
    def _build_padded_idx(toks_getter):
        toks_per_sent = [toks_getter(bi) if valid_mask[bi] else [] for bi in range(B)]
        max_t = max((len(t) for t in toks_per_sent), default=1) or 1
        idx = torch.zeros(B, max_t, dtype=torch.long)
        mask = torch.zeros(B, max_t, dtype=torch.bool)
        for bi, t in enumerate(toks_per_sent):
            if t:
                idx[bi, :len(t)] = torch.tensor(t, dtype=torch.long)
                mask[bi, :len(t)] = True
        return idx, mask

    w1_idx, w1_mask = _build_padded_idx(lambda bi: word_toks_list[bi][0])
    and_idx, and_mask = _build_padded_idx(lambda bi: word_toks_list[bi][1])
    w2_idx, w2_mask = _build_padded_idx(lambda bi: word_toks_list[bi][2])

    # Combined span for logprob computation
    span_toks_list = [
        (word_toks_list[bi][0] + word_toks_list[bi][1] + word_toks_list[bi][2])
        if valid_mask[bi] else [] for bi in range(B)
    ]
    max_span = max((len(s) for s in span_toks_list), default=0)
    if max_span == 0:
        return [None] * len(rows)
    span_idx = torch.zeros(B, max_span, dtype=torch.long)
    span_mask = torch.zeros(B, max_span, dtype=torch.bool)
    for bi in range(B):
        st = span_toks_list[bi]
        if st:
            span_idx[bi, :len(st)] = torch.tensor(st, dtype=torch.long)
            span_mask[bi, :len(st)] = True

    # Build attention mask (custom 4D if zero_cross_attention, else standard)
    enc_device = {k: v.to(device) for k, v in enc.items()}
    if zero_cross_attention:
        pad_mask_2d = enc["attention_mask"]  # (B, seq_len), 1=real, 0=pad
        # Compute position_ids from 2D padding mask (so model doesn't use 4D mask for this)
        position_ids = torch.cumsum(pad_mask_2d, dim=1) * pad_mask_2d - 1
        position_ids = position_ids.long().to(device)
        enc_device["position_ids"] = position_ids
        # Build 4D causal mask with cross-word zeroing
        min_val = torch.finfo(torch.float32).min
        causal = torch.triu(torch.full((seq_len, seq_len), min_val, dtype=torch.float32), diagonal=1)
        attn_mask = causal.unsqueeze(0).expand(B, -1, -1).clone()
        for bi in range(B):
            pad_positions = (pad_mask_2d[bi] == 0).nonzero(as_tuple=True)[0]
            if len(pad_positions) > 0:
                attn_mask[bi, :, pad_positions] = min_val
            if valid_mask[bi]:
                w1t, andt, w2t = word_toks_list[bi]
                for q in andt + w2t:
                    for k in w1t:
                        if k < q:
                            attn_mask[bi, q, k] = min_val
        enc_device["attention_mask"] = attn_mask.unsqueeze(1).to(dtype=torch.float16, device=device)

    # Forward pass
    out = model(**enc_device, output_hidden_states=True)

    # --- Vectorized logprob computation on GPU ---
    logits = out.logits.float()
    input_ids = enc_device["input_ids"]
    span_idx_d = span_idx.to(device)
    span_mask_d = span_mask.to(device)
    prev_idx = (span_idx_d - 1).clamp(min=0)
    logits_at_prev = logits.gather(1, prev_idx.unsqueeze(-1).expand(-1, -1, logits.size(-1)))
    lp_at_prev = F.log_softmax(logits_at_prev, dim=-1)
    target_ids = input_ids.gather(1, span_idx_d)
    token_lps = lp_at_prev.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    skip_mask = (span_idx_d == 0) | ~span_mask_d
    token_lps = token_lps.masked_fill(skip_mask, 0.0)
    logprobs = token_lps.sum(dim=1).cpu()
    del logits, logits_at_prev, lp_at_prev, token_lps, span_idx_d, span_mask_d, prev_idx, target_ids
    del out.logits

    # --- Per-word vectorized hidden state extraction ---
    def _pool_word(hs, word_idx, word_mask):
        expanded = word_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1))
        gathered = hs.gather(1, expanded)
        m = word_mask.unsqueeze(-1).float()
        return ((gathered * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)).numpy().astype(np.float32)

    sent_w1_vecs = {}
    sent_and_vecs = {}
    sent_w2_vecs = {}
    for lidx in layer_indices:
        hs = out.hidden_states[lidx].float().cpu()
        sent_w1_vecs[lidx] = _pool_word(hs, w1_idx, w1_mask)
        sent_and_vecs[lidx] = _pool_word(hs, and_idx, and_mask)
        sent_w2_vecs[lidx] = _pool_word(hs, w2_idx, w2_mask)

    del out

    # Assemble per-pair results: (preference, {lidx: (w1,and,w2)_alpha}, {lidx: (w1,and,w2)_non_alpha})
    output = []
    for idx in range(len(rows)):
        pi = pair_indices[idx]
        if pi is None:
            output.append(None)
            continue
        a_pos, na_pos = pi
        if not valid_mask[a_pos] or not valid_mask[na_pos]:
            output.append(None)
            continue
        a_vecs = {li: (sent_w1_vecs[li][a_pos], sent_and_vecs[li][a_pos], sent_w2_vecs[li][a_pos])
                  for li in layer_indices}
        na_vecs = {li: (sent_w1_vecs[li][na_pos], sent_and_vecs[li][na_pos], sent_w2_vecs[li][na_pos])
                   for li in layer_indices}
        output.append((logprobs[a_pos].item() - logprobs[na_pos].item(), a_vecs, na_vecs))
    return output


# Backward-compatible wrapper for attn-zeroed extraction
def extract_attn_zeroed_batch(model, tokenizer, rows, device,
                              layer_indices, extract_mode, pool_mode):
    return extract_binomial_batch(model, tokenizer, rows, device,
                                 layer_indices, extract_mode, pool_mode,
                                 zero_cross_attention=True)


# ── Isolated extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_isolated_pair(model, tokenizer, word1, word2, preference: float, device,
                          layer_indices, pool_mode, template: str):
    """
    Embed word1 and word2 each in their isolated context (e.g. "the bread").
    preference is copied from the binomial source — not re-computed here.
    Returns (preference, {lidx: va}, {lidx: vna}) or None on failure.
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
        return extract_layer_vecs(out.hidden_states, layer_indices,
                                  word_toks, "word", pool_mode)

    vas  = embed_word(word1)
    vnas = embed_word(word2)
    if vas is None or vnas is None:
        return None

    return preference, vas, vnas


@torch.no_grad()
def extract_isolated_batch(model, tokenizer, rows, device,
                           layer_indices, pool_mode, template):
    """
    Batch isolated extraction: embed all words in 'the {word}' in one forward pass.
    rows: list of (w1, w2, preference) tuples.
    Returns list of (preference, {lidx: va}, {lidx: vna}) or None per pair.
    """
    word_offset = template.find("{word}")
    if word_offset == -1:
        raise ValueError("Template must contain {word}")

    # Build all sentences: for each pair, two sentences (word1 and word2)
    all_sents = []
    all_word_spans = []
    pair_indices = []

    for idx, (w1, w2, pref) in enumerate(rows):
        a_pos = len(all_sents)
        all_sents.append(template.replace("{word}", w1))
        all_word_spans.append((word_offset, word_offset + len(w1)))
        na_pos = len(all_sents)
        all_sents.append(template.replace("{word}", w2))
        all_word_spans.append((word_offset, word_offset + len(w2)))
        pair_indices.append((a_pos, na_pos, pref))

    if not all_sents:
        return [None] * len(rows)

    B = len(all_sents)

    # Tokenize all sentences
    enc = tokenizer(all_sents, padding=True, return_offsets_mapping=True, return_tensors="pt")
    offsets_batch = enc.pop("offset_mapping")

    # Find word token indices for each sentence
    span_toks_list = []
    valid_mask = []
    for bi in range(B):
        wstart, wend = all_word_spans[bi]
        offs = offsets_batch[bi].tolist()
        word_toks = tok_indices_for_chars(offs, wstart, wend)
        if not word_toks:
            span_toks_list.append([])
            valid_mask.append(False)
        else:
            span_toks_list.append(word_toks)
            valid_mask.append(True)

    # Pad span indices for vectorized gather
    max_span = max((len(s) for s in span_toks_list), default=0)
    if max_span == 0:
        return [None] * len(rows)

    span_idx = torch.zeros(B, max_span, dtype=torch.long)
    span_mask = torch.zeros(B, max_span, dtype=torch.bool)
    for bi in range(B):
        st = span_toks_list[bi]
        if st:
            span_idx[bi, :len(st)] = torch.tensor(st, dtype=torch.long)
            span_mask[bi, :len(st)] = True

    # Forward pass
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)

    # Vectorized hidden state extraction (no logprobs needed for isolated)
    sent_vecs = {}
    for lidx in layer_indices:
        hs = out.hidden_states[lidx].float().cpu()  # (B, seq_len, hidden_dim)
        expanded_idx = span_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1))
        gathered = hs.gather(1, expanded_idx)
        mask_exp = span_mask.unsqueeze(-1).float()
        if pool_mode == "mean":
            pooled = (gathered * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        elif pool_mode == "first":
            pooled = gathered[:, 0, :]
        else:  # last
            last_valid = span_mask.long().sum(dim=1) - 1
            pooled = gathered[torch.arange(B), last_valid.clamp(min=0)]
        sent_vecs[lidx] = pooled.numpy().astype(np.float32)

    del out

    # Assemble per-pair results
    output = []
    for idx in range(len(rows)):
        a_pos, na_pos, pref = pair_indices[idx]
        if not valid_mask[a_pos] or not valid_mask[na_pos]:
            output.append(None)
            continue
        va = {lidx: sent_vecs[lidx][a_pos] for lidx in layer_indices}
        vna = {lidx: sent_vecs[lidx][na_pos] for lidx in layer_indices}
        output.append((pref, va, vna))
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    layer_args = args.layer if args.layer else ["last"]
    slug       = args.model.replace("/", "_").replace(".", "_")
    out_dir    = Path(args.out)
    is_sharded = args.shard_index is not None and args.num_shards is not None

    # Determine which layers still need extraction
    shard_suffix = f"_shard{args.shard_index}" if is_sharded else ""
    out_files   = {la: out_dir / f"{layer_tag(la)}{shard_suffix}.npz" for la in layer_args}
    needed      = [la for la in layer_args if not out_files[la].exists() or args.force]
    already_done = [la for la in layer_args if la not in needed]

    for la in already_done:
        print(f"Output already exists, skipping: {out_files[la]}  (use --force to re-extract)")
    if not needed:
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}  Layers: {needed}  Device: {device}")
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

    layer_indices = [resolve_layer_idx(model, la) for la in needed]
    print(f"Layer indices: {layer_indices} / {model.config.num_hidden_layers}")

    # Per-layer accumulators — 6 per-word vectors per layer, flushed to disk
    acc = {li: {"a_w1": [], "a_and": [], "a_w2": [],
                "na_w1": [], "na_and": [], "na_w2": []} for li in layer_indices}
    preferences   = []
    word1s        = []
    word2s        = []
    skipped       = 0
    chunk_idx     = 0
    flush_every   = 10000
    chunk_dir     = out_dir / "_chunks"

    def _flush_chunks():
        nonlocal chunk_idx
        if not preferences:
            return
        chunk_dir.mkdir(parents=True, exist_ok=True)
        np.savez(chunk_dir / f"chunk{chunk_idx}_meta.npz",
                 word1=np.array(word1s, dtype=object),
                 word2=np.array(word2s, dtype=object),
                 preference=np.array(preferences, dtype=np.float32))
        for li in layer_indices:
            np.savez(chunk_dir / f"chunk{chunk_idx}_layer{li}.npz",
                     alpha_w1=np.stack(acc[li]["a_w1"]).astype(np.float32),
                     alpha_and=np.stack(acc[li]["a_and"]).astype(np.float32),
                     alpha_w2=np.stack(acc[li]["a_w2"]).astype(np.float32),
                     non_alpha_w1=np.stack(acc[li]["na_w1"]).astype(np.float32),
                     non_alpha_and=np.stack(acc[li]["na_and"]).astype(np.float32),
                     non_alpha_w2=np.stack(acc[li]["na_w2"]).astype(np.float32))
            for k in acc[li]:
                acc[li][k].clear()
        preferences.clear()
        word1s.clear()
        word2s.clear()
        chunk_idx += 1
        import gc; gc.collect()

    # ── Binomial / attention-zeroed extraction ───────────────────────────────
    if args.context in ("binomial", "attn_zeroed"):
        pairs_df = pd.read_csv(CSV_PATHS[args.data])
        if is_sharded:
            n = len(pairs_df)
            shard_size = (n + args.num_shards - 1) // args.num_shards
            start = args.shard_index * shard_size
            end = min(start + shard_size, n)
            pairs_df = pairs_df.iloc[start:end].reset_index(drop=True)
            print(f"Shard {args.shard_index}/{args.num_shards}: rows {start}–{end} ({len(pairs_df):,} pairs)")
        print(f"Input pairs: {len(pairs_df):,}  batch_size: {args.batch_size}")

        all_rows = [
            (str(row["word1"]).lower().strip(),
             str(row["word2"]).lower().strip(),
             str(row["example_sentence"]).strip())
            for _, row in pairs_df.iterrows()
        ]

        batch_fn = extract_attn_zeroed_batch if args.context == "attn_zeroed" else extract_binomial_batch

        if args.batch_size > 1:
            for batch_start in tqdm(range(0, len(all_rows), args.batch_size),
                                    total=(len(all_rows) + args.batch_size - 1) // args.batch_size):
                batch = all_rows[batch_start:batch_start + args.batch_size]
                try:
                    results = batch_fn(
                        model, tokenizer, batch, device,
                        layer_indices, args.extract, args.pool
                    )
                except Exception as exc:
                    skipped += len(batch)
                    if skipped <= 5 * args.batch_size:
                        print(f"  WARNING batch {batch_start}: {exc}")
                    continue
                for (w1, w2, _), result in zip(batch, results):
                    if result is None:
                        skipped += 1
                        continue
                    pref, a_vecs, na_vecs = result
                    preferences.append(pref)
                    word1s.append(w1)
                    word2s.append(w2)
                    for li in layer_indices:
                        aw1, aand, aw2 = a_vecs[li]
                        nw1, nand, nw2 = na_vecs[li]
                        acc[li]["a_w1"].append(aw1)
                        acc[li]["a_and"].append(aand)
                        acc[li]["a_w2"].append(aw2)
                        acc[li]["na_w1"].append(nw1)
                        acc[li]["na_and"].append(nand)
                        acc[li]["na_w2"].append(nw2)
                if len(preferences) >= flush_every:
                    _flush_chunks()
        else:
            raise ValueError("batch_size must be > 1 for per-word extraction. Use --batch-size 256.")

    # ── Isolated extraction ───────────────────────────────────────────────────
    else:
        # Preference source: use the first needed layer's binomial npz.
        # Prefs are layer-invariant (copied from binomial context) so any layer works.
        if args.pref_source:
            src_path = Path(args.pref_source)
        else:
            src_path = BINOMIAL_OUT_DIRS[args.data] / slug / f"{layer_tag(needed[0])}.npz"

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
        print(f"Input pairs: {len(src_w1s):,}  batch_size: {args.batch_size}")

        all_iso_rows = list(zip(src_w1s, src_w2s, src_prefs))

        if args.batch_size > 1:
            for batch_start in tqdm(range(0, len(all_iso_rows), args.batch_size),
                                    total=(len(all_iso_rows) + args.batch_size - 1) // args.batch_size):
                batch = all_iso_rows[batch_start:batch_start + args.batch_size]
                try:
                    results = extract_isolated_batch(
                        model, tokenizer, batch, device,
                        layer_indices, args.pool, args.template
                    )
                except Exception as exc:
                    skipped += len(batch)
                    if skipped <= 5 * args.batch_size:
                        print(f"  WARNING batch {batch_start}: {exc}")
                    continue
                for (w1, w2, _), result in zip(batch, results):
                    if result is None:
                        skipped += 1
                        continue
                    pref, vas, vnas = result
                    preferences.append(pref)
                    word1s.append(w1)
                    word2s.append(w2)
                    for li in layer_indices:
                        # Isolated returns single vecs, store as w1 with zeros for and/w2
                        zero = np.zeros_like(vas[li])
                        acc[li]["a_w1"].append(vas[li])
                        acc[li]["a_and"].append(zero)
                        acc[li]["a_w2"].append(zero)
                        acc[li]["na_w1"].append(vnas[li])
                        acc[li]["na_and"].append(zero)
                        acc[li]["na_w2"].append(zero)
                if len(preferences) >= flush_every:
                    _flush_chunks()
        else:
            for w1, w2, pref in tqdm(zip(src_w1s, src_w2s, src_prefs), total=len(src_w1s)):
                try:
                    result = extract_isolated_pair(
                        model, tokenizer, w1, w2, float(pref), device,
                        layer_indices, args.pool, args.template
                    )
                except Exception as exc:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  WARNING {w1}/{w2}: {exc}")
                    continue
                if result is None:
                    skipped += 1
                    continue
                pref_out, vas, vnas = result
                preferences.append(pref_out)
                word1s.append(w1)
                word2s.append(w2)
                for li in layer_indices:
                    zero = np.zeros_like(vas[li])
                    acc[li]["a_w1"].append(vas[li])
                    acc[li]["a_and"].append(zero)
                    acc[li]["a_w2"].append(zero)
                    acc[li]["na_w1"].append(vnas[li])
                    acc[li]["na_and"].append(zero)
                    acc[li]["na_w2"].append(zero)

    # ── Flush remaining data and merge chunks per layer ─────────────────────
    _flush_chunks()
    del model
    torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_merge:
        print(f"Chunks saved to {chunk_dir} ({chunk_idx} chunks). Skipping merge (--no-merge).")
        return

    if chunk_idx == 0:
        print("No data extracted.")
        return

    # Merge metadata chunks (word1, word2, preference)
    meta_parts = [np.load(chunk_dir / f"chunk{c}_meta.npz", allow_pickle=True)
                  for c in range(chunk_idx)]
    w1_arr   = np.concatenate([m["word1"] for m in meta_parts])
    w2_arr   = np.concatenate([m["word2"] for m in meta_parts])
    pref_arr = np.concatenate([m["preference"] for m in meta_parts])
    total = len(pref_arr)
    print(f"Extracted {total:,} pairs  ({skipped} skipped)")

    # Merge and save each layer one at a time to bound RAM
    WORD_KEYS = ["alpha_w1", "alpha_and", "alpha_w2",
                 "non_alpha_w1", "non_alpha_and", "non_alpha_w2"]
    for la, li in zip(needed, layer_indices):
        arrays = {k: [] for k in WORD_KEYS}
        for c in range(chunk_idx):
            d = np.load(chunk_dir / f"chunk{c}_layer{li}.npz")
            for k in WORD_KEYS:
                arrays[k].append(d[k])
        merged = {k: np.concatenate(arrays[k]) for k in WORD_KEYS}
        del arrays
        np.savez(
            out_files[la],
            word1      = w1_arr,
            word2      = w2_arr,
            preference = pref_arr,
            **merged,
        )
        print(f"Saved -> {out_files[la]}")
        del merged

    # Clean up chunks
    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
