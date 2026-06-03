#!/usr/bin/env python3
"""
score_and_extract.py
--------------------
Score + extract diff_vecs for corpus binomials using a single prompt.

For each pair:
  - Builds canonical:  PROMPT + word1 + " and " + word2
  - Builds reversed:   PROMPT + word2 + " and " + word1
  - Computes preference = logprob(canonical) - logprob(reversed)
  - Extracts diff_vec  = mean_pool_hs(canonical) - mean_pool_hs(reversed)
    (mean-pooled over word1 + "and" + word2 tokens)

Input:  Data/corpus_binomials.csv  (word1, word2)
Output: Data/embeddings/{model_slug}/layer_last.npz
        Data/embeddings/{model_slug}/layer_second_to_last.npz

Usage:
  python Scripts/score_and_extract.py
  python Scripts/score_and_extract.py --models znhoughton/opt-babylm-350m-20eps-seed964
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DEFAULT_CSV     = PROJECT_ROOT / "Data" / "corpus_binomials.csv"
DEFAULT_OUT_BASE = PROJECT_ROOT / "Data" / "embeddings"
HF_CACHE_DIR    = "D:/hf_model_cache"

PROMPT = " the "

MODELS = [
    "znhoughton/opt-babylm-125m-20eps-seed964",
    "znhoughton/opt-babylm-350m-20eps-seed964",
    "znhoughton/opt-babylm-1.3b-20eps-seed964",
    "EleutherAI/pythia-1b",
]
LAYERS = ["last", "second_to_last"]


def char_spans_for_pair(prompt, word1, word2):
    text      = prompt + word1 + " and " + word2
    w1_start  = len(prompt)
    w1_end    = w1_start + len(word1)
    and_start = w1_end + 1
    and_end   = and_start + 3
    w2_start  = and_end + 1
    w2_end    = w2_start + len(word2)
    return text, (w1_start, w1_end), (and_start, and_end), (w2_start, w2_end)


def tok_indices_for_span(offsets, char_start, char_end):
    return [
        i for i, (s, e) in enumerate(offsets)
        if s < char_end and e > char_start and e > s
    ]


@torch.no_grad()
def process_pair(model, tokenizer, word1, word2, device, layer_indices):
    results = {}
    for label, w1, w2 in [("canon", word1, word2), ("rev", word2, word1)]:
        text, w1_span, and_span, w2_span = char_spans_for_pair(PROMPT, w1, w2)
        enc     = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc["offset_mapping"][0].tolist()
        inputs  = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}

        out = model(**inputs, output_hidden_states=True)
        lp  = F.log_softmax(out.logits[0], dim=-1)
        ids = inputs["input_ids"][0]
        token_lp = lp[:-1].gather(1, ids[1:].unsqueeze(1)).squeeze(1).sum().item()

        all_toks = (
            tok_indices_for_span(offsets, *w1_span)
            + tok_indices_for_span(offsets, *and_span)
            + tok_indices_for_span(offsets, *w2_span)
        )
        if not all_toks:
            return None

        hs_vecs = {}
        for ln, li in layer_indices.items():
            hs = out.hidden_states[li][0].float().cpu()
            hs_vecs[ln] = hs[all_toks].mean(0).numpy().astype(np.float32)

        results[label] = {"logprob": token_lp, "hs": hs_vecs}

    preference = results["canon"]["logprob"] - results["rev"]["logprob"]
    layer_vecs = {
        ln: results["canon"]["hs"][ln] - results["rev"]["hs"][ln]
        for ln in layer_indices
    }
    return preference, layer_vecs


def process_model(model_id, pairs_df, device, out_base):
    slug    = model_id.replace("/", "_").replace(".", "_")
    out_dir = out_base / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / "done.txt"
    if done_flag.exists():
        print(f"\n{model_id}: already done, skipping.")
        return

    print(f"\n{'='*60}\nLoading {model_id}\n{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.float16, device_map={"": device}
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    layer_indices = {
        "last":           n_layers,
        "second_to_last": n_layers - 1,
    }

    accum = {ln: {"diff_vecs": [], "word1": [], "word2": [], "preference": []}
             for ln in LAYERS}
    skipped = 0

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"  {slug}"):
        w1 = str(row["word1"]).lower().strip()
        w2 = str(row["word2"]).lower().strip()
        try:
            result = process_pair(model, tokenizer, w1, w2, device, layer_indices)
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"  WARNING {w1}/{w2}: {exc}")
            continue

        if result is None:
            skipped += 1
            continue

        preference, layer_vecs = result
        for ln in LAYERS:
            accum[ln]["diff_vecs"].append(layer_vecs[ln])
            accum[ln]["word1"].append(w1)
            accum[ln]["word2"].append(w2)
            accum[ln]["preference"].append(preference)

    print(f"  Skipped {skipped}/{len(pairs_df)}")

    for ln in LAYERS:
        a = accum[ln]
        np.savez_compressed(
            out_dir / f"layer_{ln}.npz",
            diff_vecs  = np.stack(a["diff_vecs"]).astype(np.float32),
            word1      = np.array(a["word1"],      dtype=object),
            word2      = np.array(a["word2"],      dtype=object),
            preference = np.array(a["preference"], dtype=np.float32),
        )
        print(f"  Saved layer_{ln}.npz: {len(a['diff_vecs'])} rows")

    done_flag.write_text("done")
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=MODELS)
    parser.add_argument("--gpu",        type=int,  default=0)
    parser.add_argument("--input-csv",  type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_BASE)
    args = parser.parse_args()

    device   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pairs_df = pd.read_csv(args.input_csv)
    out_base = args.output_dir

    print(f"Device:     {device}")
    print(f"Input CSV:  {args.input_csv}")
    print(f"Output dir: {out_base}")
    print(f"Pairs:      {len(pairs_df)}")
    print(f"Prompt:     '{PROMPT}'")
    print(f"Models:     {args.models}")

    out_base.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        process_model(model_id, pairs_df, device, out_base)

    print("\nAll done.")


if __name__ == "__main__":
    main()