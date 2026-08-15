"""
steering.py
-----------
Causal test: is the linear ordering direction USED by the model, or merely
decodable from it?

A probe shows information is present. That is what the readout objection
targets: information can be present without driving behaviour. Steering answers
the different question of whether intervening on the direction changes what the
model does.

Two design choices make the result informative rather than trivial:

  1. STEER AT EARLY LAYERS.  Adding the direction at the final layer just edits
     the answer immediately before readout, which proves nothing. If adding it
     at layer 6 of 24 shifts the output, the intervention had to propagate
     through 18 further layers of computation. A pure "answer register" account
     predicts efficacy only late; an upstream representation predicts early
     efficacy. We therefore sweep the layer.

  2. STEER AT THE WORD POSITIONS, NOT THE FINAL TOKEN.  We modify W1's hidden
     state before W2 attends to it, so the change is an input to the ordering
     computation rather than its output.

Direction:
    Fitted here from saved embeddings with the same antisymmetric ridge used by
    the linear probe (see by_layer_mlp.antisym_features). In `individual` mode
    the feature is h(W1) - h(W2), so the weight vector lives in the space of a
    single token's hidden state and can be added directly at a token position.

Intervention:
    h[pos(W1)] += alpha * w
    h[pos(W2)] -= alpha * w
    applied in BOTH orderings, at one layer.

Prediction if the direction is causally load-bearing:
    preference for W1-first increases monotonically with alpha.

Usage:
    python Scripts/steering.py --model znhoughton/opt-babylm-125m-20eps-seed964 \
        --slug znhoughton_opt-babylm-125m-20eps-seed964 \
        --fit-layer 12 --layers 0 3 6 9 12 --alphas -2 -1 0 1 2 --n-pairs 500
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "Scripts"))
from extract_embeddings import find_span, tok_indices_for_chars   # noqa: E402

SEED = 964


# ── locating the transformer block list across architectures ─────────────────

def get_layers(model):
    for attr in ("model.layers", "model.decoder.layers", "gpt_neox.layers",
                 "transformer.h"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate transformer block list for this model")


# ── direction ────────────────────────────────────────────────────────────────

def fit_direction(emb_dir, layer_tag, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    """Antisymmetric ridge on h(W1) - h(W2); returns a unit vector in hidden space.

    Identical constraint to the linear probe: requiring the prediction to negate
    when the two words swap forces the weight on W2 to be minus the weight on W1,
    so the model reduces to w.(h(W1) - h(W2)).
    """
    z = np.load(emb_dir / f"{layer_tag}.npz", allow_pickle=True)
    if "alpha_w1" not in z:
        raise RuntimeError(f"{layer_tag}.npz lacks per-word keys; re-extract with --extract word")
    Z = torch.from_numpy((z["alpha_w1"] - z["alpha_w2"]).astype(np.float32)).double()
    y = torch.from_numpy(z["preference"].astype(np.float32)).double()

    n = len(y)
    rng = np.random.default_rng(SEED)
    val = rng.choice(n, size=max(1, n // 10), replace=False)
    tr = np.setdiff1d(np.arange(n), val)

    mean_, std_ = Z[tr].mean(0), Z[tr].std(0).clamp(min=1e-8)
    Ztr, Zva = (Z[tr] - mean_) / std_, (Z[val] - mean_) / std_
    ZtZ, Zty = Ztr.T @ Ztr, Ztr.T @ y[tr]
    eye = torch.eye(Z.shape[1], dtype=torch.float64)

    best_w, best_a, best = None, None, float("inf")
    for a in alphas:
        w = torch.linalg.solve(ZtZ + a * eye, Zty)
        loss = ((Zva @ w) - y[val]).pow(2).mean().item()
        if loss < best:
            best_w, best_a, best = w, a, loss

    r = float(np.corrcoef((Zva @ best_w).numpy(), y[val].numpy())[0, 1])
    print(f"  direction fitted at {layer_tag}: alpha={best_a:g}  "
          f"held-out r={r:+.4f}  (R2={r**2:.4f})")
    # scale-free direction; magnitude is carried by the steering coefficient
    return (best_w / best_w.norm()).float(), r ** 2


# ── steered forward pass ─────────────────────────────────────────────────────

@torch.no_grad()
def preference(model, tok, w1, w2, sentence, device, direction=None,
               layer=None, alpha=0.0):
    """log P(W1 and W2) - log P(W2 and W1), optionally under intervention.

    Mirrors extract_binomial_batch exactly so the numbers are comparable to the
    main pipeline:
      - only ONE ordering needs to be locatable in the sentence; the other is
        constructed by swapping. A natural sentence contains just one of them.
      - the sentence is TRUNCATED at the span start (sent[:s] + first + " and "
        + second) rather than having the span replaced in situ.
      - token indices come from offset mappings, not prefix-length arithmetic,
        which is not reliable across tokenizers.

    Returns None if neither ordering can be located, or if the span tokens
    cannot be resolved.
    """
    span = find_span(sentence, w1, w2) or find_span(sentence, w2, w1)
    if span is None:
        return None
    s = span[0]

    out = []
    for first, second in ((w1, w2), (w2, w1)):
        text = sentence[:s] + first + " and " + second
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        ids = enc["input_ids"][0]

        ce_first = s + len(first)
        p_first = tok_indices_for_chars(offs, s, ce_first)
        p_and = tok_indices_for_chars(offs, ce_first + 1, ce_first + 4)
        p_second = tok_indices_for_chars(offs, ce_first + 5,
                                         ce_first + 5 + len(second))
        if not p_first or not p_and or not p_second:
            return None

        handle = None
        if direction is not None and alpha != 0.0:
            d = direction.to(device)
            # +alpha on the alphabetically-first word, -alpha on the other,
            # whichever slot each occupies in this ordering
            sign = +1.0 if first == w1 else -1.0
            pf, ps = p_first, p_second

            def hook(_mod, _inp, output):
                is_tuple = isinstance(output, tuple)
                hs = output[0] if is_tuple else output
                hs[:, pf, :] += alpha * sign * d
                hs[:, ps, :] -= alpha * sign * d
                return (hs,) + output[1:] if is_tuple else hs

            handle = get_layers(model)[layer].register_forward_hook(hook)

        try:
            logits = model(**enc).logits
        finally:
            if handle is not None:
                handle.remove()

        lp = F.log_softmax(logits[0].float(), dim=-1)
        span_pos = p_first + p_and + p_second
        out.append(sum(lp[p - 1, ids[p]].item() for p in span_pos if p > 0))

    return out[0] - out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--fit-layer", type=int, required=True, dest="fit_layer",
                    help="layer whose embeddings are used to FIT the direction "
                         "(fallback when --fit-per-layer is unavailable)")
    ap.add_argument("--fit-per-layer", action="store_true", dest="fit_per_layer",
                    help="Fit the direction at each intervention layer rather than "
                         "reusing one direction everywhere. Strongly preferred: a "
                         "direction estimated in the final layer's basis is not "
                         "meaningful when injected at layer 6, since the bases differ. "
                         "Falls back to --fit-layer where that layer's embeddings "
                         "were not extracted.")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="layers at which to APPLY the intervention")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-4, -2, -1, 0, 1, 2, 4])
    ap.add_argument("--n-pairs", type=int, default=500, dest="n_pairs")
    ap.add_argument("--data", choices=["corpus", "novel"], default="novel")
    ap.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    emb_root = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    sub = "novel_embeddings" if args.data == "novel" else "embeddings"
    emb_dir = emb_root / sub / args.slug

    print(f"model: {args.model}\ndevice: {device}")

    # One direction per intervention layer where embeddings exist. Reusing a
    # single final-layer direction across depths would be wrong: the residual
    # stream basis differs by layer, so a vector estimated at layer N carries no
    # guaranteed meaning when injected at layer 6.
    fallback, fallback_r2 = fit_direction(emb_dir, f"layer_{args.fit_layer}")
    dir_by_layer = {}
    for L in args.layers:
        if args.fit_per_layer and (emb_dir / f"layer_{L}.npz").exists():
            dir_by_layer[L] = fit_direction(emb_dir, f"layer_{L}")
        else:
            if args.fit_per_layer:
                print(f"  [warn] layer_{L}.npz absent; falling back to the "
                      f"layer_{args.fit_layer} direction for layer {L}. Its "
                      f"basis may not match, so treat this cell cautiously.")
            dir_by_layer[L] = (fallback, fallback_r2)

    csv = (BASE / "Data" /
           ("wikipedia_novel_binomials.csv" if args.data == "novel"
            else "corpus_binomials.csv"))
    pairs = pd.read_csv(csv).sample(n=args.n_pairs, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device).eval()

    rows = []
    for layer in args.layers:
        direction, fit_r2 = dir_by_layer[layer]
        for alpha in args.alphas:
            deltas = []
            for _, r in pairs.iterrows():
                p = preference(model, tok, str(r.word1), str(r.word2),
                               str(r.example_sentence), device,
                               direction=direction, layer=layer, alpha=alpha)
                if p is not None:
                    deltas.append(p)
            if not deltas:
                continue
            d = np.array(deltas)
            rows.append(dict(model=args.slug,
                             fit_layer=(layer if args.fit_per_layer else args.fit_layer),
                             fit_r2=fit_r2, layer=layer, alpha=alpha,
                             n=len(d), mean_pref=d.mean(),
                             se=d.std(ddof=1) / np.sqrt(len(d))))
            print(f"  layer {layer:>3d}  alpha {alpha:+.1f}  n={len(d):>4d}  "
                  f"mean preference = {d.mean():+.4f} "
                  f"(se {d.std(ddof=1)/np.sqrt(len(d)):.4f})", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else BASE / "Results" / args.slug / "steering.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")

    print("\nSLOPE of mean preference on alpha, by layer")
    print("  (positive and non-zero => the direction is causally load-bearing;")
    print("   efficacy at EARLY layers is what rules out an answer-register account)")
    for layer, g in df.groupby("layer"):
        if len(g) > 1 and g.alpha.std() > 0:
            slope = np.polyfit(g.alpha, g.mean_pref, 1)[0]
            print(f"  layer {layer:>3d}   slope = {slope:+.4f}")


if __name__ == "__main__":
    main()
