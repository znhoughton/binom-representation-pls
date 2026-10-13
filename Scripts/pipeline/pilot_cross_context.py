#!/usr/bin/env python3
"""
pilot_cross_context.py
----------------------
Pilot: does probing across CONTEXTS break the step-0 circularity?

The problem
-----------
The probe's input and its target come from the same forward pass. `preference`
is read off the same hidden states the probe receives, so the probe reverse-
engineers the network's frozen arithmetic rather than finding linguistic
structure. An untrained network passes, and in fact scores higher than any
trained checkpoint.

The proposal
------------
Take the probe's input from carrier sentence A and the target preference from a
DIFFERENT carrier sentence B. Sentence A's representations do not contain
sentence B's context, so the context-dependent part of the frozen computation
cannot be recomputed. Only what is invariant across contexts survives, and
context-generality is a reasonable operationalisation of an abstract preference.

What this pilot does NOT settle
-------------------------------
Cross-context removes the context-specific recomputation path. It does NOT
remove the per-word path: at initialisation there is still a context-invariant
component (the "and" token is the same in every item, so its contribution to
each word's logit is constant). So step-0 R2 is not structurally guaranteed to
be zero here, only reduced. How much it is reduced is exactly what this measures.

Design
------
`extract_binomial_batch` truncates each sentence at the binomial and keeps only
the preceding text, so a "context" is just a prefix. A donor context is then
built by splicing this pair into another item's prefix, which keeps contexts
natural, varied, and unpredictable from the probe's input. Donors are assigned
by a cyclic shift of the sampled items, so every item gets a distinct donor.

Four cells per model x condition:

                        target = own prefix      target = donor prefix
    input = own prefix   same-context (current)   CROSS-CONTEXT (proposed)

The same-context column reproduces the current design on this subsample and is
the baseline the cross-context number has to be read against.

Ridge rather than MLP, per the pilot brief: one hyperparameter, selected per
fold on an inner split, and fast enough to iterate on. Antisymmetric
augmentation is kept (swap the two words, negate the label) so the probe has the
same inductive bias as the MLP in the main pipeline.

Usage
-----
  python Scripts/pipeline/pilot_cross_context.py --n 15000 --gpu 0
  python Scripts/pipeline/pilot_cross_context.py --n 3000 --models step0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(HERE))

from extract_embeddings import extract_binomial_batch, find_span  # noqa: E402

SEED = 964
FOLDS = 10
ALPHAS = [10.0 ** k for k in range(0, 6)]

MODELS = {
    "step0": str(BASE / "Data" / "random_init"
                 / "znhoughton_opt-babylm-125m-20eps-seed964_step0"),
    "final": "znhoughton/opt-babylm-125m-20eps-seed964",
}
CONDITIONS = ["default", "attn_zeroed"]


def log(msg):
    print(msg, flush=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_rows(n: int, min_prefix: int = 0):
    """Sample n novel pairs; return own-context rows and donor-context rows.

    A row is (w1, w2, sentence) as extract_binomial_batch expects. The donor row
    splices this pair into another item's prefix, so the two rows differ only in
    the text preceding the binomial.

    min_prefix matters more than it looks. 29% of the novel set has the binomial
    at sentence start, so its "context" is just BOS, identical across those
    items. For them the preference is a pure function of the two embeddings with
    no context to vary, which makes them trivially predictable and makes
    cross-context vacuous. min_prefix=0 reproduces the main pipeline's item set;
    a positive value isolates the items that actually have context.
    """
    df = pd.read_csv(BASE / "Data" / "wikipedia_novel_binomials.csv")
    rng = np.random.default_rng(SEED)

    recs = []
    for _, r in df.sample(frac=1.0, random_state=SEED).iterrows():
        w1 = str(r["word1"]).lower().strip()
        w2 = str(r["word2"]).lower().strip()
        sent = str(r["example_sentence"]).strip()
        span = find_span(sent, w1, w2) or find_span(sent, w2, w1)
        if span is None:
            continue
        prefix = sent[:span[0]]
        if len(prefix.strip()) < min_prefix:
            continue
        recs.append((w1, w2, prefix))
        if len(recs) >= n:
            break

    log(f"  sampled {len(recs):,} pairs with usable prefixes")

    # Donor = the next item's prefix, cyclically. Deterministic, every item gets
    # a different context from its own, and donors are natural corpus text.
    shift = rng.integers(1, len(recs))
    own_rows, donor_rows = [], []
    for i, (w1, w2, prefix) in enumerate(recs):
        donor_prefix = recs[(i + shift) % len(recs)][2]
        own_rows.append((w1, w2, prefix + w1 + " and " + w2))
        donor_rows.append((w1, w2, donor_prefix + w1 + " and " + w2))
    return recs, own_rows, donor_rows


def build_rows_carriers(n: int, which: str = "novel"):
    """Rows from harvest_carrier_sentences.py: two REAL occurrences per pair.

    Supersedes build_rows() for the real design. Donor splicing was a stopgap
    from when second occurrences looked unfindable, and it produced
    ungrammatical contexts; both sentences here are genuine corpus text with at
    least 10 characters of left context, guaranteed by the harvest.
    """
    hv = pd.read_csv(BASE / "Data" / "carrier_sentences.csv")
    hv = hv[hv["sent_1"].notna() & hv["sent_2"].notna()]
    if which != "all":
        hv = hv[hv["set"] == which]
    log(f"  harvest: {len(hv):,} pairs with two real sentences ({which})")
    if n and n < len(hv):
        hv = hv.sample(n=n, random_state=SEED)
    recs, own_rows, cross_rows = [], [], []
    for w1, w2, s1, s2 in zip(hv["word1"].astype(str), hv["word2"].astype(str),
                              hv["sent_1"].astype(str), hv["sent_2"].astype(str)):
        recs.append((w1, w2, None))
        own_rows.append((w1, w2, s1))
        cross_rows.append((w1, w2, s2))
    log(f"  using {len(recs):,} pairs")
    return recs, own_rows, cross_rows


# ── Extraction ────────────────────────────────────────────────────────────────

def extract(model, tokenizer, rows, device, layer_idx, zero_attn, batch_size):
    """Return (X, y): mean-pooled span reps for both orderings, and preference."""
    X, y, ok = [], [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        res = extract_binomial_batch(model, tokenizer, chunk, device,
                                     [layer_idx], "word", "mean",
                                     zero_cross_attention=zero_attn)
        for j, r in enumerate(res):
            if r is None:
                ok.append(False)
                continue
            pref, a_vecs, na_vecs = r
            aw1, aand, aw2 = a_vecs[layer_idx]
            nw1, nand, nw2 = na_vecs[layer_idx]
            X.append(np.concatenate([(aw1 + aand + aw2) / 3.0,
                                     (nw1 + nand + nw2) / 3.0]))
            y.append(pref)
            ok.append(True)
        if (i // batch_size) % 10 == 0:
            log(f"      {min(i + batch_size, len(rows)):,}/{len(rows):,}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(ok)


# ── Probe ─────────────────────────────────────────────────────────────────────

def folds_for(split, w1s, w2s, n):
    rng = np.random.default_rng(SEED)
    if split == "pair_novel":
        perm = rng.permutation(n) % FOLDS
        return [(perm != k, perm == k) for k in range(FOLDS)]
    words = np.array(sorted(set(w1s) | set(w2s)))
    wperm = rng.permutation(len(words)) % FOLDS
    wf = {w: int(wperm[i]) for i, w in enumerate(words)}
    f1 = np.array([wf[w] for w in w1s])
    f2 = np.array([wf[w] for w in w2s])
    return [((f1 != k) & (f2 != k), (f1 == k) & (f2 == k)) for k in range(FOLDS)]


def augment(X, y):
    """Antisymmetric augmentation: swap the two orderings, negate the label."""
    d = X.shape[1] // 2
    Xf = np.concatenate([X[:, d:], X[:, :d]], axis=1)
    return np.vstack([X, Xf]), np.concatenate([y, -y])


def mlp_cv(X, y, w1s, w2s, split, device, epochs=500, patience=20, batch=4096):
    """10-fold MLP, matching the main pipeline: Linear(d,128) -> ReLU -> Linear(128,1),
    Adam lr 1e-3 / wd 1e-4, MSE, early stopping on a 10% validation split,
    antisymmetric augmentation applied per batch with p=0.5."""
    import torch.nn as nn

    d2 = X.shape[1]
    d = d2 // 2
    r2s = []
    for fi, (tr_mask, te_mask) in enumerate(folds_for(split, w1s, w2s, len(y))):
        if te_mask.sum() < 30 or tr_mask.sum() < 100:
            continue
        Xtr_all = torch.from_numpy(X[tr_mask]).float()
        ytr_all = torch.from_numpy(y[tr_mask]).float()
        n_val = max(1, int(0.1 * len(ytr_all)))
        Xva, yva = Xtr_all[:n_val].to(device), ytr_all[:n_val].to(device)
        Xtr, ytr = Xtr_all[n_val:].to(device), ytr_all[n_val:].to(device)
        Xte = torch.from_numpy(X[te_mask]).float().to(device)
        yte = y[te_mask]

        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-8
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd

        g = torch.Generator(device="cpu").manual_seed(SEED + fi)
        torch.manual_seed(SEED + fi)
        net = nn.Sequential(nn.Linear(d2, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            perm = torch.randperm(len(ytr), generator=g).to(device)
            net.train()
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                xb, yb = Xtr[idx], ytr[idx]
                flip = torch.rand(len(idx), device=device) < 0.5
                if flip.any():
                    xb = xb.clone()
                    xb[flip] = torch.cat([xb[flip][:, d:], xb[flip][:, :d]], dim=1)
                    yb = yb.clone()
                    yb[flip] = -yb[flip]
                opt.zero_grad()
                loss = ((net(xb).squeeze(-1) - yb) ** 2).mean()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(((net(Xva).squeeze(-1) - yva) ** 2).mean())
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            p = net(Xte).squeeze(-1).cpu().numpy()
        if np.std(p) < 1e-12 or np.std(yte) < 1e-12:
            r2s.append(0.0)
        else:
            r2s.append(float(np.corrcoef(p, yte)[0, 1] ** 2))
    return (float(np.mean(r2s)), float(np.std(r2s)), len(r2s)) if r2s else (np.nan, np.nan, 0)


def ridge_cv(X, y, w1s, w2s, split):
    """10-fold ridge; alpha picked per fold on an inner split. Returns mean r2."""
    r2s = []
    for tr_mask, te_mask in folds_for(split, w1s, w2s, len(y)):
        if te_mask.sum() < 30 or tr_mask.sum() < 100:
            continue
        Xtr_raw, ytr_raw = X[tr_mask], y[tr_mask]
        Xte, yte = X[te_mask], y[te_mask]

        n_in = max(1, int(0.9 * len(ytr_raw)))
        Xin, yin = augment(Xtr_raw[:n_in], ytr_raw[:n_in])
        Xva, yva = Xtr_raw[n_in:], ytr_raw[n_in:]

        mu, sd = Xin.mean(0), Xin.std(0) + 1e-8
        best, best_a = -np.inf, ALPHAS[0]
        for a in ALPHAS:
            m = Ridge(alpha=a).fit((Xin - mu) / sd, yin)
            p = m.predict((Xva - mu) / sd)
            s = -np.mean((p - yva) ** 2)
            if s > best:
                best, best_a = s, a

        Xtr, ytr = augment(Xtr_raw, ytr_raw)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        m = Ridge(alpha=best_a).fit((Xtr - mu) / sd, ytr)
        p = m.predict((Xte - mu) / sd)
        if np.std(p) < 1e-12 or np.std(yte) < 1e-12:
            r2s.append(0.0)
        else:
            r2s.append(float(np.corrcoef(p, yte)[0, 1] ** 2))
    return (float(np.mean(r2s)), float(np.std(r2s)), len(r2s)) if r2s else (np.nan, np.nan, 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--min-prefix", type=int, default=0, dest="min_prefix",
                   help="Minimum left-context length in characters. 0 reproduces "
                        "the main pipeline's item set (29%% have no prefix at all).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--contexts", default="donor",
                   choices=["donor", "carriers"],
                   help="donor: splice another item's prefix (superseded). "
                        "carriers: two real occurrences from the harvest.")
    p.add_argument("--set", default="novel", dest="which",
                   choices=["novel", "corpus", "all"],
                   help="Which item set, when --contexts carriers")
    p.add_argument("--control", action="store_true",
                   help="Hewitt & Liang control: globally shuffle the target "
                        "before CV. Near-zero r2 means the probe is recovering "
                        "structure rather than overfitting the folds.")
    p.add_argument("--probe", default="ridge", choices=["ridge", "mlp"],
                   help="ridge is fast and has one hyperparameter; mlp matches "
                        "the main pipeline exactly")
    p.add_argument("--models", nargs="+", default=["step0", "final"],
                   choices=list(MODELS))
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("No CUDA device; refusing to run this on CPU in fp16.")

    log("Building items")
    if args.contexts == "carriers":
        recs, own_rows, donor_rows = build_rows_carriers(args.n, args.which)
    else:
        recs, own_rows, donor_rows = build_rows(args.n, args.min_prefix)
    w1s_all = np.array([r[0] for r in recs])
    w2s_all = np.array([r[1] for r in recs])

    results = []
    for mkey in args.models:
        mid = MODELS[mkey]
        log(f"\n=== MODEL {mkey}  ({mid}) ===")
        tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(device)
        model.half()
        model.eval()
        layer_idx = model.config.num_hidden_layers

        for cond in CONDITIONS:
            za = cond == "attn_zeroed"
            cache = (BASE / "Data" / "pilot_cache" /
                     f"{mkey}_{cond}_n{args.n}_mp{args.min_prefix}_{args.contexts}.npz")
            if cache.exists():
                log(f"  [{mkey}/{cond}] loading cached extraction")
                z = np.load(cache)
                X, y_own, ok_a = z["X"], z["y_own"], z["ok_a"]
                y_donor, ok_b = z["y_donor"], z["ok_b"]
            else:
                log(f"  [{mkey}/{cond}] extracting OWN context")
                t0 = time.perf_counter()
                X, y_own, ok_a = extract(model, tok, own_rows, device,
                                         layer_idx, za, args.batch_size)
                log(f"  [{mkey}/{cond}] extracting DONOR context")
                _, y_donor, ok_b = extract(model, tok, donor_rows, device,
                                           layer_idx, za, args.batch_size)
                log(f"    extraction {time.perf_counter() - t0:.0f}s")
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, X=X, y_own=y_own, ok_a=ok_a,
                                    y_donor=y_donor, ok_b=ok_b)

            keep = ok_a & ok_b
            idx_a = np.cumsum(ok_a) - 1
            idx_b = np.cumsum(ok_b) - 1
            sel_a = idx_a[keep]
            sel_b = idx_b[keep]
            Xk = X[sel_a]
            y_s = y_own[sel_a]
            y_x = y_donor[sel_b]
            w1k, w2k = w1s_all[keep], w2s_all[keep]
            log(f"    usable pairs: {len(Xk):,}   "
                f"sd(y_own)={y_s.std():.4f}  sd(y_donor)={y_x.std():.4f}  "
                f"r(own,donor)={np.corrcoef(y_s, y_x)[0, 1]:.3f}")

            for split in ["pair_novel", "word_novel"]:
                for tgt_name, yy in [("same_context", y_s), ("cross_context", y_x)]:
                    if args.control:
                        # Shuffle globally, exactly as by_layer_mlp.py --control does.
                        yy = np.random.default_rng(SEED).permutation(yy)
                    if args.probe == "mlp":
                        m, s, nf = mlp_cv(Xk, yy, w1k, w2k, split, device)
                    else:
                        m, s, nf = ridge_cv(Xk, yy, w1k, w2k, split)
                    results.append(dict(model=mkey, condition=cond, split=split,
                                        target=tgt_name, r2=m, sd=s, n_folds=nf,
                                        n_pairs=len(Xk)))
                    log(f"    {split:11s} {tgt_name:14s} r2={m:.4f} +/- {s:.4f}  ({nf} folds)")

        del model
        torch.cuda.empty_cache()

    tag = "_control" if args.control else ""
    out = BASE / "Results" / f"pilot_cross_context_{args.probe}_mp{args.min_prefix}_{args.contexts}{tag}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    log(f"\nwrote {out}")

    log("\n==== SUMMARY (ridge, mean_pooled, final layer) ====")
    R = pd.DataFrame(results)
    for split in ["pair_novel", "word_novel"]:
        log(f"\n  {split}")
        log(f"    {'model':8s} {'condition':13s} {'same_context':>13s} {'cross_context':>14s}")
        for mkey in args.models:
            for cond in CONDITIONS:
                sub = R[(R.model == mkey) & (R.condition == cond) & (R.split == split)]
                if sub.empty:
                    continue
                sc = sub[sub.target == "same_context"].r2.values
                cc = sub[sub.target == "cross_context"].r2.values
                log(f"    {mkey:8s} {cond:13s} {sc[0]:13.4f} {cc[0]:14.4f}")


if __name__ == "__main__":
    main()
