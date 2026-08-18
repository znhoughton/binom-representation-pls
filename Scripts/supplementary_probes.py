"""
supplementary_probes.py
-----------------------
Supplementary probe analyses. Standalone: does not touch by_layer_mlp.py or any
of its outputs. Reads the same embedding NPZs and reuses the same CV splits, so
results are directly comparable to the main MLP results already on disk.

Two analyses, both aimed at the "the probe is just reading a compressed answer"
objection:

LINEAR PROBE
    Ridge on the antisymmetric feature, same folds as the MLP.

    The antisymmetry the MLP gets by augmentation (swap halves, negate y) has a
    closed form for a linear model: writing the input as [a; b], a linear model
    is w_a.a + w_b.b, and requiring it to negate under the swap forces
    w_b = -w_a, so it reduces to w.(a - b). That is exact, not an approximation,
    and halves the dimensionality.

    Interpretation: if linear matches the MLP, the ordering code is linearly
    formatted and the MLP's extra capacity was doing nothing (which also removes
    any Hewitt & Liang capacity concern). If the MLP wins substantially, the
    encoding is not a single projection and a scalar-summary account is wrong.

PLS COMPONENT SWEEP
    R2 as a function of the number of PLS components, K = 1..K_max.

    Directly measures how many dimensions the ordering signal occupies. Saturation
    at K=1 means ordering preference lives on one shared linear axis; a signal
    that keeps climbing means the code is distributed. This is the quantitative
    version of the "is it just a subspace encoding the answer" question.

Usage:
    python Scripts/supplementary_probes.py \
        --model-slug znhoughton_opt-babylm-125m-20eps-seed964 --num-layers 12 \
        --conditions default attn_zeroed --modes mean_pooled words_only \
        --splits pair_novel word_novel --pls-kmax 15
"""
import argparse
import csv as _csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "Scripts"))

# Reuse the main pipeline's loaders and constants so the splits, seeds and
# feature construction are identical rather than re-implemented.
from by_layer_mlp import (                                    # noqa: E402
    SEED, FOLDS, VAL_FRAC, pearsonr, nipals_pls,
    load_raw_layer, x_from_raw, resolve_npz, load_device,
)

CONDITION_DIRS = {
    "default":     {"novel": "novel_embeddings",             "corpus": "embeddings"},
    "attn_zeroed": {"novel": "novel_embeddings_attn_zeroed", "corpus": "embeddings_attn_zeroed"},
}
ALPHAS = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


# ── features ─────────────────────────────────────────────────────────────────

def antisym_features(X, mode):
    """Collapse the concatenated input to its antisymmetric feature (see module docstring).

    mean_pooled / words_only : [alpha; non_alpha]  ->  alpha - non_alpha
    individual               : [w1; and; w2]       ->  w1 - w2  ("and" drops out,
                               since antisymmetry forces its weight to zero)
    """
    D = X.shape[1]
    if mode in ("mean_pooled", "words_only"):
        h = D // 2
        return X[:, :h] - X[:, h:]
    t = D // 3
    return X[:, :t] - X[:, 2 * t:]


# ── linear probe ─────────────────────────────────────────────────────────────

def _ridge(Z, y, alphas):
    """Closed-form ridge for several alphas. No intercept: the antisymmetric
    feature is odd by construction, so the target has zero mean at Z = 0."""
    ZtZ, Zty = Z.T @ Z, Z.T @ y
    eye = torch.eye(Z.shape[1], device=Z.device, dtype=Z.dtype)
    return {a: torch.linalg.solve(ZtZ + a * eye, Zty) for a in alphas}


def linear_cv(X, y, fold_data, device, mode, save_direction=False):
    """Ridge on the same folds the MLP uses.

    Returns (r2s, mean_direction, alphas). The chosen alpha is worth recording:
    it is what determines how much of the nominal feature dimension the model
    actually uses (effective df = sum d_i^2/(d_i^2+alpha)), and it is the only
    way to tell whether the grid is wide enough. If selection piles up on the
    largest value, ALPHAS is too narrow and the reported R2 understates what a
    properly regularised linear probe would reach.
    """
    r2s, dirs, chosen = [], [], []
    for tr_np, te_np, fold_id in fold_data:
        tr = torch.from_numpy(tr_np).to(device)
        te = torch.from_numpy(te_np).to(device)

        # inner val split mirrors the MLP's early-stopping split exactly
        n_all = len(tr_np)
        n_val = max(1, int(n_all * VAL_FRAC))
        rng = np.random.default_rng(SEED + fold_id + 20000)
        val_local = rng.choice(n_all, size=n_val, replace=False)
        itr = tr[torch.from_numpy(np.setdiff1d(np.arange(n_all), val_local)).to(device)]
        iva = tr[torch.from_numpy(val_local).to(device)]

        Z_itr = antisym_features(X[itr], mode).double()
        Z_iva = antisym_features(X[iva], mode).double()
        m, s = Z_itr.mean(0), Z_itr.std(0).clamp(min=1e-8)
        ws = _ridge((Z_itr - m) / s, y[itr].double(), ALPHAS)
        best_a = min(ws, key=lambda a: (((Z_iva - m) / s @ ws[a]) - y[iva].double())
                     .pow(2).mean().item())
        chosen.append(float(best_a))
        del Z_itr, Z_iva

        Z_tr = antisym_features(X[tr], mode).double()
        Z_te = antisym_features(X[te], mode).double()
        m, s = Z_tr.mean(0), Z_tr.std(0).clamp(min=1e-8)
        w = _ridge((Z_tr - m) / s, y[tr].double(), [best_a])[best_a]
        y_pred = (((Z_te - m) / s) @ w).float().cpu()
        r2s.append(float(pearsonr(y[te].cpu(), y_pred) ** 2))
        if save_direction:
            dirs.append((w / w.norm()).float().cpu().numpy())
        del Z_tr, Z_te

    if device.type == "cuda":
        torch.cuda.empty_cache()
    direction = np.mean(dirs, axis=0) if dirs else None
    return r2s, direction, chosen


# ── PLS component sweep ──────────────────────────────────────────────────────

def pls_sweep(X, y, fold_data, device, mode, k_max=15):
    """R2 for each truncation K = 1..k_max, on the same folds."""
    per_k = {k: [] for k in range(1, k_max + 1)}
    for tr_np, te_np, _fold in fold_data:
        tr = torch.from_numpy(tr_np).to(device)
        te = torch.from_numpy(te_np).to(device)
        Z_tr = antisym_features(X[tr], mode)
        Z_te = antisym_features(X[te], mode)
        m, s = Z_tr.mean(0), Z_tr.std(0).clamp(min=1e-8)
        Z_tr, Z_te = (Z_tr - m) / s, (Z_te - m) / s
        y_tr, y_te = y[tr], y[te].cpu()

        T, W_star, _ = nipals_pls(Z_tr, y_tr, k_max, device)
        T, W_star = T.to(device), W_star.to(device)
        for k in range(1, k_max + 1):
            b_k = torch.linalg.lstsq(T[:, :k], y_tr.float().reshape(-1, 1)).solution
            pred = ((Z_te @ W_star[:, :k]) @ b_k).squeeze(-1).cpu()
            per_k[k].append(float(pearsonr(y_te, pred) ** 2))
        del Z_tr, Z_te, T, W_star

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [(k, float(np.mean(v)),
             float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for k, v in per_k.items()]


# ── folds (identical construction to by_layer_mlp.py) ────────────────────────

def build_folds(w1_all, w2_all, splits):
    out = {}
    n = len(w1_all)
    if "pair_novel" in splits:
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        out["pair_novel"] = {f: (tr, te)
                             for f, (tr, te) in enumerate(kf.split(np.arange(n)))}
    if "word_novel" in splits:
        rng = np.random.default_rng(SEED)
        words = np.array(sorted(set(w1_all) | set(w2_all)))
        perm = rng.permutation(len(words))
        w2f = {words[i]: int(perm[i] % FOLDS) for i in range(len(words))}
        f1 = np.array([w2f.get(w, -1) for w in w1_all])
        f2 = np.array([w2f.get(w, -1) for w in w2_all])
        d = {}
        for f in range(FOLDS):
            d[f] = ((f1 != f) & (f2 != f), (f1 == f) & (f2 == f))
        out["word_novel"] = d
    return out


def fold_list(fold_assignments, split):
    fd = []
    for f in range(FOLDS):
        fa = fold_assignments[split][f]
        if split == "pair_novel":
            fd.append((fa[0], fa[1], f))
        else:
            if fa[1].sum() < 10:
                continue
            fd.append((np.where(fa[0])[0], np.where(fa[1])[0], f))
    return fd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-slug", required=True, dest="model_slug")
    p.add_argument("--num-layers", type=int, required=True, dest="num_layers")
    p.add_argument("--conditions", nargs="+", default=["default", "attn_zeroed"])
    p.add_argument("--modes", nargs="+", default=["mean_pooled", "words_only"])
    p.add_argument("--splits", nargs="+", default=["pair_novel", "word_novel"])
    p.add_argument("--pls-kmax", type=int, default=15, dest="pls_kmax")
    p.add_argument("--skip-pls", action="store_true", dest="skip_pls")
    p.add_argument("--skip-linear", action="store_true", dest="skip_linear")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Restrict to these layer indices (default: all available)")
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="Where to write result CSVs (default: Data/supplementary/<slug>)")
    p.add_argument("--save-directions", action="store_true", dest="save_directions",
                   help="Save the mean linear direction per cell (used by steering.py)")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = load_device(args.gpu)
    emb_root = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"
    out_dir = (Path(args.out_dir) if args.out_dir
               else BASE / "Data" / "supplementary" / args.model_slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.model_slug}   device: {device}")
    print(f"out:   {out_dir}", flush=True)

    lin_f = open(out_dir / "linear_probe.csv", "w", newline="")
    lin_w = _csv.DictWriter(lin_f, fieldnames=["condition", "layer", "mode", "split",
                                               "mean_r2", "sd_r2", "n_folds",
                                               "alpha_median", "alpha_min",
                                               "alpha_max", "alpha_at_grid_max"])
    lin_w.writeheader()

    pls_f = open(out_dir / "pls_sweep.csv", "w", newline="")
    pls_w = _csv.DictWriter(pls_f, fieldnames=["condition", "layer", "mode", "split",
                                               "k", "mean_r2", "sd_r2"])
    pls_w.writeheader()

    directions = {}
    t0 = time.perf_counter()

    for condition in args.conditions:
        novel_dir = emb_root / CONDITION_DIRS[condition]["novel"] / args.model_slug
        tags = [f"layer_{i}" for i in range(args.num_layers + 1)]
        avail = [t for t in tags if resolve_npz(novel_dir, t, args.num_layers)]
        if args.layers is not None:
            avail = [t for t in avail if int(t.split("_")[1]) in args.layers]
        if not avail:
            print(f"  no embeddings for condition={condition}, skipping")
            continue
        print(f"\n=== {condition}: {len(avail)} layers ===", flush=True)

        ref = resolve_npz(novel_dir, avail[0], args.num_layers)
        raw0 = load_raw_layer(ref.parent, ref.stem)
        folds = build_folds(raw0["w1"], raw0["w2"], args.splits)
        del raw0

        for tag in sorted(avail, key=lambda t: int(t.split("_")[1])):
            li = int(tag.split("_")[1])
            npz = resolve_npz(novel_dir, tag, args.num_layers)
            raw = load_raw_layer(npz.parent, npz.stem)
            if raw is None:
                continue

            for mode in args.modes:
                X, y, _w1, _w2 = x_from_raw(raw, mode)
                X, y = X.to(device), y.to(device)

                for split in args.splits:
                    fd = fold_list(folds, split)

                    if not args.skip_linear:
                        r2s, d, alphas = linear_cv(
                            X, y, fd, device, mode,
                            save_direction=args.save_directions)
                        lin_w.writerow({
                            "condition": condition, "layer": li, "mode": mode,
                            "split": split, "mean_r2": round(float(np.mean(r2s)), 6),
                            "sd_r2": round(float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0, 6),
                            "n_folds": len(r2s),
                            "alpha_median": float(np.median(alphas)) if alphas else "",
                            "alpha_min": float(np.min(alphas)) if alphas else "",
                            "alpha_max": float(np.max(alphas)) if alphas else "",
                            "alpha_at_grid_max": (
                                float(np.mean(np.array(alphas) >= max(ALPHAS)))
                                if alphas else "")})
                        lin_f.flush()
                        print(f"  {tag:>10s} {mode:<12s} {split:<11s} linear "
                              f"r²={np.mean(r2s):.4f}", flush=True)
                        if d is not None and split == args.splits[0]:
                            directions[f"{condition}|{li}|{mode}"] = d

                    if not args.skip_pls:
                        for k, mr2, sr2 in pls_sweep(X, y, fd, device, mode,
                                                     k_max=args.pls_kmax):
                            pls_w.writerow({
                                "condition": condition, "layer": li, "mode": mode,
                                "split": split, "k": k,
                                "mean_r2": round(mr2, 6), "sd_r2": round(sr2, 6)})
                        pls_f.flush()
                        print(f"  {tag:>10s} {mode:<12s} {split:<11s} pls "
                              f"K=1..{args.pls_kmax} written", flush=True)

                del X, y
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del raw

    lin_f.close()
    pls_f.close()
    if directions:
        np.savez_compressed(out_dir / "linear_directions.npz", **directions)
        print(f"saved {len(directions)} directions -> {out_dir/'linear_directions.npz'}")
    print(f"\ndone in {(time.perf_counter()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
