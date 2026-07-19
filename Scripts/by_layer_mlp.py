"""
By-layer MLP analysis: train MLP probes at every layer and compare r².

Optimizations vs the full pipeline:
  - All layer embeddings loaded once, fold splits computed once and reused
  - No subprocess overhead, no intermediate file I/O
  - Saves summary CSV + per-pair CV predictions (NPZ in Results/{slug}/cv_preds/)

Usage:
    python Scripts/by_layer_mlp.py --model-slug znhoughton_opt-babylm-125m-20eps-seed964 --num-layers 12
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

def pearsonr(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-8))


def compute_scale(X):
    mean_ = X.mean(dim=0)
    std_ = X.std(dim=0).clamp(min=1e-8)
    return (X - mean_) / std_, mean_, std_


def apply_scale(X, mean_, std_):
    return (X - mean_) / std_


def load_device(gpu=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def nipals_pls(X, y, K, device):
    """NIPALS PLS1; returns (T_scores, W_star, b_coef) on CPU."""
    X = X.float().to(device)
    y = y.float().reshape(-1).to(device)
    n, p = X.shape
    W = torch.zeros(p, K, device=device)
    P = torch.zeros(p, K, device=device)
    T = torch.zeros(n, K, device=device)
    E, f = X.clone(), y.clone()
    for k in range(K):
        w = E.T @ f
        w = w / w.norm().clamp(min=1e-8)
        t = E @ w
        t_sq = (t @ t).clamp(min=1e-8)
        p_k = E.T @ t / t_sq
        q_k = float(f @ t / t_sq)
        E = E - t.unsqueeze(1) * p_k.unsqueeze(0)
        f = f - q_k * t
        W[:, k], P[:, k], T[:, k] = w, p_k, t
    W_star = W @ torch.linalg.inv(P.T @ W)
    b_coef = torch.linalg.lstsq(T, y.unsqueeze(-1)).solution.squeeze(-1)
    return T.cpu(), W_star.cpu(), b_coef.cpu()

BASE = Path(__file__).resolve().parents[1]

SEED         = 964
HIDDEN       = 128
MAX_EPOCHS   = 500
PATIENCE     = 5
VAL_FRAC     = 0.1
FOLDS        = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH        = 4096


class OrderingMLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def _incremental_mean_std(X, global_idx, chunk=4096):
    """Two-pass mean/std over X[global_idx] without creating a full-size copy."""
    total = torch.zeros(X.shape[1])
    count = 0
    for s in range(0, len(global_idx), chunk):
        part = X[torch.from_numpy(global_idx[s:s + chunk])]
        total.add_(part.sum(0))
        count += len(part)
        del part
    mean_ = total / count
    M2 = torch.zeros(X.shape[1])
    for s in range(0, len(global_idx), chunk):
        part = X[torch.from_numpy(global_idx[s:s + chunk])]
        M2.add_(((part - mean_) ** 2).sum(0))
        del part
    std_ = torch.sqrt(M2 / (count - 1))
    std_[std_ < 1e-8] = 1.0
    return mean_, std_


def train_all_folds(X, y, fold_data, device, mode):
    """Train all folds simultaneously with batched matrix multiply (torch.bmm).

    fold_data: list of (tr_idx_np, te_idx_np, fold_id)
    X, y: already on device, shape [N, D] and [N]
    Returns: list of (r2, y_te_np, y_pred_np, best_epoch, n_epochs) per fold.
    """
    F = len(fold_data)
    if F == 0:
        return []
    D = X.shape[1]

    # ── Precompute inner train/val splits and per-fold normalization ────────
    inner_tr_gpu, inner_val_gpu, inner_te_gpu = [], [], []
    y_te_cpu = []
    fold_mean = torch.zeros(F, D, device=device)
    fold_std  = torch.ones(F,  D, device=device)
    n_tr_sizes = []

    for k, (tr_np, te_np, fold_id) in enumerate(fold_data):
        n_all = len(tr_np)
        n_val = max(1, int(n_all * VAL_FRAC))
        rng_val = np.random.default_rng(SEED + fold_id + 20000)
        val_local = rng_val.choice(n_all, size=n_val, replace=False)
        itr_np  = tr_np[np.setdiff1d(np.arange(n_all), val_local)]
        ival_np = tr_np[val_local]

        itr_d  = torch.from_numpy(itr_np).to(device)
        ival_d = torch.from_numpy(ival_np).to(device)
        ite_d  = torch.from_numpy(te_np).to(device)

        X_tr_k = X[itr_d].float()
        fold_mean[k] = X_tr_k.mean(0)
        fold_std[k]  = X_tr_k.std(0, unbiased=True).clamp(min=1e-8)
        del X_tr_k

        inner_tr_gpu.append(itr_d)
        inner_val_gpu.append(ival_d)
        inner_te_gpu.append(ite_d)
        y_te_cpu.append(y[ite_d].cpu())
        n_tr_sizes.append(len(itr_d))

    # Precompute normalized val sets (kept on GPU for fast per-epoch check)
    X_val_list = [(X[inner_val_gpu[k]].float() - fold_mean[k]) / fold_std[k] for k in range(F)]
    y_val_list = [y[inner_val_gpu[k]] for k in range(F)]

    n_tr_min = min(n_tr_sizes)

    # ── Initialize batched weights [F, ...] ────────────────────────────────
    torch.manual_seed(SEED)
    W1 = torch.randn(F, D,      HIDDEN, device=device) * (2.0 / D)      ** 0.5
    b1 = torch.zeros(F, 1,      HIDDEN, device=device)
    W2 = torch.randn(F, HIDDEN, 1,      device=device) * (2.0 / HIDDEN) ** 0.5
    b2 = torch.zeros(F, 1,      1,      device=device)
    params = [W1, b1, W2, b2]
    for p in params:
        p.requires_grad_(True)

    # Manual Adam state (same hyper-params as before)
    beta1, beta2, eps_a = 0.9, 0.999, 1e-8
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    adam_t = 0

    # Best checkpoint per fold
    best_W1, best_b1 = W1.detach().clone(), b1.detach().clone()
    best_W2, best_b2 = W2.detach().clone(), b2.detach().clone()
    best_val_loss  = [float("inf")] * F
    best_epoch_arr = [0]           * F
    patience_count = [0]           * F
    active         = [True]        * F
    stopped_epoch  = [MAX_EPOCHS]  * F

    # Per-fold random generators (same seeds as sequential version)
    g_perms = [torch.Generator(device=device) for _ in range(F)]
    g_flips = [torch.Generator(device=device) for _ in range(F)]
    for k, (_tr, _te, fold_id) in enumerate(fold_data):
        g_perms[k].manual_seed(SEED + fold_id)
        g_flips[k].manual_seed(SEED + fold_id + 10000)

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(MAX_EPOCHS):
        if not any(active):
            break

        perms = [
            torch.randperm(n_tr_sizes[k], generator=g_perms[k], device=device)
            for k in range(F)
        ]
        act_f = torch.tensor(active, dtype=torch.float32, device=device)  # [F]

        for start in range(0, n_tr_min, BATCH):
            b_end = min(start + BATCH, n_tr_min)  # cap at n_tr_min so all folds give same slice size
            # Gather per-fold batch indices → [F, b]
            batch_global = torch.stack([
                inner_tr_gpu[k][perms[k][start:b_end]]
                for k in range(F)
            ])
            b_size = batch_global.shape[1]

            X_batch = X[batch_global].float()  # [F, b, D]
            y_batch = y[batch_global]  # [F, b]

            # Per-fold normalization
            X_batch = (X_batch - fold_mean.unsqueeze(1)) / fold_std.unsqueeze(1)

            # Augmentation: symmetry-flip per sample per fold
            flip = torch.stack([
                torch.rand(b_size, generator=g_flips[k], device=device) < 0.5
                for k in range(F)
            ])  # [F, b]
            if mode in ("mean_pooled", "words_only"):
                half = D // 2
                X_flip = torch.cat([X_batch[:, :, half:], X_batch[:, :, :half]], dim=2)
            else:  # individual
                t = D // 3
                X_flip = torch.cat([X_batch[:, :, 2*t:],
                                    X_batch[:, :,   t:2*t],
                                    X_batch[:, :,   :t  ]], dim=2)
            X_batch = torch.where(flip.unsqueeze(-1), X_flip, X_batch)
            y_batch = torch.where(flip, -y_batch, y_batch)

            # Forward: [F, b, D] × [F, D, H] → [F, b, H] → [F, b, 1] → [F, b]
            h    = torch.relu(torch.bmm(X_batch, W1) + b1)
            pred = (torch.bmm(h, W2) + b2).squeeze(-1)

            # MSE loss per fold; inactive folds contribute zero
            loss = ((pred - y_batch).pow(2).mean(dim=1) * act_f).sum()
            loss.backward()

            # Manual AdamW step (L2 weight decay matches PyTorch Adam default)
            adam_t += 1
            bc1 = 1.0 - beta1 ** adam_t
            bc2 = 1.0 - beta2 ** adam_t
            with torch.no_grad():
                for p, m, v in zip(params, ms, vs):
                    # mask inactive folds in the gradient
                    view = [-1] + [1] * (p.dim() - 1)
                    g = p.grad * act_f.view(*view) + WEIGHT_DECAY * p.detach()
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    step = (m / bc1) / ((v / bc2).sqrt_() + eps_a)
                    p.data.add_(step, alpha=-LR)
                    p.grad.zero_()

        # Val check every epoch — sequential over F folds (small overhead)
        with torch.no_grad():
            for k in range(F):
                if not active[k]:
                    continue
                h_v = torch.relu(X_val_list[k] @ W1[k] + b1[k].squeeze(0))
                p_v = (h_v @ W2[k] + b2[k].squeeze(0)).squeeze(-1)
                v_loss = (p_v - y_val_list[k]).pow(2).mean().item()

                if v_loss < best_val_loss[k]:
                    best_val_loss[k]  = v_loss
                    best_W1[k] = W1[k].clone()
                    best_b1[k] = b1[k].clone()
                    best_W2[k] = W2[k].clone()
                    best_b2[k] = b2[k].clone()
                    best_epoch_arr[k] = epoch
                    patience_count[k] = 0
                else:
                    patience_count[k] += 1
                    if patience_count[k] >= PATIENCE:
                        active[k] = False
                        stopped_epoch[k] = epoch + 1

    # ── Final predictions from best checkpoints ─────────────────────────────
    results = []
    with torch.no_grad():
        for k in range(F):
            X_te_k = (X[inner_te_gpu[k]].float() - fold_mean[k]) / fold_std[k]
            h_te   = torch.relu(X_te_k @ best_W1[k] + best_b1[k].squeeze(0))
            y_pred = (h_te @ best_W2[k] + best_b2[k].squeeze(0)).squeeze(-1).cpu()
            y_te   = y_te_cpu[k]
            r2     = float(pearsonr(y_te, y_pred) ** 2)
            results.append((r2, y_te.numpy(), y_pred.numpy(),
                            best_epoch_arr[k], stopped_epoch[k]))

    del X_val_list, y_val_list, fold_mean, fold_std
    del W1, b1, W2, b2, best_W1, best_b1, best_W2, best_b2, ms, vs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def train_novel_predict_corpus(X_nov, y_nov, X_cor, y_cor, mode, device, layer_name):
    """Train MLP on all novel pairs, predict on corpus pairs. Returns (y_pred, r²).
    X_nov, y_nov, X_cor must already be on device.
    """
    n_all = len(y_nov)
    n_val = max(1, int(n_all * VAL_FRAC))
    rng_val = np.random.default_rng(SEED + 99999)
    val_idx = rng_val.choice(n_all, size=n_val, replace=False)
    tr_mask = np.ones(n_all, dtype=bool); tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]

    tr_idx_d  = torch.from_numpy(tr_idx).to(device)
    val_idx_d = torch.from_numpy(val_idx).to(device)

    X_tr  = X_nov[tr_idx_d].float()
    y_tr  = y_nov[tr_idx_d]
    X_val = X_nov[val_idx_d].float()
    y_val = y_nov[val_idx_d]

    X_tr_sc, mean_, std_ = compute_scale(X_tr)
    X_val_sc = apply_scale(X_val, mean_, std_)
    X_cor_sc = apply_scale(X_cor, mean_, std_)

    input_dim = X_tr.shape[1]
    n_tr = len(y_tr)

    mlp = OrderingMLP(input_dim, HIDDEN).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    g = torch.Generator(); g.manual_seed(SEED + 77777)
    g_flip = torch.Generator(); g_flip.manual_seed(SEED + 77777 + 10000)

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        mlp.train()
        perm = torch.randperm(n_tr, generator=g)
        for start in range(0, n_tr, BATCH):
            idx = perm[start:start + BATCH]
            xb = X_tr_sc[idx]; yb = y_tr[idx]
            flip = (torch.rand(len(xb), generator=g_flip) < 0.5).to(device)
            if flip.any():
                if mode in ("mean_pooled", "words_only"):
                    half = xb.shape[1] // 2
                    flipped = torch.cat([xb[flip, half:], xb[flip, :half]], dim=1)
                elif mode == "individual":
                    third = xb.shape[1] // 3
                    flipped = torch.cat([xb[flip, 2*third:], xb[flip, third:2*third], xb[flip, :third]], dim=1)
                xb[flip] = flipped; yb[flip] = -yb[flip]
            opt.zero_grad(); loss_fn(mlp(xb), yb).backward(); opt.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = loss_fn(mlp(X_val_sc), y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    mlp.load_state_dict(best_state)
    del X_tr, X_val, y_tr, y_val, X_tr_sc, X_val_sc
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_cor_sc).cpu()

    r2 = pearsonr(y_cor, y_pred) ** 2
    print(f"  {layer_name:>10s}  corpus_pred   {mode:<14s}  r²={r2:.4f}", flush=True)
    return y_pred, r2


def load_layer(embed_dir, layer_tag, mode="mean_pooled"):
    """
    Load embeddings and construct MLP input.
    mode:
      "mean_pooled" — [mean(w1,and,w2); mean(w2,and,w1)] dim=2*D (both orderings)
      "individual"  — [H(w1); H(and); H(w2)] dim=3*D (alpha ordering only, augmented)
      "words_only"  — [H(w1); H(w2)] dim=2*D (content words only, both orderings as augmentation)

    Fix A: only loads the embedding keys required for the requested mode (saves 5-8 GB RAM
    for 1.3B individual/words_only by not loading the non_alpha* keys).
    Fix B: astype(copy=False) avoids a redundant copy when arrays are already float32.
    """
    path = embed_dir / f"{layer_tag}.npz"
    npz = np.load(path, allow_pickle=True)
    y = torch.from_numpy(npz["preference"].astype(np.float32, copy=False))
    w1 = npz["word1"].astype(str)
    w2 = npz["word2"].astype(str)

    # New per-word format
    if "alpha_w1" in npz:
        if mode == "mean_pooled":
            aw1  = torch.from_numpy(npz["alpha_w1"].astype(np.float32, copy=False))
            aand = torch.from_numpy(npz["alpha_and"].astype(np.float32, copy=False))
            aw2  = torch.from_numpy(npz["alpha_w2"].astype(np.float32, copy=False))
            nw1  = torch.from_numpy(npz["non_alpha_w1"].astype(np.float32, copy=False))
            nand = torch.from_numpy(npz["non_alpha_and"].astype(np.float32, copy=False))
            nw2  = torch.from_numpy(npz["non_alpha_w2"].astype(np.float32, copy=False))
            va  = (aw1 + aand + aw2) / 3.0
            vna = (nw1 + nand + nw2) / 3.0
            X = torch.cat([va, vna], dim=1)
            del aw1, aand, aw2, nw1, nand, nw2, va, vna
        elif mode == "individual":
            aw1  = torch.from_numpy(npz["alpha_w1"].astype(np.float32, copy=False))
            aand = torch.from_numpy(npz["alpha_and"].astype(np.float32, copy=False))
            aw2  = torch.from_numpy(npz["alpha_w2"].astype(np.float32, copy=False))
            X = torch.cat([aw1, aand, aw2], dim=1)
            del aw1, aand, aw2
        elif mode == "words_only":
            aw1 = torch.from_numpy(npz["alpha_w1"].astype(np.float32, copy=False))
            nw1 = torch.from_numpy(npz["non_alpha_w1"].astype(np.float32, copy=False))
            X = torch.cat([aw1, nw1], dim=1)
            del aw1, nw1
        else:
            raise ValueError(f"Unknown mode: {mode}")
    # Legacy format (backward compat)
    elif "vec_alpha" in npz:
        va  = torch.from_numpy(npz["vec_alpha"].astype(np.float32, copy=False))
        vna = torch.from_numpy(npz["vec_non_alpha"].astype(np.float32, copy=False))
        X = torch.cat([va, vna], dim=1)
        del va, vna
    else:
        raise ValueError(f"Unknown npz format in {path}")

    return X, y, w1, w2


def load_raw_layer(embed_dir, layer_tag):
    """Load all raw embedding arrays from an NPZ (format-agnostic).
    Avoids repeated disk reads when the same layer is needed across multiple modes.
    Returns None if the file does not exist.
    """
    path = embed_dir / f"{layer_tag}.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    out = {
        "y":  torch.from_numpy(npz["preference"].astype(np.float32, copy=False)),
        "w1": npz["word1"].astype(str),
        "w2": npz["word2"].astype(str),
    }
    if "alpha_w1" in npz:
        out["fmt"] = "per_word"
        for k in ("alpha_w1", "alpha_and", "alpha_w2",
                   "non_alpha_w1", "non_alpha_and", "non_alpha_w2"):
            out[k] = torch.from_numpy(npz[k].astype(np.float32, copy=False))
    elif "vec_alpha" in npz:
        out["fmt"] = "legacy"
        out["va"]  = torch.from_numpy(npz["vec_alpha"].astype(np.float32, copy=False))
        out["vna"] = torch.from_numpy(npz["vec_non_alpha"].astype(np.float32, copy=False))
    else:
        raise ValueError(f"Unknown npz format in {path}")
    return out


def x_from_raw(raw, mode):
    """Compute mode-specific X tensor from pre-loaded raw data (no disk I/O)."""
    y, w1, w2 = raw["y"], raw["w1"], raw["w2"]
    if raw["fmt"] == "per_word":
        if mode == "mean_pooled":
            va  = (raw["alpha_w1"] + raw["alpha_and"] + raw["alpha_w2"]) / 3.0
            vna = (raw["non_alpha_w1"] + raw["non_alpha_and"] + raw["non_alpha_w2"]) / 3.0
            X   = torch.cat([va, vna], dim=1)
        elif mode == "individual":
            X = torch.cat([raw["alpha_w1"], raw["alpha_and"], raw["alpha_w2"]], dim=1)
        elif mode == "words_only":
            X = torch.cat([raw["alpha_w1"], raw["non_alpha_w1"]], dim=1)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        X = torch.cat([raw["va"], raw["vna"]], dim=1)
    return X, y, w1, w2


def resolve_npz(directory, layer_tag, num_layers):
    layer_idx = int(layer_tag.split("_")[1])
    p = directory / f"{layer_tag}.npz"
    if p.exists():
        return p
    if layer_idx == num_layers:
        p = directory / "layer_last.npz"
    elif layer_idx == num_layers - 1:
        p = directory / "layer_second_to_last.npz"
    return p if p.exists() else None


def _load_layer_pair(novel_dir, corpus_dir, layer_tag, num_layers, do_corpus_freq):
    """Load novel + corpus NPZs for one layer tag. Designed to run in a background thread."""
    nov_npz = resolve_npz(novel_dir, layer_tag, num_layers)
    if nov_npz is None:
        return layer_tag, None, None, None, None
    raw_nov = load_raw_layer(nov_npz.parent, nov_npz.stem)
    cor_npz = resolve_npz(corpus_dir, layer_tag, num_layers) if do_corpus_freq else None
    raw_cor = (load_raw_layer(cor_npz.parent, cor_npz.stem)
               if cor_npz is not None else None)
    return layer_tag, nov_npz, raw_nov, cor_npz, raw_cor


def run_cv(X, y, w1, w2, fold_assignments, split_name, mode, device, layer_name, pred_path=None):
    n = len(w1)
    if pred_path is not None:
        y_true_out = np.full(n, np.nan, dtype=np.float32)
        y_pred_out = np.full(n, np.nan, dtype=np.float32)
        fold_out   = np.full(n, -1,    dtype=np.int8)

    fold_data = []
    for fold in range(FOLDS):
        if split_name == "pair_novel":
            tr_idx, te_idx = fold_assignments[fold]
        else:
            tr_mask, te_mask = fold_assignments[fold]
            if te_mask.sum() < 10:
                continue
            tr_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
        fold_data.append((tr_idx, te_idx, fold))

    all_results = train_all_folds(X, y, fold_data, device, mode)

    r2s, best_eps, n_eps = [], [], []
    for (r2, y_te_np, y_pred_np, best_ep, n_ep), (_tr, te_idx, fold) in zip(all_results, fold_data):
        r2s.append(r2)
        best_eps.append(best_ep)
        n_eps.append(n_ep)
        if pred_path is not None:
            y_true_out[te_idx] = y_te_np
            y_pred_out[te_idx] = y_pred_np
            fold_out[te_idx]   = fold

    if pred_path is not None:
        np.savez_compressed(pred_path,
                            word1=w1, word2=w2,
                            y_true=y_true_out, y_pred=y_pred_out,
                            fold=fold_out)

    mean_r2      = np.mean(r2s)
    sd_r2        = np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0
    mean_best_ep = float(np.mean(best_eps))
    mean_n_ep    = float(np.mean(n_eps))
    print(f"  {layer_name:>10s}  {split_name:<12s}  {mode:<14s}  "
          f"r²={mean_r2:.4f} ± {sd_r2:.4f}  "
          f"conv={mean_best_ep:.0f}ep  (n_folds={len(r2s)})", flush=True)
    return mean_r2, sd_r2, len(r2s), mean_best_ep, mean_n_ep


def main():
    global BATCH
    p = argparse.ArgumentParser()
    p.add_argument("--model-slug", required=True, dest="model_slug")
    p.add_argument("--num-layers", type=int, required=True, dest="num_layers",
                   help="Number of transformer layers (e.g., 12 for 125M, 24 for 350M)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--modes", nargs="+", default=["mean_pooled", "individual"],
                   choices=["mean_pooled", "individual", "words_only"],
                   help="Input modes to test (default: mean_pooled individual)")
    p.add_argument("--splits", nargs="+", default=None,
                   help="CV splits to run (pair_novel word_novel); omit to skip CV entirely")
    p.add_argument("--conditions", nargs="+", default=["default"],
                   choices=["default", "attn_zeroed"],
                   help="Embedding conditions to test (default: default)")
    p.add_argument("--freq-strata", action="store_true", dest="freq_strata",
                   help="Also run frequency stratum analysis (train on each freq bin of corpus, test on novel)")
    p.add_argument("--freq-bootstrap", type=int, default=50, dest="freq_bootstrap",
                   help="Bootstrap iterations for freq strata (equalizes N across bins; default: 50)")
    p.add_argument("--corpus-freq", action="store_true", dest="corpus_freq",
                   help="Train on all novel pairs, predict on corpus pairs; save per-pair predictions for frequency regression")
    p.add_argument("--control", action="store_true",
                   help="Shuffle labels before CV (selectivity control); writes to by_layer_mlp_control.csv")
    p.add_argument("--batch", type=int, default=BATCH,
                   help=f"Training mini-batch size (default: {BATCH}; increase for high-VRAM GPUs)")
    p.add_argument("--embeddings-dir", default=None, dest="embeddings_dir",
                   help="Root directory for embedding subdirs (default: <project>/Data)")
    p.add_argument("--force", action="store_true",
                   help="Re-run all computations, ignoring any existing output files")
    args = p.parse_args()

    emb_root = Path(args.embeddings_dir) if args.embeddings_dir else BASE / "Data"

    BATCH = args.batch

    device = load_device(args.gpu)
    torch.manual_seed(SEED)

    rng = np.random.default_rng(SEED)

    CONDITION_DIRS = {
        "default":     {"novel": "novel_embeddings",             "corpus": "embeddings"},
        "attn_zeroed": {"novel": "novel_embeddings_attn_zeroed", "corpus": "embeddings_attn_zeroed"},
    }

    out_fname = "by_layer_mlp_control.csv" if args.control else "by_layer_mlp.csv"
    out_path = BASE / "Results" / args.model_slug / out_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_slug}")
    print(f"Conditions: {args.conditions}")
    print(f"Splits: {args.splits}")
    print(f"Modes: {args.modes}")
    print(f"Device: {device}")
    print(f"Hidden: {HIDDEN}")
    if args.control:
        print("CONTROL RUN: labels shuffled per layer (writes to by_layer_mlp_control.csv)")

    # Set up frequency strata if requested (shared across conditions)
    freq_strata = {}
    if args.freq_strata:
        import pandas as pd
        freq_df = pd.read_csv(BASE / "Data" / "corpus_binomials.csv")
        freq_df["total_freq"] = freq_df["freq_w1_w2"] + freq_df["freq_w2_w1"]

        ref_corpus_dir = emb_root / "embeddings" / args.model_slug
        ref_corpus_path = ref_corpus_dir / "layer_last.npz"
        if not ref_corpus_path.exists():
            ref_corpus_path = ref_corpus_dir / f"layer_{args.num_layers}.npz"
        ref_corpus = np.load(ref_corpus_path, allow_pickle=True)
        w1_corpus = ref_corpus["word1"].astype(str)
        w2_corpus = ref_corpus["word2"].astype(str)

        corpus_df = pd.DataFrame({"word1": w1_corpus, "word2": w2_corpus})
        corpus_df = corpus_df.merge(freq_df[["word1","word2","total_freq"]],
                                    on=["word1","word2"], how="left")
        total_freq = corpus_df["total_freq"].fillna(1).values.astype(int)
        del ref_corpus

        freq_strata = {
            "freq_1":    total_freq == 1,
            "freq_2_5":  (total_freq >= 2) & (total_freq <= 5),
            "freq_6_20": (total_freq >= 6) & (total_freq <= 20),
            "freq_gt20": total_freq > 20,
        }
        print(f"Frequency strata: {', '.join(f'{k}={v.sum()}' for k,v in freq_strata.items())}")

    results = []
    t_start = time.perf_counter()

    import csv as _csv
    fieldnames = ["condition", "layer", "mode", "split", "mean_r2", "sd_r2", "n_folds",
                  "mean_best_epoch", "mean_n_epochs"]

    # Load already-completed rows from existing CSV (enables resume after crash).
    # Old CSVs (without mean_best_epoch column) are not resumable — start fresh.
    completed = set()
    if not args.force and args.splits and out_path.exists():
        try:
            with open(out_path, "r", newline="") as _f:
                reader = _csv.DictReader(_f)
                has_new_cols = "mean_best_epoch" in (reader.fieldnames or [])
                for row in reader:
                    if has_new_cols:
                        completed.add((row["condition"], int(row["layer"]), row["mode"], row["split"]))
            if completed:
                print(f"Resuming: {len(completed)} rows already saved, skipping those.")
            elif not has_new_cols:
                print("Old CSV format detected — starting fresh (new columns added).")
        except Exception:
            pass
    elif args.force:
        print("--force: ignoring any existing output files.")

    # Only open the main summary CSV if we're running CV splits
    if args.splits:
        _resuming = bool(completed)
        _csv_f = open(out_path, "a" if _resuming else "w", newline="")
        _csv_writer = _csv.DictWriter(_csv_f, fieldnames=fieldnames)
        if not _resuming:
            _csv_writer.writeheader()
        _csv_f.flush()
    else:
        _csv_f = None
        _csv_writer = None

    # Directory for per-pair CV prediction NPZ files
    cv_preds_dir = None
    if args.splits:
        _preds_subdir = "cv_preds_control" if args.control else "cv_preds"
        cv_preds_dir = out_path.parent / _preds_subdir
        cv_preds_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair predictions CSV for corpus-frequency regression
    completed_pred = set()
    if args.corpus_freq:
        pred_path = out_path.parent / "by_layer_corpus_pred.csv"
        if not args.force and pred_path.exists():
            try:
                with open(pred_path, "r", newline="") as _f:
                    for row in _csv.DictReader(_f):
                        completed_pred.add((row["condition"], int(row["layer"]), row["mode"]))
            except Exception:
                pass
        _pred_resuming = bool(completed_pred)
        _pred_f = open(pred_path, "a" if _pred_resuming else "w", newline="")
        _pred_writer = _csv.DictWriter(_pred_f,
            fieldnames=["condition", "layer", "mode", "word1", "word2", "y_true", "y_pred"])
        if not _pred_resuming:
            _pred_writer.writeheader()
        _pred_f.flush()

    for condition in args.conditions:
        dirs = CONDITION_DIRS[condition]
        novel_dir = emb_root / dirs["novel"] / args.model_slug
        corpus_dir = emb_root / dirs["corpus"] / args.model_slug

        # Find available layers for this condition
        layer_tags = [f"layer_{i}" for i in range(args.num_layers + 1)]
        available = [t for t in layer_tags if (novel_dir / f"{t}.npz").exists()]
        for alias in ["layer_last", "layer_second_to_last"]:
            mapped = f"layer_{args.num_layers}" if alias == "layer_last" else f"layer_{args.num_layers - 1}"
            if (novel_dir / f"{alias}.npz").exists() and mapped not in available:
                available.append(mapped)

        print(f"\n{'='*60}")
        print(f"Condition: {condition}  ({len(available)}/{args.num_layers + 1} layers available)")
        print(f"{'='*60}")

        # Fold assignments use word lists from novel data (same across layers within a condition)
        ref_path = resolve_npz(novel_dir, available[0] if available else "layer_0", args.num_layers)
        if ref_path is None:
            print(f"  No novel embeddings found for condition={condition}, skipping")
            continue
        ref = np.load(ref_path, allow_pickle=True)
        w1_all = ref["word1"].astype(str)
        w2_all = ref["word2"].astype(str)
        n_pairs = len(w1_all)

        fold_assignments = {}
        rng_cond = np.random.default_rng(SEED)

        if args.splits and "pair_novel" in args.splits:
            kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
            pair_folds = {}
            for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(n_pairs))):
                pair_folds[fold] = (tr_idx, te_idx)
            fold_assignments["pair_novel"] = pair_folds

        if args.splits and "word_novel" in args.splits:
            all_words = np.array(sorted(set(w1_all) | set(w2_all)))
            perm = rng_cond.permutation(len(all_words))
            word_to_fold = {all_words[i]: int(perm[i] % FOLDS) for i in range(len(all_words))}
            w1_folds = np.array([word_to_fold.get(w, -1) for w in w1_all])
            w2_folds = np.array([word_to_fold.get(w, -1) for w in w2_all])
            word_folds = {}
            for fold in range(FOLDS):
                te_mask = (w1_folds == fold) & (w2_folds == fold)
                tr_mask = (w1_folds != fold) & (w2_folds != fold)
                word_folds[fold] = (tr_mask, te_mask)
            fold_assignments["word_novel"] = word_folds

        del ref

        sorted_layers = sorted(available, key=lambda t: int(t.split("_")[1]))
        _pool = ThreadPoolExecutor(max_workers=1)

        def _prefetch(tag):
            return _load_layer_pair(novel_dir, corpus_dir, tag, args.num_layers, args.corpus_freq)

        _future = _pool.submit(_prefetch, sorted_layers[0]) if sorted_layers else None

        for i, layer_tag in enumerate(sorted_layers):
            layer_idx = int(layer_tag.split("_")[1])

            # Retrieve pre-loaded data; immediately kick off the next layer's disk read
            _, novel_npz, raw_nov, corpus_npz_cf, raw_cor = _future.result()
            _future = (_pool.submit(_prefetch, sorted_layers[i + 1])
                       if i + 1 < len(sorted_layers) else None)

            if novel_npz is None:
                continue

            # Fast-skip entire layer if all work for it is already done
            cv_done = all(
                (condition, layer_idx, mode, split) in completed and
                (cv_preds_dir is None or
                 (cv_preds_dir / f"{condition}_layer{layer_idx}_{mode}_{split}.npz").exists())
                for mode in args.modes for split in (args.splits or [])
            )
            freq_done = (not args.freq_strata or condition != "default" or all(
                (condition, layer_idx, "pls_diff", s) in completed
                for s in ["freq_1", "freq_2_5", "freq_6_20", "freq_gt20"]
            ))
            pred_done = (not args.corpus_freq or all(
                (condition, layer_idx, mode) in completed_pred for mode in args.modes
            ))
            if cv_done and freq_done and pred_done:
                print(f"\n--- {condition} / Layer {layer_idx} --- [skipped]", flush=True)
                del raw_nov
                if raw_cor is not None:
                    del raw_cor
                continue

            print(f"\n--- {condition} / Layer {layer_idx} ---", flush=True)

            if args.corpus_freq and corpus_npz_cf is None:
                print(f"  Skipping corpus_freq for layer {layer_idx}: corpus embeddings not found")

            for mode in args.modes:
                # Determine which splits need work: need CSV row AND pred file
                def _needs_work(sp):
                    has_csv  = (condition, layer_idx, mode, sp) in completed
                    has_pred = (cv_preds_dir is not None and
                                (cv_preds_dir / f"{condition}_layer{layer_idx}_{mode}_{sp}.npz").exists())
                    return not (has_csv and (cv_preds_dir is None or has_pred))

                splits_todo = [sp for sp in (args.splits or []) if _needs_work(sp)]
                corpus_needed = (args.corpus_freq and raw_cor is not None and
                                 (condition, layer_idx, mode) not in completed_pred)

                if not splits_todo and not corpus_needed:
                    for split in (args.splits or []):
                        print(f"  {layer_tag:>10s}  {split:<12s}  {mode:<14s}  [skipped]", flush=True)
                    if args.corpus_freq:
                        print(f"  {layer_tag:>10s}  corpus_pred   {mode:<14s}  [skipped]", flush=True)
                    X_nov = y_nov = w1_nov = w2_nov = None
                else:
                    X_nov, y_nov, w1_nov, w2_nov = x_from_raw(raw_nov, mode)
                    try:
                        X_nov = X_nov.to(device)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        X_nov = X_nov.half().to(device)  # fp16 storage: halves VRAM for large models
                        print(f"  [OOM] {layer_tag} {mode}: using fp16 storage for X_nov "
                              f"({X_nov.shape[0]}×{X_nov.shape[1]}×2B = "
                              f"{X_nov.numel()*2/1e9:.1f} GB)", flush=True)
                    y_nov = y_nov.to(device)
                    if args.control and y_nov is not None:
                        ctrl_perm = np.random.default_rng(SEED + layer_idx * 1000).permutation(len(y_nov))
                        y_nov = y_nov[torch.from_numpy(ctrl_perm).to(device)]

                for split in (args.splits or []):
                    has_csv   = (condition, layer_idx, mode, split) in completed
                    npz_path  = (cv_preds_dir / f"{condition}_layer{layer_idx}_{mode}_{split}.npz"
                                 if cv_preds_dir is not None else None)
                    has_pred  = npz_path is not None and npz_path.exists()

                    if has_csv and (npz_path is None or has_pred):
                        print(f"  {layer_tag:>10s}  {split:<12s}  {mode:<14s}  [skipped]", flush=True)
                        continue

                    mean_r2, sd_r2, n_folds, mean_best_ep, mean_n_ep = run_cv(
                        X_nov, y_nov, w1_nov, w2_nov, fold_assignments[split],
                        split, mode, device, layer_tag,
                        pred_path=None if has_pred else npz_path,
                    )

                    if not has_csv:
                        row = {
                            "condition":        condition,
                            "layer":            layer_idx,
                            "mode":             mode,
                            "split":            split,
                            "mean_r2":          round(mean_r2, 6),
                            "sd_r2":            round(sd_r2, 6),
                            "n_folds":          n_folds,
                            "mean_best_epoch":  round(mean_best_ep, 1),
                            "mean_n_epochs":    round(mean_n_ep, 1),
                        }
                        results.append(row)
                        if _csv_writer:
                            _csv_writer.writerow(row); _csv_f.flush()

                # Corpus-freq merged here: X_nov already on GPU, no re-read from disk
                if corpus_needed:
                    X_cor, y_cor, w1_cor, w2_cor = x_from_raw(raw_cor, mode)
                    X_cor = X_cor.to(device)
                    y_pred_cf, r2_cf = train_novel_predict_corpus(
                        X_nov, y_nov, X_cor, y_cor, mode, device, layer_tag)
                    results.append({
                        "condition": condition, "layer": layer_idx, "mode": mode,
                        "split": "corpus_pred", "mean_r2": round(r2_cf, 6),
                        "sd_r2": 0.0, "n_folds": 1,
                        "mean_best_epoch": "", "mean_n_epochs": "",
                    })
                    y_pred_np = y_pred_cf.numpy()
                    y_true_np = y_cor.numpy()
                    for idx in range(len(w1_cor)):
                        _pred_writer.writerow({
                            "condition": condition, "layer": layer_idx, "mode": mode,
                            "word1": w1_cor[idx], "word2": w2_cor[idx],
                            "y_true": round(float(y_true_np[idx]), 6),
                            "y_pred": round(float(y_pred_np[idx]), 6),
                        })
                    _pred_f.flush()
                    del X_cor, y_cor, y_pred_cf

                if X_nov is not None:
                    del X_nov, y_nov

            del raw_nov
            if raw_cor is not None:
                del raw_cor

            # Frequency strata (default condition only - corpus freq bins don't apply to isolated)
            if args.freq_strata and condition == "default":
                corpus_npz = resolve_npz(corpus_dir, layer_tag, args.num_layers)
                if corpus_npz is None:
                    print(f"  Skipping freq strata: corpus embeddings not found")
                else:
                    novel_npz_path = resolve_npz(novel_dir, layer_tag, args.num_layers)
                    def load_diff(npz_path):
                        npz = np.load(npz_path, allow_pickle=True)
                        if "alpha_w1" in npz:
                            va = (npz["alpha_w1"] + npz["alpha_and"] + npz["alpha_w2"]) / 3.0
                            vna = (npz["non_alpha_w1"] + npz["non_alpha_and"] + npz["non_alpha_w2"]) / 3.0
                            diff = va - vna
                        else:
                            diff = npz["diff_vecs"]
                        return torch.from_numpy(diff.astype(np.float32)), \
                               torch.from_numpy(npz["preference"].astype(np.float32))

                    X_cor_diff, y_cor = load_diff(corpus_npz)
                    X_nov_diff, y_nov_te = load_diff(novel_npz_path)
                    X_cor_diff_np = X_cor_diff.numpy()
                    X_nov_diff_np = X_nov_diff.numpy()

                    stratum_idx = {name: np.where(mask)[0] for name, mask in freq_strata.items()}
                    eq_n = min(len(idx) for idx in stratum_idx.values())
                    B = args.freq_bootstrap
                    boot_rng = np.random.default_rng(SEED)
                    K_PLS = 15

                    for stratum_name, sidx in stratum_idx.items():
                        if (condition, layer_idx, "pls_diff", stratum_name) in completed:
                            print(f"  {layer_tag:>10s}  {stratum_name:<12s}  pls_diff      [skipped]", flush=True)
                            continue
                        boot_r2s = []
                        for b in range(B):
                            sample = boot_rng.choice(sidx, size=eq_n, replace=True)
                            X_s = torch.from_numpy(X_cor_diff_np[sample])
                            y_s = y_cor[torch.from_numpy(sample)]
                            X_s_sc, mean_, std_ = compute_scale(X_s)
                            X_n_sc = apply_scale(X_nov_diff, mean_, std_)
                            _, W_star, b_coef = nipals_pls(X_s_sc, y_s, K_PLS, device)
                            T_nov = (X_n_sc.to(device) @ W_star.to(device)).cpu()
                            y_pred = T_nov @ b_coef
                            r = pearsonr(y_nov_te, y_pred)
                            boot_r2s.append(r ** 2)

                        mean_r2 = float(np.mean(boot_r2s))
                        sd_r2 = float(np.std(boot_r2s, ddof=1))
                        print(f"  {layer_tag:>10s}  {stratum_name:<12s}  pls_diff      "
                              f"r²={mean_r2:.4f} ± {sd_r2:.4f}  (B={B}, N={eq_n})", flush=True)
                        row = {
                            "condition": condition,
                            "layer": layer_idx,
                            "mode": "pls_diff",
                            "split": stratum_name,
                            "mean_r2": round(mean_r2, 6),
                            "sd_r2": round(sd_r2, 6),
                            "n_folds": B,
                            "mean_best_epoch": "",
                            "mean_n_epochs": "",
                        }
                        results.append(row)
                        if _csv_writer:
                            _csv_writer.writerow(row)
                            _csv_f.flush()

                    del X_cor_diff, y_cor, X_nov_diff, y_nov_te, X_cor_diff_np, X_nov_diff_np

            if device.type == "cuda":
                torch.cuda.empty_cache()

        _pool.shutdown(wait=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    if _csv_f:
        _csv_f.close()
        print(f"Saved: {out_path}")
    if args.corpus_freq:
        _pred_f.close()
        print(f"Saved: {pred_path}")

    # Print summary table per condition
    for condition in args.conditions:
        cond_results = [r for r in results if r["condition"] == condition]
        if not cond_results:
            continue
        all_splits = list(dict.fromkeys(r["split"] for r in cond_results))
        print(f"\n{'='*70}")
        print(f"  {condition.upper()}")
        print(f"{'='*70}")
        print(f"{'Layer':>6s}", end="")
        for split in all_splits:
            print(f"  {split[:12]:>12s}", end="")
        print()
        print("-" * (6 + 14 * len(all_splits)))
        for layer_idx in sorted(set(r["layer"] for r in cond_results)):
            print(f"{layer_idx:>6d}", end="")
            for split in all_splits:
                match = [r for r in cond_results
                         if r["layer"] == layer_idx and r["split"] == split]
                if match:
                    print(f"  {match[0]['mean_r2']:>12.4f}", end="")
                else:
                    print(f"  {'—':>12s}", end="")
            print()


if __name__ == "__main__":
    main()
