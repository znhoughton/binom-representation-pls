"""
Generate actual-vs-predicted scatter plots for words_only MLP probes.

Replicates the exact fold splits and training from by_layer_mlp.py, then
collects all held-out (y_true, y_pred) pairs and plots them as hexbin density
plots (one panel per layer). Pearson r is annotated on each panel.

Usage (run after chain finishes, needs GPU):
    python Scripts/gen_scatter_words_only.py \\
        --model-slug znhoughton_opt-babylm-1_3b-20eps-seed964 \\
        --num-layers 24 \\
        --layers 0 4 9 13 \\
        --split pair_novel \\
        --condition default \\
        --gpu 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from pls_utils import pearsonr, load_device

BASE  = Path(__file__).resolve().parents[1]
SEED  = 964
HIDDEN, MAX_EPOCHS, PATIENCE = 128, 500, 20
VAL_FRAC, FOLDS, LR, WD, BATCH = 0.1, 10, 1e-3, 1e-4, 4096


class OrderingMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def _incr_mean_std(X, idx, chunk=4096):
    tot = torch.zeros(X.shape[1]); n = 0
    for s in range(0, len(idx), chunk):
        p = X[torch.from_numpy(idx[s:s+chunk])]; tot.add_(p.sum(0)); n += len(p); del p
    m = tot / n
    M2 = torch.zeros(X.shape[1])
    for s in range(0, len(idx), chunk):
        p = X[torch.from_numpy(idx[s:s+chunk])]; M2.add_(((p - m)**2).sum(0)); del p
    s_ = torch.sqrt(M2 / (n - 1)); s_[s_ < 1e-8] = 1.0
    return m, s_


def train_fold_with_preds(X, y, tr_idx, te_idx, fold, device):
    n_all = len(tr_idx)
    n_val = max(1, int(n_all * VAL_FRAC))
    val_loc = np.random.default_rng(SEED + fold + 20000).choice(n_all, size=n_val, replace=False)
    itr = tr_idx[np.setdiff1d(np.arange(n_all), val_loc)]
    ival = tr_idx[val_loc]

    mean_, std_ = _incr_mean_std(X, itr)
    X_val = (X[torch.from_numpy(ival)] - mean_).div_(std_)
    y_val = y[torch.from_numpy(ival)]
    X_te  = (X[torch.from_numpy(te_idx)] - mean_).div_(std_)
    y_te  = y[torch.from_numpy(te_idx)]

    n_tr, D = len(itr), X.shape[1]
    use_cache = False
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if n_tr * D * 4 < free * 0.80:
            X_tr_d = (X[torch.from_numpy(itr)] - mean_).div_(std_).to(device)
            y_tr_d = y[torch.from_numpy(itr)].to(device)
            use_cache = True

    mlp = OrderingMLP(D, HIDDEN).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    g      = torch.Generator(); g.manual_seed(SEED + fold)
    g_flip = torch.Generator(); g_flip.manual_seed(SEED + fold + 10000)
    best_val, best_state, pat = float("inf"), {k: v.clone() for k,v in mlp.state_dict().items()}, 0

    for _ in range(MAX_EPOCHS):
        mlp.train()
        perm = torch.randperm(n_tr, generator=g)
        for s in range(0, n_tr, BATCH):
            idx = perm[s:s+BATCH]
            if use_cache:
                xb = X_tr_d[idx]; yb = y_tr_d[idx]
            else:
                gi = torch.from_numpy(itr[idx.numpy()])
                xb = (X[gi] - mean_).div_(std_).to(device)
                yb = y[gi].to(device)
            flip = (torch.rand(len(xb), generator=g_flip) < 0.5).to(device)
            if flip.any():
                half = xb.shape[1] // 2
                xb[flip] = torch.cat([xb[flip, half:], xb[flip, :half]], dim=1)
                yb[flip] = -yb[flip]
            opt.zero_grad(); loss_fn(mlp(xb), yb).backward(); opt.step()

        mlp.eval()
        with torch.no_grad():
            vl = loss_fn(mlp(X_val.to(device)).cpu(), y_val).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k,v in mlp.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break

    mlp.load_state_dict(best_state)
    if use_cache:
        del X_tr_d, y_tr_d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_te.to(device)).cpu()

    return y_te.numpy(), y_pred.numpy()


def load_words_only(embed_dir, layer_tag):
    npz = np.load(embed_dir / f"{layer_tag}.npz", allow_pickle=True)
    y   = torch.from_numpy(npz["preference"].astype(np.float32, copy=False))
    w1  = npz["word1"].astype(str)
    w2  = npz["word2"].astype(str)
    aw1 = torch.from_numpy(npz["alpha_w1"].astype(np.float32, copy=False))
    nw1 = torch.from_numpy(npz["non_alpha_w1"].astype(np.float32, copy=False))
    X   = torch.cat([aw1, nw1], dim=1)
    return X, y, w1, w2


def make_scatter_png(results_by_layer, split_name, out_path):
    """
    results_by_layer: dict {layer_idx: (y_true_all, y_pred_all)}
    Creates a row of hexbin scatter panels, one per layer.
    """
    layers = sorted(results_by_layer)
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6), facecolor="#fcfcfb")
    if n == 1:
        axes = [axes]

    cmap = plt.cm.Blues
    cmap.set_under("white")

    for ax, li in zip(axes, layers):
        y_true, y_pred = results_by_layer[li]
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        r2 = r ** 2

        # Hexbin density
        hb = ax.hexbin(y_pred, y_true,
                       gridsize=55, cmap="Blues",
                       mincnt=1, linewidths=0.2)

        # OLS line
        m, b = np.polyfit(y_pred, y_true, 1)
        xlo, xhi = y_pred.min(), y_pred.max()
        xs = np.array([xlo, xhi])
        ax.plot(xs, m * xs + b, color="#2a78d6", lw=1.5, zorder=5)

        # Identity reference (y = x)
        lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
        ax.plot(lims, lims, color="#c3c2b7", lw=0.8, ls="--", zorder=4)

        ax.set_title(f"Layer {li}", fontsize=10, color="#52514e", pad=6)
        ax.set_xlabel("Predicted preference", fontsize=8, color="#898781")
        if li == layers[0]:
            ax.set_ylabel("Actual preference", fontsize=8, color="#898781")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.tick_params(labelsize=7, colors="#898781")
        for spine in ax.spines.values():
            spine.set_edgecolor("#e1e0d9")
        ax.set_facecolor("#fcfcfb")

        # r annotation
        ax.annotate(f"r = {r:.3f}\nr² = {r2:.3f}",
                    xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=8, color="#52514e", va="top",
                    fontfamily="monospace")

        n_pts = len(y_true)
        ax.annotate(f"n = {n_pts:,}", xy=(0.05, 0.04), xycoords="axes fraction",
                    fontsize=7, color="#898781", va="bottom")

    split_label = "Pair-novel split" if split_name == "pair_novel" else "Word-novel split"
    fig.suptitle(f"words_only MLP — {split_label} — actual vs. predicted ordering preference",
                 fontsize=10, color="#0b0b0b", y=1.01)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-slug", required=True, dest="model_slug")
    p.add_argument("--num-layers", type=int, required=True, dest="num_layers")
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="Layer indices to include (e.g. 0 4 9 13)")
    p.add_argument("--split", default="pair_novel",
                   choices=["pair_novel", "word_novel"],
                   help="CV split to use (default: pair_novel)")
    p.add_argument("--condition", default="default",
                   choices=["default", "attn_zeroed"])
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = load_device(args.gpu)
    print(f"Device: {device}")

    COND_DIRS = {
        "default":     "novel_embeddings",
        "attn_zeroed": "novel_embeddings_attn_zeroed",
    }
    embed_dir = BASE / "Data" / COND_DIRS[args.condition] / args.model_slug
    out_dir   = BASE / "Results" / args.model_slug / "Plots"

    # Build fold assignments from first available layer
    ref_path = embed_dir / "layer_0.npz"
    ref = np.load(ref_path, allow_pickle=True)
    w1_all = ref["word1"].astype(str)
    w2_all = ref["word2"].astype(str)
    n_pairs = len(w1_all)
    del ref

    rng_cond = np.random.default_rng(SEED)

    if args.split == "pair_novel":
        kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        fold_data = [(tr, te) for tr, te in kf.split(np.arange(n_pairs))]
    else:
        all_words  = np.array(sorted(set(w1_all) | set(w2_all)))
        perm       = rng_cond.permutation(len(all_words))
        w2fold     = {all_words[i]: int(perm[i] % FOLDS) for i in range(len(all_words))}
        w1f = np.array([w2fold.get(w, -1) for w in w1_all])
        w2f = np.array([w2fold.get(w, -1) for w in w2_all])
        fold_data = []
        for fold in range(FOLDS):
            te_mask = (w1f == fold) & (w2f == fold)
            tr_mask = (w1f != fold) & (w2f != fold)
            if te_mask.sum() < 10:
                continue
            fold_data.append((np.where(tr_mask)[0], np.where(te_mask)[0]))

    results_by_layer = {}

    for li in sorted(args.layers):
        layer_tag = f"layer_{li}"
        layer_path = embed_dir / f"{layer_tag}.npz"
        if not layer_path.exists():
            print(f"  Skipping layer {li} — file not found", flush=True)
            continue

        print(f"\nLayer {li}:", flush=True)
        X, y, _, _ = load_words_only(embed_dir, layer_tag)
        y_true_all, y_pred_all = [], []

        for fold, (tr_idx, te_idx) in enumerate(fold_data):
            print(f"  fold {fold+1}/{len(fold_data)} ...", flush=True)
            yt, yp = train_fold_with_preds(X, y, tr_idx, te_idx, fold, device)
            y_true_all.append(yt)
            y_pred_all.append(yp)

        results_by_layer[li] = (
            np.concatenate(y_true_all),
            np.concatenate(y_pred_all)
        )
        r = float(np.corrcoef(*results_by_layer[li])[0, 1])
        print(f"  Layer {li}: r={r:.4f}  r²={r**2:.4f}  n={len(y_true_all[0])*len(fold_data):,}",
              flush=True)
        del X, y

    out_path = out_dir / f"words_only_scatter_{args.split}_{args.condition}.png"
    make_scatter_png(results_by_layer, args.split, out_path)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
