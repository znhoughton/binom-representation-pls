"""
mlp_comparison.py
---------------------------------------
MLP-based ordering preference prediction.

--input controls the representation:
  diff    diff-vector (alphabetical minus non-alphabetical), p-dim
  concat  concatenation of vec_alpha and vec_non_alpha, 2p-dim (from layer_{layer}.npz)
          Training includes antisymmetric augmentation.

--split controls the train/test design:
  transfer    train on corpus pairs -> test on all novel pairs
  pair_novel  10-fold CV within novel (random pair split)
  word_novel  10-fold CV within novel (word-level split; both words in held-out fold)
  word_strict train on corpus -> test on novel pairs where NEITHER word is in corpus

Architecture: Linear(input_dim, 15) -> Tanh -> Linear(15, 1)
  Hidden dim 15 matches PLS K for fair comparison.

Outputs (per slug/layer):
  mlp_{input}_{split}.csv            -- summary (mean +/- std for CV splits)
  mlp_{input}_{split}_fold_stats.csv -- per-fold r, r2, rho
  mlp_{input}_{split}_loss_curves.csv-- epoch x fold training loss (convergence)
  mlp_{input}_{split}_preds.csv      -- per-pair predictions

Usage:
  python Scripts/lib/mlp_comparison.py --input diff --split transfer
  python Scripts/lib/mlp_comparison.py --input diff --split pair_novel
  python Scripts/lib/mlp_comparison.py --input diff --split word_novel --slug ...
  python Scripts/lib/mlp_comparison.py --input concat --split word_strict --slug ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))
from pls_utils import pearsonr, spearmanr, compute_scale, apply_scale, load_device

BASE   = Path(__file__).resolve().parents[2]
HIDDEN = 15
EPOCHS = 200
FOLDS  = 10
LR     = 1e-3
BATCH  = 2048
SEED   = 964

CV_SPLITS = {"pair_novel", "word_novel"}

parser = argparse.ArgumentParser()
parser.add_argument("--slug",  default="znhoughton_opt-babylm-125m-20eps-seed964")
parser.add_argument("--gpu",   type=int, default=0)
parser.add_argument("--input", choices=["diff", "concat"], required=True)
parser.add_argument("--split", choices=["transfer", "pair_novel", "word_novel", "word_strict"],
                    required=True)
parser.add_argument("--layer", default="last", choices=["last", "second_to_last"])
args = parser.parse_args()

torch.manual_seed(SEED)
rng     = np.random.default_rng(SEED)
device  = load_device(args.gpu)
out_dir = BASE / "Results" / args.slug / f"layer_{args.layer}"
out_dir.mkdir(parents=True, exist_ok=True)
tag     = f"{args.input}_{args.split}"
print(f"Slug: {args.slug}  input: {args.input}  split: {args.split}  device: {device}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_diff(mode):
    path = (BASE / "Data/embeddings" if mode == "corpus"
            else BASE / "Data/novel_embeddings") / args.slug / f"layer_{args.layer}.npz"
    npz = np.load(path, allow_pickle=True)
    return (torch.from_numpy(npz["diff_vecs"].astype(np.float32)),
            torch.from_numpy(npz["preference"].astype(np.float32)),
            npz["word1"].astype(str), npz["word2"].astype(str))


def load_concat(mode):
    path = (BASE / "Data/embeddings" if mode == "corpus"
            else BASE / "Data/novel_embeddings") / args.slug / f"layer_{args.layer}.npz"
    npz = np.load(path, allow_pickle=True)
    va  = torch.from_numpy(npz["vec_alpha"].astype(np.float32))
    vna = torch.from_numpy(npz["vec_non_alpha"].astype(np.float32))
    return (torch.cat([va, vna], dim=1),
            torch.from_numpy(npz["preference"].astype(np.float32)),
            npz["word1"].astype(str), npz["word2"].astype(str))


def load_X(mode):
    return load_diff(mode) if args.input == "diff" else load_concat(mode)


def augment_concat(X, y):
    """Antisymmetric augmentation: add (non_alpha|alpha, -pref) to training set."""
    p = X.shape[1] // 2
    X_flip = torch.cat([X[:, p:], X[:, :p]], dim=1)
    return torch.cat([X, X_flip]), torch.cat([y, -y])


# ── model + training ──────────────────────────────────────────────────────────

class OrderingMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval(X_tr_raw, y_tr_raw, X_te, y_te, fold=0):
    """
    Train MLP on (X_tr_raw, y_tr_raw), evaluate on (X_te, y_te).
    Augmentation and scaling applied internally.
    Preloads training data to GPU when VRAM allows; falls back to per-batch
    transfer otherwise (e.g. large concat embeddings for 1.3b model).
    Returns (metrics_dict, loss_rows, y_pred_te).
    """
    # On-the-fly flip augmentation avoids pre-allocating the doubled tensor
    # (which is ~10 GB for 1.3b concat and OOMs under normal daytime RAM pressure).
    # Antisymmetric flips are applied randomly per-batch in the training loop below.
    X_tr, y_tr = X_tr_raw, y_tr_raw

    X_tr_sc, mean_, std_ = compute_scale(X_tr)
    X_te_sc = apply_scale(X_te, mean_, std_)

    input_dim = X_tr.shape[1]
    n_tr      = len(y_tr)

    # Pre-load to GPU if VRAM allows; avoids per-batch PCIe transfers
    if device.type == "cuda":
        nb = (X_tr_sc.nelement() + y_tr.nelement()) * 4
        free, _ = torch.cuda.mem_get_info(device)
        on_gpu = free > nb * 1.2
    else:
        on_gpu = False

    X_tr_d = X_tr_sc.to(device) if on_gpu else X_tr_sc
    y_tr_d = y_tr.to(device)    if on_gpu else y_tr

    mlp     = OrderingMLP(input_dim).to(device)
    opt     = torch.optim.Adam(mlp.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    g       = torch.Generator()
    g.manual_seed(SEED + fold)
    g_flip  = torch.Generator()
    g_flip.manual_seed(SEED + fold + 10000)

    total_loss = torch.zeros(1, device=device)
    loss_rows  = []
    for epoch in range(EPOCHS):
        mlp.train()
        perm = torch.randperm(n_tr, generator=g)
        total_loss.zero_()
        for start in range(0, n_tr, BATCH):
            idx = perm[start : start + BATCH]
            xb  = X_tr_d[idx]
            yb  = y_tr_d[idx]
            if not on_gpu:
                xb, yb = xb.to(device), yb.to(device)
            if args.input == "concat":
                half = xb.shape[1] // 2
                flip = (torch.rand(len(xb), generator=g_flip) < 0.5).to(xb.device)
                if flip.any():
                    flipped = torch.cat([xb[flip, half:], xb[flip, :half]], dim=1)
                    xb[flip] = flipped
                    yb[flip] = -yb[flip]
            opt.zero_grad()
            loss = loss_fn(mlp(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.detach() * len(idx)
        epoch_loss = total_loss.item() / n_tr
        loss_rows.append({"fold": fold, "epoch": epoch + 1, "train_loss": epoch_loss})
        if (epoch + 1) % 50 == 0:
            print(f"  [fold {fold}] epoch {epoch+1}/{EPOCHS}  loss={epoch_loss:.4f}", flush=True)

    del X_tr_d, y_tr_d
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mlp.eval()
    with torch.no_grad():
        X_tr_orig_sc = apply_scale(X_tr_raw, mean_, std_).to(device)
        y_pred_tr = mlp(X_tr_orig_sc).cpu()
        y_pred_te = mlp(X_te_sc.to(device)).cpu()

    r_tr = pearsonr(y_tr_raw, y_pred_tr)
    r_te = pearsonr(y_te,     y_pred_te)
    rho  = spearmanr(y_te,    y_pred_te)
    print(f"  [fold {fold}] r={r_te:.4f}  r²={r_te**2:.4f}  rho={rho:.4f}", flush=True)

    metrics = {
        "fold": fold, "n_train": len(y_tr_raw), "n_test": len(y_te),
        "train_r":  round(r_tr,     6), "train_r2":  round(r_tr**2, 6),
        "test_r":   round(r_te,     6), "test_r2":   round(r_te**2, 6),
        "test_rho": round(rho,      6),
    }
    return metrics, loss_rows, y_pred_te


# ── splits ────────────────────────────────────────────────────────────────────

fold_stats_rows = []
loss_curve_rows = []
pred_rows       = []

if args.split == "transfer":
    X_tr, y_tr, _, _   = load_X("corpus")
    X_te, y_te, w1, w2 = load_X("novel")
    print(f"Train (corpus): {len(y_tr):,}  Test (novel): {len(y_te):,}")
    metrics, loss_rows, y_pred = train_eval(X_tr, y_tr, X_te, y_te, fold=0)
    fold_stats_rows.append(metrics)
    loss_curve_rows.extend(loss_rows)
    pred_rows = [{"word1": w1[i], "word2": w2[i],
                  "preference": y_te[i].item(), "mlp_pred": y_pred[i].item(),
                  "fold": 0} for i in range(len(y_te))]

elif args.split == "pair_novel":
    X_nov, y_nov, w1_nov, w2_nov = load_X("novel")
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(y_nov)))):
        print(f"\nFold {fold+1}/{FOLDS}  train={len(tr_idx):,}  test={len(te_idx):,}")
        X_tr = X_nov[torch.from_numpy(tr_idx)]
        y_tr = y_nov[torch.from_numpy(tr_idx)]
        X_te = X_nov[torch.from_numpy(te_idx)]
        y_te = y_nov[torch.from_numpy(te_idx)]
        metrics, loss_rows, y_pred = train_eval(X_tr, y_tr, X_te, y_te, fold=fold)
        fold_stats_rows.append(metrics)
        loss_curve_rows.extend(loss_rows)
        for i, idx in enumerate(te_idx):
            pred_rows.append({"word1": w1_nov[idx], "word2": w2_nov[idx],
                               "preference": y_te[i].item(), "mlp_pred": y_pred[i].item(),
                               "fold": fold})

elif args.split == "word_novel":
    X_nov, y_nov, w1_nov, w2_nov = load_X("novel")
    all_words  = np.array(sorted(set(w1_nov) | set(w2_nov)))
    perm       = rng.permutation(len(all_words))
    fold_assign = np.empty(len(all_words), dtype=int)
    for f in range(FOLDS):
        fold_assign[perm[f::FOLDS]] = f
    word_to_fold = {w: fold_assign[i] for i, w in enumerate(all_words)}

    w1_folds = np.array([word_to_fold.get(w, -1) for w in w1_nov])
    w2_folds = np.array([word_to_fold.get(w, -1) for w in w2_nov])

    for fold in range(FOLDS):
        te_mask = (w1_folds == fold) & (w2_folds == fold)
        tr_mask = (w1_folds != fold) & (w2_folds != fold)
        n_te = te_mask.sum()
        if n_te < 10:
            print(f"Fold {fold+1}: only {n_te} test pairs — skipping.")
            continue
        print(f"\nFold {fold+1}/{FOLDS}  train={tr_mask.sum():,}  test={n_te:,}  "
              f"excluded={len(y_nov)-tr_mask.sum()-n_te:,}")
        X_tr = X_nov[torch.from_numpy(tr_mask)]
        y_tr = y_nov[torch.from_numpy(tr_mask)]
        X_te = X_nov[torch.from_numpy(te_mask)]
        y_te = y_nov[torch.from_numpy(te_mask)]
        metrics, loss_rows, y_pred = train_eval(X_tr, y_tr, X_te, y_te, fold=fold)
        fold_stats_rows.append(metrics)
        loss_curve_rows.extend(loss_rows)
        for i, idx in enumerate(np.where(te_mask)[0]):
            pred_rows.append({"word1": w1_nov[idx], "word2": w2_nov[idx],
                               "preference": y_te[i].item(), "mlp_pred": y_pred[i].item(),
                               "fold": fold})

elif args.split == "word_strict":
    X_tr, y_tr, _, _             = load_X("corpus")
    X_nov, y_nov, w1_nov, w2_nov = load_X("novel")
    corpus_npz   = np.load(BASE / "Data/embeddings" / args.slug / f"layer_{args.layer}.npz",
                           allow_pickle=True)
    corpus_words = set(corpus_npz["word1"].astype(str)) | set(corpus_npz["word2"].astype(str))
    te_mask = np.array([w1_nov[i] not in corpus_words and w2_nov[i] not in corpus_words
                        for i in range(len(y_nov))])
    n_te = te_mask.sum()
    print(f"Word-strict test pairs (neither word in corpus): {n_te:,} / {len(y_nov):,}")
    if n_te < 100:
        print("WARNING: fewer than 100 qualifying test pairs — results may be unreliable.")
    X_te = X_nov[torch.from_numpy(te_mask)]
    y_te = y_nov[torch.from_numpy(te_mask)]
    w1, w2 = w1_nov[te_mask], w2_nov[te_mask]
    metrics, loss_rows, y_pred = train_eval(X_tr, y_tr, X_te, y_te, fold=0)
    fold_stats_rows.append(metrics)
    loss_curve_rows.extend(loss_rows)
    pred_rows = [{"word1": w1[i], "word2": w2[i],
                  "preference": y_te[i].item(), "mlp_pred": y_pred[i].item(),
                  "fold": 0} for i in range(len(y_te))]


# ── aggregate and save ────────────────────────────────────────────────────────

fold_df = pd.DataFrame(fold_stats_rows)

if args.split in CV_SPLITS:
    summary = {
        "input": args.input, "split": args.split,
        "folds": len(fold_df), "epochs": EPOCHS,
        "mean_n_train":  round(fold_df["n_train"].mean(), 1),
        "mean_n_test":   round(fold_df["n_test"].mean(),  1),
        "mean_test_r":   round(fold_df["test_r"].mean(),  6),
        "std_test_r":    round(fold_df["test_r"].std(),   6),
        "mean_test_r2":  round(fold_df["test_r2"].mean(), 6),
        "std_test_r2":   round(fold_df["test_r2"].std(),  6),
        "mean_test_rho": round(fold_df["test_rho"].mean(),6),
        "std_test_rho":  round(fold_df["test_rho"].std(), 6),
        "mean_train_r":  round(fold_df["train_r"].mean(), 6),
    }
    print(f"\n{FOLDS}-fold summary:  "
          f"r={summary['mean_test_r']:.4f} ± {summary['std_test_r']:.4f}  "
          f"r²={summary['mean_test_r2']:.4f} ± {summary['std_test_r2']:.4f}")
else:
    row = fold_stats_rows[0]
    summary = {
        "input": args.input, "split": args.split,
        "folds": 1, "epochs": EPOCHS,
        "n_train": row["n_train"],  "n_test":   row["n_test"],
        "test_r":  row["test_r"],   "test_r2":  row["test_r2"],
        "test_rho": row["test_rho"], "train_r": row["train_r"],
    }
    print(f"\nResult:  r={summary['test_r']:.4f}  r²={summary['test_r2']:.4f}  "
          f"rho={summary['test_rho']:.4f}")

pd.DataFrame([summary]).to_csv(out_dir / f"mlp_{tag}.csv", index=False)
fold_df.to_csv(out_dir / f"mlp_{tag}_fold_stats.csv", index=False)
pd.DataFrame(loss_curve_rows).to_csv(out_dir / f"mlp_{tag}_loss_curves.csv", index=False)
pd.DataFrame(pred_rows).to_csv(out_dir / f"mlp_{tag}_preds.csv", index=False)

print(f"Saved mlp_{tag}.csv  mlp_{tag}_fold_stats.csv  "
      f"mlp_{tag}_loss_curves.csv  mlp_{tag}_preds.csv  -> {out_dir}")
