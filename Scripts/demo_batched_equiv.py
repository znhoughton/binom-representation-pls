"""
Demonstrates that batched torch.bmm training is mathematically equivalent
to a sequential fold loop, producing identical predictions.

Run with any Python >= 3.10 + PyTorch installation (CPU is fine):
    python Scripts/demo_batched_equiv.py
"""
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# ── Toy data ───────────────────────────────────────────────────────────────────
N, D, H, F = 200, 16, 8, 4   # samples, features, hidden units, folds
X = torch.randn(N, D)
y = torch.randn(N)

# Assign each sample to one of F folds (round-robin for simplicity)
fold_ids = torch.arange(N) % F
folds = [(torch.where(fold_ids != k)[0], torch.where(fold_ids == k)[0])
         for k in range(F)]

# ── Shared initialization (same seed → same weights for both approaches) ───────
torch.manual_seed(42)
W1_init = torch.randn(F, D, H) * (2.0 / D) ** 0.5
b1_init = torch.zeros(F, 1, H)
W2_init = torch.randn(F, H, 1) * (2.0 / H) ** 0.5
b2_init = torch.zeros(F, 1, 1)

LR, EPOCHS, BATCH = 1e-2, 5, 64


# ── Sequential training (one fold at a time) ───────────────────────────────────
print("Sequential training...")
seq_preds = []
for k in range(F):
    tr, te = folds[k]
    X_tr, y_tr = X[tr], y[tr]
    X_te = X[te]

    mu    = X_tr.mean(0)
    sigma = X_tr.std(0).clamp(min=1e-8)
    X_tr_n = (X_tr - mu) / sigma
    X_te_n = (X_te - mu) / sigma

    W1 = W1_init[k].clone().requires_grad_(True)
    b1 = b1_init[k].clone().requires_grad_(True)
    W2 = W2_init[k].clone().requires_grad_(True)
    b2 = b2_init[k].clone().requires_grad_(True)
    opt = torch.optim.SGD([W1, b1, W2, b2], lr=LR)

    for epoch in range(EPOCHS):
        perm = torch.randperm(len(tr),
                              generator=torch.Generator().manual_seed(k * 100 + epoch))
        for s in range(0, len(tr), BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = X_tr_n[idx], y_tr[idx]
            h    = torch.relu(xb @ W1 + b1)
            pred = (h @ W2 + b2).squeeze(-1)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        h_te = torch.relu(X_te_n @ W1 + b1)
        seq_preds.append((h_te @ W2 + b2).squeeze(-1))


# ── Batched training (all folds in one torch.bmm call per mini-batch) ──────────
print("Batched training...")
W1b = W1_init.clone().requires_grad_(True)
b1b = b1_init.clone().requires_grad_(True)
W2b = W2_init.clone().requires_grad_(True)
b2b = b2_init.clone().requires_grad_(True)
opt = torch.optim.SGD([W1b, b1b, W2b, b2b], lr=LR)

# Per-fold normalization stats — [F, D]
mu_f    = torch.stack([X[folds[k][0]].mean(0) for k in range(F)])
sigma_f = torch.stack([X[folds[k][0]].std(0).clamp(min=1e-8) for k in range(F)])

n_tr = len(folds[0][0])

for epoch in range(EPOCHS):
    # One independent permutation generator per fold, same seeds as sequential
    perms = [torch.randperm(n_tr,
                            generator=torch.Generator().manual_seed(k * 100 + epoch))
             for k in range(F)]
    for s in range(0, n_tr, BATCH):
        # Gather one mini-batch per fold → [F, b, D]
        idx_k   = [folds[k][0][perms[k][s:s + BATCH]] for k in range(F)]
        X_batch = torch.stack([X[idx_k[k]] for k in range(F)])    # [F, b, D]
        y_batch = torch.stack([y[idx_k[k]] for k in range(F)])    # [F, b]

        # Per-fold normalize (broadcast over batch dimension)
        X_batch = (X_batch - mu_f.unsqueeze(1)) / sigma_f.unsqueeze(1)

        # Batched forward:  [F,b,D] @ [F,D,H] → [F,b,H] → [F,b,1] → [F,b]
        h    = torch.relu(torch.bmm(X_batch, W1b) + b1b)          # [F, b, H]
        pred = (torch.bmm(h, W2b) + b2b).squeeze(-1)              # [F, b]

        # Sum of per-fold MSE losses (each fold contributes equally)
        loss = ((pred - y_batch) ** 2).mean(dim=1).sum()
        opt.zero_grad(); loss.backward(); opt.step()

# Evaluate each fold from the batched weights
batched_preds = []
with torch.no_grad():
    for k in range(F):
        te = folds[k][1]
        X_te_n = (X[te] - mu_f[k]) / sigma_f[k]
        h_te   = torch.relu(X_te_n @ W1b[k] + b1b[k].squeeze(0))
        batched_preds.append((h_te @ W2b[k] + b2b[k].squeeze(0)).squeeze(-1))


# ── Compare results ────────────────────────────────────────────────────────────
print("\nFold  max_abs_diff")
print("-" * 25)
all_close = True
for k in range(F):
    diff = (seq_preds[k] - batched_preds[k]).abs().max().item()
    print(f"  {k}   {diff:.2e}")
    if diff > 1e-5:
        all_close = False

print(f"\nAll folds identical to 1e-5: {all_close}")
assert all_close, "Mismatch — batched and sequential are NOT equivalent!"
print("Assertion passed.")
