"""
Shared NIPALS PLS1 utilities — imported by all analysis scripts.
"""
import torch


def nipals_pls(X, y, n_components, device):
    """NIPALS PLS1 on GPU/CPU. Returns (T, W_star, b) on CPU."""
    X = X.to(device); y = y.to(device)
    n, p = X.shape
    W = torch.zeros(p, n_components, device=device)
    T = torch.zeros(n, n_components, device=device)
    P = torch.zeros(p, n_components, device=device)
    X_res = X.clone(); y_res = y.clone()
    for k in range(n_components):
        w = X_res.T @ y_res
        w = w / (w.norm() + 1e-12)
        t = X_res @ w
        tss = (t @ t).item() + 1e-14
        p_load = (X_res.T @ t) / tss
        q = (y_res @ t).item() / tss
        X_res -= torch.outer(t, p_load); y_res -= t * q
        W[:, k] = w; T[:, k] = t; P[:, k] = p_load
    PtW = (P.T @ W).double()
    W_star = (W.double() @ torch.linalg.inv(PtW)).float()
    TtT = (T.T @ T).double()
    b = torch.linalg.solve(TtT, (T.T @ y.to(device)).double()).float()
    return T.cpu(), W_star.cpu(), b.cpu()


def pearsonr(x, y):
    x = x.double(); y = y.double()
    xc = x - x.mean(); yc = y - y.mean()
    return float((xc @ yc) / (torch.sqrt((xc**2).sum() * (yc**2).sum()) + 1e-12))


def spearmanr(x, y):
    def _rank(a):
        a = a.double(); o = torch.argsort(a)
        r = torch.empty_like(a); r[o] = torch.arange(1, len(a) + 1, dtype=torch.float64)
        return r
    return pearsonr(_rank(x), _rank(y))


def compute_scale(X):
    """Returns (X_scaled, mean, std). std floored at 1e-8."""
    mean = X.mean(0); std = X.std(0); std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def apply_scale(X, mean, std):
    return (X - mean) / std


def load_device(gpu=0):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")