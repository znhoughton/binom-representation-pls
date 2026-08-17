"""
cross_model_agreement.py
------------------------
Are the models' preferences for NOVEL binomials arbitrary?

A pair the model never saw cannot have a memorised preference, but the model
still assigns one: some deterministic function of the two words' representations.
That function could be linguistically grounded, or it could be whatever falls
out of the weights. Both are systematic, both are encoded, and both generalise
to held-out words, so probe generalisation cannot separate them.

Cross-model agreement can. An arbitrary function of one model's weights has no
reason to match another model's. If independently trained models agree on which
ordering a novel pair prefers, the agreement must come from something they
share, and what they share is the language.

CONFOUND, stated up front: every model sees the same sentence frame for a given
pair, drawn from a real corpus sentence in which one ordering actually occurred.
The prefix therefore predicts the first word for every model, inflating absolute
agreement. The frame is shared identically across all four cells below, so the
CONTRASTS (attested vs novel, default vs attention-zeroed) are interpretable
even where the absolute level is not.

Outputs (Results/):
  cross_model_agreement.csv         pairwise r for every model pair, per cell
  cross_model_agreement_summary.csv per-cell summary incl. within/between family

Usage:
    python Scripts/cross_model_agreement.py
    python Scripts/cross_model_agreement.py --sets novel
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SEED = 964
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "Results"

# Models in the paper. Pythia-2.8B and OLMo-2-1124-1B exist in Results/ but are
# not part of the paper, so they are excluded.
FAMILY = {
    "znhoughton_opt-babylm-125m-20eps-seed964": "BabyLM",
    "znhoughton_opt-babylm-350m-20eps-seed964": "BabyLM",
    "znhoughton_opt-babylm-1_3b-20eps-seed964": "BabyLM",
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2",
    "gpt2-large": "GPT-2",
    "gpt2-xl": "GPT-2",
    "EleutherAI_pythia-160m": "Pythia",
    "EleutherAI_pythia-410m": "Pythia",
    "EleutherAI_pythia-1b": "Pythia",
    "allenai_OLMo-1B-hf": "OLMo",
    "allenai_OLMo-7B-hf": "OLMo",
    "allenai_OLMo-2-0425-1B": "OLMo",
    "allenai_OLMo-2-1124-7B": "OLMo",
    "meta-llama_Llama-3.2-1B": "Llama",
    "meta-llama_Meta-Llama-3-8B": "Llama",
}


def load_novel(slug, condition):
    d = RES / slug / "cv_preds"
    if not d.is_dir():
        return None
    pat = re.compile(rf"^{re.escape(condition)}_layer(\d+)_mean_pooled_pair_novel\.npz$")
    best, path = -1, None
    for f in d.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best:
            best, path = int(m.group(1)), f
    if path is None:
        return None
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame({"key": np.char.add(np.char.add(
        z["word1"].astype(str), "\t"), z["word2"].astype(str)),
        slug: z["y_true"].astype(float)})


def load_attested(slug, condition):
    p = RES / slug / "by_layer_corpus_pred.csv"
    if not p.exists():
        p = p.with_suffix(".csv.xz")
        if not p.exists():
            return None
    df = pd.read_csv(p, usecols=["condition", "layer", "mode",
                                 "word1", "word2", "y_true"])
    df = df[(df["condition"] == condition) & (df["mode"] == "mean_pooled")]
    if df.empty:
        return None
    df = df[df["layer"] == df["layer"].max()].dropna(
        subset=["word1", "word2", "y_true"])
    return pd.DataFrame({
        "key": df["word1"].astype(str) + "\t" + df["word2"].astype(str),
        slug: df["y_true"].astype(float)})


def pair_predictors(keys):
    """Trivially shared lexical statistics: relative frequency and relative
    length. Partialling these out leaves agreement on something beyond what any
    two models would share just by both knowing English word statistics."""
    from wordfreq import zipf_frequency

    w1, w2 = zip(*(k.split("\t") for k in keys))
    types = sorted(set(w1) | set(w2))
    zf = {t: zipf_frequency(t.lower(), "en") for t in types}
    ln = {t: len(t) for t in types}
    dfreq = np.array([zf[a] - zf[b] for a, b in zip(w1, w2)])
    dlen = np.array([ln[a] - ln[b] for a, b in zip(w1, w2)])
    return np.column_stack([dfreq, dlen])


def residualise(Y, Z):
    """Remove the part of each model's y_true predictable from Z."""
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return Y - A @ beta


def build_matrix(slugs, condition, item_set):
    """Join every model's y_true on the pair key, so rows are aligned pairs."""
    loader = load_novel if item_set == "novel" else load_attested
    frames, present = [], []
    for s in slugs:
        f = loader(s, condition)
        if f is None or f.empty:
            continue
        f = f.drop_duplicates(subset="key").set_index("key")
        frames.append(f)
        present.append(s)
    if len(frames) < 2:
        return None, []
    M = pd.concat(frames, axis=1, join="inner")
    return M, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["attested", "novel"])
    ap.add_argument("--conditions", nargs="*", default=["default", "attn_zeroed"])
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    slugs = [s for s in FAMILY if (RES / s).is_dir()]
    print(f"model dirs found: {len(slugs)}/{len(FAMILY)}\n", flush=True)

    rows, summaries = [], []
    for item_set in args.sets:
        for cond in args.conditions:
            M, present = build_matrix(slugs, cond, item_set)
            if M is None:
                print(f"[skip] {item_set} / {cond}: <2 models", flush=True)
                continue
            print(f"=== {item_set} / {cond}: {len(present)} models, "
                  f"{len(M)} aligned pairs ===", flush=True)

            Z = pair_predictors(M.index.to_list())
            Yr = {s: residualise(M[s].to_numpy(), Z) for s in present}

            rs = []
            for a, b in itertools.combinations(present, 2):
                x, y = M[a].to_numpy(), M[b].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 100:
                    continue
                r = pearsonr(x[ok], y[ok])[0]
                rho = spearmanr(x[ok], y[ok])[0]
                r_res = pearsonr(Yr[a][ok], Yr[b][ok])[0]
                same = FAMILY[a] == FAMILY[b]
                rows.append(dict(item_set=item_set, condition=cond,
                                 model_a=a, model_b=b,
                                 family_a=FAMILY[a], family_b=FAMILY[b],
                                 same_family=same, pearson=r, spearman=rho,
                                 pearson_resid=r_res, n=int(ok.sum())))
                rs.append((r, r_res, same, FAMILY[a] if same else None))

            arr = np.array([r for r, _, _, _ in rs])
            arr_res = np.array([q for _, q, _, _ in rs])
            same_mask = np.array([s for _, _, s, _ in rs])
            fam_of = [f for _, _, _, f in rs]

            # sanity null: shuffle one model's values, agreement must vanish
            a, b = present[0], present[1]
            x = M[a].to_numpy()
            y = rng.permutation(M[b].to_numpy())
            ok = np.isfinite(x) & np.isfinite(y)
            null_r = pearsonr(x[ok], y[ok])[0]

            def add(scope, mask):
                if not np.any(mask):
                    return
                summaries.append(dict(
                    item_set=item_set, condition=cond, scope=scope,
                    n_model_pairs=int(mask.sum()), n_pairs=len(M),
                    mean_r=arr[mask].mean(), median_r=np.median(arr[mask]),
                    min_r=arr[mask].min(), max_r=arr[mask].max(),
                    mean_r_resid=arr_res[mask].mean(),
                    shuffled_null_r=null_r))

            add("all pairs of models", np.ones(len(arr), bool))
            add("within family", same_mask)
            add("between family", ~same_mask)
            # Per-family breakouts. BabyLM is the decisive one: same corpus and
            # data order, different capacity, and the only family for which the
            # novel pairs are genuinely absent from training.
            for fam in sorted({f for f in fam_of if f}):
                add(f"within {fam}", np.array([f == fam for f in fam_of]))

            print(f"    mean r = {arr.mean():.3f}  (within-family "
                  f"{arr[same_mask].mean() if same_mask.any() else float('nan'):.3f}, "
                  f"between {arr[~same_mask].mean():.3f})  "
                  f"residualised {arr_res.mean():.3f}  "
                  f"shuffled null {null_r:+.4f}", flush=True)
            bl = np.array([f == "BabyLM" for f in fam_of])
            if bl.any():
                print(f"    BabyLM only: r = {arr[bl].mean():.3f}  "
                      f"residualised {arr_res[bl].mean():.3f}\n", flush=True)

    if not rows:
        raise SystemExit("no cells produced results")

    pd.DataFrame(rows).to_csv(RES / "cross_model_agreement.csv", index=False)
    S = pd.DataFrame(summaries)
    S.to_csv(RES / "cross_model_agreement_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=" * 78)
    print("CROSS-MODEL AGREEMENT ON y_true")
    print("  high agreement on NOVEL pairs = the preference is not arbitrary")
    print("  absolute level inflated by the shared sentence frame; read CONTRASTS")
    print("=" * 78)
    print(S.round(3).to_string(index=False))
    print(f"\nwrote {RES / 'cross_model_agreement.csv'}")
    print(f"wrote {RES / 'cross_model_agreement_summary.csv'}")


if __name__ == "__main__":
    main()
