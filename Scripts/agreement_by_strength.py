"""
agreement_by_strength.py
------------------------
Is the cross-model agreement on novel pairs carried by a minority of
strong-preference items, with the bulk agreeing at chance?

Three checks, on the three BabyLM models and the novel set (the only cell where
the pairs are genuinely absent from training):

1. The distribution of |y_true|. If novel-pair preferences were uniformly weak,
   the premise of the objection would hold; if they are not, it does not.

2. Directional (sign) agreement. What fraction of pairs do two models order the
   same way? Chance is 0.50. This weights every pair equally and ignores
   magnitude entirely.

3. Agreement stratified by preference strength, where the stratifying variable
   is the THIRD model's |y_true|. With exactly three models each pair of models
   has a unique third, so the bins are independent of the two series being
   correlated and there is no selection on the outcome.

If agreement holds in the weakest stratum, it is not an outlier artefact.

Usage:
    python Scripts/agreement_by_strength.py
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SEED = 964
BASE = Path(__file__).resolve().parents[1]
RES = BASE / "Results"

BABYLM = [
    "znhoughton_opt-babylm-125m-20eps-seed964",
    "znhoughton_opt-babylm-350m-20eps-seed964",
    "znhoughton_opt-babylm-1_3b-20eps-seed964",
]
SHORT = {s: s.split("babylm-")[1].split("-20eps")[0] for s in BABYLM}
N_BINS = 5


def load_novel(slug, condition):
    d = RES / slug / "cv_preds"
    pat = re.compile(rf"^{re.escape(condition)}_layer(\d+)_mean_pooled_pair_novel\.npz$")
    best, path = -1, None
    for f in d.iterdir():
        m = pat.match(f.name)
        if m and int(m.group(1)) > best:
            best, path = int(m.group(1)), f
    z = np.load(path, allow_pickle=True)
    key = np.char.add(np.char.add(z["word1"].astype(str), "\t"),
                      z["word2"].astype(str))
    return pd.DataFrame({"key": key, slug: z["y_true"].astype(float)}) \
             .drop_duplicates("key").set_index("key")


def main():
    rng = np.random.default_rng(SEED)
    rows, dist_rows = [], []

    for cond in ("default", "attn_zeroed"):
        M = pd.concat([load_novel(s, cond) for s in BABYLM], axis=1, join="inner")
        print(f"\n=== {cond}: {len(M)} aligned novel pairs ===", flush=True)

        print("  |y_true| distribution")
        for s in BABYLM:
            a = np.abs(M[s].to_numpy())
            dist_rows.append(dict(condition=cond, model=SHORT[s],
                                  median_abs_y=np.median(a),
                                  q25=np.quantile(a, .25), q75=np.quantile(a, .75),
                                  pct_below_0_5=float((a < 0.5).mean() * 100)))
            print(f"    {SHORT[s]:>5s}  median |y| = {np.median(a):.2f}"
                  f"   IQR {np.quantile(a,.25):.2f}-{np.quantile(a,.75):.2f}"
                  f"   {100*(a<0.5).mean():.1f}% below 0.5", flush=True)

        # overall directional agreement
        print("  directional agreement (chance = 0.500)")
        for a, b in itertools.combinations(BABYLM, 2):
            x, y = M[a].to_numpy(), M[b].to_numpy()
            sign_agree = float((np.sign(x) == np.sign(y)).mean())
            print(f"    {SHORT[a]} vs {SHORT[b]}: {sign_agree:.3f}", flush=True)
            rows.append(dict(condition=cond, model_a=SHORT[a], model_b=SHORT[b],
                             bin="all", bin_lo=np.nan, bin_hi=np.nan,
                             n=len(x), pearson=pearsonr(x, y)[0],
                             sign_agreement=sign_agree))

        # stratified by the THIRD model's |y_true|
        print(f"  stratified by the third model's |y_true| ({N_BINS} bins)")
        for a, b in itertools.combinations(BABYLM, 2):
            c = next(s for s in BABYLM if s not in (a, b))
            strength = np.abs(M[c].to_numpy())
            edges = np.quantile(strength, np.linspace(0, 1, N_BINS + 1))
            edges[-1] += 1e-9
            x, y = M[a].to_numpy(), M[b].to_numpy()
            for i in range(N_BINS):
                m = (strength >= edges[i]) & (strength < edges[i + 1])
                if m.sum() < 200:
                    continue
                rows.append(dict(condition=cond, model_a=SHORT[a],
                                 model_b=SHORT[b], bin=f"Q{i+1}",
                                 bin_lo=edges[i], bin_hi=edges[i + 1],
                                 n=int(m.sum()), pearson=pearsonr(x[m], y[m])[0],
                                 sign_agreement=float(
                                     (np.sign(x[m]) == np.sign(y[m])).mean())))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "agreement_by_strength.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(RES / "agreement_y_distribution.csv",
                                   index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 74)
    print("AGREEMENT BY PREFERENCE STRENGTH (BabyLM, novel pairs)")
    print("  bins = quintiles of the THIRD model's |y_true|, so the binning is")
    print("  independent of the two series being correlated")
    print("  Q1 = weakest preferences.  sign agreement chance = 0.500")
    print("=" * 74)
    summ = (df.groupby(["condition", "bin"], as_index=False)
              .agg(n=("n", "mean"), pearson=("pearson", "mean"),
                   sign_agreement=("sign_agreement", "mean")))
    print(summ.round(3).to_string(index=False))
    print(f"\nwrote {RES / 'agreement_by_strength.csv'}")


if __name__ == "__main__":
    main()
