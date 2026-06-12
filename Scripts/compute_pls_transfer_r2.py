import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SLUGS = [
    ("125m", "znhoughton_opt-babylm-125m-20eps-seed964"),
    ("350m", "znhoughton_opt-babylm-350m-20eps-seed964"),
    ("1.3b", "znhoughton_opt-babylm-1_3b-20eps-seed964"),
]
LAYERS = ["last", "second_to_last"]
COMPS  = [f"C{i}" for i in range(1, 16)]

rows = []
for label, slug in SLUGS:
    for layer in LAYERS:
        d = BASE / "Results" / slug / f"layer_{layer}"
        corpus = pd.read_csv(d / "corpus_pls_scores.csv")
        novel  = pd.read_csv(d / "novel_pls_scores.csv")

        X_corp = corpus[COMPS].values
        y_corp = corpus["preference"].values
        X_aug  = np.c_[np.ones(len(X_corp)), X_corp]
        beta   = np.linalg.lstsq(X_aug, y_corp, rcond=None)[0]

        X_nov = np.c_[np.ones(len(novel)), novel[COMPS].values]
        y_nov = novel["preference"].values
        r2_transfer = float(np.corrcoef(y_nov, X_nov @ beta)[0, 1] ** 2)

        cv_pair = pd.read_csv(d / "novel_cv_summary.csv")
        cv_word = pd.read_csv(d / "novel_wordcv_summary.csv")
        r2_pair = float(cv_pair["cv_r2"].iloc[0])
        r2_word = float(cv_word["cv_r2"].iloc[0])

        p = d / "corpus_wordcv_summary.csv"
        r2_corp_word = float(pd.read_csv(p)["cv_r2"].iloc[0]) if p.exists() else None

        rows.append(dict(model=label, layer=layer,
                         transfer=round(r2_transfer, 4),
                         pair_novel=round(r2_pair, 4),
                         word_novel=round(r2_word, 4),
                         corpus_word=round(r2_corp_word, 4) if r2_corp_word is not None else "N/A"))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
