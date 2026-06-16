import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

BASE  = Path(__file__).resolve().parents[1] / "Results"
SLUGS = {
    "125m": "znhoughton_opt-babylm-125m-20eps-seed964",
    "350m": "znhoughton_opt-babylm-350m-20eps-seed964",
    "1.3b": "znhoughton_opt-babylm-1_3b-20eps-seed964",
}
TAGS = {
    "default":    "layer_last",
    "last_token": "layer_last_last_token",
    "isolated":   "layer_last_isolated",
}

def pr2(path, obs, pred):
    if not path.exists(): return "—"
    df = pd.read_csv(path)[[obs, pred]].dropna()
    if len(df) < 5: return "—"
    r, _ = pearsonr(df[obs], df[pred])
    return round(r ** 2, 3)

def r2csv(path, col):
    if not path.exists(): return "—"
    df = pd.read_csv(path)
    if col not in df.columns: return "—"
    return round(float(df[col].iloc[0]), 3)

for cond, tag in TAGS.items():
    print(f"\n=== {cond} ===")
    for m, slug in SLUGS.items():
        d = BASE / slug / tag

        # --- real results ---
        pls_tr = pr2(d / "novel_pls_scores.csv",          "preference", "pls_pred")
        pls_pn = r2csv(d / "novel_cv_summary.csv",        "cv_r2")
        pls_wn = r2csv(d / "novel_wordcv_summary.csv",    "cv_r2")
        pls_wc = r2csv(d / "corpus_wordcv_summary.csv",   "cv_r2")
        pls_ws = r2csv(d / "pls_word_strict.csv",         "r2")
        md_tr  = r2csv(d / "mlp_diff_transfer.csv",       "test_r2")
        md_pn  = r2csv(d / "mlp_diff_pair_novel.csv",     "mean_test_r2")
        md_wn  = r2csv(d / "mlp_diff_word_novel.csv",     "mean_test_r2")
        md_wc  = r2csv(d / "mlp_diff_word_corpus.csv",    "mean_test_r2")
        md_ws  = r2csv(d / "mlp_diff_word_strict.csv",    "test_r2")
        mc_tr  = r2csv(d / "mlp_concat_transfer.csv",     "test_r2")
        mc_pn  = r2csv(d / "mlp_concat_pair_novel.csv",   "mean_test_r2")
        mc_wn  = r2csv(d / "mlp_concat_word_novel.csv",   "mean_test_r2")
        mc_wc  = r2csv(d / "mlp_concat_word_corpus.csv",  "mean_test_r2")
        mc_ws  = r2csv(d / "mlp_concat_word_strict.csv",  "test_r2")

        # --- control results ---
        c_pls_tr = pr2(d / "control_novel_pls_scores.csv",       "preference", "pls_pred")
        c_pls_pn = r2csv(d / "control_novel_cv_summary.csv",     "cv_r2")
        c_pls_wn = r2csv(d / "control_novel_wordcv_summary.csv", "cv_r2")
        c_pls_wc = r2csv(d / "control_corpus_wordcv_summary.csv","cv_r2")
        c_pls_ws = r2csv(d / "control_pls_word_strict.csv",      "r2")
        c_md_tr  = r2csv(d / "control_mlp_diff_transfer.csv",    "test_r2")
        c_md_pn  = r2csv(d / "control_mlp_diff_pair_novel.csv",  "mean_test_r2")
        c_md_wn  = r2csv(d / "control_mlp_diff_word_novel.csv",  "mean_test_r2")
        c_md_wc  = r2csv(d / "control_mlp_diff_word_corpus.csv", "mean_test_r2")
        c_md_ws  = r2csv(d / "control_mlp_diff_word_strict.csv", "test_r2")
        c_mc_tr  = r2csv(d / "control_mlp_concat_transfer.csv",  "test_r2")
        c_mc_pn  = r2csv(d / "control_mlp_concat_pair_novel.csv","mean_test_r2")
        c_mc_wn  = r2csv(d / "control_mlp_concat_word_novel.csv","mean_test_r2")
        c_mc_wc  = r2csv(d / "control_mlp_concat_word_corpus.csv","mean_test_r2")
        c_mc_ws  = r2csv(d / "control_mlp_concat_word_strict.csv","test_r2")

        print(f"{m}|PLS      |{pls_tr}|{pls_pn}|{pls_wn}|{pls_wc}|{pls_ws}")
        print(f"{m}|PLS-ctrl |{c_pls_tr}|{c_pls_pn}|{c_pls_wn}|{c_pls_wc}|{c_pls_ws}")
        print(f"{m}|Mdif     |{md_tr}|{md_pn}|{md_wn}|{md_wc}|{md_ws}")
        print(f"{m}|Mdif-ctrl|{c_md_tr}|{c_md_pn}|{c_md_wn}|{c_md_wc}|{c_md_ws}")
        print(f"{m}|Mcon     |{mc_tr}|{mc_pn}|{mc_wn}|{mc_wc}|{mc_ws}")
        print(f"{m}|Mcon-ctrl|{c_mc_tr}|{c_mc_pn}|{c_mc_wn}|{c_mc_wc}|{c_mc_ws}")
