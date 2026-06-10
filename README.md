# binom-corpus-pls

PLS analysis of binomial ordering preferences in OPT-BabyLM models.
Compares attested corpus binomials against novel Wikipedia binomials to test
whether learned ordering representations generalize to unseen word pairs.

Full methods and results: [`Results/methods_and_results.md`](Results/methods_and_results.md)

---

## Models

Three OPT models fine-tuned on the BabyLM 150M-token corpus for 20 epochs (seed 964):

| Slug | Size | Hidden dim |
|------|------|-----------|
| `znhoughton_opt-babylm-125m-20eps-seed964` | 125M | 768 |
| `znhoughton_opt-babylm-350m-20eps-seed964` | 350M | 1,024 |
| `znhoughton_opt-babylm-1_3b-20eps-seed964` | 1.3B | 2,048 |

---

## Directory Structure

```
binom-corpus-pls/
├── Data/
│   ├── corpus_binomials.csv          # 48,965 attested binomial pairs with frequencies
│   ├── babylm_vocab.txt              # BabyLM tokenizer vocabulary (423,810 types)
│   ├── babylm_word_freqs.csv         # Unigram frequencies from BabyLM corpus
│   ├── wikipedia_novel_binomials.csv # Novel Wikipedia binomial pairs
│   ├── embeddings/{slug}/
│   │   └── layer_last.npz            # Corpus diff_vecs + preferences
│   └── novel_embeddings/{slug}/
│       └── layer_last.npz            # Novel diff_vecs + preferences
│
├── Scripts/
│   │
│   │   ── Data pipeline ──
│   ├── extract_corpus_binomials.py   # Extract binomials from BabyLM corpus
│   ├── extract_wikipedia_binomials.py # Extract novel candidates from Wikipedia
│   ├── score_and_extract_all.py      # Score pairs + save embeddings (--mode corpus|novel)
│   ├── filter_strict_nouns_v3.py     # Final noun-filtering step (WordNet morphy)
│   │
│   │   ── Analysis ──
│   ├── run_full_pipeline.py          # Master runner for all steps per slug
│   ├── pls_analysis.py               # PLS (K=15) fit on corpus, transfer to novel
│   ├── novel_kfold_cv.py             # 10-fold pair-level CV within novel set
│   ├── novel_wordlevel_cv.py         # 10-fold word-level CV within novel set
│   ├── compute_delta_features.py     # Feature differences (freq, length, syllables, animacy)
│   ├── semantic_direction.py         # W1→W2 semantic direction per component
│   ├── semantic_clustering.py        # Semantic coherence of high/low scoring words per component
│   ├── feature_correlations.R        # Pearson/Spearman r: features × PLS components
│   ├── frequency_stratum_pls.py      # PLS fit separately per corpus frequency stratum
│   └── frequency_stratum_bootstrap.py # Bootstrap (B=500, N=1416) equalized stratum analysis
│
├── Results/
│   ├── methods_and_results.md        # Full methods and results writeup
│   └── {slug}/                       # Per-model results
│       ├── corpus_pls_scores.csv
│       ├── novel_pls_scores.csv
│       ├── novel_cv_{summary,predictions,fold_stats}.csv
│       ├── novel_wordcv_{summary,predictions,fold_stats}.csv
│       ├── delta_features.csv
│       ├── feature_correlations.csv
│       ├── semantic_direction.csv
│       ├── semantic_clustering.csv
│       ├── word_bias_scores.csv
│       ├── stratum_summary.csv
│       ├── stratum_{name}_{corpus,novel}.csv
│       ├── stratum_bootstrap_summary.csv
│       ├── stratum_bootstrap_diffs.csv
│       └── stratum_bootstrap_r2.npy
│
└── logs/
    ├── corpus_scoring.log
    ├── novel_scoring.log
    ├── analysis_pipeline.log
    └── wordcv_rerun.log
```

---

## Running the Analysis Pipeline

Assumes embeddings already exist in `Data/embeddings/{slug}/` and
`Data/novel_embeddings/{slug}/`.

```powershell
# Run all analysis steps for all three models
& "C:\Users\zacha\anaconda3\envs\PRenv\python.exe" Scripts\run_full_pipeline.py

# Run for a specific model
& "C:\Users\zacha\anaconda3\envs\PRenv\python.exe" Scripts\run_full_pipeline.py `
    --slugs znhoughton_opt-babylm-350m-20eps-seed964

# Run specific steps only
& "C:\Users\zacha\anaconda3\envs\PRenv\python.exe" Scripts\run_full_pipeline.py `
    --steps pls_analysis.py novel_kfold_cv.py

# Frequency stratum bootstrap (run separately)
& "C:\Users\zacha\anaconda3\envs\PRenv\python.exe" Scripts\frequency_stratum_bootstrap.py `
    --slug znhoughton_opt-babylm-125m-20eps-seed964 --B 500
```

Requires `PRenv` conda environment (CUDA-enabled PyTorch for GPU/cuBLAS).

---

## Key Results (see Results/methods_and_results.md for full details)

| Model | Corpus R² | Novel r² (frozen) | Transfer |
|-------|-----------|-------------------|----------|
| 125m  | .495      | .126              | 25.5%    |
| 350m  | .496      | .144              | 29.0%    |
| 1.3B  | .364      | .138              | 37.9%    |

Novel pair-level CV r²: .284 / .303 / .325 (125m / 350m / 1.3B)
Novel word-level CV r²: .138 / .135 / .082