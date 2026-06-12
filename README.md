# binom-corpus-pls

Probing OPT-BabyLM models for binomial ordering preferences.

Attested binomials from the BabyLM training corpus (~49k pairs) are used to
train probes (PLS and MLP); predictions are evaluated on novel binomials from
Wikipedia (~340k pairs) across several generalization conditions. Three
embedding extraction conditions and a Hewitt & Liang (2019) label-shuffle
control are supported throughout.

Pilot results: [`pilot_results.md`](pilot_results.md)

---

## Models

Three OPT models fine-tuned on the BabyLM 150M-token corpus (20 epochs, seed 964):

| Key | HuggingFace ID | Size | Hidden dim |
|-----|---------------|------|-----------|
| `125m` | `znhoughton/opt-babylm-125m-20eps-seed964` | 125M | 768 |
| `350m` | `znhoughton/opt-babylm-350m-20eps-seed964` | 350M | 1,024 |
| `1.3b` | `znhoughton/opt-babylm-1.3b-20eps-seed964` | 1.3B | 2,048 |

---

## Directory Structure

```
binom-corpus-pls/
│
├── Data/
│   ├── corpus_binomials.csv               # ~49k attested binomials with frequencies
│   ├── wikipedia_novel_binomials.csv      # ~340k novel Wikipedia binomials
│   ├── babylm_vocab.txt                   # BabyLM tokenizer vocab
│   ├── babylm_word_freqs.csv              # Unigram frequencies
│   ├── embeddings/{slug}/                 # Corpus embeddings (binomial context, span mean-pool)
│   ├── novel_embeddings/{slug}/           # Novel embeddings (same condition)
│   ├── embeddings_last/{slug}/            # Corpus embeddings (binomial context, last token)
│   ├── novel_embeddings_last/{slug}/      # Novel embeddings (same condition)
│   ├── embeddings_isolated/{slug}/        # Corpus embeddings (isolated "the {word}" context)
│   └── novel_embeddings_isolated/{slug}/  # Novel embeddings (same condition)
│       └── layer_{last,second_to_last}.npz
│           # Contains: word1, word2, preference, vec_alpha, vec_non_alpha, diff_vecs
│
├── Scripts/
│   ├── extract_embeddings.py              # Unified embedding extraction (all conditions)
│   ├── run_pipeline.py                    # Main entry point: extract + analyze
│   │
│   ├── lib/
│   │   ├── pls_analysis.py                # PLS (K=15): corpus fit → novel transfer
│   │   ├── cross_validation.py            # 10-fold CV (pair-level and word-level)
│   │   ├── mlp_comparison.py              # MLP probe (diff and concat inputs)
│   │   ├── permutation_test.py            # Permutation test for transfer significance
│   │   ├── compute_delta_features.py      # Frequency, length, syllable, animacy features
│   │   ├── frequency_analysis.py          # Per-stratum and holdout frequency analyses
│   │   ├── semantic_analysis.py           # Semantic direction + clustering per PLS component
│   │   └── pls_utils.py                   # Shared: NIPALS PLS, Pearson/Spearman, scaling
│   │
│   └── preprocessing/
│       ├── extract_corpus_binomials.py    # Extract binomials from BabyLM corpus
│       ├── extract_wikipedia_binomials.py # Extract novel candidates from Wikipedia
│       ├── merge_extraction_shards.py     # Merge parallel extraction shards
│       ├── run_parallel_extraction.py     # Orchestrate parallel binomial extraction
│       └── filter_strict_nouns_v3.py      # Post-hoc noun filter (WordNet morphy)
│
├── Results/
│   └── {slug}/
│       └── layer_{last,second_to_last}[_{condition}]/
│           # condition tag: empty = default, _isolated, _last_token
│           # Real results: corpus_pls_scores.csv, novel_pls_scores.csv,
│           #   novel_cv_*.csv, novel_wordcv_*.csv, corpus_wordcv_*.csv,
│           #   mlp_{diff,concat}_{transfer,pair_novel,word_novel,word_strict}.csv, ...
│           # Control results: same files prefixed with control_
│
├── Plots/
│   └── pred_vs_obs_{pls,mlp_concat}.png, pls_vs_mlp_pair_novel.png
│
└── logs/
```

---

## Embedding Extraction Conditions

Three conditions are supported, each producing identically structured `.npz` files:

| Condition | `--context` | `--extract` | What it captures |
|-----------|------------|------------|-----------------|
| `default` | `binomial` | `word` | Mean-pooled span [w1, and, w2] in binomial sentence |
| `last_token` | `binomial` | `last` | Final token of span in binomial sentence |
| `isolated` | `isolated` | `word` | Each word separately in "the {word}" context |

Each `.npz` contains: `word1`, `word2`, `preference` (log-prob ratio from binomial context),
`vec_alpha`, `vec_non_alpha`, `diff_vecs` (= `vec_alpha - vec_non_alpha`).

---

## Evaluation Conditions

| Condition | Train | Test | What it tests |
|-----------|-------|------|--------------|
| Transfer | Corpus (~49k) | All novel (340k) | Frozen generalization across datasets |
| Pair-novel CV | Novel (10-fold) | Novel held-out fold | Within-novel generalization, pairs as units |
| Word-novel CV | Novel (10-fold, word-split) | Novel, both words held out | Generalization to unseen word pairs |
| Word-strict | Corpus (~49k) | Novel, neither word in corpus | Transfer to words with no ordering signal |

---

## Running the Pipeline

Requires the `PRenv` conda environment (CUDA-enabled PyTorch, RTX 3060 Ti).

```powershell
$py = "C:\Users\zacha\anaconda3\envs\PRenv\python.exe"

# Full default pipeline — extract embeddings + run all analysis (all models, both layers):
& $py Scripts\run_pipeline.py

# Run a specific subset:
& $py Scripts\run_pipeline.py --models 350m --layers last --steps pls cv_pair

# Extract new embedding conditions + analyze:
& $py Scripts\run_pipeline.py --conditions last_token isolated

# Run Hewitt & Liang control for default condition (embeddings already exist):
& $py Scripts\run_pipeline.py --conditions default --skip-extraction --control-only

# Everything — all conditions, real + control:
& $py Scripts\run_pipeline.py --conditions default last_token isolated --run-control
```

### `run_pipeline.py` arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--models` | all three | `125m`, `350m`, `1.3b` |
| `--layers` | `last second_to_last` | Layer(s) to extract/analyze |
| `--conditions` | `default` | `default`, `last_token`, `isolated` |
| `--steps` | all | Subset of step names (see `--help`) |
| `--skip-extraction` | off | Skip embedding extraction |
| `--skip-analysis` | off | Skip analysis |
| `--run-control` | off | Run H&L control pass after each analysis step |
| `--control-only` | off | Run control pass only (skip real analysis) |
| `--gpu` | 0 | GPU index |
| `--force` | off | Re-extract even if `.npz` already exists |

### Running extraction directly

```powershell
# Default condition:
& $py Scripts\extract_embeddings.py `
    --model znhoughton/opt-babylm-350m-20eps-seed964 `
    --data corpus --layer last `
    --out Data/embeddings/znhoughton_opt-babylm-350m-20eps-seed964

# Isolated context:
& $py Scripts\extract_embeddings.py `
    --model znhoughton/opt-babylm-350m-20eps-seed964 `
    --data corpus --layer last --context isolated `
    --out Data/embeddings_isolated/znhoughton_opt-babylm-350m-20eps-seed964
```

### Running analysis scripts directly

All three core analysis scripts share the same new arguments:

```powershell
& $py Scripts\lib\pls_analysis.py `
    --slug znhoughton_opt-babylm-350m-20eps-seed964 --layer last `
    --embed-dir-corpus Data/embeddings_isolated/znhoughton_opt-babylm-350m-20eps-seed964 `
    --embed-dir-novel  Data/novel_embeddings_isolated/znhoughton_opt-babylm-350m-20eps-seed964 `
    --out-dir          Results/znhoughton_opt-babylm-350m-20eps-seed964/layer_last_isolated `
    --control   # add for Hewitt & Liang label-shuffle control
```

---

## Probes

**PLS** (`lib/pls_analysis.py`, `lib/cross_validation.py`)
- NIPALS PLS, K=15 components, fit on `diff_vecs` (= `vec_alpha - vec_non_alpha`)
- Corpus fit frozen; components applied to novel via the same W\*, β

**MLP** (`lib/mlp_comparison.py`)
- Architecture: `Linear(dim, 15) → ReLU → Linear(15, 1)` (hidden dim matches PLS K)
- Two input modes: `diff` (p-dim) and `concat` (2p-dim with antisymmetric augmentation)
- L2 weight decay (1e-4), early stopping on validation loss (patience=20)
- Seed: 964 throughout

**Hewitt & Liang control** (`--control` flag)
- Preference labels globally shuffled (seed 964 for corpus, 965 for novel) before fitting
- Same CV/transfer splits applied to shuffled labels
- Output files prefixed `control_`; selectivity = real r² − control r²

---

## Key Results (last layer, default condition)

| Model | PLS transfer r² | MLP-concat pair-CV r² | MLP-concat word-CV r² |
|-------|----------------|-----------------------|-----------------------|
| 125m  | 0.095          | 0.321                 | 0.186                 |
| 350m  | 0.104          | 0.348                 | 0.217                 |
| 1.3b  | 0.060          | 0.238                 | 0.127                 |

MLP-diff ≈ PLS; MLP-concat substantially outperforms both, indicating nonlinear
structure in the joint embedding space. Word-strict r² (0.02–0.05) is low across
all models. Full tables and plots: [`pilot_results.md`](pilot_results.md).
