# Binomial Ordering in Language Models: Methods and Results

## Code

```
Scripts/
  run_full_pipeline.py          # main entry point — runs all analyses
  feature_correlations.R        # frequentist correlation table
  bayesian_analysis.Rmd         # Bayesian analyses for paper
  run_bayesian_all.R            # batch runner for bayesian_analysis.Rmd
  lib/
    pls_utils.py                # shared NIPALS PLS, pearsonr, spearmanr
    pls_analysis.py             # core PLS fit + novel transfer
    cross_validation.py         # pair-level and word-level CV (--mode)
    frequency_analysis.py       # stratum, holdout, bootstrap (--mode)
    semantic_analysis.py        # direction + clustering (spaCy)
    compute_delta_features.py   # feature differences CSV
    permutation_test.py         # null distribution for transfer r²
    mlp_comparison.py           # MLP neural baseline (--input diff/concat, --split)
  preprocessing/
    extract_corpus_binomials.py    # BabyLM corpus extraction (benepar constituency)
    extract_wikipedia_binomials.py # Wikipedia novel extraction (benepar constituency)
    score_and_extract_all.py       # score + save embeddings (--mode corpus|novel)
    extract_individual_vecs.py     # extract vec_alpha/vec_non_alpha for MLP concat mode
```

To reproduce all results from existing embeddings:
```
python Scripts/run_full_pipeline.py
```

## Models and Data

Three OPT models fine-tuned on BabyLM (150M tokens of child-directed speech, Wikipedia, and books) for 20 epochs, differing only in size: 125M, 350M, and 1.3B parameters.

Two datasets of binomials covering nouns, verbs, adjectives, and adverbs: **attested** pairs extracted from the BabyLM corpus (canonical = more frequent order) and **novel** pairs drawn from Wikipedia (neither order appears in BabyLM).

## Extraction Method

Binomials are extracted using benepar (Berkeley Neural Parser) constituency parsing. For each sentence, a pair "W1 and W2" is retained only when:

1. Both words are lowercase `[a-z]{2,}` and open-class (NOUN, VERB, ADJ, or ADV; no proper nouns)
2. Both words and the conjunction "and" form a **tight constituent** — the lowest common ancestor of W1 and W2 in the parse tree is a phrase-level node (NP, VP, ADJP, etc.) that spans exactly those three tokens and no others

This strict constituency check eliminates cases like "France and **Great** Britain" (where "Britain" trails "Great" inside the same constituent) and ensures both words are genuinely coordinated as equals.

## Method

**Preference score.** For each pair (W1, W2), both orderings are scored in the natural sentence context in which the pair was observed. The sentence prefix (all tokens before the span) is held constant; only the span [W1, and, W2] vs [W2, and, W1] differs:

> preference = Σ log P(tₖ | prefix) for tₖ ∈ {W1, and, W2} − Σ log P(tₖ | prefix) for tₖ ∈ {W2, and, W1}

Since the prefix is identical across orderings, it cancels in the difference, making this equivalent to full-sentence log-probability differencing. Positive values indicate the model prefers W1-first.

**Diff-vectors.** For each pair we extract the mean last-layer hidden state over the span tokens {W1, and, W2} conditioned on the same sentence prefix, then subtract the non-alpha ordering:

> **d**(W1, W2) = **h̄**([W1, and, W2] | prefix) − **h̄**([W2, and, W1] | prefix)

Subtracting both orders cancels what is constant across orderings (each word's identity) and isolates the order-sensitive component of the representation.

**PLS (K = 15, NIPALS).** Partial Least Squares finds 15 weight vectors **w**₁…**w**₁₅ such that the component scores **t**ₖ = **X****w**ₖ have maximum covariance with the preference vector **y**. Out-of-sample projection uses the corrected weight matrix **W*** = **W**(**P**ᵀ**W**)⁻¹, where **P** are the X-loadings from each deflation step. A linear regression is then fit in component space: ŷ = β₀ + β₁C₁ + … + β₁₅C₁₅.

**MLP baseline.** A two-layer MLP (hidden dim = 15, matching PLS K) is trained to predict ordering preference from the diff-vector (or concatenated individual vectors). Three train/test splits are tested: corpus→novel transfer, word-level split within novel, and strict transfer (pairs where neither word appears in corpus).

**Transfer design.** All parameters (**W***, scaling, regression coefficients) are estimated on the attested corpus pairs and frozen. The same parameters are applied without refitting to the novel pairs; *r*² between predicted and observed novel preferences measures out-of-sample generalization.

---

## Results

*(Results will be updated after re-extraction with the new constituency-based pipeline.)*

### Main Results (PLS)

All metrics are *r*² (predicted vs. observed preference).

| Analysis | 125M | 350M | 1.3B |
|---|---:|---:|---:|
| Corpus PLS (in-sample) | — | — | — |
| → Novel, frozen | — | — | — |
| Pair-level CV (novel) | — | — | — |
| Word-level CV (novel) | — | — | — |
| Word-level CV (corpus) | — | — | — |

### Feature–Component Correlations

*(To be updated after re-analysis.)*

### Bayesian Mixed-Effects Results

*(To be updated after re-analysis.)*
