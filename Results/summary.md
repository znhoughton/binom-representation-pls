# Binomial Ordering in Language Models: Methods and Results

## Code

```
Scripts/
  run_full_pipeline.py          # main entry point — runs all analyses
  feature_correlations.R        # frequentist correlation table
  bayesian_analysis.Rmd         # Bayesian analyses for paper
  lib/
    pls_utils.py                # shared NIPALS PLS, pearsonr, spearmanr
    pls_analysis.py             # core PLS fit + novel transfer
    cross_validation.py         # pair-level and word-level CV (--mode)
    frequency_analysis.py       # stratum, holdout, bootstrap (--mode)
    semantic_analysis.py        # direction + clustering (spaCy)
    compute_delta_features.py   # feature differences CSV
  preprocessing/
    run_preprocessing_pipeline.py  # orchestrates steps 1-5 below
    extract_corpus_binomials.py
    extract_wikipedia_binomials.py
    filter_strict_nouns_v3.py
    score_and_extract.py
    score_and_extract_novel.py
```

To reproduce all results from existing embeddings:
```
python Scripts/run_full_pipeline.py
```

## Models and Data

Three OPT models fine-tuned on BabyLM (150M tokens of child-directed speech, Wikipedia, and books) for 20 epochs, differing only in size: 125M, 350M, and 1.3B parameters.

Two datasets of noun binomials: **attested** pairs extracted from the BabyLM corpus (*N* = 212,946; canonical = more frequent order) and **novel** pairs drawn from Wikipedia (*N* = 53,685; neither order appears in BabyLM).

## Method

**Preference score.** For each pair (W1, W2):

> preference = log P("the W1 and W2") − log P("the W2 and W1")

Positive values indicate the model prefers W1-first; the magnitude captures preference strength.

**Diff-vectors.** For each pair we extract the mean last-layer hidden state for both orderings and take their difference:

> **d**(W1, W2) = **h̄**("the W1 and W2") − **h̄**("the W2 and W1")

Subtracting both orders cancels what is constant across orderings (each word's identity) and isolates the order-sensitive component of the representation.

**PLS (K = 15, NIPALS).** Partial Least Squares finds 15 weight vectors **w**₁…**w**₁₅ such that the component scores **t**ₖ = **X****w**ₖ have maximum covariance with the preference vector **y**. Out-of-sample projection uses the corrected weight matrix **W*** = **W**(**P**ᵀ**W**)⁻¹, where **P** are the X-loadings from each deflation step. A linear regression is then fit in component space: ŷ = β₀ + β₁C₁ + … + β₁₅C₁₅.

**Transfer design.** All parameters (**W***, scaling, regression coefficients) are estimated on the attested corpus pairs and frozen. The same parameters are applied without refitting to the novel pairs; *r*² between predicted and observed novel preferences measures out-of-sample generalization.

---

## Results

### Main Results

All metrics are *r*² (predicted vs. observed preference).

| Analysis | *N* | 125M | 350M | 1.3B |
|---|---|---:|---:|---:|
| Corpus PLS (in-sample) | 212,946 | .495 | .496 | .364 |
| → Novel, frozen | 53,685 | .126 | .144 | .138 |
| Pair-level CV (novel) | 53,685 | .284 | .303 | .325 |
| Word-level CV (novel) | 5,444 | .138 | .135 | .082 |
| Word-level CV (corpus) | 21,429 | .442 | .449 | .280 |

*Pair-level CV*: 10-fold CV on novel pairs; words may appear in both folds. *Word-level CV*: unique words split into folds; a pair is testable only when both words fall in the same held-out fold (~10% of pairs). Word-level CV estimates generalization to new words, not just new combinations.

**Model-size effects.** Pair-level CV increases with model size (.284 → .325), indicating larger models exploit more lexical information when words are seen in other binomials. Word-level CV decreases with model size in both the novel and corpus sets (.138 → .082; .442 → .280), indicating that larger models encode more pair-specific preferences — knowing a word's tendency in some binomials predicts its tendency in others less well as model size increases.

### Feature–Component Correlations

Pearson *r* between Δfeature (W1 − W2) and component score (OPT-125M):

| Feature | C1 | C2 | C4 | C6 | C8 | C9 |
|---|---:|---:|---:|---:|---:|---:|
| BabyLM log-freq | −.315 | −.404 | +.274 | +.117 | −.049 | +.280 |
| Word length | +.004 | +.231 | +.097 | −.307 | +.206 | −.095 |
| Syllable count | +.030 | +.255 | +.081 | −.322 | +.228 | −.073 |
| Animacy | +.094 | +.208 | −.004 | +.056 | +.068 | +.067 |

Direct feature–preference *r*: all ≤ .08. No single feature accounts for ordering. Components dissociate competing principles: C2 loads rarity + length (rare, long words first); C4 loads frequency-first; C6 loads short-word-first; C8 loads long-word-first.

### Frequency and Generalization

**Stratum analysis** (train on stratum, test on novel):

| Stratum | *n* corpus | 125M *r*² | 350M *r*² | 1.3B *r*² |
|---|---:|---:|---:|---:|
| freq = 1 | 168,938 | .129 | .145 | .139 |
| freq = 2–5 | 36,638 | .111 | .123 | .115 |
| freq = 6–20 | 5,954 | .062 | .069 | .049 |
| freq > 20 | 1,416 | .027 | .023 | .016 |

Pairs seen once generalize 4–9× better to novel pairs than pairs seen >20 times. A bootstrap equalizing sample size across strata confirms the decline is not a sample-size artifact (freq=1 − freq>20: 125M Δ.025*, 350M Δ.037*).

**Holdout analysis** (train on low-frequency complement, test on held-out stratum):

| Holdout | *n* test | 125M *r*² | 350M *r*² | 1.3B *r*² |
|---|---:|---:|---:|---:|
| freq > 1 | 44,008 | .511 | .506 | .375 |
| freq > 5 | 7,370 | .476 | .472 | .360 |
| freq > 20 | 1,416 | .439 | .407 | .315 |

High-frequency pairs are largely predictable from principles learned on low-frequency pairs (*r*² ≈ .31–.44), but this predictability decreases monotonically as the holdout threshold rises. High-frequency pairs are not opaque — they conform to the abstract ordering principles — but they carry additional pair-specific variance that those principles do not capture and that does not transfer outward to novel pairs.

---

## Summary

1. Models learn generalizable ordering preferences: in-sample *r*² ≈ .36–.50; novel frozen *r*² ≈ .13–.14.
2. With scale, preferences become more lexically specific: pair-level CV improves but word-level CV declines.
3. PLS dissociates multiple competing principles (rarity-first, frequency-first, short-first, long-first) as separate dimensions. No single surface feature predicts ordering.
4. Frequent co-occurrence entrenches pair-specific preferences that do not generalize to novel pairs, though high-frequency pairs remain largely predictable from principles learned on low-frequency data.
