# Pilot Results: BabyLM Binomial Ordering

We probe three BabyLM causal language models (125m, 350m, and 1.3b parameters) to assess whether their contextual embeddings encode binomial ordering preferences. For each noun pair (e.g., *bread* and *butter*), the target is the log-preference score log P(*bread and butter*) − log P(*butter and bread*), derived from the model's own token probabilities in a binomial context sentence. Three probe types and three embedding conditions are crossed and evaluated across five generalization splits. All embeddings come from the last hidden layer (second-to-last is within ±0.002 throughout).

---

## Probes

Three probes were fit to predict ordering preference from pairs of word embeddings.

**PLS (Partial Least Squares).** For each pair, the input is the difference vector `vec_alpha − vec_non_alpha`, where `vec_alpha` is the embedding extracted from the alpha-order sentence (e.g., "bread and butter") and `vec_non_alpha` from the reverse. PLS finds 15 latent components (NIPALS algorithm) that maximize covariance between this difference vector and the scalar preference score, then regresses preference on those components. Because the input is already the antisymmetric difference, PLS is effectively a linear model on the contrast between the two orderings.

**MLP-diff.** Takes the same difference vector as PLS, but applies a nonlinear two-layer network: Linear(dim, 15) → ReLU → Linear(15, 1), with L2 weight decay (1e-4) and early stopping (patience = 20). Comparing MLP-diff to PLS isolates the contribution of nonlinearity while holding the input representation constant.

**MLP-concat.** Takes the full concatenation of both vectors, `[vec_alpha; vec_non_alpha]`, as input (2× dimensionality). Trained with antisymmetric augmentation: for each pair (A, B) with preference p, the reversed pair (B, A) with preference −p is added to the training set, enforcing that the learned function satisfies f(A, B) = −f(B, A). Comparing MLP-concat to MLP-diff tests whether access to the full joint representation, rather than just the difference, carries additional ordering signal beyond what a linear contrast captures.

---

## Embedding conditions

Three conditions vary how word embeddings are extracted from the model for each binomial pair.

| Condition | Context | Extraction | What it captures |
|---|---|---|---|
| **Default** | Binomial sentence (e.g., "dogs and cats") | Mean-pool over span tokens [w1, and, w2] | Both words represented simultaneously in binomial ordering context |
| **Last-token** | Binomial sentence | Final subtoken of each word's span | Boundary representation of each word in binomial context |
| **Isolated** | "the {word}" (each word in isolation) | Mean-pool over word tokens | Out-of-context word representation, no binomial ordering frame |

The preference DV is always derived from the binomial context regardless of condition; only the embeddings used as probe input change. For the last-token condition, `vec_alpha` is the hidden state at the final subtoken of the second word in the alpha-order sentence (e.g., "butter" in "bread and **butter**"), and `vec_non_alpha` is the final subtoken of the second word in the reverse sentence (e.g., "bread" in "butter and **bread**") — in both cases the model processes the full sentence up to and including that token. The isolated condition tests whether context-free word representations carry ordering signal without any sentence-level framing.

---

## Evaluation splits

Five splits test generalization at increasing levels of strictness.

| Split | Train | Test | Notes |
|---|---|---|---|
| **Transfer** | Corpus (~49k) | All novel (340k) | Probe frozen after corpus fit |
| **Pair-CV** | Novel (10-fold) | Novel held-out fold | Pairs as held-out units |
| **Word-CV (novel)** | Novel (10-fold, word-split) | Novel, both words unseen in train | Generalization to unseen word pairs |
| **Corpus word-CV** | Corpus (10-fold, word-split) | Corpus, both words unseen in train | Within-corpus word generalization |
| **Word-strict** | Corpus (~49k) | Novel pairs, neither word in corpus | Transfer to words with zero corpus ordering signal |

"Word-strict" words are still in BabyLM's pretraining data; what's missing is any binomial ordering context for those words during probe training.

"—" indicates the metric wasn't computed.

---

## Results

### Default (binomial sentence, span mean-pooled)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.095 | 0.083 | 0.120 | 0.104 | 0.095 | 0.121 | 0.060 | 0.059 | 0.065 |
| Pair-CV | 0.200 | 0.230 | 0.321 | 0.225 | 0.263 | 0.348 | 0.171 | 0.182 | 0.238 |
| Word-CV (novel) | 0.152 | 0.138 | 0.186 | 0.170 | 0.163 | 0.217 | 0.117 | 0.107 | 0.127 |
| Corpus word-CV | 0.172 | 0.172 | 0.183 | 0.139 | 0.141 | 0.176 | 0.129 | 0.144 | 0.142 |
| Word-strict | 0.034 | 0.027 | 0.045 | 0.043 | 0.037 | 0.049 | 0.024 | 0.021 | 0.023 |

MLP-concat substantially outperforms PLS on pair-CV (r² 0.238–0.348 vs. 0.171–0.225), indicating nonlinear structure in the joint embedding space that a linear projection on the difference vector doesn't capture. MLP-diff tracks PLS closely throughout, confirming that PLS recovers essentially the same information as a nonlinear probe on the same difference input. Word-strict performance is low across all probes (r² 0.021–0.049), reflecting that the learned preference mapping doesn't generalize well to word pairs absent from binomial ordering training data. The 350m model consistently leads; 1.3b underperforms relative to its size, suggesting the larger BabyLM model may be less sensitive to the surface features driving ordering preferences.

### Last-token (binomial sentence, last subtoken of span)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.084 | 0.077 | 0.101 | 0.078 | 0.072 | 0.098 | 0.061 | 0.056 | 0.064 |
| Pair-CV | 0.157 | 0.162 | 0.272 | 0.168 | 0.176 | 0.290 | 0.144 | 0.146 | 0.207 |
| Word-CV (novel) | 0.118 | 0.112 | 0.147 | 0.132 | 0.118 | 0.146 | 0.098 | 0.091 | 0.114 |
| Corpus word-CV | 0.196 | 0.187 | 0.200 | 0.168 | 0.164 | 0.184 | 0.152 | 0.157 | 0.169 |
| Word-strict | 0.036 | 0.031 | 0.046 | 0.044 | 0.038 | 0.052 | 0.029 | 0.025 | 0.028 |

Last-token performance is modestly lower than default on most splits, with the same probe hierarchy intact. One exception is corpus word-CV, which is higher under last-token than default for all models and probes (e.g., 125m MLP-concat 0.200 vs. 0.183), suggesting the final subtoken position carries particularly stable within-corpus ordering signal.

### Isolated ("the {word}", word tokens mean-pooled)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.052 | 0.048 | 0.060 | 0.046 | 0.044 | 0.047 | 0.041 | 0.038 | 0.043 |
| Pair-CV | 0.115 | 0.128 | 0.217 | 0.114 | 0.130 | 0.226 | 0.113 | 0.122 | 0.178 |
| Word-CV (novel) | 0.064 | 0.058 | 0.069 | 0.064 | 0.059 | 0.064 | 0.049 | 0.048 | 0.050 |
| Corpus word-CV | 0.126 | 0.136 | 0.140 | 0.099 | 0.111 | 0.114 | 0.093 | 0.104 | 0.112 |
| Word-strict | 0.014 | 0.011 | 0.015 | 0.014 | 0.014 | 0.016 | 0.005 | 0.004 | 0.007 |

Isolated embeddings carry substantially less ordering signal across all probes and splits. MLP-concat pair-CV (0.178–0.226) remains the strongest isolated probe but falls well below the default condition (0.238–0.348), confirming that the binomial ordering frame itself contributes information beyond what's encoded in individual word representations.

---

## Shuffled-label control

All analyses were also run with shuffled preference labels (same embeddings, random target assignment). Control r² was ≤ 0.005 across all conditions, probes, and eval splits.

### Default (shuffled-label control)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Pair-CV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Word-CV (novel) | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| Corpus word-CV | 0.000 | 0.001 | 0.002 | 0.001 | 0.002 | 0.001 | 0.000 | 0.001 | 0.005 |
| Word-strict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Last-token (shuffled-label control)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Pair-CV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Word-CV (novel) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
| Corpus word-CV | 0.000 | 0.001 | 0.004 | 0.000 | 0.002 | 0.002 | 0.000 | 0.001 | 0.003 |
| Word-strict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Isolated (shuffled-label control)

| Eval split | 125m PLS | 125m MLP-diff | 125m MLP-concat | 350m PLS | 350m MLP-diff | 350m MLP-concat | 1.3b PLS | 1.3b MLP-diff | 1.3b MLP-concat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Pair-CV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Word-CV (novel) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Corpus word-CV | 0.000 | 0.002 | 0.002 | 0.000 | 0.001 | 0.002 | 0.000 | 0.001 | 0.002 |
| Word-strict | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Predicted vs observed: by eval split

MLP-concat, default embedding, all models. Columns: transfer → pair-CV → word-CV (novel) → word-strict → corpus word-CV.

![Predicted vs observed by eval split](Plots/pred_vs_obs_by_split_last.png)

---

## Predicted vs observed: by embedding condition

MLP-concat, pair-CV, all models. Columns: default → last-token → isolated.

![Predicted vs observed by embedding condition](Plots/pred_vs_obs_by_condition_last.png)

---

## Predicted vs observed: by probe type

Default embedding, all models. Columns: PLS (transfer) → PLS (pair-CV) → MLP-diff (pair-CV) → MLP-concat (pair-CV) → MLP-concat (word-CV novel).

![Predicted vs observed by probe type](Plots/pred_vs_obs_by_probe_last.png)
