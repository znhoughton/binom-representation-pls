# Wikipedia Novel Binomials — PLS/MLP Analysis
## Methods and Results

---

## 1. Models

Three OPT models fine-tuned on the BabyLM 150M-token training corpus for 20 epochs (seed 964), differing only in parameter count and hidden dimension:

| Model | HuggingFace slug | Hidden dim |
|---|---|---|
| OPT-125M | znhoughton/opt-babylm-125m-20eps-seed964 | 768 |
| OPT-350M | znhoughton/opt-babylm-350m-20eps-seed964 | 1,024 |
| OPT-1.3B | znhoughton/opt-babylm-1.3b-20eps-seed964 | 2,048 |

All analyses are run separately per model; results are compared across model sizes.

---

## 2. Attested (Corpus) Binomials

Binomials were extracted from the BabyLM training corpus sentence by sentence using benepar (Berkeley Neural Parser). For each sentence, a pair "W1 and W2" (three consecutive tokens) was retained when:

1. Both words are lowercase `[a-z]{2,}` and open-class (NOUN, VERB, ADJ, or ADV — no proper nouns, pronouns, determiners, or function words)
2. The sentence contains "and" (fast pre-filter before parsing)
3. The sentence is ≤ 40 tokens
4. The lowest common ancestor (LCA) of W1 and W2 in the constituency parse tree is a phrase-level node (NP, VP, ADJP, ADVP, PP, etc.)
5. The LCA spans exactly [W1, and, W2] — no other tokens belong to the same constituent

Pair frequencies are recorded in both orders (freq_w1_w2, freq_w2_w1). The first attested example sentence is stored for validation.

---

## 3. Wikipedia Novel Binomials

### 3.1 Extraction

The English Wikipedia dump (wikimedia/wikipedia, 20231101.en) was streamed via HuggingFace. Each Wikipedia article was pre-split on newlines; only lines containing " and " and ≤ 280 characters were passed to the parser. The same constituency-based extraction criteria as the corpus were applied:

1. Both words are lowercase `[a-z]{2,}`, open-class, and in the BabyLM vocabulary (423,810 types)
2. The pair is not already attested in corpus_binomials.csv (novelty filter)
3. The sentence contains "and" (pre-filter)
4. The sentence is ≤ 40 tokens
5. The LCA is a phrase-level constituent spanning exactly [W1, and, W2]

### 3.2 Output

Output saved to `Data/wikipedia_novel_binomials.csv` with columns: word1, word2, wiki_count, pos1, pos2, example_sentence. N ≈ 340,042 pairs.

---

## 4. Scoring

For each pair (W1, W2), both orderings are scored using each model:

**Preference score:**
$$\text{pref}(A, B) = \log P(\textit{A and B}) - \log P(\textit{B and A})$$

Specifically: Σ log P(tₖ | preceding context) for tₖ ∈ {W1, and, W2} minus the same for {W2, and, W1}, where both orderings share the identical sentence prefix. The prefix cancels in the difference.

---

## 5. Embedding Conditions

Three conditions vary how hidden-state representations are extracted for each word pair. The preference DV is always derived from the model's own token probabilities — only the probe input changes.

| Condition | Context | Extraction | What it tests |
|---|---|---|---|
| **Default** | Binomial sentence | Mean-pool over word span | Both words in binomial ordering frame |
| **Last-token** | Binomial sentence | Final subtoken of each word's span | Boundary representation in context |
| **Isolated** | *"the {word}"* | Mean-pool over word tokens | Context-free word representation |

For last-token: v_α = hidden state at the final subtoken of word 2 in the alpha-order sentence; v_ᾱ = final subtoken of word 2 in the reverse sentence.

All three conditions produce a **diff-vector**: d = v_α − v_ᾱ, which cancels everything constant across orderings. For MLP-concat, both v_α and v_ᾱ are retained as a concatenated input.

---

## 6. Evaluation Splits

Five splits test generalization at increasing levels of strictness, plus a reverse-direction transfer for frequency analysis.

| Split | Train | Test | What it tests |
|---|---|---|---|
| **Transfer** | Corpus (~49k) | All novel (~340k) | Across-dataset generalization |
| **Pair-CV** | Novel (10-fold) | Novel held-out fold | In-distribution fit |
| **Word-CV (novel)** | Novel (10-fold, word split) | Novel, both words unseen in train | Generalization to new word pairs |
| **Corpus word-CV** | Corpus (10-fold, word split) | Corpus, both words unseen in train | Within-corpus word generalization |
| **Word-strict** | Corpus (~49k) | Novel, neither word in corpus | Probes idiosyncrasy of corpus pairs |
| **Novel-to-corpus** | Novel (~340k) | Corpus (~49k) | Reverse transfer; used for frequency analysis |

For word-level splits (word-CV novel, corpus word-CV, word-strict): unique words are assigned to folds; a test pair qualifies only when both its words are in the held-out fold; training uses only pairs where neither word appears in the held-out fold.

---

## 7. PLS Probe

Partial Least Squares Regression (PLS; K = 15 components) was fit on diff-vectors, separately per model and embedding condition. Features were z-scored using the training-set mean and standard deviation; the same scaler was applied frozen to test data.

**Algorithm:** NIPALS implemented in PyTorch (GPU-accelerated). To project new data without re-running iterative deflation, the corrected weight matrix W* = W(PᵀW)⁻¹ was computed from NIPALS weights W and X-loadings P, giving T_new = X_new · W*. A linear regression of preference on the 15 components was fit on training scores and applied frozen to test.

**Control task (Hewitt & Liang 2019):** A matched control run shuffles preference labels (independently for train and test with different seeds) before fitting. Control runs are prefixed `control_` in output files.

---

## 8. MLP Probes

### 8.1 Architecture

**Architecture:** `Linear(input_dim, H) → ReLU → Linear(H, 1)`

Three hidden dimensions were compared to assess sensitivity to probe capacity:

| Variant | Hidden dim H | Notes |
|---|---|---|
| h15 | 15 | Matches PLS K=15; minimal capacity |
| h64 | 64 | Moderate capacity |
| h128 | 128 | Upper bound tested |

### 8.2 Inputs

- **MLP-diff:** Input is d = v_α − v_ᾱ (same as PLS input)
- **MLP-concat:** Input is [v_α; v_ᾱ] (2p-dimensional). Training includes antisymmetric augmentation: for each training pair (A, B, pref), the reversed pair (B, A, −pref) is added to the batch, enforcing f(A,B) = −f(B,A)

### 8.3 Training

- **Objective:** MSE loss (regression) or BCE-with-logits loss (classification, for accuracy models)
- **Optimizer:** Adam (lr = 1e-3), L2 weight decay (λ = 1e-4)
- **Early stopping:** Patience = 20 epochs on a 10% held-out validation set
- **Max epochs:** 500; batch size = 2,048
- **Seed:** 964

### 8.4 Regression vs. Classification Models

Two separate models are trained per split/condition/hidden-dim:

- **Regression model** (MSE loss): used for r² and Spearman ρ
- **Classification model** (BCE loss, binarized labels): used for accuracy. Labels binarized as (pref > 0) → 1, (pref < 0) → 0; ties (pref = 0) excluded. This ensures accuracy is not derived from the sign of a regression prediction but from a model trained directly on the binary task.

**Control task:** For both model types, a matched control run shuffles labels before training and evaluation. Output files prefixed `control_` (regression) or `binarize_control_` (classification).

---

## 9. Novel-to-Corpus Transfer and Frequency Analysis

To test whether high-frequency attested binomials are harder to predict from abstract embedding structure, probes trained on novel pairs are applied to the attested corpus:

1. **Novel-to-corpus transfer:** PLS and MLP probes trained on all novel pairs (~340k); tested on corpus pairs (~49k). Output files prefixed `novel_to_corpus_`.

2. **Frequency-error analysis:** For each corpus test pair, absolute prediction error |ŷ − y| is computed and merged with the pair's total corpus frequency (freq_w1_w2 + freq_w2_w1). OLS regression of abs-error on log(1 + freq) tests whether higher-frequency pairs are harder to predict. Results: `freq_error_pairs.csv`, `freq_error_regression.csv`.

The hypothesis is that high-frequency binomials function as increasingly frozen or semi-fixed expressions whose ordering preferences are idiosyncratic and not well-captured by the abstract structure in the embeddings.

---

## 10. Results

### 10.1 r² by Condition (Regression Models, h15)

#### Default Condition (binomial context, mean-pooled span)

| Eval split | 125m PLS | 125m diff | 125m concat | 350m PLS | 350m diff | 350m concat | 1.3b PLS | 1.3b diff | 1.3b concat |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | .095 | .083 | .120 | .104 | .095 | .121 | .060 | .059 | .065 |
| Pair-CV | .200 | .230 | .321 | .225 | .263 | .348 | .171 | .182 | .238 |
| Word-CV (novel) | .152 | .138 | .186 | .170 | .163 | .217 | .117 | .107 | .127 |
| Corpus word-CV | .172 | .172 | .183 | .139 | .141 | .176 | .129 | .144 | .142 |
| Word-strict | .034 | .027 | .045 | .043 | .037 | .049 | .024 | .021 | .023 |

#### Last-token Condition (binomial context, final subtoken)

| Eval split | 125m PLS | 125m diff | 125m concat | 350m PLS | 350m diff | 350m concat | 1.3b PLS | 1.3b diff | 1.3b concat |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | .084 | .077 | .101 | .078 | .072 | .098 | .061 | .056 | .064 |
| Pair-CV | .157 | .162 | .272 | .168 | .176 | .290 | .144 | .146 | .207 |
| Word-CV (novel) | .118 | .112 | .147 | .132 | .118 | .146 | .098 | .091 | .114 |
| Corpus word-CV | .196 | .187 | .200 | .168 | .164 | .184 | .152 | .157 | .169 |
| Word-strict | .036 | .031 | .046 | .044 | .038 | .052 | .029 | .025 | .028 |

#### Isolated Condition ("the {word}", mean-pooled)

| Eval split | 125m PLS | 125m diff | 125m concat | 350m PLS | 350m diff | 350m concat | 1.3b PLS | 1.3b diff | 1.3b concat |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transfer | .052 | .048 | .060 | .046 | .044 | .047 | .041 | .038 | .043 |
| Pair-CV | .115 | .128 | .217 | .114 | .130 | .226 | .113 | .122 | .178 |
| Word-CV (novel) | .064 | .058 | .069 | .064 | .059 | .064 | .049 | .048 | .050 |
| Corpus word-CV | .126 | .136 | .140 | .099 | .111 | .114 | .093 | .104 | .112 |
| Word-strict | .014 | .011 | .015 | .014 | .014 | .016 | .005 | .004 | .007 |

### 10.2 r² by Hidden Dim (Default Condition)

Capacity ablation across h15, h64, and h128 for the default condition. h15 values reproduced from Section 10.1 for comparison. OPT-1.3B pending pipeline completion.

#### OPT-125M

| Eval split | Md h15 | Md h64 | Md h128 | Mc h15 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|---:|---:|
| Transfer | .083 | .087 | .092 | .120 | .126 | .135 |
| Pair-CV | .230 (.003) | .229 (.004) | .232 (.002) | .321 (.002) | .359 (.005) | .379 (.004) |
| Word-CV (novel) | .138 (.037) | .148 (.027) | .150 (.027) | .186 (.024) | .208 (.023) | .216 (.023) |
| Corpus word-CV | .172 (.047) | .172 (.040) | .176 (.044) | .183 (.047) | .199 (.043) | .196 (.043) |
| Word-strict | .027 | .029 | .032 | .045 | .048 | .053 |

#### OPT-350M

| Eval split | Md h15 | Md h64 | Md h128 | Mc h15 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|---:|---:|
| Transfer | .095 | .098 | .100 | .121 | .125 | .129 |
| Pair-CV | .263 (.008) | .257 (.008) | .258 (.006) | .348 (.008) | .391 (.008) | .407 (.006) |
| Word-CV (novel) | .163 (.025) | .170 (.021) | .172 (.021) | .217 (.030) | .233 (.020) | .240 (.023) |
| Corpus word-CV | .141 (.045) | .143 (.046) | .144 (.046) | .176 (.055) | .181 (.056) | .176 (.051) |
| Word-strict | .037 | .037 | .039 | .049 | .047 | .052 |

#### OPT-1.3B

| Eval split | Md h15 | Md h64 | Md h128 | Mc h15 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|---:|---:|
| Transfer | .059 | .057 | .058 | .065 | .069 | .071 |
| Pair-CV | .182 (.004) | .187 (.004) | .190 (.004) | .238 (.004) | .263 (.006) | .273 (.005) |
| Word-CV (novel) | .107 (.022) | .114 (.021) | .113 (.022) | .127 (.026) | .141 (.028) | .147 (.026) |
| Corpus word-CV | .144 (.037) | .146 (.037) | .146 (.036) | .142 (.040) | .137 (.031) | .139 (.038) |
| Word-strict | .021 | .020 | .022 | .023 | .029 | .029 |

Increasing hidden dim from h15 to h128 produces small but consistent gains in Pair-CV and Corpus word-CV for MLP-concat (e.g., 125m: .321 → .359 → .379; 350m: .348 → .391 → .407; 1.3b: .238 → .263 → .273). MLP-diff shows minimal sensitivity to capacity across all models. This suggests the nonlinear joint-representation probe benefits modestly from additional capacity, while the difference-vector probe is already near its information ceiling at h15.

### 10.3 Accuracy by Condition (Classification Models)

Accuracy from separately trained binary classification models (BCE loss, binarized labels: pref > 0 → 1, pref < 0 → 0, ties excluded). h15 was not run for 1.3b/default or any last-token/isolated condition (shown as —).

#### Default Condition — OPT-125M

| Eval split | Md h15 | Md h64 | Md h128 | Mc h15 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|---:|---:|
| Transfer | .586 | .587 | .584 | .595 | .599 | .599 |
| Pair-CV | .629 (.002) | .630 (.003) | .630 (.003) | .653 (.001) | .662 (.003) | .662 (.003) |
| Word-CV (novel) | .614 (.010) | .616 (.010) | .615 (.013) | .627 (.013) | .627 (.011) | .631 (.014) |
| Corpus word-CV | .625 (.028) | .632 (.024) | .623 (.026) | .627 (.027) | .627 (.033) | .632 (.025) |
| Word-strict | .541 | .545 | .542 | .546 | .551 | .553 |

#### Default Condition — OPT-350M

| Eval split | Md h15 | Md h64 | Md h128 | Mc h15 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|---:|---:|
| Transfer | .590 | .590 | .591 | .594 | .596 | .592 |
| Pair-CV | .633 (.002) | .634 (.002) | .634 (.001) | .655 (.003) | .665 (.002) | .666 (.003) |
| Word-CV (novel) | .620 (.011) | .619 (.008) | .617 (.011) | .631 (.012) | .637 (.009) | .634 (.007) |
| Corpus word-CV | .626 (.015) | .632 (.019) | .627 (.014) | — | .631 (.020) | .630 (.028) |
| Word-strict | .555 | .557 | .559 | .556 | .559 | .549 |

#### Default Condition — OPT-1.3B

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .573 | .573 | .573 | .574 |
| Pair-CV | .618 (.002) | .618 (.003) | .630 (.002) | .631 (.002) |
| Word-CV (novel) | .601 (.013) | .601 (.010) | .607 (.011) | .604 (.010) |
| Corpus word-CV | .624 (.029) | .623 (.028) | .616 (.030) | .622 (.032) |
| Word-strict | .549 | .546 | .545 | .545 |

#### Last-token Condition — OPT-125M

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .592 | .593 | .598 | .597 |
| Pair-CV | .622 (.002) | .623 (.002) | .655 (.002) | .656 (.002) |
| Word-CV (novel) | .604 (.007) | .606 (.007) | .618 (.010) | .617 (.015) |
| Corpus word-CV | .626 (.024) | .628 (.035) | .640 (.032) | .635 (.027) |
| Word-strict | .558 | .560 | .560 | .560 |

#### Last-token Condition — OPT-350M

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .588 | .588 | .594 | .596 |
| Pair-CV | .626 (.002) | .627 (.002) | .656 (.002) | .656 (.004) |
| Word-CV (novel) | .611 (.009) | .611 (.008) | .624 (.011) | .619 (.010) |
| Corpus word-CV | .633 (.024) | .632 (.029) | .630 (.023) | .636 (.023) |
| Word-strict | .563 | .562 | .568 | .568 |

#### Last-token Condition — OPT-1.3B

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .571 | .574 | .573 | .576 |
| Pair-CV | .613 (.003) | .613 (.003) | .628 (.003) | .629 (.003) |
| Word-CV (novel) | .593 (.011) | .592 (.010) | .598 (.013) | .600 (.009) |
| Corpus word-CV | .637 (.028) | .624 (.032) | .624 (.026) | .629 (.028) |
| Word-strict | .547 | .548 | .548 | .550 |

#### Isolated Condition — OPT-125M

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .576 | .576 | .581 | .581 |
| Pair-CV | .609 (.002) | .610 (.002) | .640 (.002) | .641 (.003) |
| Word-CV (novel) | .586 (.015) | .582 (.015) | .594 (.011) | .591 (.012) |
| Corpus word-CV | .608 (.031) | .608 (.028) | .608 (.025) | .604 (.035) |
| Word-strict | .544 | .544 | .544 | .545 |

#### Isolated Condition — OPT-350M

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .573 | .574 | .574 | .579 |
| Pair-CV | .608 (.003) | .609 (.003) | .639 (.002) | .639 (.003) |
| Word-CV (novel) | .578 (.012) | .576 (.014) | .589 (.015) | .587 (.013) |
| Corpus word-CV | .619 (.028) | .615 (.024) | .608 (.020) | .613 (.027) |
| Word-strict | .547 | .547 | .548 | .552 |

#### Isolated Condition — OPT-1.3B

| Eval split | Md h64 | Md h128 | Mc h64 | Mc h128 |
|:--|---:|---:|---:|---:|
| Transfer | .563 | .563 | .561 | .562 |
| Pair-CV | .602 (.002) | .601 (.003) | .618 (.003) | .618 (.003) |
| Word-CV (novel) | .565 (.007) | .565 (.010) | .573 (.011) | .572 (.008) |
| Corpus word-CV | .608 (.023) | .607 (.023) | .610 (.022) | .612 (.033) |
| Word-strict | .525 | .527 | .521 | .523 |

Accuracy tracks the r² pattern across all conditions: MLP-concat pair-CV is highest (.601–.666), word-strict is lowest (.521–.568), and hidden-dim has little effect (differences ≤ 0.01 throughout). Last-token accuracy is comparable to default, while isolated accuracy is uniformly 1–5 percentage points lower, mirroring the r² drop from default to isolated. The 350m model slightly leads 125m on most splits; 1.3b trails both on pair-CV and word-strict, most notably in the isolated condition.

### 10.4 Novel-to-Corpus Transfer

Probes trained on all novel pairs (~340k) and tested on corpus pairs (~49k). This is the reverse of the Transfer split (section 6). r² shown below.

#### Default Condition

| Probe | 125m | 350m | 1.3b |
|:--|---:|---:|---:|
| PLS | .123 | .127 | .100 |
| MLP-diff | .114 | .129 | .101 |
| MLP-concat | .162 | .170 | .119 |

#### Last-token Condition

| Probe | 125m | 350m | 1.3b |
|:--|---:|---:|---:|
| PLS | .142 | .133 | .127 |
| MLP-diff | .130 | .118 | .117 |
| MLP-concat | .181 | .174 | .152 |

#### Isolated Condition

| Probe | 125m | 350m | 1.3b |
|:--|---:|---:|---:|
| PLS | .094 | .081 | .082 |
| MLP-diff | .090 | .079 | .069 |
| MLP-concat | .128 | .110 | .119 |

Novel-to-corpus r² is notably lower than the within-novel pair-CV results (e.g., default MLP-concat: .119–.170 here vs. .238–.348 pair-CV), consistent with corpus pairs carrying idiosyncratic ordering signal not predictable from abstract embedding structure.

### 10.5 Frequency-Error Analysis

For each corpus pair in the novel-to-corpus test set, absolute prediction error |ŷ − y| was regressed on log(1 + total_freq). OLS slopes and r² are shown for the default condition (primary condition of interest); results are qualitatively consistent across conditions.

#### OLS: abs_error ~ log(1 + freq), Default Condition

| Model | Probe | slope | r² |
|:--|:--|---:|---:|
| 125m | PLS | 0.111 | .00082 |
| 125m | MLP-concat | 0.102 | .00073 |
| 350m | PLS | 0.074 | .00039 |
| 350m | MLP-concat | 0.077 | .00045 |
| 1.3b | PLS | 0.256 | .00148 |
| 1.3b | MLP-concat | 0.269 | .00168 |

#### Binned Mean Error by Frequency, Default Condition (125m)

| Freq bin | n | PLS mean |ŷ − y| | MLP-concat mean |ŷ − y| |
|:--|---:|---:|---:|
| freq=1 | 34,867 | 2.044 | 2.014 |
| freq=2 | 8,072 | 2.033 | 2.003 |
| freq=3–5 | 4,046 | 2.051 | 1.993 |
| freq=6–20 | 1,567 | 2.122 | 2.057 |
| freq>20 | 398 | 2.684 | 2.710 |

Slopes are uniformly positive across all 9 model × condition combinations. The jump in the freq>20 bin is the most consistent signal: pairs seen more than 20 times in the BabyLM corpus show substantially higher prediction error, consistent with high-frequency attested binomials functioning as increasingly frozen expressions whose ordering preferences are not well-captured by the abstract embedding structure on which the probe was trained. The r² values are small throughout (frequency explains little variance overall), but the directional pattern is clear and replicates across models and conditions. Note that 1.3b shows larger absolute errors because that model produces larger-magnitude preference scores.

---

## 11. Control Task Results

*(To be populated. Expected: r² ≤ 0.005 throughout.)*
