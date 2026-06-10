# Wikipedia Novel Binomials — PLS/MLP Analysis
## Methods

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

Output saved to `Data/wikipedia_novel_binomials.csv` with columns: word1, word2, wiki_count, pos1, pos2, example_sentence.

---

## 4. Scoring

For each pair (W1, W2), both orderings are scored using each model:

- **Preference** = Σ log P(tₖ | preceding context) for tₖ ∈ {W1, and, W2} minus the same for {W2, and, W1}, where both orderings share the identical sentence prefix (all tokens before the span). Since the prefix cancels in the difference, this is equivalent to full-sentence log-probability differencing.
- **diff_vec** = mean-pooled last-layer hidden states over the span tokens {W1, and, W2} conditioned on the same sentence prefix, minus the same for {W2, and, W1}.

---

## 5. PLS Analysis

Partial Least Squares Regression (PLS; K = 15 components) was fit on the attested corpus diff_vecs (last-layer hidden states), separately for each model. Features were z-scored using the corpus mean and standard deviation; the same scaler was applied frozen to novel pairs. A linear regression of preference on the 15 PLS components was fit on the corpus scores and applied frozen to the novel set.

PLS was implemented via the NIPALS algorithm in PyTorch (GPU/cuBLAS). To project new data without re-running the iterative deflation, the corrected weight matrix W* = W(PᵀW)⁻¹ was computed from the raw NIPALS weights W and loadings P, giving the direct mapping T_new = X_new · W*. All random seeds = 964.

---

## 6. MLP Analysis

A two-layer MLP (`Linear(input_dim, 15) → Tanh → Linear(15, 1)`) was trained to predict ordering preference from hidden-state representations. Hidden dimension = 15 matches PLS K for fair comparison. Trained with MSE loss, Adam (lr=1e-3), 200 epochs, batch=2048, z-score scaling (corpus statistics applied frozen to test).

Two input representations:
- **diff**: Diff-vector only (same input as PLS)
- **concat**: Concatenate(vec_alpha, vec_non_alpha) with antisymmetric augmentation during training

Three train/test splits:
- **transfer**: Train on corpus → test on novel (main generalization test)
- **word_novel**: 80/20 word split within novel (test new word combinations)
- **word_strict**: Train on corpus → test novel pairs where neither word appears in corpus

---

## 7. Evaluation Designs

| Design | Description |
|---|---|
| **Corpus in-sample** | Full-sample fit: R² of lm(preference ~ C1–C15) on corpus |
| **Novel: frozen coefficients** | Corpus PLS + corpus lm coefficients applied directly to novel pairs |
| **Pair-level CV** | 10-fold cross-validation; words may overlap between train and test folds |
| **Word-level CV** | 10-fold CV; unique words assigned to folds; test set = pairs where both words are in the held-out fold (~10% of pairs testable); train set = pairs where neither word is in the held-out fold |
| **MLP: transfer** | MLP trained on corpus, tested on novel |
| **MLP: word_novel** | MLP trained on 80% of novel words, tested on remaining 20% |
| **MLP: word_strict** | MLP trained on corpus, tested on novel pairs with no corpus word overlap |

---

## 8. Results

*(Results to be populated after re-extraction and re-analysis with the new constituency-based pipeline.)*

