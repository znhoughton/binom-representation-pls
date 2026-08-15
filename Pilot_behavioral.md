# Behavioral Pilot: Memorization and Abstraction Without the Probe

**Scripts:** `Scripts/pilot_behavioral.py`, `Scripts/pilot_behavioral_ushape.py`
**Outputs:** `Results/behavioral_ushape_all.csv`
**Status:** pilot. Part A = one model. Part B = all 17 final models + 84 checkpoint cells.
**Last run:** 2026-08-14

---

## 1. Why this exists

The probe results are vulnerable to a specific objection: at the final layer the
model's output is a deterministic function of its representations, so ordering
information is present *by construction*. A probe recovering it may be reading a
compressed encoding of the answer rather than anything linguistically structured.
Every probing design we considered (word-held-out CV, isolated words, cross-context
recombination, zero-co-occurrence stratification) is deflected by the same move:
whatever produces the behavior must be in the representation, so the probe can read it.

This work sidesteps the objection by never using the representations. Every number
comes from **`y_true`**, the model's own output preference:

```
y_true = log P("W1 and W2") − log P("W2 and W1")
```

It is already stored in `Results/{model}/by_layer_corpus_pred.csv.xz` (attested pairs)
and `Results/{model}/cv_preds/*.npz` (novel pairs). It does not vary by `layer` or
`mode` (verified: max within-pair spread = 0.00e+00); those columns only matter for
the probe.

**The goal is not to replace the probe.** It is convergent validity: if a measure with
none of the probing assumptions reproduces the probe's conclusions, the probe's story
does not rest on the inference being contested.

---

## 2. Part A — Exposure and correctness (`pilot_behavioral.py`)

### 2.1 Design

`corpus_binomials.csv` was extracted from the **BabyLM corpus**. Pythia trained on the
**Pile**. That splits the roles cleanly:

| quantity | source | role for Pythia |
|---|---|---|
| Pile directional counts | infinigram | **exposure** — what the model saw |
| BabyLM ordering counts | `Data/corpus_binomials.csv` | **correctness criterion** — never trained on |

**Abstraction** = the zero-exposure row: pairs with zero Pile occurrences, scored
against a corpus the model never saw. Not memorization (pair absent from training),
not noise (noise does not track an external criterion). This is a *correctness* test,
not a consistency test — two models can converge on a shared error, but matching real
usage for unseen pairs means the generalization is right.

**Memorization** = the slope across exposure rows, and how it changes under ablation.

### 2.2 Results (`EleutherAI_pythia-1b`, n = 48,950)

Correlation of `y_true` with BabyLM-corpus ordering log-odds:

| Pile exposure | n | default | attn_zeroed |
|---|---|---|---|
| **unseen (0)** | 1,726 | **+0.276** [0.23, 0.32] | +0.200 [0.15, 0.25] |
| 1–5 | 2,748 | +0.270 | +0.153 |
| 6–50 | 8,014 | +0.315 | +0.161 |
| 51–500 | 15,272 | +0.360 | +0.162 |
| >500 | 21,190 | +0.469 | +0.194 |

**Exposure effect (>500 minus unseen):** +0.192 [+0.151, +0.235] intact;
**−0.007 [−0.053, +0.038]** ablated.

The entire benefit of having seen a pair is delivered through cross-word attention.
Block it and 500+ exposures buy nothing.

**Confounds (unseen pairs only):** partialling out word length and unigram frequency
moves the correlation from +0.276 to **+0.260**. The generalization is not reducible
to those two constraints.

**Note on a null:** `|Δ| = |y_true(default) − y_true(attn_zeroed)|` is flat across
exposure (r = −0.034 with log frequency). It is a poor index — it counts change in
either direction and is dominated by noise. Recorded so it is not rediscovered.

---

## 3. Part B — Behavioral analogue of the U-shape (`pilot_behavioral_ushape.py`)

### 3.1 The two pipelines are the same experiment

| step | probe (Exp 1–3) | behavioral (Exp 4) |
|---|---|---|
| 1 | 340k novel pairs, absent from BabyLM's corpus | same pairs |
| 2 | train MLP on their **representations** | fit additive model on their **preferences** |
| 3 | → abstraction model, built where memorization is impossible | → same |
| 4 | apply to attested pairs → `ŷ` | apply to attested pairs → `ŷ` |
| 5 | `residual = \|y_true − ŷ\|` | same |
| 6 | regress residual on `log freq + log freq²` → β₂ | same |

Only step 2 differs. The conceptual work is done by **training on the novel set**:
those pairs cannot have been memorized, so anything learned from them is generalizable
structure, and the residual on attested pairs is what memorization added.

### 3.2 The two regression models

**Abstraction model** (fit on novel pairs). One row per binomial; one predictor per
word type, coded +1 for the first word, −1 for the second, 0 otherwise:

```
y_i = Σ_w β_w · x_iw + ε_i
```

Equivalently, a by-word random intercept entering positively in slot 1 and negatively
in slot 2. In brms:

```r
y_true ~ 1 + (1 | mm(word1, word2, weights = cbind(1, -1), scale = FALSE))
```

Prediction for a new pair is `β_{W1} − β_{W2}`, so it generalizes to unseen *pairs* of
known words. It cannot generalize to unseen *words* — see §5.

**Residual model** (fit on attested pairs), identical to the writeup's:

```
|y_true − ŷ| ~ β₀ + β₁ log(freq) + β₂ log(freq)²
```

The pilot uses OLS for both; the paper version should use brms.

### 3.3 Final models (17)

β₂, full context vs attention-zeroed, both methods computed with the same estimator:

| model | β₂ probe (def) | β₂ behav (def) | β₂ probe (abl) | β₂ behav (abl) |
|---|---|---|---|---|
| BabyLM-125M | +0.0729 | +0.0736 | +0.0243 | +0.0110 |
| BabyLM-350M | +0.0641 | +0.0565 | +0.0206 | +0.0125 |
| BabyLM-1.3B | +0.0689 | +0.0386 | +0.0083 | −0.0162 |
| GPT-2 | +0.0589 | +0.0643 | +0.0050 | +0.0265 |
| GPT-2-medium | +0.0385 | +0.0727 | +0.0121 | +0.0154 |
| GPT-2-large | +0.0241 | +0.0735 | +0.0056 | +0.0205 |
| GPT-2-xl | +0.0223 | +0.0684 | +0.0161 | +0.0251 |
| Pythia-160M | +0.0379 | +0.0512 | +0.0188 | +0.0078 |
| Pythia-410M | +0.0247 | +0.0524 | +0.0131 | +0.0068 |
| Pythia-1B | +0.0420 | +0.0659 | +0.0132 | +0.0143 |
| Pythia-2.8B | +0.0190 | +0.0752 | +0.0198 | +0.0208 |
| OLMo-1B | +0.0220 | +0.0631 | +0.0143 | +0.0172 |
| OLMo-7B | +0.0039 | +0.0579 | +0.0115 | +0.0202 |
| OLMo-2-1B | +0.0231 | +0.0658 | +0.0318 | +0.0253 |
| OLMo-2-7B | +0.0224 | +0.0541 | +0.0290 | +0.0231 |
| Llama-3.2-1B | +0.0241 | +0.0728 | +0.0180 | +0.0278 |
| Llama-3-8B | +0.0313 | +0.0796 | +0.0151 | +0.0312 |

**β₂ > 0 under full context in 17/17 models, both methods.**

**Attenuation under ablation:**

| method | mean β₂ default | mean β₂ ablated | attenuation | models attenuated |
|---|---|---|---|---|
| probe | +0.0353 | +0.0163 | +0.0190 | 13/17 |
| behavioral | +0.0638 | +0.0170 | **+0.0468** | **17/17** |

The behavioral attenuation is larger and fully consistent. The four probe exceptions
are Pythia-2.8B and all three OLMo-7B/OLMo-2 models.

### 3.4 Checkpoint trajectories (default condition)

Both methods trace the same developmental arc: β₂ **starts negative and crosses to
positive** as training proceeds.

| model | earliest step | β₂ probe | β₂ behav | final | β₂ probe | β₂ behav |
|---|---|---|---|---|---|---|
| BabyLM-125M | 24 | −0.0083 | −0.0401 | final | +0.0729 | +0.0736 |
| BabyLM-350M | 48 | −0.0287 | −0.0362 | final | +0.0641 | +0.0565 |
| BabyLM-1.3B | 97 | −0.0134 | −0.0298 | step 9021 | +0.0935 | +0.0837 |
| Pythia-160M | 16 | −0.0036 | −0.0394 | final | +0.0379 | +0.0512 |
| Pythia-410M | 16 | −0.0012 | −0.0417 | final | +0.0247 | +0.0524 |
| Pythia-1B | 16 | −0.0093 | −0.0458 | final | +0.0420 | +0.0659 |

The crossing is monotonic in both methods and happens at a comparable point in
training. This is the strongest convergence in the dataset.

### 3.5 Convergence between methods

Each point is a model × condition cell, not a pair.

| set | condition | n | r(β₂ probe, β₂ behav) |
|---|---|---|---|
| all (incl. checkpoints) | default | 59 | **+0.699** |
| all (incl. checkpoints) | attn_zeroed | 59 | **+0.793** |
| **final models only** | default | 17 | **−0.223** |
| **final models only** | attn_zeroed | 17 | +0.214 |

---

## 4. What replicates and what does not

| probe claim | behavioral twin | verdict |
|---|---|---|
| β₂ > 0, U-shaped error | β₂ > 0 in 17/17 | ✅ replicates |
| β₂ attenuated under ablation | 17/17 behavioral vs 13/17 probe | ✅ replicates, more consistently |
| β₂ negative early, positive late | same crossing, same direction | ✅ replicates |
| memorization tied to cross-word attention | exposure effect → 0 under ablation | ✅ replicates (Part A) |
| ordering predictable at all | `y_true` predicts corpus ordering | ✅ replicates |
| BabyLM β₂ > Pile-trained families | behaviorally Llama-3-8B is highest (+0.0796); BabyLM-1.3B is among the lowest (+0.0386) | ❌ **does not replicate** |
| word-held-out R² > 0 | no clean twin | ⚠️ partial |
| pair-vs-word gap = lexical component | additive model *is* the lexical component; direct estimate, not a gap | ⚠️ different quantity |
| layer-wise emergence | none possible | ❌ probe-only |

---

## 5. Limitations and problems found

**Convergence is carried by the trajectory, not by model ranking.** The r ≈ 0.70–0.79
figures include checkpoints, where both measures move from negative to positive
together. Among the 17 fully-trained models alone, where β₂ varies over a narrow
range, the two measures do **not** agree on ranking (r = −0.223 in the default
condition). The honest claim is that the methods agree on the *developmental
trajectory* and on the *ablation effect*, not on which fully-trained model memorizes most.

**Probe β₂ here does not match the writeup for all families.** The OLS values reproduce
the writeup's BabyLM range (0.064–0.073) exactly, but not the reported Pile-family
range (writeup: 0.007–0.018; here: 0.004–0.059, e.g. GPT-2 at +0.0589). Needs
reconciling against the brms spec before either set of numbers is trusted.

**Pythia-2.8B checkpoint data looks duplicated.** Every checkpoint (steps 16–1000)
returns identical values to four decimal places, matching the final model. Almost
certainly the checkpoint extraction reused the final model. Should be checked and
re-run.

**BabyLM-1.3B slug inconsistency.** Checkpoints use
`znhoughton_opt-babylm-1.3b-...` (dot) while the final model uses
`...-1_3b-...` (underscore), so they do not join automatically.

**The additive model cannot generalize to unseen words.** Words absent from the novel
set get β = 0. The probe generalizes to novel words (word-held-out R² > 0); the
behavioral abstraction model structurally cannot. This is the sharpest disanalogy.

**The behavioral abstraction model is much weaker in R².** For BabyLM it is close to
the probe (0.11–0.17 vs 0.10–0.22), but for large models it is far behind (0.04–0.15
vs 0.28–0.55). β₂ nevertheless converges, because β₂ measures the *shape* of the
residual against frequency rather than its size: structure the additive model misses
raises the intercept without bending the curve.

**Criterion noise in Part A.** Pairs unseen in the Pile are also rare in BabyLM, so the
criterion is estimated from small counts and +0.276 is a lower bound. Only 12 unseen
pairs have BabyLM count ≥ 5.

**Part A is one model.** Should be extended, with roles reversed for BabyLM.

---

## 6. Supplementary probe analyses

Separate track, run on the server via `Scripts/supplementary_analyses.sh`. Job list
is generated by `Scripts/supplementary_jobs.py` from `Results/*/by_layer_mlp.csv`, so
coverage matches the paper cell for cell (52 cells; Pythia-2.8B and OLMo-2-1124-1B
excluded as unreported). Outputs go to `Data/supplementary/<slug>/`; nothing touches
main-pipeline files.

### 6.1 Smoke-test results (2026-08-14)

**Linear probe — validated.** On synthetic embeddings with a known planted signal it
recovers exactly the analytic expectation:

| quantity | observed | predicted |
|---|---|---|
| `mean_pooled` R² | 0.9621 | 0.957 |
| `words_only` R² | 0.4792 | 0.478 |
| direction concentration | \|w[0]\| = 0.999 vs 0.004 elsewhere | signal was planted in dim 0 |

**PLS component sweep — does NOT measure intrinsic dimensionality.** On a signal that
is one-dimensional *by construction*, R² still climbs from 0.833 at K=1 to 0.962 at
K=6. PLS components maximise covariance in the standardised space, so finite-sample
noise across irrelevant dimensions dilutes the first component; ridge shrinks that
noise, PLS cannot.

Consequence: **"R² keeps climbing past K=1" is not evidence of distributed coding.**
Report the sweep as descriptive only. The discriminating measures are the
concentration of the fitted ridge direction, and linear-vs-MLP R². More generally,
for a scalar target the readout is always one direction, so "how many dimensions" is
better posed as whether separate *constraint* dimensions are independently decodable
— a feature-decoding question, not a PLS question.

**Steering — three bugs found and fixed.** It returned `None` for every pair before
the fix. It required both orderings to be locatable in the sentence (a natural
sentence contains only one); it kept the sentence tail where the pipeline truncates
at the span start; and it used prefix-length token arithmetic instead of offset
mappings. Now mirrors `extract_binomial_batch` exactly: 8/8 pairs compute, hook fires
and shifts the output.

**Not yet tested:** checkpoint loading via `--revision`, and the `.sh` end to end.
Both need a real extraction.

---

## 7. Pair-specific convergence: memorization without a probe or a criterion

**Script:** `Scripts/pilot_pairspecific_convergence.py` → `Results/pairspecific_convergence.csv`

### 7.1 Method

Decompose the model's **final** preference, out-of-fold, into

- **per-word component** — everything expressible as a comparison of the two words'
  individual ordering biases (additive model, one parameter per word type)
- **residual** — what remains, i.e. anything depending on *which two words* are combined

Then at each checkpoint compute `cor(y_true(t), residual_final)` within exposure strata,
where exposure is the pair's count in that model's **own** training corpus (Pile via
infinigram for Pythia; BabyLM corpus counts for BabyLM).

**Why a gradient means memorization.** The per-word decomposition has already removed
every lexical-frequency effect, since those are per-word comparisons. And an abstract
constraint learned across many pairs applies uniformly regardless of any individual
pair's count, so it cannot generate exposure-dependence. A residual that converges
faster for frequently-seen pairs means the model learned something about *those pairs*.

**Why it avoids the confounds that sank earlier attempts.** The target is the model's
own final state, so there is no external criterion (no sparse-count measurement error)
and no prefix priming — the example sentence contributes identically at every
checkpoint and to the final state, so it cannot create a difference between them.

### 7.2 Results — gradient (top exposure stratum minus bottom)

| model | early checkpoints | mid | late |
|---|---|---|---|
| Pythia-160M | — | +0.043 | **+0.175** [.097,.247] |
| Pythia-410M | +0.022, +0.010, +0.009 | +0.050 | **+0.142** [.046,.234], **+0.173** [.093,.251] |
| Pythia-1B | +0.066, +0.079, +0.086 | +0.060 | **+0.177** [.089,.265], **+0.244** [.169,.313] |
| BabyLM-125M | +0.082, +0.086, +0.087 | **+0.131** [.028,.234] | **+0.172** [.096,.251] |
| BabyLM-350M | +0.069, +0.078, +0.081 | **+0.156** [.042,.265] | **+0.139** [.050,.216] |
| BabyLM-1.3B | +0.024, +0.066, +0.066 | **+0.120** [.019,.210] | **+0.235** [.176,.288] |

Bold = 95% bootstrap CI excludes zero. **6/6 models** show a reliable positive gradient.

The gradient appears earlier in BabyLM because its earliest checkpoints already carry
real exposure (20 epochs over a small corpus), whereas at Pythia step 16 a 500-count
pair has been seen 0.06 times.

**Not an SNR artifact.** Residual variance is U-shaped across exposure, not monotonic.
Pythia-1B's zero-exposure and >50k strata have near-identical residual SD (3.40 vs 3.57)
but correlations of +0.348 vs +0.592 — matched on noise, differing ~1,100× in exposure.

**Ignore the `final` row.** Correlating the final state against its own residual
saturates near 0.95, and the ceiling compresses the gradient toward zero (slightly
negative for Pythia). It is an artifact of the design, not a result.

### 7.3 Scope

The residual **level** is not memorization — it is non-zero even at zero exposure, where
nothing can have been memorized, because the residual also contains compositional
abstraction and sentence-context effects. Only the **gradient** is the memorization
signal. State it as: *the pair-specific component of the final preference is established
faster for pairs the model saw more often, holding per-word effects constant.*

This supports memorization only. It says nothing about abstraction, and nothing about
which developed first — see §5 on why the timing question is not resolvable with the
current data.

---

## 8. Next steps

1. Reconcile OLS β₂ with the brms spec, then rerun both methods in brms.
2. Fix the BabyLM-1.3B slug mismatch. (Pythia-2.8B is moot — excluded from the paper.)
3. Extend Part A to all models, reversing exposure/criterion roles for BabyLM.
4. Run the one analysis that answers "isn't this just per-word preferences": on unseen
   pairs, compare `y_true` vs the external criterion (+0.276, known) against the
   additive model's prediction vs the same criterion (not yet computed).
5. Random-init baseline, to confirm the unseen-pair correlation is learned rather than
   architectural.
6. Decide how to present convergence honestly: lead with the trajectory agreement and
   the ablation agreement, and report the null model-ranking correlation rather than
   the inflated pooled figure.
