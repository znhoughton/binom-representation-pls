# Wikipedia Novel Binomials — PLS Analysis
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

Binomials were extracted from the BabyLM training corpus. A pair (word1, word2) was included if "word1 and word2" occurred at least once; canonical order was determined by frequency (more frequent order = canonical). Final corpus: **N = 212,946 pairs**.

---

## 3. Wikipedia Novel Binomials

### 3.1 Extraction

The English Wikipedia dump (wikimedia/wikipedia, 20231101.en) was streamed via HuggingFace. A regex extracted all "X and Y" surface forms. A pair was retained if:

1. Both words appear in the BabyLM training vocabulary (tokenizer of znhoughton/babylm-150m-v3; 423,810 unique lowercase types).
2. Neither "X and Y" nor "Y and X" appears in the attested corpus (i.e., the pair is novel to the model's training data).
3. The pair occurs in Wikipedia at least **5 times** (wiki_count ≥ 5).

### 3.2 POS Filtering

Pairs were POS-tagged at the phrase level using spaCy (en_core_web_sm) on the string "word1 and word2". Only **strict NOUN+NOUN** pairs were retained (PROPN excluded). This reduced the candidate pool to approximately 299K pairs.

### 3.3 Verb-form Exclusion

Even after phrase-level POS filtering, inflected verb forms (e.g., *puts*, *reaches*, *loves*) that spaCy consistently labels as NOUN in the "X and Y" construction slipped through. An additional filter excluded any word for which NLTK WordNet's morphological analyser (`wn.morphy(word, wn.VERB)`) returns a non-null result — i.e., any word that can be morphologically analysed as a verb inflection — plus a list of known contractions (*wasn*, *didn*, *doesn*, etc.).

### 3.4 Scoring

The filtered pairs were scored using the model. For each pair:

- **Preference** = log P("the word1 and word2") − log P("the word2 and word1"), where log P is the sum of token log-probabilities over the full string.
- **diff_vec** = mean-pooled last-layer hidden states over the tokens of "word1 and word2" (canonical) minus the same for the reversed string.

### 3.5 Final Novel Set

After all filtering and scoring: **N = 53,685 pairs**, covering **8,955 unique words**.

---

## 4. PLS Analysis

Partial Least Squares Regression (PLS; K = 15 components) was fit on the attested corpus diff_vecs (last-layer hidden states; 768-, 1,024-, or 2,048-dim depending on model), separately for each model. Features were z-scored using the corpus mean and standard deviation; the same scaler was applied frozen to novel pairs. A linear regression of preference on the 15 PLS components was fit on the corpus scores and applied frozen to the novel set.

PLS was implemented via the NIPALS algorithm in PyTorch (GPU/cuBLAS). To project new data without re-running the iterative deflation, the corrected weight matrix W* = W(PᵀW)⁻¹ was computed from the raw NIPALS weights W and loadings P, giving the direct mapping T_new = X_new · W*. All random seeds = 964.

---

## 5. Evaluation Designs

| Design | Description |
|---|---|
| **Corpus in-sample** | Full-sample fit: R² of lm(preference ~ C1–C15) on corpus |
| **Novel: frozen coefficients** | Corpus PLS + corpus lm coefficients applied directly to novel pairs |
| **Novel: unique-word split** | Same as above, but corpus model trained only on pairs where neither word appears anywhere in the novel set (n = 149,239 corpus pairs) |
| **Pair-level CV** | 10-fold cross-validation; words may overlap between train and test folds |
| **Word-level CV** | 10-fold CV; unique words assigned to folds; test set = pairs where both words are in the held-out fold (~10% of pairs testable); train set = pairs where neither word is in the held-out fold |

---

## 6. Results

### 6.1 Summary Table

| Set | Analysis | *N* (pairs) | 125m *r*² | 350m *r*² | 1.3B *r*² |
|---|---|---|---|---|---|
| **Attested** | In-sample PLS fit | 212,946 | .495 | .496 | .364 |
| **Novel** | Frozen corpus coefficients | 53,685 | .126 | .144 | .138 |
| **Novel** | Pair-level CV (K = 10) | 53,685 | .284 | .303 | .325 |
| **Novel** | Word-level CV (K = 10) | 5,444 | .138 | .135 | .082 |

*Variance transfer = novel frozen r² / corpus in-sample R².*

| Model | Corpus R² | Novel *r* | Novel *r*² | Novel *ρ* | Transfer |
|---|---|---|---|---|---|
| 125m | .495 | .355 | .126 | .342 | 25.5% |
| 350m | .496 | .379 | .144 | .357 | 29.0% |
| 1.3B | .364 | .372 | .138 | .364 | 37.9% |

### 6.2 Interpretation

**Attested pairs.** The corpus PLS explains ~50% of variance in-sample for 125m and 350m. The 1.3B model yields a lower in-sample R² (.364), likely because 2,048-dim representations provide a richer but more dispersed feature space that is harder to compress into K = 15 components without overfitting.

**Novel pairs (frozen coefficients).** Transfer from corpus to novel increases with model size in absolute terms (novel r² = .126 → .144 → .138) and even more sharply in relative terms (variance transfer = 25.5% → 29.0% → 37.9%). The 1.3B model transfers the largest fraction of its corpus signal to novel pairs despite the lower corpus R², suggesting larger models encode more abstract ordering information.

**Cross-validation within novel.** Pair-level CV r² increases monotonically with model size (.284 → .303 → .325), confirming that larger models capture more of the novel ordering signal. Word-level CV r² is more variable (125m: .138, 350m: .135, 1.3B: .082); the sharp drop for 1.3B may reflect that its higher-dimensional representations require more training data per fold for stable PLS estimation. The word-level CV covers only ~10% of novel pairs (n = 5,444) since both words of a pair must fall in the same held-out fold.

---

## 7. Component Interpretation (OPT-125M; Word-level CV test pairs)

Inspection of the top and bottom scoring pairs per PLS component (restricted to word-level CV test pairs, so neither word appeared in training):

| Component | *r* with preference | BabyLM Δfreq *r* | Δsyllables *r* | Apparent dimension |
|---|---|---|---|---|
| C1 | −.141 | −.315 | +.030 | **Rarity-first**: less frequent word tends to come first |
| C2 | +.127 | −.404 | +.255 | **Rarity + length + animacy**: rarer, longer, animate words first; W2-group semantically coherent (z=+3.76) |
| C3 | −.171 | −.237 | +.048 | **Domain specificity**: specific/technical terms first (cybersecurity, stochastic, kindergarten) over generic terms (days, years, examples) |
| C4 | −.044 | +.274 | +.081 | **Frequency-first**: more frequent words first (opposite direction from C1/C2) |
| C5 | +.079 | −.031 | +.156 | Weak; partially driven by single high-frequency word (*community*) |
| C6 | — | +.117 | −.322 | **Short-word-first (phonological weight)**: shorter/fewer-syllable words prefer first position |
| C8 | — | −.049 | +.228 | **Long-word-first**: longer words prefer first; W1-group semantically coherent (z=+3.16) |
| C9 | — | +.280 | −.073 | **Common animate agents first**: frequent social role words (men, students, players, actors) prefer W1 |

C2 and C3 align with known principles of binomial ordering. C2's dominant signal is BabyLM frequency (Δfreq *r* = −.404), secondarily syllable count and animacy — the component encodes a "marked/specific entity first" cluster of properties, not animacy alone. C6 and C8 form a complementary phonological pair (shorter-first vs. longer-first). C4 and C9 represent a "frequency-first" dimension, counter to C1/C2.

---

## 8. Frequency Stratum Analysis

### 8.1 Full-sample stratum results

PLS was fit separately on each corpus frequency stratum (total occurrences of the pair in the BabyLM corpus), then applied to the same 53,685 novel pairs. Novel r² is reported for each model.

| Corpus stratum | *n* corpus | 125m *r*² | 350m *r*² | 1.3B *r*² |
|---|---|---|---|---|
| freq = 1 | 168,938 | .129 | .145 | .139 |
| freq = 2–5 | 36,638 | .111 | .123 | .115 |
| freq = 6–20 | 5,954 | .062 | .069 | .049 |
| freq > 20 | 1,416 | .027 | .023 | .016 |
| All | 212,946 | .126 | .144 | .138 |

Novel generalisation decreases monotonically with corpus frequency across all three models: pairs seen once generalise 4–9× better than pairs seen more than 20 times. However, the strata differ substantially in size (n = 168,938 vs. n = 1,416), so differences in r² could partly reflect estimation variance rather than true differences in the PLS solution.

### 8.2 Bootstrap analysis (equalised sample size)

To control for sample-size differences, a bootstrap analysis was conducted (B = 500 iterations). In each iteration, N = 1,416 pairs (the size of the smallest stratum, freq > 20) were drawn with replacement from each stratum, PLS was fit on the subsample, and novel r² was recorded. Results are reported as mean r² with 95% bootstrap confidence intervals (2.5th–97.5th percentile).

**Transfer is measured as follows**: the full prediction pipeline (z-scoring with the training sample's mean/SD → NIPALS PLS projection via W\* → linear regression in score space) is fit entirely on the bootstrap corpus sample and applied frozen to all 53,685 novel pairs. The r² between predicted and observed novel preferences is recorded for each iteration.

| Stratum | 125m mean *r*² [95% CI] | 350m mean *r*² [95% CI] | 1.3B mean *r*² [95% CI] |
|---|---|---|---|
| freq = 1 | .047 [.031, .064] | .058 [.041, .077] | .025 [.014, .037] |
| freq = 2–5 | .048 [.031, .066] | .053 [.036, .071] | .027 [.016, .040] |
| freq = 6–20 | .035 [.022, .049] | .039 [.025, .053] | .022 [.011, .034] |
| freq > 20 | .022 [.013, .031] | .020 [.012, .031] | .015 [.009, .022] |

**Pairwise contrasts** (lower − higher frequency; * = 95% CI excludes 0):

| Contrast | 125m | 350m | 1.3B |
|---|---|---|---|
| freq=1 − freq=2–5 | .000 [−.023, .023] | .005 [−.019, .032] | −.003 [−.020, .015] |
| freq=2–5 − freq=6–20 | .013 [−.009, .036] | .014 [−.008, .037] | .006 [−.011, .023] |
| freq=6–20 − freq>20 | .012 [−.003, .030] | .019 [.002, .036] * | .006 [−.006, .020] |
| freq=1 − freq>20 | .025 [.006, .044] * | .037 [.017, .058] * | .010 [−.003, .023] |

The monotonic decline in transfer survives equalisation for 125m and 350m, confirming it is not entirely attributable to sample-size differences. The effect is statistically significant only for the largest contrast (freq=1 vs. freq>20) in 125m and 350m; adjacent contrasts do not reach significance. For 1.3B, no contrast is significant, likely because the overall transfer r² is lower, reducing power to detect differences at N = 1,416.

A caveat applies to all strata: "freq > 20" in the BabyLM corpus corresponds to a maximum of 3,047 occurrences, with a median that remains low by any conventional standard. The BabyLM corpus (150M tokens) is too small to produce truly high-frequency bigrams, so these results speak to a modest frequency gradient rather than the canonical low- vs. high-frequency distinction.

---

## 9. Feature-Difference Correlations

To characterise what the PLS components encode, feature differences were computed for each novel pair: Δfeature = feature(word1) − feature(word2). Four features were available: BabyLM log-frequency (log(corpus count + 1), counted by streaming the full BabyLM 150M corpus; 100% coverage), word length (character count), syllable count (CMU Pronouncing Dictionary; 80.7% coverage), and animacy (binary: 1 = animate, 0 = inanimate; from the VanArsdall/THINGS animacy norms; 3.2% coverage). Brysbaert concreteness and Kuperman AoA norms were not available locally. Pearson *r* and Spearman *ρ* between Δfeature and each component score were computed in R.

### 9.1 Feature × Component Correlations (Pearson *r*)

| Feature | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | Preference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BabyLM log-freq | −.315 | −.404 | −.237 | +.274 | −.031 | +.117 | −.035 | −.049 | +.280 | −.249 | −.186 | +.160 | −.083 | +.035 | +.079 | +.024 |
| Word length | +.004 | +.231 | +.001 | +.097 | +.111 | −.307 | −.138 | +.206 | −.095 | +.067 | +.103 | −.035 | −.039 | +.039 | +.092 | +.017 |
| Syllable count | +.030 | +.255 | +.048 | +.081 | +.156 | −.322 | −.153 | +.228 | −.073 | +.082 | +.061 | +.074 | −.073 | +.029 | +.017 | −.005 |
| Animacy (n=507) | +.094 | +.208 | −.070 | −.004 | −.066 | +.056 | −.129 | +.068 | +.067 | −.239 | −.424 | −.036 | +.198 | +.099 | −.082 | +.080 |

### 9.2 Interpretation

**Direct feature-preference correlations are negligible** (all |*r*| ≤ .08), confirming that ordering preferences are not reducible to any single-feature difference. The components, however, show substantial structure:

- **BabyLM log-frequency** is the dominant correlate overall. For C2 (*r* = −.404) and C1 (*r* = −.315): lower-frequency words prefer W1 (first) position. For C4 (*r* = +.274) and C9 (*r* = +.280): higher-frequency words prefer W1. This indicates that PLS separates a "rarity-first" (C1/C2) from a "frequency-first" (C4/C9) ordering dimension.

- **Syllable count and word length** are strongest for C6 (negative: shorter words first, *r* ≈ −.32) and C2/C8 (positive: longer words first, *r* ≈ +.23–.26). The opposing signs for C2 (longer-first) and C6 (shorter-first) confirm these are distinct phonological ordering principles captured as separate dimensions.

- **Animacy** (n = 507, low coverage) correlates most strongly with C11 (*r* = −.424: animate words prefer W2) and C2 (*r* = +.208: animate words prefer W1). The opposite direction on C11 vs C2 suggests these components capture different contexts in which animacy modulates ordering.

---

## 10. Semantic Clustering

To ask whether words that share a W1 (or W2) bias on a given component form semantically coherent clusters, per-word W1-bias scores were computed: bias_k(w) = mean(C_k | word1=w) − mean(C_k | word2=w), restricted to words appearing ≥5 times in either position (3,749 words). The top and bottom 100 words by bias were taken as W1- and W2-preferred groups. Pairwise cosine similarities within each group were computed using `en_core_web_md` 300-dim word vectors. Results are expressed as z-scores relative to a bootstrap distribution of random groups of N=100 (1,000 iterations).

### 10.1 Semantic Clustering Results

| Component | W1-group similarity (z) | W2-group similarity (z) | Notable pattern |
|---|---:|---:|---|
| C1 | −2.03 | −1.07 | Both groups dispersed; no clustering |
| **C2** | −2.16 | **+3.76** | W2-preferred words form tight cluster |
| C3 | +1.50 | +0.24 | W1 marginal |
| C4 | +1.58 | −0.77 | W1 marginal |
| **C8** | **+3.16** | −1.65 | W1-preferred words form tight cluster |
| **C11** | −0.17 | **+3.54** | W2-preferred words form tight cluster |
| C15 | +1.99 | +0.84 | W1 moderate |

Remaining components (C5–C7, C9–C10, C12–C14): all |z| < 1.7, no strong clustering.

### 10.2 Word Lists for Clustered Components

**C2** (W2-coherent, z = +3.76): W1-preferred — *campsites, ceilings, soybeans, corridors, estuaries, wharves, athleticism, verandahs, cameraman, protagonist, burglary, harmonies, pathos, editorials, royalties, genius, ceramics, folktales, criticisms, robotics*. W2-preferred — *moments, waterways, neighborhood, fur, battery, passageways, footage, difference, night, none, day, past, minutes, things, door*. The W2 group is semantically coherent and consists of high-frequency everyday nouns (night, day, past, minutes, things, door), consistent with the frequency correlation (Δfreq r = −.404): W2-preferred = common words, W1-preferred = rare/specific.

**C8** (W1-coherent, z = +3.16): W1-preferred — *manual, campgrounds, software, civilizations, appearance, maternity, operettas, urology, construction, lightning, quality, knowledge, mangroves, airspace, wellness, mystery, horror, editorial, gunpowder, depressions*. The W1 cluster is semantically coherent around institutional/domain-specific concepts.

**C11** (W2-coherent, z = +3.54): W1-preferred — *workshops, setbacks, holders, hatred, magazines, computers, paintings, figs, workers, stickers, collection, rediscovery, bombings, cadets, standings*. W2-preferred — *strength, hearts, appliances, happiness, soul, regions, vocal, income, relief, lips, independence, death, heart, sonar, representatives*. The W2 group is semantically coherent around abstract collective/emotive concepts (happiness, strength, soul, independence, death, heart).

### 10.3 Interpretation

The clustering results are consistent with the feature correlations: C2's semantically coherent W2-group (common everyday nouns) aligns with its strong frequency signal. C11's W2-coherent group (abstract emotive concepts) coincides with its strong animacy correlation (r = −.424), suggesting animate/emotive terms prefer W2 position for the pattern encoded by C11. C8's W1-coherent group (institutional/domain-specific nouns) adds a dimension not captured by simple frequency or length alone.

Overall, the pattern of semantic coherence is one-sided for the notable components (one pole coherent, the other dispersed), suggesting these components encode categorical distinctions rather than continuous gradients.

---

## 11. Semantic Vector Difference Analysis

Two further analyses used `en_core_web_md` 300-dim word vectors (99.5% pair coverage).

### 11.1 Pair Cosine Similarity

For each pair, the cosine similarity between vec(W1) and vec(W2) was computed (how semantically related the two words are to each other). Correlation with preference: *r* = .000, *p* = .97. Correlations with all 15 components: all |*r*| ≤ .033. **Complete null**: how similar W1 and W2 are to each other is entirely irrelevant to ordering. Semantic distance between the words does not predict which comes first.

### 11.2 W1→W2 Semantic Direction

For each component Ck, the mean difference vector vec(W1) − vec(W2) was computed over the top and bottom quartile of pairs (by Ck score). The difference of these means defines a "semantic axis" for each component; each pair's projection onto this axis was then correlated with the component score.

For each component, the axis is derived from the mean W1−W2 difference vector in high-scoring pairs minus that in low-scoring pairs. Vocabulary words are then ranked by their dot product with this axis; the top 10 at each end are reported. *r* is the correlation between each pair's projection onto the axis and its component score; *r*² is variance explained.

| Comp | *r* | *r*² | W1 pole (top 10) | W2 pole (top 10) |
|---|---:|---:|---|---|
| C1 | .244 | .059 | fatimids, cays, accordionist, parodist, chairmanships, annalists, regencies, eulogies, epitaphs, presentment | music, songs, remixes, ringtones, remixer, crewmen, sons, policemen, gunshot, deaths |
| C2 | .334 | .111 | crossbench, muralists, biotechnologist, prows, accordionist, parodist, hothouses, cays, fatimids, acculturation | humidity, temperature, volume, improvements, downwards, junction, plateau, extension, steeper, elevation |
| C3 | .220 | .049 | crossbench, hothouses, biotechnologist, muralists, prows, patroness, speyer, sportswoman, hymnwriter, sinologist | editorial, readers, columns, column, columnist, suggestions, topics, theories, description, facts |
| C4 | .294 | .087 | efforts, role, community, opportunities, resources, contribution, areas, concentration, concentrations, activity | regencies, annalists, accordionist, fatimids, cays, urbanist, arabization, proclamations, glassblowing, tabloids |
| C5 | .302 | .091 | businesses, investments, advisors, sector, investor, assistance, institution, foundations, assets, advisory | goddess, wig, furry, vixen, cheerleader, doll, actress, woman, girl, hairstyle |
| C6 | .309 | .095 | reincarnation, existence, creature, mortals, visions, persons, sphere, extraterrestrials, deities, gods | reporter, journalists, documentation, disclosure, payroll, appraisal, valuation, prospectus, departments, aviation |
| C7 | .276 | .076 | merchantmen, airbases, humanitarians, sheikhs, berbers, persians, civilians, bystanders, corpses, wops | completion, criteria, objective, basis, indicators, guidance, feasibility, inspection, rehabilitation, evaluation |
| C8 | .285 | .081 | insight, sophistication, intellect, intelligence, knowledge, implementation, concepts, assumptions, philosophies, pedagogy | midges, portages, bullwinkle, relegations, coves, dockworkers, klansmen, cadillacs, entrenchments, cylinders |
| C9 | .277 | .077 | infrastructure, appliances, maintenance, unit, interfaces, appliance, facilities, adaptive, systems, enclosure | consolations, beatitude, predestination, liberality, unction, partisanship, gridlock, rancor, heterodox, empiricism |
| C10 | .292 | .085 | capacities, conveniences, efficiencies, audibility, infrasound, compositions, particles, atmospheres, solids, combinations | opinion, opinions, viewpoints, imagination, tranquility, destiny, contemplation, mysteries, psyche, consciousness |
| C11 | .327 | .107 | controllers, interfaces, maintenance, distribution, infrastructures, telecommunication, software, systems, modules, computers | contemplation, peace, divinity, tranquility, horizon, womanhood, planet, harmony, imagination, destiny |
| C12 | .272 | .074 | invention, procedure, improvisation, demonstration, elements, hypnosis, instruction, procedures, aspects, practitioners | stipends, appropriations, hospices, lobbyist, deduction, penalties, forfeiture, tonnage, admirals, lieutenants |
| C13 | .292 | .085 | beatings, vigils, repressions, extortions, avocations, intimidation, profundity, erudition, misanthropy, snobbery | assets, sector, asset, affairs, advisory, advisors, businesses, investment, investor, organizations |
| C14 | .271 | .073 | incomes, income, households, household, revenues, acquisitions, revenue, subscribers, savings, transportation | sarcasm, subtlety, irony, elocution, meanings, numerals, idiom, vowels, notations, diction |
| C15 | .335 | .112 | transportation, savings, withdrawals, transactions, payments, expenditure, liabilities, duties, expenditures, childcare | septuagint, tafsir, template, tutorial, instructions, sprite, crossbones, avatars, sprites, icons |

The semantic direction between W1 and W2 explains 5–11% of variance in each component score (*r* = .22–.34). The null result for cosine similarity (§11.1) and the moderate effect for direction (§11.2) together indicate that ordering is not about how *related* two words are, but about *where* each word sits in semantic space relative to the other.

---

## 12. Notes on POS Filtering

Three rounds of filtering were applied to remove non-noun items from the novel set:

1. **Phrase-level spaCy** on "word1 and word2": removes pairs where either word is tagged VERB or ADJ in context.
2. **Always-NOUN constraint**: excludes words ever tagged as non-NOUN across all their Wikipedia pair appearances.
3. **WordNet morphy**: excludes words for which `wn.morphy(word, VERB)` is non-null (i.e., any morphological verb reading exists).

Filtering reduced the novel set from ~15.8M raw candidates to 53,685 final pairs. The strictest filter (step 3) reduced the always-NOUN set from 100,742 to 53,685 pairs, removing 2,847 unique words with any verb morphological analysis. This excludes some legitimate nouns with incidental verb readings (e.g., *bronzes*, *sculptures*), trading recall for precision in the noun classification.