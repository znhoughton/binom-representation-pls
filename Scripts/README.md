# Scripts

Four stages, run in this order. Everything resolves paths relative to the
project root, so scripts can be invoked from anywhere.

```
corpus/     build the binomial sets and count them in each training corpus
pipeline/   extract hidden states, train probes            (GPU, slow)
analysis/   fit the regressions the paper reports          (CPU, ~7 h)
supplementary/  appendix analyses
```

The two files the paper renders from are `Data/derived/brms_relfreq_rawscale.rds` and
`Data/derived/relfreq_slope_curve.csv`. Both are tracked, so `Writeup/writeup.qmd`
builds from a clean clone without re-running anything below.

---

## corpus/ — datasets and frequency counts

| script | what it does |
|---|---|
| `corpus_creation/extract_corpus_binomials.py` | Extracts open-class binomials from the BabyLM corpus, using benepar to confirm each is a minimal coordinate phrase |
| `corpus_creation/extract_wikipedia_binomials.py` | Same procedure over Wikipedia, for the novel set |
| `corpus_creation/filter_strict_nouns_v3.py` | Drops any word WordNet can analyse as a verb form |
| `corpus_creation/run_parallel_extraction.py` | Shards either extractor across N processes |
| `corpus_creation/merge_extraction_shards.py` | Merges the shard CSVs |
| `query_infinigram.py` | Counts both orderings of each pair via infini-gram |
| `run_all_infinigram.py` | Runs the four infini-gram jobs in sequence |

Outputs `Data/corpus_binomials.csv`, `Data/wikipedia_novel_binomials.csv`, and
`Results/corpus_binomials_infinigram_piletrain.csv`.

## pipeline/ — representations and probes

| script | what it does |
|---|---|
| `run_pipeline.py` | Main entry point; runs the phases below |
| `extract_embeddings.py` | Hidden states for both orderings, under both attention conditions |
| `by_layer_mlp.py` | Trains the MLP probes and writes the corpus predictions |
| `run_bylayer.py` | Extraction plus probing for the layer sweep |
| `run_scale_models.py` | The 13 large-scale models (final layer) |
| `run_babylm_checkpoints.py` | BabyLM training checkpoints |
| `run_pythia_checkpoints.py` | Pythia checkpoints, **all layers**. The main paper uses `run_scale_models.py --checkpoints` (final layer only); this one is kept because the supplementary driver below needs every layer |
| `run_pythia_supplementary_babylm_checkpoints.sh` | Pythia by-layer checkpoints at BabyLM-comparable token counts |

Writes `Results/{slug}/by_layer_mlp.csv`, `by_layer_mlp_control.csv`, and
`by_layer_corpus_pred.csv.xz`. The `.npz` embeddings and `.xz` predictions are
gitignored: together they are ~17 GB and regenerable from here.

## analysis/ — the regressions in the paper

| script | what it does |
|---|---|
| `corpus_relfreq_brms_all.R` | Fits Eq. 1 in all 104 model x condition cells (32 final checkpoints, 72 training checkpoints) |
| `prepare_relfreq_slopes.R` | Turns those posteriors into the simple slopes the figures draw |

`corpus_relfreq_brms_all.R` takes about seven hours. It is resume-safe: each
cell is cached under `Data/brms_models/` and skipped if present, so an
interrupted run continues where it stopped.

**Predictor scaling** is chosen by the `SCALE` environment variable, and it also
sets the output filenames so the two variants never collide:

```bash
Rscript Scripts/analysis/corpus_relfreq_brms_all.R              # SCALE=z   (default)
SCALE=raw Rscript Scripts/analysis/corpus_relfreq_brms_all.R    # what the paper reports
```

* `raw` leaves `rel_freq` on its -0.5..+0.5 proportion scale and centres
  `log_freq` without scaling. Both are corpus properties whose units mean the
  same thing everywhere, so dividing by a corpus-specific SD would replace a
  shared unit with a local one. Writes `Data/derived/brms_relfreq_rawscale.rds`.
* `z` z-scores all four variables. Writes `Data/derived/brms_relfreq_prop.rds`.

`y_true` and `y_pred` are z-scored under both settings. Their spread is a
property of the model rather than of binomial knowledge: `sd(y_pred)` tracks
probe fit at r = 0.97, and `sd(y_true)` reflects how peaked a model's output
distribution is, running opposite to probe R².

`prepare_relfreq_slopes.R` has a `SUFFIX` constant that must match the fits on
disk. It controls the frequency back-conversion, which differs between the two
scalings — centred needs an addition, z-scored needs the SD as well — so a
mismatch silently mislabels every frequency axis.

## supplementary/ — appendix analyses

| script | what it does |
|---|---|
| `supplementary_analyses.sh` | Driver for the appendix probes |
| `supplementary_jobs.py` | Emits that driver's job list |
| `supplementary_probes.py` | The probe analyses themselves; standalone |
| `supplementary_backfill_markers.py` | Writes completion markers so a re-run skips finished cells |
| `cross_model_agreement.py` | Do independently trained models agree on novel binomials? |
| `agreement_by_strength.py` | Is that agreement carried by a few high-magnitude items? |
| `per_word_preference.py` | Word-level ordering preferences |
| `probe_convergence.py` | Do probes on different models converge on the same solution? |
| `steering.py` | Causal test of whether the ordering direction is used, not merely present. Called as step 4 of `supplementary_analyses.sh`; not reported in the paper, but do not delete it without editing that driver |

---

## Environment

GPU work (`pipeline/`, `supplementary/`) needs the `PRenv` conda environment,
the only one here with CUDA-enabled PyTorch. R work needs R 4.5 with `brms` and
`cmdstanr`; the fitting script uses the cmdstanr backend with within-chain
threading. `.Renviron` at the project root points R's temp directory at `D:`,
because decompressing the `.xz` prediction files through `vroom` has exhausted
`C:` mid-render.
