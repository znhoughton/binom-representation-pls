# Scripts

**Run `./run_paper_pipeline.sh` from the project root.** It executes every
analysis the paper reports and nothing else, in dependency order, and is
resume-safe at every stage. The sections below describe the pieces it calls.

```
corpus/     build the binomial sets, harvest carrier sentences
pipeline/   extract hidden states, train probes            (GPU, slow)
analysis/   fit the regressions the paper reports          (CPU, ~7 h)
supplementary/  the two appendix analyses that are reported
archive/    everything the paper does not use
```

`Scripts/PAPER_INVENTORY.md` is the audit behind that split: it traces every
file `Writeup/writeup.qmd` reads back to the script that produces it. Anything
not on that list lives in `archive/`.

## The probe target

Extraction now reads `Data/carrier_sentences.csv`, which gives each binomial
**two** occurrences with real left context. The first supplies the
representations and the same-context preference; the second supplies
`preference_cross`, and `by_layer_mlp.py --target cross` predicts that.

This matters because the old design let the probe recompute its own target:
input and target came from one forward pass, so the probe recovered the
network's frozen arithmetic rather than anything learned. A randomly
initialised BabyLM-125M scored **higher** than any trained checkpoint.

| BabyLM-125M, word_novel, attn_zeroed | step 0 | final | shuffled control |
|---|---|---|---|
| old design | 0.822 | 0.143 | 0.003 |
| filtered items only | 0.004 | 0.105 | 0.003 |
| filtered + cross-context | 0.003 | 0.039 | 0.004 |

Two things do the work: dropping the 29% of items whose binomial sits at
sentence start, where every item shares one constant context, and measuring the
target somewhere the probe cannot see.

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
| `run_paper_cells.py` | **Entry point for stage 1.** Enumerates the 58 reported cells (16 final models + 42 checkpoints including step 0) and runs each |
| `extract_embeddings.py` | Hidden states for both orderings under both attention conditions; `--carriers` adds the cross-context target |
| `by_layer_mlp.py` | MLP probes and corpus predictions; `--target {same,cross}` |
| `run_bylayer.py` | Extraction plus probing for one model |
| `run_scale_models.py` | `run_model()`, the per-cell worker every driver reuses |
| `make_random_init.py` | Rebuilds the BabyLM untrained initializations (seed 964, config from the published repo) for the step-0 cells |
| `pilot_cross_context.py` | The validation behind the design change. Not part of the pipeline; kept as the evidence |

Writes `Results/{slug}/by_layer_mlp.csv`, `by_layer_mlp_control.csv`, and
`by_layer_corpus_pred.csv.xz`. The `.npz` embeddings and `.xz` predictions are
gitignored: together they are ~17 GB and regenerable from here.

### Checkpoint 0

Step 0 is an ordinary reported cell now, run by `run_paper_cells.py` alongside
every other checkpoint, and it is the floor Experiment 3 needs: it says what the
probe scores before the model has seen a single token.

Pythia publishes its initialization as revision `step0`. The BabyLM models were
trained from scratch under the HF Trainer, which writes nothing before the first
`save_steps`, so `make_random_init.py` rebuilds their init: `set_seed(964)` then
`AutoModelForCausalLM.from_config(...)`, the same two calls in the same order as
`train_autoreg.py`, with the config pulled from the published repo rather than
from `facebook/opt-*` so the init cannot disagree with the trained model about
architecture. It prints `do_layer_norm_before` and `word_embed_proj_dim`, which
is where the post-LN bug entered.

The reconstruction is an independent draw from the same init distribution, not
the draw those runs started from: the checksum differs across transformers
versions, so the same seed on a different machine gives different weights. That
is sufficient for a floor, and `--seeds` puts several draws on the plot if one
looks too convenient.

Results land at `Results/{slug}_step0/`, so adding `0` to the step vectors in
`Writeup/writeup.qmd` and in `corpus_relfreq_brms_all.R`'s `CKPT_REGISTRY` pulls
step 0 into the Experiment 3 figures, tables, and Eq. 1 fits.

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

## supplementary/ — the reported appendix analyses

| script | what it does |
|---|---|
| `supplementary_probes.py` | Ridge probe on the same folds as the MLP. Appendix D. Run with `--modes mean_pooled --skip-pls`, which is all the appendix reads |
| `cross_model_agreement.py` | Do independently trained models agree on novel binomials? Appendix H and the Discussion's r = 0.45. `--slug-suffix`/`--out-tag` point it at a checkpoint slice |

Everything else that used to live here is in `archive/`, with a note on each.

---

## Environment

GPU work (`pipeline/`, `supplementary/`) needs the `PRenv` conda environment,
the only one here with CUDA-enabled PyTorch. R work needs R 4.5 with `brms` and
`cmdstanr`; the fitting script uses the cmdstanr backend with within-chain
threading. `.Renviron` at the project root points R's temp directory at `D:`,
because decompressing the `.xz` prediction files through `vroom` has exhausted
`C:` mid-render.
