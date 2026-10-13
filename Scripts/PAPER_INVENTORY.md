# What the paper actually reports

Audited against `Writeup/writeup.qmd` by tracing every file the document reads.
This is the contract the pipeline has to satisfy: everything here must be
produced, and nothing outside it needs to be.

## Outputs the writeup reads

| consumed file | produced by | used for |
|---|---|---|
| `Results/{slug}/by_layer_mlp.csv` | `by_layer_mlp.py` | Exp 1-3 probe R², all tables and figures |
| `Results/{slug}/by_layer_mlp_control.csv` | `by_layer_mlp.py --control` | label-shuffle control, reported beside every R² |
| `Results/{slug}/by_layer_corpus_pred.csv.xz` | `by_layer_mlp.py --corpus-freq` | input to the brms fits (not read directly) |
| `Data/derived/brms_relfreq_rawscale.rds` | `corpus_relfreq_brms_all.R` (`SCALE=raw`) | Eq. 1, all experiments, Appendix F |
| `Data/derived/relfreq_slope_curve.csv` | `prepare_relfreq_slopes.R` | simple-slope figures |
| `Data/supplementary/{slug}/linear_probe.csv` | `supplementary_probes.py` | Appendix D, ridge replication |
| `Results/cross_model_agreement_summary.csv` | `cross_model_agreement.py` | Appendix H, Discussion (r = 0.45) |
| `Results/novel_binomials_infinigram_dolma.csv` | `query_infinigram.py` | Methods, novel-set frequency |
| `Results/corpus_binomials_infinigram_piletrain.csv` | `query_infinigram.py` | Appendix G, corpus frequency distributions |

Both infini-gram files are tracked in the repo and do not need regenerating.

## Probe grid, required in full

Every cell of `conditions × modes × splits` is reported. `words_only` carries the
word-level replications in Appendices H, I and J; dropping it would remove three
appendix subsections.

- conditions: `default`, `attn_zeroed`
- modes: `mean_pooled`, `words_only`
- splits: `pair_novel`, `word_novel`
- plus `--control` over the same grid, and `--corpus-freq`
- final layer only (every reader in the writeup reduces to `layer == max(layer)`)

## Models

16 final-checkpoint models, from `tbl-models`:

    BabyLM      125M, 350M, 1.3B
    GPT-2       gpt2, medium, large, xl
    Pythia      160M, 410M, 1B
    OLMo        OLMo-1B-hf, OLMo-2-0425-1B, OLMo-7B-hf, OLMo-2-1124-7B
    Llama       Llama-3.2-1B, Meta-Llama-3-8B

NOT in the paper: `EleutherAI/pythia-2.8b`, `allenai/OLMo-2-1124-1B`. Both exist
in `Results/` from earlier exploration and are excluded from `tbl-models`.

Checkpoints (Exp 3): BabyLM 125M/350M/1.3B and Pythia 160M/410M/1B, six steps
each, plus **step 0**, which the checkpoint-0 baseline added as the floor.

## Reported but requiring no computation

- Appendix A `tbl-models`, Appendix E `tbl-cv-sizes`: hardcoded tables. The CV
  table quotes item counts and must be updated for the new item set.
- Appendices B and C: prose.

## Analyses NOT in the paper

Referenced nowhere in `writeup.qmd`; archived rather than run:

    agreement_by_strength.py     probe_convergence.py
    per_word_preference.py       steering.py
    supplementary_probes.py --freq-strata (the strata arm only; the ridge arm is Appendix D)
    by-layer sweeps (run_bylayer.py over all layers, run_pythia_checkpoints.py)
