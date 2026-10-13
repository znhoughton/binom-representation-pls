# Archived scripts

None of these produce anything `Writeup/writeup.qmd` reads. They were checked
against the document file by file; see `Scripts/PAPER_INVENTORY.md` for the
audit. They are kept rather than deleted because several were genuinely useful
during development and a future paper may want them back.

## Superseded by the carrier-sentence pipeline

| script | why |
|---|---|
| `run_pipeline.py` | phase driver for the old item set; `run_paper_cells.py` replaces it and enumerates only reported cells |
| `run_babylm_checkpoints.py` | BabyLM checkpoint sweep; now cells in `run_paper_cells.py` |
| `run_step0_baseline.py` | checkpoint 0 was a one-off diagnostic; step 0 is now an ordinary reported cell |
| `supplementary_analyses.sh` | ran the whole appendix battery; only the ridge probe is reported, so the new driver calls `supplementary_probes.py` directly |
| `supplementary_jobs.py`, `supplementary_backfill_markers.py` | job spec and completion markers for that driver |

## Never reported

| script | what it did |
|---|---|
| `per_word_preference.py` | whether `ĝ(w)` is predicted by length, syllables, frequency, stress. Probe-free and a good measure of abstraction in its own right; worth reviving if the probe framing changes again |
| `agreement_by_strength.py` | whether cross-model agreement is carried by high-magnitude items |
| `probe_convergence.py` | whether probes on different models converge on the same solution |
| `steering.py` | causal test that the ordering direction is used rather than merely present |
| `run_pythia_checkpoints.py` | Pythia checkpoints at **all layers**; the paper is final-layer only |
| `run_pythia_supplementary_babylm_checkpoints.sh` | driver for the above |

## Still live, elsewhere

`cross_model_agreement.py` and `supplementary_probes.py` stay in
`Scripts/supplementary/`: the first produces Appendix H and the Discussion's
r = 0.45, the second produces Appendix D. `pilot_cross_context.py` stays in
`Scripts/pipeline/` because it is the evidence for the design change, not part
of the pipeline.
