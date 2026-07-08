# write-introspection

**Can a language model *control* where a concept appears in its own residual
stream — and how reliably?**

We tell a model to transcribe a fixed sentence while *thinking about* (or *not
thinking about*) a concept at varying intensity, record its residual stream
token-by-token, and measure how the activations move relative to that concept's
direction. On top of that data we build a suite of **controllability measures**
that separate *how far* the model can push a concept (gain) from *how reliably*
it can dial it (fidelity), per token and per layer, and split the effect into a
**direction** channel and a **magnitude** channel.

---

## The idea in one paragraph

For each trial `(concept, sentence, condition)` the model is prompted:

> `Write "<sentence>" exactly. Think intensely about <concept> while you write. Don't write anything else.`

It transcribes the sentence; we capture the residual stream at chosen layers for
every generated token. A **concept vector** (the baseline-subtracted activation
of "Tell me about `<concept>`") gives a direction in activation space. The core
per-token readout is the **cosine** between the residual and that direction —
plus its **norm** for the magnitude channel. By comparing readouts across the
prompt conditions (a neutral baseline, "don't think", "think", and an intensity
ramp 1→4), we quantify control.

---

## The controllability measures

All live in **`scripts/controllability_heatmap.py`**, which renders one heatmap
per measure: **x = layer, y = token**, averaged over concepts, for a single
sentence (the register-rich "cat" sentence by default). These are **generated
automatically at the end of `run_experiment.py`** (for the `cos` and `relnorm`
channels); the script can also be run standalone on any run directory.

Every heatmap uses a diverging **blue (below/negative) → white (0) → red
(above/positive)** colormap with symmetric limits; significance is from
permutation nulls with Benjamini–Hochberg FDR, and **numbers are printed only in
significant cells**.

Two **readouts** (`--metric`), giving two channels:

| readout | channel | definition |
|---|---|---|
| `cos` (default) | **direction** | `cos(concept_vec, residual)` — magnitude-invariant |
| `relnorm` / `norm` | **magnitude** | residual L2 norm (÷ trial's content-token mean for `relnorm`) |

The measures built on a readout:

| measure | question it answers | range | cross-model comparable? |
|---|---|---|---|
| **Rank** | graded "dial" control: does the readout track intensity 1→4 monotonically? (mean **signed** Spearman over concepts) | [-1, 1] | yes (rank-based) |
| **gain** | effect size: how far does the readout swing across the ramp? (`Δreadout` = int4 − int1) | signed | **relnorm: yes**; cos: partial |
| **engagement / suppression** | two-panel: each condition minus the neutral baseline (`think − neutral`, `dont − neutral`); red = above neutral, blue = below | signed | **relnorm: yes**; cos: partial |

> **Cohen's d is disabled.** The standardized ramp gain (`meanΔ / sdΔ`) is no
> longer computed or rendered — its methodology is still documented in
> `MEASURES.md` §4 and the code is retained (commented) in
> `controllability_heatmap.py` for easy restoration.

`Rank` has a **`_specific`** variant on the direction channel that subtracts the
mean off-concept readout, isolating control of *this* concept from a generic
"think harder about everything" effect. The magnitude channel is concept-blind
(no `_specific`).

The two magnitude readouts differ in comparability:
- **`relnorm`** (÷ trial's content-mean norm) is a *dimensionless ratio*, so its
  gain `Δrelnorm` cancels both the model's norm scale *and* hidden-dimension — it
  is **fully cross-model comparable**, so `gain` is reported.
- **`norm`** (raw `‖r‖`) is in model-specific units, so `gain` is **suppressed**
  for it — only the scale-invariant `Rank` is kept.

On the direction side, `Δcos` is invariant to the norm scale but *drifts with
hidden dimension* (cosines shrink ~1/√d), so cosine gains are only "comparable
across models of similar width"; the fully comparable cosine option is the
rank measures.

**Gain vs consistency are different axes.** A token can move a lot but
inconsistently across concepts (a large `gain` whose sign flips concept to
concept — e.g. some register slots) or move a little but very consistently.
`gain` gives the interpretable magnitude; `Rank` gives the sign-consistent,
cross-model-comparable graded-control signal; read them together.

---

## Install

One machine: the **GPU box** generates the data (loads the model, runs the
experiments) and also runs the **analysis**. The analysis never loads the model,
but parts of it read a run's multi-GB `results.pkl` (activations), so it needs
the box that holds the runs and enough host RAM (> pickle size, ~100 GB for the
full-depth runs).

```bash
pip install -r requirements.txt          # torch pulls the CUDA build
```

`bitsandbytes` (quantization) and `wandb` (logging) are optional — skip them for
a lean install; pass `--no-wandb` to the runner if wandb isn't present.

Then create a `.env` with `HF_TOKEN=...` (and optional `WANDB_API_KEY=...`).

## Quick start

```bash
# 1-2. Install + .env (see above)

# 3. Generate data (GPU). The MAIN experiment is the primary one: always-on
#    controls + selectable condition sets (intensity / token_location / persistence /
#    layer_location). Each active set's analyses auto-run at the end
#    (heatmaps, trace plots, targeting/temporal figures).
python scripts/run_experiment.py --config experiments/main/config.yaml
#    ... or a subset of sets:
python scripts/run_experiment.py --config experiments/main/config.yaml \
    --set 'experiment.sets=[intensity,persistence]'

# 4. (Optional) re-render standalone -- a different sentence/layer/channel.
#    No model load, but may read the run's results.pkl (large RAM).
python scripts/controllability_heatmap.py \
    --run-dir results/raw/<RUN_DIR> --model-name gemma3_27b --metric cos
python scripts/plot_results.py --run-dir results/raw/<RUN_DIR> --layers 55 61

# 5. (Optional) cosine summary CSV
python scripts/run_analysis.py --run-dir results/raw/<RUN_DIR>

# 6. (Layer-targeting runs ONLY) the plot7-12 target-layer diagonal figures
python scripts/plot_layer_targeting.py --run-dir results/raw/<RUN_DIR>
```

Steps 3's plots come out automatically. Re-running the analysis scripts reads
the saved `results.pkl` / `results.json` + cached concept vectors — no model
load, but run them on the box holding the runs (results.pkl needs the RAM).

Each run's `plots/` then contains:
- **controllability heatmaps**, per channel (`cos`, `relnorm`):
  `heatmap_rank_raw_*`, `heatmap_rank_specific_*` (cos only), `heatmap_gain_*`,
  `heatmap_engage_suppress_*`, + `controllability_heatmap_<metric>.csv`.
- **trace plots**: `plot1_cos_L<layer>_s<idx>.png`, `plot1_norms_L<layer>_s<idx>.png`.
- **layer-targeting** (only if you run `plot_layer_targeting.py` on a
  layer-targeting run): `plot7*`–`plot12*`.

### Environment note

If your venv's `python` is not the interpreter whose site-packages hold the
deps, call the interpreter directly and set `PYTHONPATH`, e.g.
`PYTHONPATH=<venv>/lib/python3.11/site-packages /usr/bin/python3.11 scripts/...`.

---

## Paper figures (`results/paper/`)

A separate, hand-curated figure set for the write-up — **distinct from the
per-run auto-plots above**. Each `scripts/figN_*.py` loads no model — it reads a
saved run's `results.json` (+ `no_instruction_cache.pkl`, cached concept vectors,
and for some analyses the run's `results.pkl`) and renders one figure. Promoted PNGs plus a per-figure `.md` handoff
(task, measures, exact statistics, draft caption) live in `results/paper/`
(untracked — regenerate with the driver below).

| fig | script | what |
|---|---|---|
| 1 | see `Fig1.md` (`plot1_concept.py` / `fig1_*`) | prompted concept-modulation traces (single concept) |
| 2 | `fig2_engage_suppress.py` | engage/suppress heatmap across depth, one sentence |
| 3 | `fig3_position_engage_suppress.py` | …by absolute token position (length-clipped) |
| 4 | `fig4_fraction_engage_suppress.py` | …by fractional sentence position |
| 5 | `fig5_rank_intensity.py` | rank (signed Spearman vs intensity) by position |
| 6 | `fig6_gain_intensity.py` | ramp gain by position |
| 7 | `fig7_pos_categories.py` | four metrics by POS category (bars + CIs) |
| 8 | `fig8_location_position.py` | positional targeting (beginning/end) profiles |
| 9 | `fig9_type_targeting.py` | type targeting (punctuation/adjectives) by POS |
| 10 | `fig10_persistence.py` | temporal persistence profiles (Fig 10a/10b) |
| 11 | `fig11_localization_metrics.py` | localization metrics (inside gain / leakage / selectivity / center-of-mass) |
| 12 | `fig12_persistence_metrics.py` | persistence timing metrics (onset/offset/persistence/rebound/leakage) — **draft, unpromoted** |
| 13 | `fig13_target_matrix.py` | layer-targeting 8×8 matrices (+ supplementary demeaned); see the "significance dilemma" in `Fig13.md` |

Shared conventions (Figs 2–13): the "signal" is **Δ vs `no_instruction`,
differenced within each (sentence, concept) unit** (cancels the per-concept
offset); averages are over **50 sentences × 10 concepts**; error bars/bands are
**95% bootstrap CIs over units**; significance is a **paired sign-flip permutation
with BH-FDR**. Layers are each channel's peak depth (cos deep, relnorm mid).
`Fig11-12_explained.md` is a plain-language walkthrough of the two metric figures.

**Regenerate the whole suite with the driver** (analysis is fully decoupled from
the experiment code; no model load, reads saved runs):

```bash
python scripts/make_paper_figures.py           # Figs 1-13 (12 pending), ~5 min
python scripts/make_paper_figures.py --only 4,7,13
python scripts/make_paper_figures.py --list
```

> **Status: work in progress.** Figure content, layers, and some interpretations
> (e.g. Fig 9) are still being finalized; Fig 12 is an unpromoted draft (skipped
> by the driver, see the comment in `make_paper_figures.py`). Individual figures
> can still be run via their own scripts, e.g.
> `python scripts/fig7_pos_categories.py --run-dir results/raw/<RUN> --out ...`

---

## Configs

| config | purpose |
|---|---|
| **`experiments/main/config.yaml`** | **THE primary config**: always-on controls + 4 selectable condition sets, deep granular layers (40+), first 50 sentences — see `experiments/README.md` |
| `experiments/intensity_scales/config.yaml` | one-off exploratory ramps (open-ceiling + 1..8); run AFTER the main experiment (controls sourced from it) |
| `experiments/_base.yaml` | shared base (model, concepts, `sentences_file`, compliance) that the main experiment / one-offs `extends:` |
| `configs/*.yaml` (legacy) | historical configs matching past runs (`experiment.yaml`, `experiment_layers*.yaml`, `experiment_layer_target_deep.yaml`, `experiment_attn.yaml`) — kept as records |
| `configs/models/*.yaml` | per-model HF id + dtype |

Override any value from the CLI with `--set dotted.key=value` (e.g.
`--set 'experiment.sets=[intensity]'`, `--set 'experiment.sentence_indices=[0,6]'`,
`--set 'analysis_layers.fractions=[0.75,0.90,1.0]'`).

---

## Repo layout

```
experiments/              THE experiment definitions (see experiments/README.md)
  main/                   primary config: controls + 4 condition sets + analyses
  intensity_scales/       one-off exploratory ramps (needs a main run first)
  token_location/, persistence/   analyze.py register custom analysis kinds
  _base.yaml              shared base (extends:)
configs/                  legacy experiment configs (historical records) + models/
sentences.txt             the 100 baseline sentences (s0..s99; labels = 0-based indices)
exaggerated_phrases.txt   register-rich probes (p0..p6; hello x15, cat family, lists)
pos_tags.json             word-level spaCy POS tags (UPOS+PTB) for both files; model-independent
scripts/
  run_experiment.py       main runner: prompt -> generate+record -> cosine -> save,
                          then dispatches the config's analysis: steps (set-gated)
  builtin_analyses.py     registers controllability_heatmap / trace_plots / layer_targeting
  controllability_heatmap.py   THE controllability suite (per-token heatmaps)
  plot_results.py         per-concept trace plots plot1_cos / plot1_norms
  plot_layer_targeting.py layer-targeting plots 7-12 (also registered as an analysis kind)
  make_paper_figures.py   THE analysis driver: regenerates results/paper/Fig1-13
  fig{2..13}_*.py         paper-figure scripts (no model load) -- see "Paper figures"
  run_analysis.py         cosine summary CSV by (condition, layer)
  run_attn_experiment.py  per-layer attention-mass recording
  analyze_attn.py         attention aggregation + plots
src/
  config.py               YAML + extends + condition_sets + CLI-override -> dataclasses;
                          sentences_file loader; sentence_indices subsetting
  analysis/registry.py    analysis-kind registry: config `analysis:` -> registered fns
  models/{registry,wrapper}.py   short-name->HF id; loading; batched generation+recording
  vectors/extraction.py   concept-vector extraction (baseline-subtracted, cached)
  activation/recorder.py  forward hooks; per-token residual capture with per-row masking
  prompts/builder.py      prompt formatting ({sentence}/{concept}/{layer}/{total},
                          {total}=model layer count) + full-cross trial schedule
  analysis/{alignment,cosine}.py   align recorded window to sentence span; cosine traces
  utils/{env,io,compliance,layers,wandb_utils}.py
```

See also: **`experiments/README.md`** (the main experiment: sets, analysis kinds, indexing),
**`experiments.md`** (every run + the hardware/timing record), and
**`Update_5-14-26.md`** (the register/attention-sink investigation that motivated
the channel split).
