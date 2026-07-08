# experiments/

Self-contained, independently-runnable experiments. Each is a folder with a
`config.yaml` (what to generate + which analyses to run) and, only if it needs a
novel analysis, an `analyze.py`. Adding an experiment touches **nothing** in
`src/` or `scripts/` — that's the point.

## The main experiment (primary entry point)

**`experiments/main/config.yaml` is THE experiment config going forward.** It
unifies the previous per-experiment configs into one run with four condition
**sets** plus four **always-on controls**:

| set | conditions | analyses |
|---|---|---|
| `intensity` | `think_intensity_{1..4}_of_4`, `think_intensely` | ramp heatmap (`intensity_1to4`), trace plots |
| `token_location` | `loc_punctuation/adjectives/beginning/end`, `loc_not_beginning/end` | per-condition heatmaps, `location_targeting`, trace plots |
| `persistence` | `persist_first_half/throughout/once/after_fourth` | per-condition heatmaps, `temporal_profile`, trace plots |
| `layer_location` | `think_at_layer`, `think_intensely_at_layer`, `ctrl_think_intensely_at_layer`, `ctrl_think_at_layer` | `layer_targeting` |

The controls — `think_about` (positive), `dont_think_about` (negative),
`no_instruction` (baseline), `ctrl_think_intensely` (control) — live in
`prompt_conditions:` and run exactly **once per run regardless of which sets
are active**; every set's analyses reference them.

```bash
# all four sets (experiment.sets unset = everything active)
python scripts/run_experiment.py --config experiments/main/config.yaml

# just some sets
python scripts/run_experiment.py --config experiments/main/config.yaml \
    --set 'experiment.sets=[intensity,persistence]'
```

**How sets work (config schema).** A config may declare a top-level
`condition_sets:` mapping — set-name → list of conditions (same shape as
`prompt_conditions` entries). `experiment.sets` selects which sets are active:
**absent/null = all declared sets**, explicit `[]` = none (controls only), an
unknown name errors. The final condition list = the top-level
`prompt_conditions` (always-on controls) + each active set's conditions, in
declaration order; duplicate ids across the merged list error. The run's saved
`config.yaml` records `active_sets`.

Analysis steps tagged with `set: <name>` run only when that set is active;
untagged steps always run. For the `layer_location` set,
`prompt_layers.fractions` supplies the `{layer}` values — the same deep
fractions as the recorded sweep (`[0.65..1.00]` → 40, 43, 46, 49, 52, 55, 58, 61
on gemma3-27b), so the full 8×8 prompt-layer × analysis-layer diagonal is
measurable.

**Recorded layers:** the main experiment (and intensity_scales) override `_base`'s full
20-fraction sweep with the **deep granular sweep** — every 5% of depth from 65%
up (`fractions: [0.65..1.00]` → layers 40, 43, 46, 49, 52, 55, 58, 61 on
gemma3-27b). Engagement is negligible before ~layer 37, so the front of the
network is not recorded by default. Trim further per run if disk is tight, e.g.
`--set 'analysis_layers.fractions=[0.75,0.90,1.0]'`.

## The one-off: intensity_scales

`experiments/intensity_scales/config.yaml` is an **exploratory one-off**
(gemma3-27b only, not part of the multi-model main experiment): the open-ceiling
intensity ramp (`think_intensity_open_1..4`) and the 1..8 ramp
(`think_intensity_{1..8}_of_8`) in a single run, with **no control conditions**
of its own. **Run it only after the main experiment** — its analyses source the
control conditions from the main run (in-run engagement/suppression panels
auto-skip; the cross-run comparison against controls happens in later
analysis).

## Run one

```bash
python scripts/run_experiment.py --config experiments/main/config.yaml
```

The runner generates the trials, saves `results.pkl` + `results.json` into a new
`results/raw/<run>/` dir, then runs the analyses declared in that config's
`analysis:` block. Experiments are independent: each writes its own run dir and
never touches another's data.

## Anatomy of an experiment

```
experiments/<name>/
  config.yaml     extends ../_base.yaml; declares prompt_conditions + analysis:
  analyze.py      (optional) registers experiment-specific analysis kinds
```

**config.yaml** inherits the shared model / layers / concepts / sentences from
`_base.yaml` via `extends:`, and overrides only what differs:

```yaml
extends: ../_base.yaml
experiment:
  name: "write_introspection_<name>"
prompt_conditions:
  - id: ...            # this experiment's conditions (+ any reference poles)
analysis:
  - kind: <registered-analysis>
    <params>           # splatted as kwargs into the analysis function
```

**analysis kinds** resolve through `src/analysis/registry.py`:

| kind | source | params (common) |
|---|---|---|
| `controllability_heatmap` | `scripts/builtin_analyses.py` → `generate_heatmaps` | `metrics`, `ramp`, `ramp_name`, `pos_cond`, `neg_cond`, `baseline_cond`, `sentence`, `alpha`; `pos_conds` (list → one heatmap set per condition in `plots/<cond>/`); `subdir` (write under `plots/<subdir>/` — needed when one config runs this step twice, else filenames collide) |
| `trace_plots` | `scripts/builtin_analyses.py` → `make_trace_plots` | `condition_ids`, `sentence`, `layers` |
| `layer_targeting` | `scripts/builtin_analyses.py` | — |
| `location_targeting` | `experiments/token_location/analyze.py` | `baseline_cond`, `sentence`, `layer`, `begin_frac`, `end_frac` |
| `temporal_profile` | `experiments/persistence/analyze.py` | `baseline_cond`, `sentence`, `layer`, `nbins` |

A config with **no** `analysis:` block falls back to the historical default
(cos + relnorm heatmaps + trace plots on the default sentence — s6 "She stacked
the folders...", see `DEFAULT_SENTENCE` in the plot scripts), so the legacy
`configs/*.yaml` behave as before.

## Add a new experiment

1. `mkdir experiments/<name>`, write `config.yaml` with `extends: ../_base.yaml`,
   your `prompt_conditions`, and an `analysis:` block.
2. If the standard analyses suffice (e.g. any intensity/ramp variant — see
   `intensity_scales`), you're done — **no code**.
3. If you need a new analysis, add `analyze.py` and register it:

   ```python
   from src.analysis.registry import register

   @register("my_analysis")
   def my_analysis(*, run_dir, results, model_name, cfg, **params):
       ...  # write plots/CSVs under run_dir/"plots"; best-effort, may raise
   ```

   The runner imports `analyze.py` (if present, next to the config) before
   dispatching, so the `kind` becomes available. See `token_location/analyze.py`
   for the standard shape (load cached vectors, read per-trial activations,
   render a figure + CSV).

## Notes

- **The main experiment supersedes the old per-experiment configs.** The old
  `intensity_open` / `intensity_of_8` configs were REMOVED (their conditions
  live on in `intensity_scales`). `token_location` and `persistence` keep their
  dirs: their `analyze.py` files register the custom kinds the main experiment reuses
  (`location_targeting`, `temporal_profile`) via the shim
  `experiments/main/analyze.py` (the runner only auto-imports the analyze.py
  sitting next to the config being run); their config.yaml files remain usable
  for standalone re-runs.

- **Baseline sentences come from `sentences.txt`.** `_base.yaml` sets
  `sentences_file: sentences.txt` (project root); the loader strips the `sN:`
  labels and reads the text in file order. Edit that file to change the baseline
  set. An explicit `sentences:` list in a config overrides the file. The
  register-rich probes live in `exaggerated_phrases.txt` (`p0..p6`), not sourced
  by default.
  - **Indexing note:** the files are now labelled 0-based — `sentences.txt` as
    `s0..s99` and `exaggerated_phrases.txt` as `p0..p6` — so file labels MATCH
    the 0-based code indices exactly (no more off-by-one).
    `sentence_indices=[0..49]` selects `s0..s49`.
- **POS tags for future analyses.** `pos_tags.json` (project root) holds
  word-level spaCy tags (UPOS + Penn Treebank, with char spans) for every
  sentence in `sentences.txt` AND `exaggerated_phrases.txt`. It is
  **model-independent** (tags attach to words; the per-model word→subword
  alignment happens at analysis time from each run's saved `anchored_token_strs`).
  Regenerate only if the sentence files change. No analysis consumes it yet —
  it is the substrate for the planned POS-grouped readout analysis.
- **Run a subset of sentences.** Set `experiment.sentence_indices` (0-based global
  indices into the sentence list) to run only those, e.g.
  `--set 'experiment.sentence_indices=[0,5,9]'`. The full list is still saved to
  the run's `config.yaml`, so plot labels stay global even in a subset run. Empty
  (default) runs all sentences.
- **Layers.** `_base.yaml` declares the full every-5%-of-depth sweep (20
  fractions), but the main experiment (and intensity_scales) **override it with the deep
  sweep** (`0.65..1.00` → layers 40..61 on gemma3-27b; see the main experiment section
  above). All resolved layers are already in `results/vector_cache/` for these
  concepts + `method: baseline`, so **no concept vectors are re-extracted**.
- **Custom analyses pin layer 55** (where engagement saturates); the built-in
  `controllability_heatmap` renders all recorded layers as its x-axis
  automatically.
- **Reference poles.** The intensity experiments carry `think_about` /
  `dont_think_about` / `no_instruction` so the engagement/suppression panels are
  self-contained. Those few conditions get re-generated per experiment (a small
  cost for full independence). To skip them, drop the poles — the ramp measures
  (Rank / gain) still compute; only the engage/suppress panels are
  omitted.
- **Analyses are re-runnable standalone** on a finished run dir, CPU-only, since
  they read only `results.pkl` + the vector cache. To re-render without
  regenerating, call the underlying script (`controllability_heatmap.py`,
  `plot_results.py`) or import the registered function directly.
