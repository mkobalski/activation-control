# activation-control

Measuring how much **intrinsic control** an instruction-tuned LLM has over its own
activation space — using only natural-language instructions, no training or steering
vectors. For a fixed sentence and a concept X, the model is told to *think about X*,
*don't think about X*, *think about X at intensity k/4*, *think about X only at the
end*, and so on, while we record the residual stream token-by-token. From those
recordings a battery of measures asks how reliably each instruction moves X's
representation, and a single conjunctive scalar **S ∈ [0, 1]** summarises them.

## Design in one screen

- **Readout: projection.** Everything is scored on the projection of the residual
  stream onto the (unit-normalised) concept vector, `⟨r, ĉ⟩ = ‖r‖·cos`. `cos` and
  `relnorm` are still emitted for the appendix but the paper uses `proj`.
- **The scalar S** is a measure-equal, **conjunctive** (geometric-mean) composite of
  **five** capability measures — Engage, Dial Rank, Dial Resolution, Temporal
  control, Coverage — each mapped to `[0, 1]` by a **linear-clip link** against a
  per-measure ceiling `D_REF` (the score meaning "essentially perfect control"), then
  `S = clip(2G − 1, 0, 1)`. It is *absolute* (a model's S doesn't depend on the panel)
  and rewards broad competence over any single strong axis.
- **Reported but excluded from S** (diagnostics, not capabilities): Suppress (a
  white-bear rebound, not a suppression ability), Layer targeting (a designed null),
  Onset/offset error, Token group.
- **The maths and every calibration constant** (links, `D_REF`, exclusions, CIs) live
  in **`METRICS_2026-07-16.md`** — read that for anything about *what a number means*.

## Pipeline

```
run.sh <config>
  └─ run_experiment.py         generate + save residual-stream recordings  (needs a GPU)
  └─ postprocess.py            score + figures, on completion (CPU-only):
       ├─ compute_scores.py    SCORES_<model>.json + PROFILES_<model>.json  (the battery)
       ├─ scalar_ci.py         SCALAR_CI_<model>.json                        (S + joint CI)
       ├─ explore.py           per-run exploratory figures                   (in the run dir)
       └─ superplot.py         model_comparison / null_measures / degenerate_measures .png (cross-model)
```

Each model is run twice: a **main** run (the `intensity`, `token_location` and
`persistence` sets — 8600 trials) and a **layer-targeting** run
(`experiment.sets=[layer_location]`, auto-tagged `_lt` — 8800 trials). The two are
disjoint: `layer_location` must NOT run inside the main run. It is scored from the
separate `_lt` dir, so a main run that also generates it just computes it twice and
throws one copy away. `experiments/main/config.yaml` pins `sets:` accordingly — do not
delete that line to "run everything". The scoring reads only the
cheap stored artifacts (`results.json` + the small `no_instruction_cache.pkl` + concept
vectors) — never the large `results.pkl` of raw activations.

### Control under task load

`control-under-load/` is a self-contained, analysis-ready release of the final
Gemma-3-27B polynomial-load study: 800 problems, 25,600 generated answers, 768,000
layerwise concept projections, the earlier v3 comparison, and four reproducible plots.
The final slope is −0.02545 d′ per ordered difficulty bin with a 95% interval of
[−0.04276, +0.01093]; the interval includes zero and narrowly misses the registered
precision target. See `control-under-load/README.md` for the score definition, table
schemas, provenance, and limitations.

```bash
python scripts/control_under_load.py verify   # checks SHA-256, schemas, counts, and frozen values
python scripts/control_under_load.py plots    # remakes all four SVG/PNG figure bundles
python scripts/control_under_load.py all      # also reruns the 2,000-draw CPU analysis
```

## Layout

| path | what |
|---|---|
| `src/` | the experiment engine (model loading, prompting, recording, concept vectors) |
| `experiments/` , `configs/` | experiment specifications (conditions, layers, models) |
| `scripts/run_experiment.py` | the runner — pure generate-and-save |
| `scripts/{compute_scores,aggregate_scalar,scalar_ci}.py` | the scoring layer |
| `scripts/onset_offset_word.py` | word-based onset/offset supersession (`ONSET_OFFSET_WORD_<model>.json`); run manually, not part of `postprocess` |
| `scripts/{explore,superplot,figstyle,model_family_colors}.py` | figures (`model_family_colors` = the shared model-family palette, a verbatim copy of the paper repo's) |
| `scripts/{run.sh,postprocess.py}` | the orchestrator |
| `src/control_under_load/` | Parquet loading, exact polynomial grading, frozen d′/bootstrap analysis, and figure construction |
| `control-under-load/` | shipped final/comparison data, SHA-256 manifest, documentation, and four figure bundles |
| `scripts/control_under_load.py` | verify, deterministic CPU reanalysis, frozen-output comparison, and plot regeneration |
| `results/raw/<run>/` | one run's recordings (gitignored) |
| `results/*.json` | the derived SCORES / PROFILES / SCALAR / CI artifacts (gitignored) |

The paper's publication figures live in the **paper repo**
(`activation-controllability/figure_scripts/`); they read these JSONs via
`--data-root` / `$AC_DATA` and are not shipped here.

## Running

```bash
pip install -r requirements.txt
export AC_DATA=$PWD/results          # where the derived JSONs live

# one model (main run, then its layer-targeting run). The per-model YAML pins the
# model; the main run needs no overrides, the LT run needs only the set selector.
scripts/run.sh --config experiments/main/<model>.yaml
scripts/run.sh --config experiments/main/<model>.yaml --set 'experiment.sets=[layer_location]'
```

The override flag is `--set` (repeatable, `dotted.key=value`). Sanity-check a run by
its trial count before letting it finish: **8600** for main, **8800** for `_lt`. Anything
else means the set selection is wrong — 17400 is the classic one (all four sets active).

Scoring/figure steps are also runnable standalone (see `--help` on each; `METRICS`
§8). Selectable knobs on the collapse: `--channels {projection,legacy,all}` and
`--link {linear,phi}` (defaults `projection`, `linear`).

### Adding a model

1. `src/models/registry.py` — short name -> HF id in `MODEL_NAME_MAP`, plus any quirk
   set that applies (`MODELS_WITHOUT_SYSTEM_ROLE`, `BASE_MODELS`, and exactly one of
   `HARMONY_MODELS` / `THINK_TAG_MODELS` if it emits a reasoning trace).
2. `experiments/main/<short>.yaml` — `extends: config.yaml` + the `model:` block.
   `configs/models/<short>.yaml` is documentation only; nothing loads it.
3. `scripts/superplot.py` — add to `MODELS` / `DETAIL` (and `FAMILY_ORDER` for a new
   family) or it is scored but invisible in the comparison. Family colors come from
   `scripts/model_family_colors.py` (add the family there too if it is new).

Then run the two commands above and check the trial counts. `models.txt` records the
per-model caveats (thinking toggles, missing chat templates) — read its header first.
Two traps the loader now handles automatically, both worth knowing about:
**multimodal checkpoints** keep their chat template in `chat_template.json` on the
processor, so the tokenizer reports none and prompts would silently fall back to a
base-model `User:/Assistant:` scaffold; and they are registered only under
image-text-to-text, so `AutoModelForCausalLM` cannot load them. `ModelWrapper` detects
both (`_ensure_chat_template`, the auto-class probe) and logs when it does.

## Notes

- σ in every d′ is **across-sentence baseline variability**, not measurement noise —
  generation is deterministic. "d′ = 6" means 6 sentence-to-sentence SDs, not 6σ of
  precision.
- **Every CI and figure band is a joint two-way cluster bootstrap over sentences AND
  concepts** (`scalar_ci.joint_bootstrap`, `compute_scores.dprime_stats`,
  `explore._cluster_band` — one shared resample, so covariance between measures
  survives into the conjunctive S). Because generation is deterministic there is no
  measurement noise to report: the only meaningful uncertainty is whether a result
  would survive a *different sample of stimuli*, so the resampled unit is one
  (sentence × concept) pair. Pooling those ~500 units as if independent understates
  the spread by up to ~4x — concept is the larger cluster, and its share grows toward
  the end of the sentence. With only 10 concepts that axis is coarse; treat CI
  endpoints as approximate.
- `D_REF` and the measure set are the scalar's calibration choices; they're dated and
  documented in `METRICS` and recorded in each emitted JSON, and are meant to be
  revisited as the model panel grows.
- **Onset/offset error — word-based supersession (2026-07-22).** The shipped
  onset/offset error in `SCORES_<model>.json` (`compute_scores._persistence_edges`)
  scores the `persist_after_fourth` ONSET gate against the **4th token** (requested
  onset `4/(n-1)`). But the instruction is worded "after the fourth **word**", and
  because `anchored_token_strs` are `tokenizer.decode()`d, a word boundary is a plain
  leading space for every tokenizer — yet tokenizers that split words into multiple
  sub-tokens (e.g. GPT-OSS, Mistral) place the 4th-word boundary a token or two after
  the 4th token. `scripts/onset_offset_word.py` recomputes the onset gate against the
  actual 4th-word boundary (requested onset = mean fractional position of the
  5th-word-start token) and writes `ONSET_OFFSET_WORD_<model>.json`. The **original
  score is left untouched** (frozen; still what the paper's prior numbers cite); the
  word-based file supersedes it for the temporal figures. Because `req` enters
  `_persistence_edges` only as a constant subtraction (`error = detected − req`), the
  word-based onset is an exact constant shift of the token-based onset — the offset
  gate ("first half" = 0.5) and all detection/bootstrap are unchanged — so the script
  reuses `_persistence_edges` verbatim and asserts its replicated unit set matches.
  Conditions are unchanged: onset = `persist_after_fourth`, offset = `persist_first_half`.
