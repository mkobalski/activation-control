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
       └─ superplot.py         model_comparison.png + null_measures_...png   (cross-model)
```

Each model is run twice: a **main** run (all condition sets) and a **layer-targeting**
run (`experiment.sets=[layer_location]`, auto-tagged `_lt`). The scoring reads only the
cheap stored artifacts (`results.json` + the small `no_instruction_cache.pkl` + concept
vectors) — never the large `results.pkl` of raw activations.

## Layout

| path | what |
|---|---|
| `src/` | the experiment engine (model loading, prompting, recording, concept vectors) |
| `experiments/` , `configs/` | experiment specifications (conditions, layers, models) |
| `scripts/run_experiment.py` | the runner — pure generate-and-save |
| `scripts/{compute_scores,aggregate_scalar,scalar_ci}.py` | the scoring layer |
| `scripts/{explore,superplot,figstyle}.py` | figures |
| `scripts/{run.sh,postprocess.py}` | the orchestrator |
| `results/raw/<run>/` | one run's recordings (gitignored) |
| `results/*.json` | the derived SCORES / PROFILES / SCALAR / CI artifacts (gitignored) |

The paper's publication figures live in the **paper repo**
(`activation-controllability/figure_scripts/`); they read these JSONs via
`--data-root` / `$AC_DATA` and are not shipped here.

## Running

```bash
pip install -r requirements.txt
export AC_DATA=$PWD/results          # where the derived JSONs live

# one model (main run, then its layer-targeting run):
scripts/run.sh --config experiments/main/config.yaml --overrides model.name=<model>
scripts/run.sh --config experiments/main/config.yaml --overrides model.name=<model> experiment.sets='[layer_location]'
```

Scoring/figure steps are also runnable standalone (see `--help` on each; `METRICS`
§8). Selectable knobs on the collapse: `--channels {projection,legacy,all}` and
`--link {linear,phi}` (defaults `projection`, `linear`).

## Notes

- σ in every d′ is **across-sentence baseline variability**, not measurement noise —
  generation is deterministic. "d′ = 6" means 6 sentence-to-sentence SDs, not 6σ of
  precision.
- `D_REF` and the measure set are the scalar's calibration choices; they're dated and
  documented in `METRICS` and recorded in each emitted JSON, and are meant to be
  revisited as the model panel grows.
