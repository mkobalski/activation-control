# activation-control

Code and derived data for **Measuring Activation Control in Large Language
Models**. The experiment has an instruction-tuned model transcribe a fixed
sentence while following an embedded instruction about a concept — *think about
X*, *do not think about X*, *think about X at intensity k/4*, *think about X only
at the end* — and records the residual stream token by token. A battery of
measures scores how reliably each instruction moves that concept's
representation, and collapses them into one scalar `S ∈ [0, 1]`.

## Layout

| path | what |
|---|---|
| `src/` | the experiment engine: model loading, prompting, recording, concept vectors |
| `src/control_under_load/` | Parquet loading, polynomial grading, and the frozen analysis for the task-load study |
| `experiments/` | experiment specifications; `experiments/main/config.yaml` is the protocol, `experiments/main/<model>.yaml` pins the model |
| `scripts/run_experiment.py` | the runner — pure generate-and-save |
| `scripts/{score_data,compute_scores,aggregate_scalar,scalar_ci}.py` | the scoring layer |
| `scripts/onset_offset_word.py` | word-boundary onset/offset supersession |
| `scripts/{explore,superplot}.py` | per-run and cross-model figures; `scripts/figstyle.py` and `scripts/model_family_colors.py` are their shared style |
| `scripts/{run.sh,postprocess.py}` | the orchestrator |
| `scripts/{_launch.sh,dl_qwen35.sh}` | venv-isolating launcher, and a resumable weight pre-download |
| `scripts/control_under_load.py` | verify, reanalyse, and plot the task-load study |
| `figure_scripts/` | the paper's publication figures |
| `control-under-load/` | shipped task-load data, SHA-256 manifest, and figure bundles |
| `results/*.json` | derived scoring artifacts (tracked — the figure inputs) |
| `results/raw/<run>/`, `results/vector_cache/` | recordings and concept vectors (gitignored, multi-TB) |
| `exploratory/<model>/` | the per-run exploratory figures, rendered for 20 models |
| `compliance/` | the per-model compliance table, its source JSON, and the script that rebuilds them |
| `tests/` | the pytest suite: the task-load study, plus a guard on the LT-set invariant |
| `sentences.txt`, `pos_tags.json` | the 50 transcription sentences and their POS tagging |
| `models.txt` | per-model caveats: thinking toggles, quantization, gated repos, hardware |
| `extra_sentences.txt`, `exaggerated_phrases.txt` | unused spares, and the register-rich probes used only in exploratory runs |

## Pipeline

```
scripts/run.sh <config>
  ├─ run_experiment.py        generate + record            (needs a GPU)
  └─ postprocess.py           score + figures, on success  (CPU only)
       ├─ compute_scores.py      SCORES_<model>.json, PROFILES_<model>.json
       ├─ scalar_ci.py           SCALAR_CI_<model>.json
       ├─ onset_offset_word.py   ONSET_OFFSET_WORD_<model>.json
       ├─ explore.py             per-run figures, into the run directory
       └─ superplot.py           cross-model comparison figures
```

Each model needs **two runs**: a main run (the `intensity`, `token_location` and
`persistence` condition sets, **8,600** trials) and a layer-targeting run
(`layer_location` alone, auto-tagged `_lt`, **8,800** trials). They must stay
disjoint — `experiments/main/config.yaml` pins `sets:` accordingly, and Layer
Targeting is scored from the `_lt` directory. Any other trial count means the set
selection is wrong.

Scoring reads only the cheap artifacts a run leaves behind — `results.json`,
`no_instruction_cache.pkl`, and the cached concept vectors — never the
multi-gigabyte `results.pkl` of raw activations. No model is loaded anywhere in
the scoring or figure path.

## Running

```bash
pip install -r requirements.txt

# one model: the main run, then its layer-targeting run
scripts/run.sh --config experiments/main/<model>.yaml
scripts/run.sh --config experiments/main/<model>.yaml --set 'experiment.sets=[layer_location]'
```

`--set` is repeatable and takes `dotted.key=value`. The interpreter is
`$AC_PYTHON`, else an active virtualenv, else `python3`; `$AC_ENV_FILE` names an
optional env file for `HF_TOKEN` and friends.

The scoring steps also run standalone:

```bash
python scripts/compute_scores.py --main-run <RUN> [--lt-run <LT_RUN>] \
    --json results/SCORES_<model>.json --profiles-json results/PROFILES_<model>.json
python scripts/scalar_ci.py --main-run <RUN> --json results/SCALAR_CI_<model>.json
python scripts/onset_offset_word.py --models <model>
python scripts/aggregate_scalar.py --json results/SCALAR.json   # all models at once
```

The collapse takes `--channels {projection,legacy,all}` and `--link
{linear,phi}`, defaulting to `projection` and `linear`; both are recorded in the
emitted JSON, along with the `D_REF` constants from `aggregate_scalar.D_REF`.

### Adding a model

1. `src/models/registry.py` — short name to HF id in `MODEL_NAME_MAP`, plus any
   quirk set that applies: `MODELS_WITHOUT_SYSTEM_ROLE`, `BASE_MODELS`, and at
   most one of `HARMONY_MODELS` / `THINK_TAG_MODELS` if it emits a reasoning
   trace.
2. `experiments/main/<short>.yaml` — `extends: config.yaml` plus the `model:`
   block. This is the only per-model config.
3. `figure_scripts/roster.py` — add to `MODELS` (or `MOE`, for a large model
   scored from stored SCORES with no retained raw run) and `DETAIL`, plus
   `FAMILY_ORDER` for a new family. This single roster feeds every paper figure
   and `scripts/superplot.py`. To make your model the focal one for the
   single-model illustrative panels, set `AC_FOCAL=<short>` (or pass a script's
   `--model`).

Then run the two commands above and check the trial counts. Multimodal
checkpoints are handled automatically: `ModelWrapper` recovers a chat template
from `chat_template.json` when the tokenizer reports none, and falls back to
`AutoModelForImageTextToText` for architectures `AutoModelForCausalLM` cannot
load. Both are logged when they trigger.

## Emitted artifacts

One set per model, in `results/`:

| file | contents |
|---|---|
| `SCORES_<model>.json` | every measure, on all three readout channels, with per-measure intervals |
| `PROFILES_<model>.json` | depth-profile curves for the engage/suppress/gain/rank panels |
| `SCALAR_CI_<model>.json` | `S`, its joint confidence interval, and the per-component breakdown |
| `ONSET_OFFSET_WORD_<model>.json` | the onset/offset timing diagnostic, scored against the 4th-word boundary |
| `FIGDATA_<model>.json` | frozen focal-panel figure inputs: the layer-targeting grid and the temporal-control/precision positional profiles (so those figures render from a clone) |

The measure keys in `SCORES_<model>.json`:

| key | paper name | conditions contrasted | in `S` |
|---|---|---|---|
| `engage` | Engage | `think_about` vs `no_instruction` | yes |
| `suppress` | Suppress | `dont_think_about` vs `no_instruction` | yes |
| `dial_rank` | Dial Rank | `think_intensity_{1..4}_of_4` | yes |
| `temporal_control` | Temporal Control | `loc_beginning`, `persist_once`, `loc_end` vs `think_about` | yes |
| `coverage` | Coverage | `think_about` vs `no_instruction`, split by POS category | yes |
| `layer_targeting` | Layer Targeting | `think_at_layer`, from the `_lt` run | yes |
| `dial_resolution`, `dial_resolution_pool` | — | the same intensity ramp, adjacent pairs | no |
| `token_group` | Token Group | `loc_punctuation`, `loc_adjectives` vs `think_about` | no |
| `onset_offset_error` | — | `persist_after_fourth`, `persist_first_half` | no |

Each measure carries all three readout channels — `proj` (the paper's),
`cos`, `relnorm` — and the channel set is chosen at collapse time. Intervals are
two-way cluster bootstraps over sentences and concepts; `scalar_ci.py` draws a
single shared resample so the interval on `S` reflects the covariance between
measures.

### Models without retained recordings

Five models — `qwen3_coder_480b`, `qwen35_397b_a17b`, `llama4_maverick`,
`glm52`, `qwen3_235b_a22b_2507` — were scored from recordings that were not
kept. Their `SCALAR_CI_<model>.json` therefore carries `point_estimate: true`
with `ci_lo` and `ci_hi` null, and `scalar_ci.py` cannot be rerun for them.
Per-measure intervals in `SCORES_<model>.json` are unaffected.

The five Olmo 3.1 32B training snapshots (`olmo31_32b_s1_328k`, `_s1_final`,
`_base`, `_sft`, `_dpo`) have a registry entry and a config, but their upstream
checkpoints are not retrievable as of 2026-08-20, so they cannot be re-run from
scratch. Their scoring JSONs are the surviving record. The 7B snapshots are
unaffected — `scripts/olmo_snapshot_lane.sh` still resolves those.

## Per-model exploratory figures

`scripts/explore.py` renders a per-run capability glance, written into the run's
own directory by `postprocess.py`. These are working figures, not the paper's:
they keep titles, axis labels and legends. A rendered set is committed under
`exploratory/<model>/` for the 20 models whose recordings were retained; the
five largest have no raw to render from.

| figure | what it shows |
|---|---|
| `raw_trace_example` | per-token projection for one sentence and concept, across the instruction conditions |
| `channels_depth` | engage and suppress sensitivity vs depth, one panel per readout channel |
| `intensity_rank` | how well the readout tracks the instructed intensity order, vs depth |
| `pos_coverage` | engage and suppress sensitivity by part-of-speech category |
| `temporal_control` | where in the sentence the concept lands under the three region instructions |
| `temporal_precision` | how sharply the concept switches on and off at a commanded edge |
| `engage_heatmap` | engagement over tokens x layers, on all three channels |

These use conventional d': positive means the readout rose under the
instruction, so suppression sits at or above zero where it rebounds. That is the
convention in the paper's depth and part-of-speech figures. Note that the
Suppress score in the paper's Table 1 and in Figures 3 and 13 is signed in the
*instructed* direction, so positive there means pushed **below** baseline — which
is why Gemma 4 31B reads as the strongest suppressor at +1.45.

```bash
python scripts/explore.py --run-dir results/raw/<RUN>          # into <RUN>/figures/
python scripts/explore.py --run-dir <RUN> --out <DIR> --only channels_depth,pos_coverage
```

## Regenerating the paper's figures

`figure_scripts/` renders thirteen of the paper's nineteen figures. Of the other
six, one comes from `scripts/superplot.py` (the full cross-model panel) and two
from the `control-under-load` bundle (see below). Three are not produced here: the
hero schematic and the two monitor-evasion figures (see *Not included*). Render
the thirteen with one command, which reports what rendered, what was skipped, and
what failed:

```bash
python figure_scripts/render_all.py --data-root results --out-dir /tmp/figs
```

**Eleven of the thirteen read only the committed derived JSONs** (`SCORES_`,
`PROFILES_`, `SCALAR_CI_`, `ONSET_OFFSET_WORD_`, `FIGDATA_`), so they rebuild from
a clone with no GPU and no raw data.

**Two still need raw** — `think_intensity` (single-trial per-token traces) and
`token_coverage` (per-POS readouts averaged across models). Those quantities are
per-trial, not aggregates, so they are not frozen to JSON; the scripts read each
model's `raw/` run and `vector_cache/`, which are not shipped. When the raw is
absent they print a one-line notice and are reported **skipped** (they exit
`paths.SKIP_EXIT`, not an error), so `render_all.py` still exits 0 and rebuilds the
other eleven. Regenerate the raw for those two with `scripts/run.sh` (see
*Running*), or point `--data-root` at your own run store.

The focal-panel figures `temporal_control`, `temporal_precision` and
`layer_targeting` read their per-model aggregates from `FIGDATA_<model>.json`, which
`postprocess.py` freezes during scoring via `scripts/figdata.py` — so they, too,
rebuild from a clone. `engage_suppress.py` and `roster.py` emit no figure; they are
the loader and the shared model roster the others import.

Paths resolve relative to the repository and are overridable with `AC_ROOT`,
`AC_DATA` and `AC_FIG_OUT`; the focal model is `AC_FOCAL` (or a script's `--model`).
See `figure_scripts/paths.py` and `figure_scripts/roster.py`.

### Not included

The monitor-evasion pipelines behind the paper's probe, Jacobian-lens,
activation-oracle and natural language autoencoder figures were run outside this
repository, as recorded in the reproducibility checklist. Everything else —
recording, scoring, the scalar, and the figures — is here.

## Control under task load

`control-under-load/` is a self-contained release of the Gemma 3 27B
polynomial-load study: 800 problems, 25,600 answers, 768,000 layerwise
projections, the earlier comparison run, and four figure bundles. See its own
README for table schemas and provenance.

```bash
python scripts/control_under_load.py verify   # hashes, schemas, counts, frozen values
python scripts/control_under_load.py plots    # remake the figure bundles
python scripts/control_under_load.py all      # also rerun the 2,000-draw analysis
```

## Tests

```bash
pip install -r requirements.txt      # pytest, plotly and pyarrow are in there
python -m pytest -q tests
```

Runs in a few seconds, CPU only, no raw data needed. Four files cover the
task-load study: table loading and schema validation plus the frozen analysis
(`test_analysis.py`), the polynomial grader (`test_grader.py`), the figure
bundles (`test_plotting.py`), and the CLI (`test_cli.py`). The fifth,
`tests/test_lt_sets_in_sync.py`, guards a cross-file invariant that comments
alone cannot: `run_experiment.LT_ONLY_SETS` and the `standalone_sets` in
`experiments/main/config.yaml` must name the same condition set. If they drift,
a layer-targeting run silently generates the wrong trial count under the right
name — the failure the trial-count check in *Pipeline* is there to catch, one
step too late.

## Paper

**Measuring Activation Control in Large Language Models.** Marek Mateusz
Kowalski, Joshua Fonseca Rivera, Uzay Macar, David Demitri Africa.

```bibtex
@misc{kowalski2027activation,
  title         = {Measuring Activation Control in Large Language Models},
  author        = {Kowalski, Marek Mateusz and Fonseca Rivera, Joshua and
                   Macar, Uzay and Africa, David Demitri},
  year          = {2027},
  eprint        = {arXiv:XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
}
```

## License

MIT — see `LICENSE`.
