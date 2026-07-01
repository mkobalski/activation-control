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
| **Cohen's d** | standardized ramp gain (`meanΔ / sdΔ`) — big *and* consistent across concepts | unbounded | **yes** (ratio cancels units) |
| **engagement / suppression** | two-panel: each condition minus the neutral baseline (`think − neutral`, `dont − neutral`); red = above neutral, blue = below | signed | **relnorm: yes**; cos: partial |

`Rank` has a **`_specific`** variant on the direction channel that subtracts the
mean off-concept readout, isolating control of *this* concept from a generic
"think harder about everything" effect. The magnitude channel is concept-blind
(no `_specific`).

The two magnitude readouts differ in comparability:
- **`relnorm`** (÷ trial's content-mean norm) is a *dimensionless ratio*, so its
  gain `Δrelnorm` cancels both the model's norm scale *and* hidden-dimension — it
  is **fully cross-model comparable**, so `gain` is reported.
- **`norm`** (raw `‖r‖`) is in model-specific units, so `gain` is **suppressed**
  for it — only the scale-invariant `Cohen's d` and `Rank` are kept.

On the direction side, `Δcos` is invariant to the norm scale but *drifts with
hidden dimension* (cosines shrink ~1/√d), so cosine gains are only "comparable
across models of similar width"; the fully comparable cosine options are the
rank measures and the ratio-form Cohen's d.

**Gain vs consistency are different axes.** A token can move a lot but
inconsistently across concepts (high `gain`, low `Cohen's d` — e.g. some register
slots) or move a little but very consistently (low `gain`, high `Cohen's d`).
`gain` gives the interpretable magnitude; `Cohen's d` gives the standardized,
cross-model-comparable signal-to-noise; read them together.

---

## Install

One machine is enough: a **GPU** generates the data (loads the model, runs the
experiments) and the **analysis** reads only saved tensors (no model, no GPU),
so it runs in the same environment. You *can* optionally move the analysis to a
cheaper CPU-only box later — the minimal install below is for that case.
`pip install -r requirements.txt` on the GPU box covers both.

```bash
# --- GPU box (generate data): full deps + CUDA torch ---
pip install -r requirements.txt          # torch here pulls the CUDA build

# --- CPU-only box (analysis): minimal deps ---
pip install numpy matplotlib pyyaml      # + a CPU build of torch (to load the
pip install torch --index-url https://download.pytorch.org/whl/cpu   # .pt vectors)
# (add pandas only if you also run run_analysis.py)
```

`bitsandbytes` (quantization) and `wandb` (logging) are optional — skip them for
a lean install; pass `--no-wandb` to the runner if wandb isn't present.

Then create a `.env` with `HF_TOKEN=...` (and optional `WANDB_API_KEY=...`).

## Quick start

```bash
# 1-2. Install + .env (see above)

# 3. Generate data (GPU). This ALSO auto-writes, into results/raw/<RUN_DIR>/plots/:
#      - the controllability heatmaps (cos + relnorm), and
#      - the per-concept trace plots (plot1_cos, plot1_norms).
python scripts/run_experiment.py --config configs/experiment_layers.yaml

# 4. (Optional) re-render standalone -- a different sentence/layer/channel, or on
#    a CPU-only box. No GPU / model load needed.
python scripts/controllability_heatmap.py \
    --run-dir results/raw/<RUN_DIR> --model-name gemma3_27b --metric cos
python scripts/plot_results.py --run-dir results/raw/<RUN_DIR> --layers 55 61

# 5. (Optional) cosine summary CSV
python scripts/run_analysis.py --run-dir results/raw/<RUN_DIR>

# 6. (Layer-targeting runs ONLY) the plot7-12 target-layer diagonal figures
python scripts/plot_layer_targeting.py --run-dir results/raw/<RUN_DIR>
```

Steps 3's plots come out automatically. Re-running the analysis scripts reads
only the saved `results.pkl` + cached concept vectors — **no GPU and no model
load** — so they also run on a CPU-only box.

Each run's `plots/` then contains:
- **controllability heatmaps**, per channel (`cos`, `relnorm`):
  `heatmap_Rank_raw_*`, `heatmap_Rank_specific_*` (cos only), `heatmap_gain_*`,
  `heatmap_cohensd_*`, `heatmap_engage_suppress_*`, + `controllability_heatmap_<metric>.csv`.
- **trace plots**: `plot1_cos_L<layer>_s<idx>.png`, `plot1_norms_L<layer>_s<idx>.png`.
- **layer-targeting** (only if you run `plot_layer_targeting.py` on a
  layer-targeting run): `plot7*`–`plot12*`.

### Environment note

If your venv's `python` is not the interpreter whose site-packages hold the
deps, call the interpreter directly and set `PYTHONPATH`, e.g.
`PYTHONPATH=<venv>/lib/python3.11/site-packages /usr/bin/python3.11 scripts/...`.

---

## Configs

| config | purpose |
|---|---|
| `experiment.yaml` | main config: gemma3-27b, fractional analysis + prompt layers, all 14 conditions |
| `experiment_layers.yaml` | the dataset config: explicit deep layers `[40,45,55]`, layer-targeting off |
| `experiment_layers_granular.yaml` | granular depth sweep: every 5% of depth (20 layers) — see size note in the file |
| `experiment_layer_target_deep.yaml` | layer-targeting study: prompts name every layer 55–61 to test per-layer concentration |
| `experiment_attn.yaml` | attention-mass experiment (register/sink test; uses `run_attn_experiment.py`) |
| `configs/models/*.yaml` | per-model HF id + dtype |

Override any value from the CLI with `--set dotted.key=value`.

---

## Repo layout

```
configs/                  experiment + model configs (above)
scripts/
  run_experiment.py       main runner: prompt -> generate+record -> cosine -> save
                          (auto-runs the heatmaps + trace plots at the end)
  controllability_heatmap.py   THE controllability suite (per-token heatmaps; auto-run)
  plot_results.py         per-concept trace plots plot1_cos / plot1_norms (auto-run)
  plot_layer_targeting.py layer-targeting plots 7-12 (manual; layer-targeting runs only)
  controllability.py      controllability aggregated by (layer, token-class, concept)
  run_analysis.py         cosine summary CSV by (condition, layer)
  run_attn_experiment.py  per-layer attention-mass recording
  analyze_attn.py         attention aggregation + plots
src/
  config.py               YAML + CLI-override -> dataclasses
  models/{registry,wrapper}.py   short-name->HF id; loading; batched generation+recording
  vectors/extraction.py   concept-vector extraction (baseline-subtracted, cached)
  activation/recorder.py  forward hooks; per-token residual capture with per-row masking
  prompts/builder.py      prompt formatting + full-cross trial schedule
  analysis/{alignment,cosine}.py   align recorded window to sentence span; cosine traces
  utils/{env,io,compliance,layers,wandb_utils}.py
```

See also: **`experiments.md`** (every run + the hardware/timing record),
**`sentences.txt`** (the 12 sentences), and **`Update_5-14-26.md`** (the
register/attention-sink investigation that motivated the channel split).

---

## What we've found so far (gemma3-27b, 20-layer depth sweep)

- **Engagement is deep and emergent.** "Think about X" pushes the residual
  significantly *toward* X (above the no-instruction baseline), but only in the
  back third of the network — negligible before ~layer 37, then rising sharply
  and saturating (all tokens) around layers 52–58.
- **Engagement is not symmetric with suppression.** Directionally, "don't think
  about X" is **indistinguishable from neutral at every depth** — the model
  engages but does not rotate the residual *away* from the concept.
- **Suppression lives in the magnitude channel.** "Don't think" *does* move the
  residual **norm** (below neutral at deep layers) even though it doesn't move the
  direction — a dissociation the cosine-only view hides.
- **Register slots are high-gain, low-fidelity.** Structural tokens (`the`, the
  period) show the largest cosine swing with intensity but the *least reliable*
  per-trial ordering; content tokens and the period are where the concept can be
  *dialed* cleanly.
- **Graded direction control is consistent; graded magnitude control is not.**
  `Rank(cos)` is reliably positive at the deep layers (the concept-cosine climbs
  monotonically with intensity). `Rank(relnorm)` is near zero / mixed-sign — the
  magnitude readout dials *up* with intensity at some spots and *down* at others
  — so "reliable graded control of magnitude" is not supported.

## Credits

Concept-vector extraction + Gemma rotary patch adapted from
*introspection-master* (Experiment 2); layer-targeted prompting + fractional-depth
layer selection adapted from *think-hard* (Experiment 1).
