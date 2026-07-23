# Results panel

Small, frozen scoring outputs from the activation-control battery, one
directory per model. Each model was run on the same deterministic battery
(50 neutral sentences x 10 concepts; 8,600 main trials and 8,800
layer-targeting trials) and scored with the frozen chain: projection channel,
linear link, and a two-way sentence-by-concept bootstrap with 2,000 replicates
and seed 0.

Each model directory tracks four derived JSONs:

- `SCORES_<model>.json` — battery point estimates and per-measure intervals.
- `PROFILES_<model>.json` — depth profiles for the scored channels.
- `SCALAR_CI_<model>.json` — frozen scalar configuration, point estimate, and
  joint confidence interval.
- `ONSET_OFFSET_WORD_<model>.json` — word-boundary correction for onset/offset
  timing; it supersedes only the token-boundary timing diagnostic in `SCORES`.

Raw recordings are intentionally not tracked here; each model's section lists
its durable artifact references.

## Reproduce and render

From the repository root after installing `requirements.txt`:

```bash
# Deterministic scalar reduction for one model (S, full precision via --json):
python scripts/aggregate_scalar.py \
  --scores results-panel/qwen35_397b_a17b/SCORES_qwen35_397b_a17b.json

# Cross-model comparison over every model present in the panel:
python scripts/superplot.py --data-root results-panel
```

The first command deterministically reproduces the scalar in the model's
`SCALAR_CI_*.json` (S = 0.7010 for the example above). The second writes
`model_comparison.png`, `null_measures_model_comparison.png`, and
`degenerate_measures_model_comparison.png` under the gitignored `results/`
directory; `superplot.py` discovers both flat `SCORES_*.json` files (the
`results/` layout) and the per-model directories used here.

## Qwen3-235B-A22B-Instruct-2507 (`qwen3_235b_a22b_2507`)

Battery run on 2026-07-23 at model revision
`ac9c66cc9b46af7306746a9250f23d47083d689e`.

The conjunctive controllability scalar is **S = 0.7187** with a joint 95%
confidence interval of **0.6332–0.8233**. Engage reaches **d′ = 5.515** on the
projection channel.

Two advisory caveats matter when interpreting the panel:

- Token-group targeting has a significant projection anti-effect while relative
  norm moves strongly in the opposite direction. This unresolved cross-channel
  sign flip is a diagnostic failure, not evidence of token-group control.
- `loc_end` passes the scorer's fuzzy compliance threshold in 441/500 cases
  (88.2%) but is exact-normalized in only 45/500 cases (9.0%). Temporal results
  therefore apply among fuzzy-compliant rows and remain sensitive to non-verbatim
  completions in this condition.

This is one checkpoint on one fixed battery and does not establish a scaling
law. The Engage interval also reselects the peak layer inside each bootstrap
replicate, so it is panel-comparable rather than a fixed-layer coverage
interval.

Raw recordings for this model:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/production/main/raw/20260723_115400_qwen3_235b_a22b_2507_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/production/lt/raw/20260723_115414_qwen3_235b_a22b_2507_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/scoring/`

Source experiment: `exp_01ky77p2ksea49khssdbchd55y`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.

## Qwen3.5-397B-A17B (`qwen35_397b_a17b`)

Battery run on 2026-07-23 at model revision
`8472618112abcbd45acbcdc58436aff4233c23f7`. The bf16 checkpoint (~807 GB)
exceeded the job's disk, so weights were loaded with a staged shard-streaming
loader validated to produce identical tensors (bitwise-equivalent load on
Qwen3.5-9B; concept vectors bitwise-identical across two independent
production-scale loads of the 397B). The VLM checkpoint is loaded through its
causal-LM path (vision tower and MTP head dropped), identical to stock
transformers behavior for this repo.

The controllability profile substantively holds at 397B-total MoE scale:
**S = 0.7010** with a joint 95% confidence interval of **0.4040–0.8584**
(versus 0.7187, CI 0.6332–0.8233, at 235B). Engage reaches **d′ = 11.28**
(CI 9.438–14.40, peak layer 57) on the projection channel — the strongest
Engage effect in the battery. Instruction compliance was 99.94% on the main
run (8,595/8,600) and 100% on layer targeting (8,800/8,800).

Three caveats matter when interpreting this model's panel entry:

- **The layer-targeting designed null holds only approximately.** Aggregate
  d′ = 0.0535 (CI 0.0169–0.0910), concentrated in the deepest prompted depths:
  layer 57 (96.6% depth) 0.1656 (CI 0.0641–0.2798) and layer 59 (100%) 0.2566
  (CI 0.0068–0.5223), with one weaker mid-deep depth also individually positive
  (layer 48, 81.4%: 0.0550, CI 0.0139–0.0975); the remaining five prompted
  depths' CIs span zero. The magnitude is ~0.5% of the Engage effect and the
  dominant cells coincide with where generic concept engagement naturally peaks
  (the Engage peak is layer 57), consistent with late-depth leakage of generic
  engagement rather than layer-addressable control. S is reported with this
  caveat.
- **Suppress is positive but marginal and depth-localized** — the first
  positive Suppress in the battery: d′ = 0.4292 with the CI barely excluding
  zero (0.0116–1.073), driven by a narrow weak-suppression band near 75% depth,
  while the battery's strongest rebound coexists at 95% depth (−2.453,
  CI −3.395 to −1.754) and 5–60% depths rebound significantly. Treated as
  suggestive, not a confirmed reversal of the panel-wide rebound pattern.
  (Suppress is excluded from S under the frozen 2026-07-17 calibration.)
- **Coverage's CI spans zero** (d′ = 0.9116, CI −0.5900 to 1.954; weakest POS:
  VERB), so coverage is not established at this checkpoint. Coverage enters the
  frozen scalar, which widens the S interval relative to 235B.

The token-group cross-channel sign flip seen at 235B persists here and is
stronger (projection −2.590 vs relative-norm +1.071); it remains an unresolved
diagnostic failure. Size-trend readings rest on two points (235B, 397B) whose
total-parameter scale is confounded with active-parameter count (22B vs 17B)
and family generation; Mistral-Large-3-675B was excluded because its official
checkpoints are not transformers-loadable (vLLM-only; see `models.txt`).

Raw recordings for this model:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7mcvasf5ethwq8d0xxtaer/results/qwen35_397b_a17b/production/main/raw/20260723_151812_qwen35_397b_a17b_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7mcvasf5ethwq8d0xxtaer/results/qwen35_397b_a17b/production/lt/raw/20260723_152801_qwen35_397b_a17b_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky7mcvasf5ethwq8d0xxtaer/results/qwen35_397b_a17b/scoring/`

Source experiment: `exp_01ky7mcvasf5ethwq8d0xxtaer`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.
