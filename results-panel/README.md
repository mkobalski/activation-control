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
```

This reduces the stored per-measure scores to S under the repository's current
scalar definition. It will **not** match the `S` recorded in this directory's
`SCALAR_CI_*.json`: those files were produced separately, under an earlier
scalar definition, and are quoted as-is by the per-model sections below.

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

## The completed large panel (>150B, 2026-07-23)

With the three runs below, the large end of the panel has five models, all
scored under the same frozen 2026-07-17 calibration:

| Model | Short name | S | Joint 95% CI |
|---|---|---|---|
| Qwen3-Coder-480B-A35B-Instruct | `qwen3_coder_480b` | **0.8311** | 0.7391–0.9087 |
| Llama-4-Maverick (400B) | `llama4_maverick` | **0.7332** | 0.6703–0.7934 |
| Qwen3-235B-A22B-Instruct-2507 | `qwen3_235b_a22b_2507` | **0.7187** | 0.6332–0.8233 |
| Qwen3.5-397B-A17B | `qwen35_397b_a17b` | **0.7010** | 0.4040–0.8584 |
| GLM-5.2 745B (official FP8) | `glm52` | **0.6249** | 0.4994–0.7028 |

An observation, not a claim: S does not rise with total parameter count across
these five points — it varies by 0.21 across 235B–745B with overlapping CIs
except at the extremes (GLM-5.2 vs Coder are disjoint). Total-parameter scale is
confounded with active-parameter count (17–39B), model family/generation, and —
at the GLM-5.2 point — numeric precision, so no scaling read should be taken
from this panel. The one within-family contrast available is that the coding
specialist (Qwen3-Coder-480B) scores higher than its same-architecture
generalist sibling (Qwen3-235B; CIs overlap slightly at the edge).

## Qwen3-Coder-480B-A35B-Instruct (`qwen3_coder_480b`)

Battery run on 2026-07-23 at model revision
`9d90cf8fca1bf7b7acca42d3fc9ae694a2194069` (bf16, 8x B200, batch 24). The
960 GB checkpoint exceeded the job's ~500 GiB local disk, so weights were
loaded with the conversion-aware staged shard-streaming loader (feeding
transformers' own v5 loading engine in group-complete batches), certified
stock-equivalent on same-class proxy gates: identical greedy ids, logit max
abs diff 0.0, identical structure census.

The conjunctive controllability scalar is **S = 0.8311** with a joint 95%
confidence interval of **0.7391–0.9087** — the highest in the battery. Engage
reaches **d′ = 10.93** on the projection channel (lower bound 9.35). The
layer-targeting designed null holds cleanly (d′ = 0.005, CI −0.0074 to 0.0198).

Caveats from the reviewed findings:

- **Engage's upper CI bound is degenerate** (2.67e12) via the known
  bootstrap-tail artifact when a replicate's baseline σ nears zero; the
  hypothesis read uses the lower bound only.
- **Compliance conditioning is much deeper per-condition than the 83.5%
  aggregate suggests:** four conditions fall below the 70% aggregate gate level
  (`persist_after_fourth` 43.0%, `think_intensely` 52.8%, `think_about` 59.4%,
  `loc_end` 67.6% — each n=500), so the Engage and persistence reads rest on
  substantially reduced compliant coverage in exactly the conditions that drive
  them. Failures are elaboration-after-transcription (the model transcribes the
  sentence, then keeps writing). Layer-targeting compliance was 100%.

Raw recordings for this model:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/qwen3_coder_480b/production/main/raw/20260723_225436_qwen3_coder_480b_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/qwen3_coder_480b/production/lt/raw/20260723_225527_qwen3_coder_480b_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/qwen3_coder_480b/scoring/`

Source experiment: `exp_01ky7tchmsfqfs414h2zr2m0ft`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.

## Llama-4-Maverick-17B-128E-Instruct (`llama4_maverick`)

Battery run on 2026-07-24 at model revision
`73d14711bcc77c16df3470856949c3764056b617` (bf16, 8x B200, batch 24; gated
repo — the Meta license must be accepted on HF before the checkpoint
downloads). The 803 GB checkpoint was loaded with the same staged
shard-streaming loader (renames only; the checkpoint ships pre-fused expert
tensors).

The conjunctive controllability scalar is **S = 0.7332** with a joint 95%
confidence interval of **0.6703–0.7934** — the cleanest large-model CIs of the
panel (no degenerate bootstrap tails). Engage reaches **d′ = 7.083**
(CI 5.810–9.602) on the projection channel. The layer-targeting designed null
holds cleanly (d′ = 0.0135, CI −0.0027 to 0.0321). Main-run compliance was
95.59%; layer targeting 100%.

Caveat from the reviewed findings:

- **The smoke gate ran on 6x B200, not the 8x production shape** (the shape was
  widened after the warm benchmark showed batch 24 is 2.6x faster than batch 8
  but leaves one 6x-map device at 0.8% free). The 8x production jobs are
  therefore not covered by an exact-shape smoke; their own validation (exact and
  unique trial counts, green checks, completion at batch 24) is the shape
  evidence, and per-device headroom strictly increases with the extra GPUs.

Raw recordings for this model:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/llama4_maverick/production/main/raw/20260724_002143_llama4_maverick_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/llama4_maverick/production/lt/raw/20260724_002241_llama4_maverick_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/llama4_maverick/scoring/`

Source experiment: `exp_01ky7tchmsfqfs414h2zr2m0ft`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.

## GLM-5.2 745B, official FP8 (`glm52`)

Battery run on 2026-07-24 at model revision
`ba978f7d347eaf65d22f1a86833408afdb953541` of `zai-org/GLM-5.2-FP8` (6x B200,
batch 24, `enable_thinking: false` pinned — the closed think block was verified
both in the rendered template and behaviorally: 99.91% main-run compliance with
zero think tags). The 756 GB FP8 checkpoint was loaded with the staged loader's
fine-grained-FP8 path (stock transformers module conversion on the meta
skeleton before placement).

The conjunctive controllability scalar is **S = 0.6249** with a joint 95%
confidence interval of **0.4994–0.7028** — the lowest of the large panel.
Engage reaches **d′ = 7.112** (CI 6.516–8.976). The layer-targeting designed
null holds cleanly (d′ = 0.0123, CI −0.0115 to 0.0345).

Caveats from the reviewed findings:

- **Precision confound, stated plainly:** GLM-5.2 runs the official FP8
  checkpoint because its bf16 repo (1.51 TB) cannot fit an 8x B200 node at any
  GPU count, while the rest of the panel (except native-MXFP4 gpt-oss) runs
  bf16. Scale and precision are confounded at the panel's largest point:
  GLM-5.2's lower S cannot be attributed to scale, architecture, or precision
  individually.
- **The distinctive finding:** GLM-5.2 is the only large-panel model with
  positive token-group control (proj d′ = 1.398, CI 0.386–2.537); Coder
  (−2.190) and Maverick (−1.601) show the panel-typical negative effect.
  (Token-group is reported outside S and is a near-universal panel failure.)
- **Dial rank is the weakest ordered dial** among the new large models
  (ρ = 0.498, CI 0.336–0.644).
- **Loader evidence is proxy-plus-value-level:** the staged-loader equivalence
  gates ran on Qwen3-30B proxies (same conversion classes) because GLM-5.2
  itself cannot be stock-loaded at this size on this hardware. Its lane rests on
  those proxy gates plus its exact-index addressing audit and bitwise cross-load
  consistency: 20/20 concept-vector files identical (max abs diff 0.0) across
  two independent staged loads, with 99.19% identical generations on the
  860-trial smoke–main overlap.

Raw recordings for this model:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/glm52/production/main/raw/20260724_015723_glm52_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/glm52/production/lt/raw/20260724_015813_glm52_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky7tchmsfqfs414h2zr2m0ft/results/glm52/scoring/`

Source experiment: `exp_01ky7tchmsfqfs414h2zr2m0ft`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.
