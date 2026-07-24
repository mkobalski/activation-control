# Qwen3-235B-A22B-2507 results panel

This directory contains the small, frozen scoring outputs for
`Qwen/Qwen3-235B-A22B-Instruct-2507` from the activation-control battery run on
2026-07-23. The evaluated model revision is
`ac9c66cc9b46af7306746a9250f23d47083d689e`.

The deterministic battery used 50 neutral sentences and 10 concepts, producing
8,600 main trials and 8,800 layer-targeting trials. Scoring used the projection
channel, the frozen linear link, and a two-way sentence-by-concept bootstrap
with 2,000 replicates and seed 0.

## Headline result

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

## Files

- `SCORES_qwen3_235b_a22b_2507.json` — battery point estimates and per-measure
  intervals.
- `PROFILES_qwen3_235b_a22b_2507.json` — depth profiles for the scored channels.
- `SCALAR_CI_qwen3_235b_a22b_2507.json` — frozen scalar configuration, point
  estimate, and joint confidence interval.
- `ONSET_OFFSET_WORD_qwen3_235b_a22b_2507.json` — word-boundary correction for
  onset/offset timing; it supersedes only the token-boundary timing diagnostic in
  `SCORES`.

Raw recordings are intentionally not tracked here. They remain in the source
experiment's durable artifact store:

- Main run (8,600 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/production/main/raw/20260723_115400_qwen3_235b_a22b_2507_activation_control/`
- Layer-targeting run (8,800 trials):
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/production/lt/raw/20260723_115414_qwen3_235b_a22b_2507_activation_control_lt/`
- Frozen scoring bundle:
  `artifact://resolution-6d0064/experiments/exp_01ky77p2ksea49khssdbchd55y/results/scoring/`

Source experiment: `exp_01ky77p2ksea49khssdbchd55y`. Source thread:
`th_01ky73gqg2fsaa7s9gfsxp8ksp`.

## Reproduce and render

From the repository root after installing `requirements.txt`:

```bash
python scripts/aggregate_scalar.py \
  --scores results-panel/qwen3_235b_a22b_2507/SCORES_qwen3_235b_a22b_2507.json

python scripts/superplot.py \
  --data-root results-panel/qwen3_235b_a22b_2507
```

The first command deterministically reproduces `S = 0.7187`. The second writes
`model_comparison.png`, `null_measures_model_comparison.png`, and
`degenerate_measures_model_comparison.png` under the gitignored `results/`
directory. Point `--data-root` at a directory containing additional models'
`SCORES_*`, `SCALAR_CI_*`, and optional `ONSET_OFFSET_WORD_*` files to render a
cross-model panel.
