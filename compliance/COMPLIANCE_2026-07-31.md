# Instruction compliance by model — activation-control battery

Generated 2026-07-31 from the artifacts on the RunPod volume. Regenerate with the script this file was built by; do not hand-edit the numbers.

**Definition.** A trial is compliant when the generated transcription reaches Ratcliff–Obershelp sequence similarity ≥ 0.85 to the target sentence (`results.json → metrics.compliance_rate`). Base checkpoints (the pre-training and mid-training Olmo snapshots) are scored on the first |target| characters, since they lack end-of-sequence discipline; reasoning models (gpt-oss) are scored on the final channel only. Non-compliant trials are saved but flagged `is_compliant=False`, and every scorer drops them. See the paper's Supplementary Material, paragraph *Trial compliance*.

Main runs are 8,600 trials (`intensity`, `token_location`, `persistence`); layer-targeting runs are 8,800 (`layer_location`).

## Panel models — raw retained (measured)

Read directly from each model's `results.json`, resolved through the `main_run`/`lt_run` recorded in its `SCORES_*.json` (so the superseded `20260722_015304_gptoss_20b_low_lt` is correctly ignored).

| Model | Main run (n=8,600) | Layer-targeting (n=8,800) |
|---|---|---|
| `gemma2_9b` | 86.50% (7,439/8,600) | 100.00% (8,800/8,800) |
| `gemma3_27b` | 93.34% (8,027/8,600) | 100.00% (8,800/8,800) |
| `gemma4_12b` | 99.99% (8,599/8,600) | 100.00% (8,800/8,800) |
| `gemma4_31b` | 100.00% (8,600/8,600) | 100.00% (8,800/8,800) |
| `glm46v` | 96.06% (8,261/8,600) | 100.00% (8,800/8,800) |
| `glm47_flash` | 95.98% (8,254/8,600) | 100.00% (8,800/8,800) |
| `gptoss_120b_low` | 99.78% (8,581/8,600) | 100.00% (8,800/8,800) |
| `gptoss_20b_low` | 99.99% (8,599/8,600) | 99.75% (8,778/8,800) |
| `llama33_70b` | 94.00% (8,084/8,600) | 100.00% (8,800/8,800) |
| `llama4_scout` | 90.97% (7,823/8,600) | 100.00% (8,800/8,800) |
| `llama_8b` | 88.17% (7,583/8,600) | 99.72% (8,775/8,800) |
| `mistral_small_31_24b` | 89.56% (7,702/8,600) | 100.00% (8,800/8,800) |
| `mistral_small_4` | 92.59% (7,963/8,600) | 100.00% (8,800/8,800) |
| `qwen35_122b_a10b` | 99.95% (8,596/8,600) | 100.00% (8,800/8,800) |
| `qwen35_4b` | 99.88% (8,590/8,600) | 100.00% (8,800/8,800) |
| `qwen35_9b` | 99.85% (8,587/8,600) | 100.00% (8,800/8,800) |
| `qwen36_27b` | 99.69% (8,573/8,600) | 100.00% (8,800/8,800) |
| `qwen_72b` | 98.33% (8,456/8,600) | 100.00% (8,800/8,800) |

## Olmo training-snapshot lane — raw pruned (recovered from W&B)

`olmo_snapshot_lane.sh` deletes raw activations and weights on completion, and the derived JSONs store no trial counts — but the runner logs `compliance_rate` to W&B, which survives. Each run is bound to its scoring run dir by created_at (W&B init lags the dir stamp by ~10-90 s), by `active_sets`, and by trial count. The two Instruct anchors, which still have raw, reproduce their measured values exactly; the five 7B main values reproduce the stamps hardcoded in `scripts/snapshot_superplot.py`.

| Model | Main run (n=8,600) | Layer-targeting (n=8,800) |
|---|---|---|
| `olmo31_32b` † | 94.45% (8,123/8,600) | 100.00% (8,800/8,800) |
| `olmo31_32b_base` | 96.09% (8,264/8,600) | 100.00% (8,800/8,800) |
| `olmo31_32b_dpo` | 90.37% (7,772/8,600) | 100.00% (8,800/8,800) |
| `olmo31_32b_s1_328k` | 69.57% (5,983/8,600) | 85.06% (7,485/8,800) |
| `olmo31_32b_s1_final` | 77.84% (6,694/8,600) | 95.80% (8,430/8,800) |
| `olmo31_32b_sft` | 97.98% (8,426/8,600) | 100.00% (8,800/8,800) |
| `olmo3_7b` † | 68.43% (5,885/8,600) | 99.24% (8,733/8,800) |
| `olmo3_7b_base` | 99.53% (8,560/8,600) | 100.00% (8,800/8,800) |
| `olmo3_7b_dpo` | 41.22% (3,545/8,600) | 52.65% (4,633/8,800) |
| `olmo3_7b_s1_700k` | 74.67% (6,422/8,600) | 80.33% (7,069/8,800) |
| `olmo3_7b_s1_final` | 82.63% (7,106/8,600) | 90.42% (7,957/8,800) |
| `olmo3_7b_sft` | 89.98% (7,738/8,600) | 100.00% (8,800/8,800) |

† Instruct anchor — raw retained, so these two rows are measured from `results.json`, not recovered. They are what validates the other ten: the W&B `compliance_rate` for each anchor equals its measured value exactly.

## Large panel (>150B) — raw never on this volume (documented)

Transcribed from `models.txt` and the run write-ups. These runs used the separate experiment platform and were never logged to W&B, so the two 'not recorded' cells can only come from the `artifact://resolution-6d0064/...` sidecars referenced per-model in the panel README.

| Model | Main run (n=8,600) | Layer-targeting (n=8,800) |
|---|---|---|
| `qwen3_coder_480b` | 83.50% | 100.00% |
| `llama4_maverick` | 95.59% | 100.00% |
| `qwen35_397b_a17b` | 99.94% (8,595/8,600) | 100.00% (8,800/8,800) |
| `glm52` | 99.91% | not recorded |
| `qwen3_235b_a22b_2507` | not recorded | not recorded |

## Per-cell notes

- **`glm52` main** — verified with zero think tags (enable_thinking=false pinned)
- **`qwen3_coder_480b` main** — aggregate; four conditions below the 70% gate (persist_after_fourth 43.0%, think_intensely 52.8%, think_about 59.4%, loc_end 67.6%, each n=500)

## Gaps

- `qwen3_235b_a22b_2507` — neither run recorded. The only compliance figure documented anywhere for it is the `loc_end` condition: 441/500 (88.2%) fuzzy-compliant, 45/500 (9.0%) exact-normalized.
- `glm52` — layer-targeting run not recorded (main is).

Neither is recoverable from this volume or in W&B carries them.

