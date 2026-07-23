#!/bin/bash
# One Olmo snapshot end-to-end: download -> main -> lt -> score -> PRUNE.
# Usage: scripts/olmo_snapshot_lane.sh <short_name>   (e.g. olmo3_7b_sft)
#
# Snapshot runs keep only the score JSONs (SCORES/PROFILES/SCALAR_CI/...);
# raw activations AND weights are deleted at the end. Retention rationale +
# monitoring protocol: RUNBOOK_olmo_snapshots_1xH100.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/workspace/.venv/bin/python

declare -A REPO=(
  [olmo3_7b_s1_700k]="allenai/Olmo-3-1025-7B"
  [olmo3_7b_s1_final]="allenai/Olmo-3-1025-7B"
  [olmo3_7b_base]="allenai/Olmo-3-1025-7B"
  [olmo3_7b_sft]="allenai/Olmo-3-7B-Instruct-SFT"
  [olmo3_7b_dpo]="allenai/Olmo-3-7B-Instruct-DPO"
)
declare -A REV=(
  [olmo3_7b_s1_700k]="stage1-step700000"
  [olmo3_7b_s1_final]="stage1-step1413814"
)

m="${1:?usage: olmo_snapshot_lane.sh <short_name>}"
repo="${REPO[$m]:?unknown snapshot short name: $m}"
rev="${REV[$m]:-}"

# 1. Download to the snapshot's own dir (trivial to wipe; no shared HF-cache blobs).
hf download "$repo" ${rev:+--revision "$rev"} --local-dir "/ckpts/$m"

# 2. Battery: main + lt (run.sh auto-scores each run dir on completion).
scripts/run.sh --config "experiments/main/$m.yaml" > "logs/night_${m}_main.log" 2>&1
scripts/run.sh --config "experiments/main/$m.yaml" \
    --set 'experiment.sets=[layer_location]'       > "logs/night_${m}_lt.log" 2>&1

# 3. Aggregate model-level scores (reads the raw run dirs -> MUST precede pruning).
"$PY" scripts/postprocess.py --model "$m"

# 4. Prune: raw activations + weights. JSONs in results/ are the keepers.
#    Glob is exact on the short name, so the precious existing olmo3_7b runs
#    (final-Instruct point) can never match.
rm -rf results/raw/*_"${m}"_activation_control results/raw/*_"${m}"_activation_control_lt
rm -rf "/ckpts/$m"
echo "[lane] $m complete: scored + pruned."
