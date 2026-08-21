#!/bin/bash
# Wrapper: run an experiment, then (on completion) score it + refresh figures.
#
# This is the completion trigger the pipeline is built around -- invoke THIS instead
# of run_experiment.py directly. run_experiment.py itself stays a pure generate-and-
# save step; all post-processing (compute_scores -> scalar_ci -> explore -> superplot)
# runs here afterward, via scripts/postprocess.py.
#
# Usage:
#   scripts/run.sh --config experiments/main/config.yaml [run_experiment args...]
#   scripts/run.sh --config experiments/main/config.yaml --overrides experiment.sets='[layer_location]'
#
# The interpreter is $AC_PYTHON, else the active virtualenv, else python3.
# Post-processing only runs if the experiment succeeds.
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional: a machine-wide env file (HF_TOKEN etc.) plus a repo-local .env.
set -a
[ -n "${AC_ENV_FILE:-}" ] && source "$AC_ENV_FILE" 2>/dev/null || true
[ -f .env ] && source .env 2>/dev/null || true
set +a
# Interpreter: $AC_PYTHON, else an active venv, else python3 from PATH.
PY="${AC_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PY="${PY:-python3}"
export PYTHONUNBUFFERED=1

# Run the experiment; stream output live AND capture it to recover the run dir.
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
"$PY" scripts/run_experiment.py "$@" | tee "$tmp"

run_dir="$(grep -oE '^Output: .*' "$tmp" | head -1 | sed 's/^Output: //')"
if [ -z "$run_dir" ]; then
  echo "[run.sh] could not determine run dir from run_experiment output; skipping post-processing" >&2
  exit 1
fi

echo ""
echo "[run.sh] experiment done -> post-processing $run_dir"
"$PY" scripts/postprocess.py --run-dir "$run_dir"
