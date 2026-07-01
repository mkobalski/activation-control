#!/bin/bash
# Launcher: source env, set PYTHONPATH for python3.11 (deps), run experiment.
set -euo pipefail
set -a
source /workspace/.env
set +a
cd /workspace/write-introspection-main
export PYTHONPATH="/workspace/venv/lib/python3.11/site-packages:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec /usr/bin/python3.11 "$@"
