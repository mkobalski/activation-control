#!/bin/bash
# Launcher: source env, then run via the ISOLATED venv interpreter.
#
# We deliberately exec /workspace/.venv/bin/python rather than /usr/bin/python3.11
# with the venv merely on PYTHONPATH: the venv has include-system-site-packages=false,
# so its own python excludes the system dist-packages. The system site-packages hold a
# torchvision built against the old system torch (2.4.1); if it stays importable it
# breaks transformers' Gemma3 import with an ABI error (torchvision::nms missing). The
# venv python (torch 2.12, no torchvision) avoids that entirely. Do NOT reintroduce the
# PYTHONPATH+system-python pattern.
set -euo pipefail
set -a
source /workspace/.env
set +a
cd /workspace/activation-control
export PYTHONUNBUFFERED=1
exec /workspace/.venv/bin/python "$@"
