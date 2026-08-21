#!/bin/bash
# Launcher: source env, then run via the ISOLATED venv interpreter.
#
# Set AC_PYTHON (or activate a venv) to an interpreter whose venv has
# include-system-site-packages=false, so its own python excludes the system
# dist-packages. Do NOT instead put a venv merely on PYTHONPATH and run the
# system python: that leaves the system site-packages importable. They hold a
# torchvision built against the old system torch (2.4.1); if it stays importable it
# breaks transformers' Gemma3 import with an ABI error (torchvision::nms missing). The
# venv python (torch 2.12, no torchvision) avoids that entirely. Do NOT reintroduce the
# PYTHONPATH+system-python pattern.
set -euo pipefail
set -a
[ -n "${AC_ENV_FILE:-}" ] && source "$AC_ENV_FILE"
[ -f "$(dirname "$0")/../.env" ] && source "$(dirname "$0")/../.env"
set +a
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
exec "${AC_PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}" "$@"
