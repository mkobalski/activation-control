#!/usr/bin/env bash
# Pre-download Qwen3.5-122B-A10B weights to GPU-local disk (HF_HOME on overlay).
# Resumable; logs progress. Launched in background by Claude Code.
set -euo pipefail
set -a; source /workspace/.env; set +a
echo "START $(date -u +%H:%M:%S) HF_HOME=$HF_HOME"
/workspace/.venv/bin/python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
t0=time.time()
p=snapshot_download(
    "Qwen/Qwen3.5-122B-A10B",
    token=os.environ.get("HF_TOKEN"),
    allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*","*.model","*.jinja","merges*","vocab*"],
    max_workers=8,
)
print("DONE snapshot at", p, "in %.0f min" % ((time.time()-t0)/60))
PY
echo "END $(date -u +%H:%M:%S)"
