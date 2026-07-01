"""Environment setup: load .env and default HF cache path."""

import os
from dotenv import load_dotenv


def setup_env():
    """Load .env files and set sensible defaults for HF + W&B.

    Call this once at the start of any script. We load the workspace-level
    .env first, then a local .env (which can override it), so per-project
    secrets/tokens take precedence. We only set HF_HOME / WANDB_SILENT if they
    are not already defined, so an explicitly exported value always wins.
    """
    # Workspace-wide secrets/config first...
    load_dotenv("/workspace/.env")
    # ...then a directory-local .env, which overrides the workspace defaults.
    load_dotenv()
    # Point the HuggingFace cache at a writable workspace path unless the caller
    # has already chosen one (avoids re-downloading large model weights).
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    # Keep Weights & Biases from spamming stdout during runs by default.
    if "WANDB_SILENT" not in os.environ:
        os.environ["WANDB_SILENT"] = "true"
