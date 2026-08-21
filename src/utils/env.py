"""Environment setup: load .env and default HF cache path."""

import os
from dotenv import load_dotenv


def setup_env():
    """Load .env files and set sensible defaults for HF + W&B.

    Call this once at the start of any script. We load an optional machine-wide
    .env named by AC_ENV_FILE first, then a directory-local .env (which can
    override it), so per-project secrets/tokens take precedence. HF_HOME and
    WANDB_SILENT are only set if not already defined, so an explicitly exported
    value always wins.
    """
    # An optional machine-wide .env first (AC_ENV_FILE, e.g. a shared secrets
    # file outside the checkout), then a directory-local .env, which overrides it.
    shared = os.environ.get("AC_ENV_FILE")
    if shared:
        load_dotenv(shared)
    load_dotenv()
    # Leave HF_HOME alone unless the caller set AC_HF_HOME; the HuggingFace
    # default (~/.cache/huggingface) is correct on most machines.
    if "HF_HOME" not in os.environ and os.environ.get("AC_HF_HOME"):
        os.environ["HF_HOME"] = os.environ["AC_HF_HOME"]
    # Keep Weights & Biases from spamming stdout during runs by default.
    if "WANDB_SILENT" not in os.environ:
        os.environ["WANDB_SILENT"] = "true"
