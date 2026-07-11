"""Model registry: short config-friendly names -> HuggingFace ids + capability sets.

Throughout the codebase and the YAML configs, models are referred to by terse
short names (e.g. ``gemma3_27b``) instead of full HuggingFace repo ids. This
module is the single source of truth that maps those short names to the actual
ids to load, plus a few sets that record per-model quirks (which families lack a
system role, which checkpoints are base rather than instruct-tuned). To add a
model, add its short name here and to any capability set that applies.
"""

# Short name -> HuggingFace repo id passed to the loader.
MODEL_NAME_MAP = {
    # Gemma (primary)
    "gemma2_2b": "google/gemma-2-2b-it",
    "gemma2_9b": "google/gemma-2-9b-it",
    "gemma2_9b_base": "google/gemma-2-9b",
    "gemma2_27b": "google/gemma-2-27b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma4_31b": "google/gemma-4-31b-it",
    # Qwen
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen36_27b": "Qwen/Qwen3.6-27B",
    # Llama
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama33_70b": "meta-llama/Llama-3.3-70B-Instruct",
}

# Capability/quirk sets, all keyed by short name:
#   GEMMA_MODELS            - the Gemma family.
#   MODELS_WITHOUT_SYSTEM_ROLE - chat templates that reject a "system" message
#                             (Gemma); callers must fold system text into the
#                             user turn instead. Aliased to GEMMA_MODELS today.
#   BASE_MODELS             - non-instruct checkpoints lacking a chat template,
#                             so prompt wrapping falls back to "User:/Assistant:".
GEMMA_MODELS = {"gemma2_2b", "gemma2_9b", "gemma2_9b_base", "gemma2_27b", "gemma3_27b",
                "gemma4_31b"}
MODELS_WITHOUT_SYSTEM_ROLE = GEMMA_MODELS
BASE_MODELS = {"gemma2_9b_base"}
