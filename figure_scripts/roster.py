#!/usr/bin/env python3
"""Single source of truth for the cross-model roster, family order, display labels,
and the focal model -- shared by every paper figure.

Before this module the (short_name, family, size_B) roster and the pretty-name DETAIL
map were copy-pasted across engage_suppress / temporal_control / scalar, and the MoE
extension across temporal_precision / token_targeting. Adding a model, or repointing
the focal model, is now a one-file edit here.

  MODELS      the 20 models with retained raw runs (bootstrap CIs available)
  MOE         5 large MoE models scored from stored SCORES only (point estimate, no CI)
  ROSTER      MODELS + MOE (the full cross-model panel, in canonical order)
  FAMILY_ORDER family plotting order (also re-sorted by top-S at render time)
  DETAIL      display label per short-name (every label carries the model size)
  MOE_DETAIL  display labels for the MoE extension
  DETAIL_ALL  DETAIL merged with MOE_DETAIL
  FOCAL       focal model for the single-model illustrative panels; override per run
              with $AC_FOCAL or a script's --model flag
"""
import os

# (short_name, family, size_B) -- 20 models with retained raw runs.
MODELS = [("gemma2_9b", "Gemma", 9), ("gemma4_12b", "Gemma", 12), ("gemma3_27b", "Gemma", 27),
          ("gemma4_31b", "Gemma", 31),
          ("qwen35_4b", "Qwen", 4), ("qwen35_9b", "Qwen", 9), ("qwen36_27b", "Qwen", 27),
          ("qwen_72b", "Qwen", 72), ("qwen35_122b_a10b", "Qwen", 122),
          ("llama_8b", "Llama", 8), ("llama33_70b", "Llama", 70), ("llama4_scout", "Llama", 109),
          ("gptoss_20b_low", "GPT-OSS", 20), ("gptoss_120b_low", "GPT-OSS", 120),
          ("olmo3_7b", "Olmo", 7), ("olmo31_32b", "Olmo", 32),
          ("mistral_small_31_24b", "Mistral", 24), ("mistral_small_4", "Mistral", 119),
          ("glm47_flash", "GLM", 31), ("glm46v", "GLM", 106)]

# Large MoE models scored from stored SCORES only -- no raw runs, so these carry a
# POINT ESTIMATE of the six-measure S with no bootstrap CI (drawn open, no whisker).
MOE = [("qwen3_235b_a22b_2507", "Qwen", 235), ("qwen35_397b_a17b", "Qwen", 397),
       ("qwen3_coder_480b", "Qwen", 480), ("llama4_maverick", "Llama", 400),
       ("glm52", "GLM", 744)]

ROSTER = MODELS + MOE

FAMILY_ORDER = ["Gemma", "Qwen", "Llama", "GPT-OSS", "Olmo", "Mistral", "GLM"]

DETAIL = {"gemma2_9b": "Gemma 2 9B", "gemma4_12b": "Gemma 4 12B", "gemma3_27b": "Gemma 3 27B",
          "gemma4_31b": "Gemma 4 31B", "qwen35_4b": "Qwen 3.5 4B", "qwen35_9b": "Qwen 3.5 9B",
          "qwen36_27b": "Qwen 3.6 27B", "qwen_72b": "Qwen 2.5 72B", "qwen35_122b_a10b": "Qwen 3.5 122B",
          "llama_8b": "Llama 3.1 8B", "llama33_70b": "Llama 3.3 70B", "llama4_scout": "Llama 4 Scout 109B",
          "gptoss_20b_low": "GPT-OSS 20B", "gptoss_120b_low": "GPT-OSS 120B",
          "olmo3_7b": "Olmo 3 7B", "olmo31_32b": "Olmo 3.1 32B",
          "mistral_small_31_24b": "Mistral Small 3.1 24B", "mistral_small_4": "Mistral Small 4 119B",
          "glm47_flash": "GLM 4.7 Flash 31B", "glm46v": "GLM 4.6V 106B"}

MOE_DETAIL = {"qwen3_235b_a22b_2507": "Qwen 3 235B", "qwen35_397b_a17b": "Qwen 3.5 397B",
              "qwen3_coder_480b": "Qwen 3 Coder 480B", "llama4_maverick": "Llama 4 Maverick 400B",
              "glm52": "GLM 5.2 744B"}

DETAIL_ALL = {**DETAIL, **MOE_DETAIL}

# Release month per model (YYYY-MM), for within-family ordering; same-month ties
# break by size.
RELEASE = {"gemma2_9b": "2024-06", "gemma3_27b": "2025-03", "gemma4_31b": "2026-03", "gemma4_12b": "2026-06",
           "qwen_72b": "2024-09", "qwen3_235b_a22b_2507": "2025-07", "qwen3_coder_480b": "2025-07",
           "qwen35_122b_a10b": "2026-02", "qwen35_397b_a17b": "2026-02", "qwen35_4b": "2026-03",
           "qwen35_9b": "2026-03", "qwen36_27b": "2026-04",
           "llama_8b": "2024-07", "llama33_70b": "2024-12", "llama4_scout": "2025-04", "llama4_maverick": "2025-04",
           "gptoss_20b_low": "2025-08", "gptoss_120b_low": "2025-08",
           "olmo3_7b": "2025-11", "olmo31_32b": "2025-12",
           "mistral_small_31_24b": "2025-03", "mistral_small_4": "2026-03",
           "glm47_flash": "2026-01", "glm46v": "2025-12", "glm52": "2026-06"}


# Focal model for the single-model illustrative panels (think_intensity, temporal_*,
# layer_targeting, token_coverage). Override for a run with AC_FOCAL, or per script
# with --model. Kept as one name so all focal figures move together.
FOCAL = os.environ.get("AC_FOCAL") or "gemma3_27b"
