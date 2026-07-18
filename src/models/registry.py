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
    # 270M: the panel's smallest model by two orders of magnitude, and the only one
    # with FEWER layers (18) than the 20-fraction depth sweep asks for -- the
    # fractions dedupe to all 18, so its depth axis has 18 points, not 20.
    "gemma3_270m": "google/gemma-3-270m-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "gemma4_31b": "google/gemma-4-31b-it",
    # Qwen
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen_32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen36_27b": "Qwen/Qwen3.6-27B",
    # Qwen3.5-122B-A10B (MoE: 122B total, ~10B active). A single unified
    # checkpoint that toggles chain-of-thought via the chat template's
    # `enable_thinking` flag. Two short names map to the SAME weights but pin
    # opposite modes via their experiment YAMLs (model.enable_thinking): the
    # non-thinking entry transcribes directly, the `_thinking` entry emits a
    # <think>...</think> trace before the answer (parsed like gpt-oss's harmony
    # `final` channel, but with think tags — see src/utils/think_tags.py). Kept
    # distinct so run dirs, concept-vector caches, SCORES_*.json and the cross-
    # model comparison treat the two modes as separate points (enable_thinking
    # changes the prompt, so their activations — and thus concept vectors —
    # genuinely differ).
    "qwen35_122b_a10b": "Qwen/Qwen3.5-122B-A10B",
    "qwen35_122b_a10b_thinking": "Qwen/Qwen3.5-122B-A10B",
    # Llama
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama33_70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Olmo (AI2, Apache-2.0, fully open). Dense, standard Olmo3ForCausalLM with a
    # system role and no thinking toggle -- the plain -Instruct checkpoint, NOT
    # the -Think sibling -- so it joins none of the quirk sets below.
    "olmo31_32b": "allenai/Olmo-3.1-32B-Instruct",
    # Mistral. Dense, non-reasoning, has a system role -> no quirk sets. NOTE the
    # checkpoint is MULTIMODAL (Mistral3ForConditionalGeneration: a vision encoder
    # + projector wrapped around a 24B text model), so unlike every other panel
    # entry the decoder layers live at model.model.language_model.layers and the
    # layer count is config.text_config.num_hidden_layers (40), not the top level.
    # wrapper.py already falls back to both -- see ModelWrapper.n_layers/_layers.
    "mistral_small_31_24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # OpenAI gpt-oss (MoE, harmony chat format, native MXFP4). Two entries map to
    # the SAME weights but pin different harmony `reasoning_effort` levels via
    # their experiment YAMLs; kept as distinct short names so run dirs, concept-
    # vector caches, SCORES_*.json and the cross-model comparison treat low vs
    # high as separate points (the effort changes the system prompt, so their
    # activations -- and thus concept vectors -- genuinely differ).
    "gptoss_120b_low": "openai/gpt-oss-120b",
    "gptoss_120b_high": "openai/gpt-oss-120b",
}

# Capability/quirk sets, all keyed by short name:
#   GEMMA_MODELS            - the Gemma family.
#   MODELS_WITHOUT_SYSTEM_ROLE - chat templates that reject a "system" message
#                             (Gemma); callers must fold system text into the
#                             user turn instead. Aliased to GEMMA_MODELS today.
#   BASE_MODELS             - non-instruct checkpoints lacking a chat template,
#                             so prompt wrapping falls back to "User:/Assistant:".
GEMMA_MODELS = {"gemma2_2b", "gemma2_9b", "gemma2_9b_base", "gemma2_27b", "gemma3_270m", "gemma3_27b",
                "gemma4_31b"}
MODELS_WITHOUT_SYSTEM_ROLE = GEMMA_MODELS
BASE_MODELS = {"gemma2_9b_base"}

#   REASONING_MODELS - checkpoints that emit a chain-of-thought before the final
#                      answer. Their generated text must be split into reasoning
#                      vs. final answer before compliance/alignment; only the
#                      final answer is the requested sentence. The runner treats
#                      every REASONING_MODELS entry the same way (record all
#                      generated steps, then slice the final span), and dispatches
#                      to the right parser by the sub-sets below:
#   HARMONY_MODELS   - reasoning via the harmony channel format (gpt-oss):
#                      <|channel|>analysis...<|channel|>final... — parsed by
#                      src/utils/harmony.py:final_channel_span.
#   THINK_TAG_MODELS - reasoning via <think>...</think> tags before the answer
#                      (Qwen3-style, enable_thinking=True) — parsed by
#                      src/utils/think_tags.py:final_answer_span.
# REASONING_MODELS is their union; add a model to exactly one sub-set.
HARMONY_MODELS = {"gptoss_120b_low", "gptoss_120b_high"}
THINK_TAG_MODELS = {"qwen35_122b_a10b_thinking"}
REASONING_MODELS = HARMONY_MODELS | THINK_TAG_MODELS
