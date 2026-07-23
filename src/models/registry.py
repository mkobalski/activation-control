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
    "gemma4_12b": "google/gemma-4-12B-it",
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
    # Qwen3.5 dense hybrids (thinking OFF by default via the builder's
    # enable_thinking=False when unset -- see models.txt header; not in
    # REASONING_MODELS, they transcribe directly).
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    # Qwen3-235B-A22B-2507: MoE 235B total / 22B active. Non-thinking ONLY by
    # design (the -2507 Instruct split; there is no <think> turn to suppress), so
    # the builder default is a no-op and it joins no reasoning set.
    "qwen3_235b_a22b_2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    # Llama
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama33_70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Llama-4-Scout: MoE 109B total / 17B active, 16 experts, natively MULTIMODAL
    # (registered only under image-text-to-text -> ModelWrapper's auto-class probe
    # loads it with AutoModelForImageTextToText; decoder layers via
    # language_model.layers). Has a system role, non-reasoning -> no quirk sets.
    "llama4_scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # Mistral. Ministral-3-3B is MULTIMODAL (~3.4B LM + 0.4B vision encoder),
    # like mistral_small_31_24b: registered only under image-text-to-text, chat
    # template in chat_template.json -- ModelWrapper auto-detects both. Has a
    # system role, dense, non-reasoning -> no quirk sets.
    "ministral3_3b": "mistralai/Ministral-3-3B-Instruct-2512",
    # Olmo (AI2, Apache-2.0, fully open). Dense, standard Olmo3ForCausalLM with a
    # system role and no thinking toggle -- the plain -Instruct checkpoint, NOT
    # the -Think sibling -- so it joins none of the quirk sets below.
    "olmo3_7b": "allenai/Olmo-3-7B-Instruct",
    # GLM (Z.ai). GLM-4.7-Flash is a reasoning/hybrid whose chat template reads
    # `enable_thinking`: with it False (the builder default when unset) the
    # generation prompt ends "<|assistant|></think>" -- the think block is closed
    # immediately, so the model transcribes directly. Verified 2026-07-22. Hence
    # NOT in REASONING_MODELS (same treatment as the Gemma4/Qwen3.5 hybrids); has
    # a system role, so no quirk sets.
    "glm47_flash": "zai-org/GLM-4.7-Flash",
    # GLM-4.6V: MoE 106B total / ~12B active, natively MULTIMODAL (vision). HYBRID:
    # the thinking-off switch is UNVERIFIED for 4.6V (documented only for 4.5V) --
    # the builder default (enable_thinking=False) is applied, but the smoke run's
    # compliance MUST be checked before trusting it as non-thinking. Has a system
    # role. Loaded via AutoModelForImageTextToText (auto-class probe).
    "glm46v": "zai-org/GLM-4.6V",
    # Moonshot Kimi-Linear-48B-A3B: MoE 48B total / 3B active with linear attention
    # (may warn about a missing fla/causal-conv1d fast path and fall back to the
    # torch impl -- functional, just slower, as seen with Qwen3.5-4B). Plain
    # -Instruct, system role, non-reasoning -> no quirk sets.
    "kimi_linear_48b_a3b": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
    "olmo31_32b": "allenai/Olmo-3.1-32B-Instruct",
    # Olmo 3 7B training snapshots (Experiment C, AblationPlan). Values are LOCAL
    # paths, not HF ids: each snapshot is downloaded to /ckpts/<name> with
    # `hf download <repo> [--revision <rev>] --local-dir /ckpts/<name>` and wiped
    # after its battery run (see scripts/olmo_snapshot_lane.sh + the H100 runbook).
    # name              repo                            revision
    # s1_700k           allenai/Olmo-3-1025-7B          stage1-step700000
    # s1_final          allenai/Olmo-3-1025-7B          stage1-step1413814
    # base              allenai/Olmo-3-1025-7B          main (post stage2+3)
    # sft               allenai/Olmo-3-7B-Instruct-SFT  main
    # dpo               allenai/Olmo-3-7B-Instruct-DPO  main
    # (final RLVR point = the existing `olmo3_7b` results; no new run.)
    "olmo3_7b_s1_700k": "/ckpts/olmo3_7b_s1_700k",
    "olmo3_7b_s1_final": "/ckpts/olmo3_7b_s1_final",
    "olmo3_7b_base": "/ckpts/olmo3_7b_base",
    "olmo3_7b_sft": "/ckpts/olmo3_7b_sft",
    "olmo3_7b_dpo": "/ckpts/olmo3_7b_dpo",
    # Mistral. Dense, non-reasoning, has a system role -> no quirk sets. NOTE the
    # checkpoint is MULTIMODAL (Mistral3ForConditionalGeneration: a vision encoder
    # + projector wrapped around a 24B text model), so unlike every other panel
    # entry the decoder layers live at model.model.language_model.layers and the
    # layer count is config.text_config.num_hidden_layers (40), not the top level.
    # wrapper.py already falls back to both -- see ModelWrapper.n_layers/_layers.
    "mistral_small_31_24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # Mistral-Small-4-119B: MoE 119B total / 6.5B active (NOT a 24B dense model).
    # HYBRID reasoning: the builder only passes reasoning_effort when set, so its
    # YAML MUST pin model.reasoning_effort: "none" to suppress the CoT (see
    # experiments/main/mistral_small_4.yaml and the models.txt header). Has a
    # system role; non-thinking once effort=none -> no reasoning-set membership.
    "mistral_small_4": "mistralai/Mistral-Small-4-119B-2603",
    # OpenAI gpt-oss (MoE, harmony chat format, native MXFP4). Two entries map to
    # the SAME weights but pin different harmony `reasoning_effort` levels via
    # their experiment YAMLs; kept as distinct short names so run dirs, concept-
    # vector caches, SCORES_*.json and the cross-model comparison treat low vs
    # high as separate points (the effort changes the system prompt, so their
    # activations -- and thus concept vectors -- genuinely differ).
    "gptoss_120b_low": "openai/gpt-oss-120b",
    "gptoss_120b_medium": "openai/gpt-oss-120b",
    "gptoss_120b_high": "openai/gpt-oss-120b",
    # gpt-oss-20b (MoE, harmony, native MXFP4). Same harmony reasoning path as
    # the 120b; pinned to low effort via its YAML. -> HARMONY_MODELS below.
    "gptoss_20b_low": "openai/gpt-oss-20b",
    "gptoss_20b_medium": "openai/gpt-oss-20b",
    "gptoss_20b_high": "openai/gpt-oss-20b",
}

# Capability/quirk sets, all keyed by short name:
#   GEMMA_MODELS            - the Gemma family.
#   MODELS_WITHOUT_SYSTEM_ROLE - chat templates that reject a "system" message
#                             (Gemma); callers must fold system text into the
#                             user turn instead. Aliased to GEMMA_MODELS today.
#   BASE_MODELS             - non-instruct checkpoints lacking a chat template,
#                             so prompt wrapping falls back to "User:/Assistant:".
GEMMA_MODELS = {"gemma2_2b", "gemma2_9b", "gemma2_9b_base", "gemma2_27b", "gemma3_270m", "gemma3_27b",
                "gemma4_12b", "gemma4_31b"}
MODELS_WITHOUT_SYSTEM_ROLE = GEMMA_MODELS
BASE_MODELS = {"gemma2_9b_base",
               # Olmo pretraining/base snapshots: no chat template ->
               # User:/Assistant: fallback + base-model compliance method.
               # (sft/dpo have chat templates and are NOT base.)
               "olmo3_7b_s1_700k", "olmo3_7b_s1_final", "olmo3_7b_base"}

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
HARMONY_MODELS = {"gptoss_120b_low", "gptoss_120b_medium", "gptoss_120b_high",
                  "gptoss_20b_low", "gptoss_20b_medium", "gptoss_20b_high"}
THINK_TAG_MODELS = {"qwen35_122b_a10b_thinking",
                    # GLM-4.6V: template's enable_thinking=False switch verified
                    # (renders /nothink + a closed <think></think>), but the 2xB200
                    # smoke still came back 57.6% compliant -> stray CoT suspected.
                    # Routed through the think-tag parser, which degrades to
                    # "everything is final" on trials with no think block.
                    "glm46v"}
REASONING_MODELS = HARMONY_MODELS | THINK_TAG_MODELS
