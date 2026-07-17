"""Prompt formatting + trial schedule construction.

Prompts are formatted with the model's chat template. Templates may contain
the placeholders {sentence}, {concept}, {layer}, {total} (any subset). We pass
only the placeholders the template declares so concept-less / layer-less
conditions can share the same function. {total} is the model's layer count
(wrapper.n_layers), so a layer-targeting prompt can tell the model how many
layers it has ("layer {layer} of your {total} layers").
"""

from typing import List, Dict, Optional

from src.config import PromptCondition
from src.models.wrapper import ModelWrapper


def _chat_wrap(model: ModelWrapper, msg: str) -> str:
    # Wrap a bare user message in the model's chat format. Instruct-tuned models
    # ship a `chat_template`; if present we use it (with add_generation_prompt so
    # the string ends ready for the assistant turn). Otherwise fall back to a
    # plain "User:/Assistant:" scaffold so base models still get a sane prompt.
    if getattr(model.tokenizer, "chat_template", None):
        # enable_thinking: Qwen3-style templates default to a <think> scratchpad
        # turn, which would break verbatim transcription, so we pin it. Default
        # False (transcribe directly); a reasoning model sets model.enable_thinking
        # =True in its config to opt into the <think> trace. Templates without the
        # variable simply ignore the kwarg. Read off the wrapper (set by the
        # runner) so it stays consistent with the extraction prompt.
        et = getattr(model, "enable_thinking", None)
        kwargs = dict(tokenize=False, add_generation_prompt=True,
                      enable_thinking=(False if et is None else et))
        # reasoning_effort: harmony (gpt-oss) templates read this ("low"/"medium"/
        # "high"); passed only when set so other templates keep their defaults.
        effort = getattr(model, "reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], **kwargs,
        )
    return f"User: {msg}\n\nAssistant:"


def format_prompt(model: ModelWrapper, template: str,
                  sentence: str,
                  concept: Optional[str] = None,
                  layer: Optional[int] = None) -> str:
    # Only feed `.format` the placeholders the template actually declares, so a
    # control template with no {concept} doesn't require a concept argument.
    # `sentence` is always supplied; concept/layer are added conditionally and
    # raise if the template demands one but the caller passed None.
    kwargs = {"sentence": sentence}
    if "{concept}" in template:
        if concept is None:
            raise ValueError("Template needs {concept} but none provided")
        kwargs["concept"] = concept
    if "{layer}" in template:
        if layer is None:
            raise ValueError("Template needs {layer} but none provided")
        kwargs["layer"] = layer
    if "{total}" in template:
        kwargs["total"] = model.n_layers          # tell the model its layer count
    return _chat_wrap(model, template.format(**kwargs))


def build_trials(
    concepts: List[str],
    sentences: List[str],
    conditions: List[PromptCondition],
    prompt_layers: List[int],
    num_repetitions: int,
) -> List[Dict]:
    """Full-cross trial schedule.

    A trial dict has:
        condition_id, condition_kind, template,
        concept (or None), sentence, prompt_layer (or None), rep_idx

    The schedule is the full Cartesian product of
    repetitions x conditions x concepts x sentences x prompt_layers, but two
    axes are condition-dependent so the loops below vary their iterables:

      * Concept axis: only positive/negative conditions actually mention a
        concept word, so they iterate over `concepts`. Control/baseline
        conditions iterate over the single sentinel `[None]` -- one trial per
        sentence, with concept recorded as None.
      * Layer axis: conditions whose template embeds {layer} (`has_layer`)
        iterate over `prompt_layers`, producing one trial per target layer.
        Layer-less conditions iterate over `[None]`. A has_layer condition with
        no configured prompt_layers is skipped entirely (the feature is off).

    Using `[None]` as the placeholder iterable keeps a single nested-loop body
    for every condition kind instead of branching the append logic.
    """
    uses_concept = lambda c: c.kind in ("positive", "negative")

    trials: List[Dict] = []
    for rep in range(num_repetitions):
        for cond in conditions:
            if cond.has_layer and not prompt_layers:
                continue  # layer-targeted condition disabled

            # Pick the real axis or the [None] sentinel per the rules above.
            layers_iter = prompt_layers if cond.has_layer else [None]
            concepts_iter = concepts if uses_concept(cond) else [None]

            for concept in concepts_iter:
                for sentence in sentences:
                    for pl in layers_iter:
                        trials.append({
                            "condition_id": cond.id,
                            "condition_kind": cond.kind,
                            "template": cond.template,
                            "concept": concept,
                            "sentence": sentence,
                            "prompt_layer": pl,
                            "rep_idx": rep,
                        })
    return trials
