"""Concept vector extraction with per-(model, layer) caching.

A concept vector for `word` at layer `L` is the activation of the last prompt token on `"Tell me about {word}"` at
layer L, optionally with a baseline subtracted (mean of the same operation
over a pool of unrelated baseline words).

The baseline subtraction is the key idea: a raw last-token activation carries
a lot of generic "answering a Tell-me-about prompt" signal shared by every
word. Averaging that activation over many semantically-random baseline words
estimates the shared component; subtracting it leaves a vector that points in
the direction specific to the concept word. Results are cached to disk per
(model, layer, method) so repeated runs skip the expensive forward passes.
"""

import torch
from pathlib import Path
from typing import Dict, List

from src.models.wrapper import ModelWrapper


DEFAULT_BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]


# Pool of semantically unrelated words used to estimate the generic
# "answering a prompt" activation that gets subtracted off (see module docstring).
def get_baseline_words(n: int = 100) -> List[str]:
    return DEFAULT_BASELINE_WORDS[:n]


def format_extraction_prompt(model: ModelWrapper, word: str,
                             template: str = "Tell me about {word}") -> str:
    # Build the prompt whose last-token activation becomes the concept vector.
    # Same chat-template-or-fallback wrapping as builder._chat_wrap, kept here so
    # vector extraction has no dependency on the prompts package.
    msg = template.format(word=word)
    if getattr(model.tokenizer, "chat_template", None):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        # Match builder._chat_wrap exactly so the concept-vector prompt format is
        # identical to the experiment prompts (same last-token context):
        #  - enable_thinking: Qwen3-style templates default to a <think> turn;
        #    pin it (default False) so the extraction prompt isn't silently in a
        #    different mode than the experiment prompt.
        #  - reasoning_effort: harmony (gpt-oss) templates read this.
        et = getattr(model, "enable_thinking", None)
        kwargs["enable_thinking"] = False if et is None else et
        effort = getattr(model, "reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort
        return model.tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], **kwargs,
        )
    return f"User: {msg}\n\nAssistant:"


def _extract_for_layer(model: ModelWrapper, concept_words: List[str],
                       baseline_words: List[str], layer_idx: int,
                       extraction_method: str, template: str,
                       token_idx: int, normalize: bool) -> Dict[str, torch.Tensor]:
    """Compute the concept vectors for a single layer (no caching here).

    Runs one forward pass over all concept prompts to get their activations,
    then computes a `subtract` term whose meaning depends on the method, and
    returns concept_act - subtract per word.
    """
    fmt = lambda w: format_extraction_prompt(model, w, template)
    concept_prompts = [fmt(w) for w in concept_words]
    # (n_concepts, d_model) activation at token_idx of layer layer_idx.
    concept_acts = model.extract_activations(concept_prompts, layer_idx, token_idx)

    # Choose what to subtract off each concept activation:
    if extraction_method == "baseline":
        # Mean activation over the unrelated baseline pool -> removes the
        # generic prompt-shaped signal, isolating the concept direction.
        baseline_prompts = [fmt(w) for w in baseline_words]
        baseline_acts = model.extract_activations(baseline_prompts, layer_idx, token_idx)
        subtract = baseline_acts.mean(dim=0)
    elif extraction_method == "simple":
        # Cheap proxy: subtract a single neutral word's activation.
        subtract = model.extract_activations([fmt("The")], layer_idx, token_idx)[0]
    elif extraction_method == "no_baseline":
        subtract = None  # use the raw activation as-is
    else:
        raise ValueError(f"Unknown extraction_method: {extraction_method}")

    out = {}
    for i, w in enumerate(concept_words):
        vec = concept_acts[i] - subtract if subtract is not None else concept_acts[i]
        if normalize:
            vec = vec / (vec.norm() + 1e-8)  # unit-length; eps guards zero vectors
        out[w] = vec
    return out


def extract_concept_vectors(
    model: ModelWrapper,
    concept_words: List[str],
    layers: List[int],
    cache_dir: str = "results/vector_cache",
    extraction_method: str = "baseline",
    template: str = "Tell me about {word}",
    token_idx: int = -1,
    normalize: bool = False,
    n_baseline_words: int = 100,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Return {layer_idx: {concept_word: tensor(d_model)}}, with FS caching.

    For each requested layer, the cache file is keyed by (model, layer, method)
    so different methods never collide. A cache hit is only honoured when it
    contains *every* requested concept word; a partial cache (e.g. new concepts
    added since it was written) triggers a full re-extraction for that layer.
    Note this re-extracts/overwrites all words rather than merging in the
    missing ones.
    """
    baseline_words = get_baseline_words(n_baseline_words)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    vectors_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx in layers:
        # One cache file per (model, layer, method); see docstring on hit rules.
        cache_path = cache_root / f"{model.model_name}_layer{layer_idx}_{extraction_method}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            if all(w in cached for w in concept_words):
                print(f"  layer {layer_idx}: loaded {len(concept_words)} cached vectors")
                # Return only the requested subset, in case the cache has extras.
                vectors_by_layer[layer_idx] = {w: cached[w] for w in concept_words}
                continue
            print(f"  layer {layer_idx}: cache incomplete, re-extracting...")

        print(f"  layer {layer_idx}: extracting {len(concept_words)} vectors...")
        vecs = _extract_for_layer(
            model, concept_words, baseline_words, layer_idx,
            extraction_method, template, token_idx, normalize,
        )
        torch.save(vecs, cache_path)
        vectors_by_layer[layer_idx] = vecs

    return vectors_by_layer
