"""Configuration loading from YAML + CLI overrides.

The whole experiment is driven by a single YAML file (default
``configs/experiment.yaml``). This module mirrors that YAML structure as a
tree of frozen-ish ``@dataclass`` objects so the rest of the codebase reads
typed attributes (``cfg.model.name``) instead of raw dict lookups.

Flow: ``parse_cli_args`` reads ``--config`` and any ``--set a.b.c=value``
overrides off the command line, ``load_config`` loads the YAML, splices the
overrides into the raw dict, then hand-builds each dataclass from the
corresponding YAML sub-section (filling defaults for anything missing).
"""

import argparse
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Project root (src/config.py -> ..); used to resolve a relative `sentences_file`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_sentences_file(path: str) -> List[str]:
    """Read sentences from a ``sentences.txt``-style file -> list of strings.

    Each non-blank, non-``#`` line is one sentence; an optional ``s<N>:`` / ``p<N>:``
    label prefix is stripped (so the file stays human-readable/renumberable while
    the code just consumes the text, in file order). A relative path resolves
    against the project root, so ``sentences_file: sentences.txt`` works from any CWD.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    out: List[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^[sp]\d+:\s*(.*)$", line)
        out.append(m.group(1) if m else line)
    return out


def _resolve_sentences(raw: Dict[str, Any]) -> List[str]:
    """Sentences from an explicit ``sentences:`` list, else from ``sentences_file``."""
    if raw.get("sentences"):
        return raw["sentences"]
    if raw.get("sentences_file"):
        return load_sentences_file(raw["sentences_file"])
    return []


@dataclass
class ModelConfig:
    name: str = "gemma3_27b"
    config_path: str = "configs/models/gemma3_27b.yaml"
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    device: str = "cuda"
    attn_implementation: Optional[str] = None
    # Reasoning effort for models whose chat template supports it (e.g. gpt-oss
    # harmony: "low"/"medium"/"high"). None = don't pass the kwarg (templates
    # that don't declare it ignore it anyway). Threaded into the chat template
    # via ModelWrapper.reasoning_effort (see run_experiment + builder).
    reasoning_effort: Optional[str] = None
    # Chain-of-thought toggle for Qwen3-style templates that read `enable_thinking`.
    # None keeps the codebase default (False — transcribe directly, no <think>
    # turn); True opts a model into emitting a <think>...</think> trace before the
    # answer (the reasoning path then records/aligns on the post-</think> answer;
    # see src/utils/think_tags.py). Threaded into the chat template via
    # ModelWrapper.enable_thinking (see run_experiment + builder). Templates
    # without the variable ignore it.
    enable_thinking: Optional[bool] = None


@dataclass
class WandbConfig:
    project: str = "activation-control"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)


# Controls how concept vectors are built (see src/vectors/extraction.py).
# `method` selects the baseline-subtraction strategy; `template` is the
# extraction prompt; `token_idx` picks which token's activation to read.
@dataclass
class ConceptVectorConfig:
    method: str = "baseline"
    template: str = "Tell me about {word}"
    token_idx: int = -1
    normalize: bool = False
    n_baseline_words: int = 100
    cache_dir: str = "results/vector_cache"


@dataclass
class ComplianceConfig:
    method: str = "normalized_levenshtein"
    threshold: float = 0.85


# One experimental prompt variant. `kind` drives whether a concept word is
# substituted into the trial (positive/negative use a concept; control/baseline
# do not -- see builder.uses_concept). `has_layer` flags templates that embed a
# layer index ({layer}) and therefore fan out across prompt_layers.
@dataclass
class PromptCondition:
    id: str
    template: str
    kind: str                  # positive | negative | control | baseline
    has_layer: bool = False


@dataclass
class LayerSpec:
    """Layer selection: fractional depths (0-1) and/or explicit integer indices.

    Both lists are merged and de-duped at runtime against wrapper.n_layers.
    Set both to [] to disable the corresponding layer-targeted feature.
    """
    fractions: List[float] = field(default_factory=list)
    layers: List[int] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    name: str = "activation_control"
    seed: int = 42
    temperature: float = 0.0
    max_new_tokens: int = 64
    num_repetitions: int = 1
    token_buffer: int = 10
    batch_size: int = 8
    # Save residual-stream recordings for SPECIAL tokens too: (a) the generated
    # tail after the sentence span (e.g. <end_of_turn>/EOS, already recorded but
    # previously discarded at save) and (b) the prompt's special tokens
    # (<start_of_turn> etc.), captured during prefill.
    record_special_tokens: bool = True
    # Store results.pkl activation tensors in the lossless bf16 codec (uint16
    # carrier) instead of fp32 -- halves the pickle. Loaders auto-detect, so
    # legacy fp32 pickles still load. Set false to force fp32.
    pickle_bf16: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis_layers: LayerSpec = field(default_factory=LayerSpec)
    prompt_layers: LayerSpec = field(default_factory=LayerSpec)
    concept_vectors: ConceptVectorConfig = field(default_factory=ConceptVectorConfig)
    concepts: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    # Optional: source `sentences` from a text file (see load_sentences_file). An
    # explicit `sentences:` list in the config takes precedence over this.
    sentences_file: Optional[str] = None
    # Optional subset: indices into `sentences` to actually run this run. Empty =
    # run all. Indices are global/stable (s6 == cat), so plots keep their labels.
    sentence_indices: List[int] = field(default_factory=list)
    prompt_conditions: List[PromptCondition] = field(default_factory=list)
    # Named groups of extra conditions (top-level `condition_sets:` YAML key).
    # Active sets' conditions are merged into `prompt_conditions` at build time;
    # this field keeps ALL declared sets for reference/bookkeeping.
    condition_sets: Dict[str, List[PromptCondition]] = field(default_factory=dict)
    # Resolved list of active set names (see `experiment.sets`): absent/null ->
    # every declared set; explicit [] -> none (controls only). Stored in
    # condition_sets declaration order (the merge order), not selection order.
    active_sets: List[str] = field(default_factory=list)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_base_dir: str = "results"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(raw: Dict, overrides: Dict[str, Any]):
    """Dot-path overrides applied in-place to the raw dict.

    Each override key like ``experiment.seed`` is split on ``.`` into a path of
    nested dict keys. We walk every key except the last with ``setdefault`` so
    intermediate dicts are created on demand (an override can target a section
    the YAML never mentioned), then assign ``value`` at the final leaf key.
    """
    for key, value in overrides.items():
        parts = key.split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})  # descend, creating empty dicts as needed
        d[parts[-1]] = value


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: ``over`` layered on top of ``base``.

    Nested dicts are merged recursively (so a child config can tweak a single key
    inside, say, the ``experiment`` block without restating the whole block); any
    non-dict value (scalar OR list) in ``over`` replaces the base value wholesale.
    Lists are replaced rather than concatenated on purpose: an experiment that
    lists its own ``prompt_conditions`` gets exactly those, with no surprise
    leftovers from the base.
    """
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_extends(raw: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Resolve a chain of ``extends: <relative/path.yaml>`` inheritance.

    A config may set ``extends`` to a path (relative to that config's own
    directory) of a base config to inherit from. We load the base, resolve ITS
    ``extends`` first (so chains work), then layer the child on top via
    ``_deep_merge``. This lets each experiment config declare only what differs
    from a shared base (its conditions + analysis) instead of copying the whole
    concepts/sentences/model block.
    """
    parent = raw.pop("extends", None)
    if not parent:
        return raw
    parent_path = (Path(base_dir) / parent).resolve()
    parent_raw = load_yaml(str(parent_path))
    parent_raw = _resolve_extends(parent_raw, parent_path.parent)
    return _deep_merge(parent_raw, raw)


def load_config(yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load the YAML (resolving ``extends``), apply CLI overrides, and assemble
    the typed config tree.

    Order matters: base configs are merged in first (``extends``), THEN the
    ``--set`` CLI overrides are spliced on top, so a command-line override always
    wins over both the experiment config and any base it inherits from.
    """
    raw = load_yaml(yaml_path)
    raw = _resolve_extends(raw, Path(yaml_path).resolve().parent)
    if overrides:
        _apply_overrides(raw, overrides)
    return _build_config(raw)


def _build_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Assemble the typed config tree from a fully-merged raw dict.

    Every sub-dataclass is built by reading its section with ``.get`` defaults, so
    a sparse config still produces a fully-populated tree. The ``**raw.get(...)``
    calls expand a section dict directly into dataclass kwargs, meaning the keys
    must line up exactly with the field names. Kept separate from ``load_config``
    (no file/YAML I/O) so the assembly logic can be unit-tested with a plain dict.
    """
    # Pull each top-level section once; absent sections fall back to {} so the
    # dataclass constructors below just use their own field defaults.
    exp = raw.get("experiment", {})
    model_raw = raw.get("model", {})
    cv_raw = raw.get("concept_vectors", {})
    comp_raw = raw.get("compliance", {})
    wandb_raw = raw.get("wandb", {})

    # prompt_conditions is a list of dicts, so build each PromptCondition by
    # hand (id/template are required; kind/has_layer have defaults).
    def _parse_condition(p: Dict[str, Any]) -> PromptCondition:
        return PromptCondition(
            id=p["id"],
            template=p["template"],
            kind=p.get("kind", "positive"),
            has_layer=p.get("has_layer", False),
        )

    # Top-level `prompt_conditions` are the always-on controls.
    prompt_conds = [_parse_condition(p) for p in raw.get("prompt_conditions", [])]

    # Optional named condition sets: {set_name: [condition dicts...]}. Each
    # entry has the same shape as a prompt_conditions entry.
    condition_sets: Dict[str, List[PromptCondition]] = {
        name: [_parse_condition(p) for p in (conds or [])]
        for name, conds in (raw.get("condition_sets") or {}).items()
    }

    # `experiment.sets` selects which sets are active:
    #   absent / null -> ALL declared sets; explicit [] -> NONE (controls only);
    #   unknown name  -> error. active_sets is stored (and merged) in
    #   condition_sets declaration order, regardless of selection order.
    sets_requested = exp.get("sets")
    if sets_requested is None:
        active_sets = list(condition_sets.keys())
    else:
        unknown = [s for s in sets_requested if s not in condition_sets]
        if unknown:
            raise ValueError(
                f"Unknown condition set(s) {unknown} in experiment.sets; "
                f"known sets: {sorted(condition_sets.keys())}"
            )
        active_sets = [name for name in condition_sets if name in sets_requested]

    # Final conditions = controls, then each active set's conditions in
    # declaration order. Condition ids must be globally unique after the merge.
    merged_conds = list(prompt_conds)
    for name in active_sets:
        merged_conds.extend(condition_sets[name])
    seen_ids = set()
    for c in merged_conds:
        if c.id in seen_ids:
            raise ValueError(
                f"Duplicate prompt condition id {c.id!r} after merging "
                f"condition_sets (active sets: {active_sets})"
            )
        seen_ids.add(c.id)

    return ExperimentConfig(
        name=exp.get("name", "activation_control"),
        seed=exp.get("seed", 42),
        temperature=exp.get("temperature", 0.0),
        max_new_tokens=exp.get("max_new_tokens", 64),
        num_repetitions=exp.get("num_repetitions", 1),
        token_buffer=exp.get("token_buffer", 10),
        batch_size=exp.get("batch_size", 8),
        record_special_tokens=exp.get("record_special_tokens", True),
        pickle_bf16=exp.get("pickle_bf16", True),
        model=ModelConfig(**model_raw),
        analysis_layers=LayerSpec(**raw.get("analysis_layers", {"fractions": []})),
        prompt_layers=LayerSpec(**raw.get("prompt_layers", {"fractions": []})),
        concept_vectors=ConceptVectorConfig(**cv_raw),
        concepts=raw.get("concepts", []),
        sentences=_resolve_sentences(raw),
        sentences_file=raw.get("sentences_file"),
        sentence_indices=exp.get("sentence_indices", []),
        prompt_conditions=merged_conds,
        condition_sets=condition_sets,
        active_sets=active_sets,
        compliance=ComplianceConfig(**comp_raw),
        wandb=WandbConfig(**wandb_raw),
        output_base_dir=raw.get("output", {}).get("base_dir", "results"),
    )


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI args and turn repeated ``--set`` flags into an overrides dict.

    ``--set`` uses ``action="append"`` so it can be passed many times. Each
    value is ``dotted.key=raw`` and is split on the first ``=`` only (so the
    value may itself contain ``=``). The right-hand side is run through
    ``yaml.safe_load`` so scalars get their natural types -- ``7`` -> int,
    ``true`` -> bool, ``[a, b]`` -> list -- instead of staying strings; if that
    parse raises, we leave the value as the original string.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--set", dest="sets", action="append", default=[],
                   help="Override config: --set experiment.seed=7")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-pickle", action="store_true",
                   help="skip saving the full results.pkl (per-trial activation "
                        "arrays; tens-to-hundreds of GB). results.json (traces) "
                        "and no_instruction_cache.pkl (baseline activations) are "
                        "always saved and are all the score/figure pipeline needs.")
    args = p.parse_args()
    overrides = {}
    for s in args.sets:
        if "=" not in s:
            continue  # malformed override, skip silently
        k, v = s.split("=", 1)
        try:
            v = yaml.safe_load(v)  # coerce scalar to its YAML type
        except Exception:
            pass  # not parseable as YAML -> keep the raw string
        overrides[k] = v
    args.overrides = overrides
    return args
