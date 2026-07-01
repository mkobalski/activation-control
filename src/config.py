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
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ModelConfig:
    name: str = "gemma3_27b"
    config_path: str = "configs/models/gemma3_27b.yaml"
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    device: str = "cuda"
    attn_implementation: Optional[str] = None


@dataclass
class WandbConfig:
    project: str = "write-introspection-main"
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
    name: str = "write_introspection_main"
    seed: int = 42
    temperature: float = 0.0
    max_new_tokens: int = 64
    num_repetitions: int = 1
    token_buffer: int = 10
    batch_size: int = 8
    model: ModelConfig = field(default_factory=ModelConfig)
    analysis_layers: LayerSpec = field(default_factory=LayerSpec)
    prompt_layers: LayerSpec = field(default_factory=LayerSpec)
    concept_vectors: ConceptVectorConfig = field(default_factory=ConceptVectorConfig)
    concepts: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    prompt_conditions: List[PromptCondition] = field(default_factory=list)
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


def load_config(yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load the YAML, apply CLI overrides, and assemble the typed config tree.

    Every sub-dataclass is built by reading its YAML section with ``.get``
    defaults, so a sparse YAML still produces a fully-populated config. The
    ``**raw.get(...)`` calls expand a section dict directly into dataclass
    kwargs, meaning the YAML keys must line up exactly with the field names.
    """
    raw = load_yaml(yaml_path)
    if overrides:
        _apply_overrides(raw, overrides)

    # Pull each top-level section once; absent sections fall back to {} so the
    # dataclass constructors below just use their own field defaults.
    exp = raw.get("experiment", {})
    model_raw = raw.get("model", {})
    cv_raw = raw.get("concept_vectors", {})
    comp_raw = raw.get("compliance", {})
    wandb_raw = raw.get("wandb", {})

    # prompt_conditions is a list of dicts, so build each PromptCondition by
    # hand (id/template are required; kind/has_layer have defaults).
    prompt_conds = [
        PromptCondition(
            id=p["id"],
            template=p["template"],
            kind=p.get("kind", "positive"),
            has_layer=p.get("has_layer", False),
        )
        for p in raw.get("prompt_conditions", [])
    ]

    return ExperimentConfig(
        name=exp.get("name", "write_introspection_main"),
        seed=exp.get("seed", 42),
        temperature=exp.get("temperature", 0.0),
        max_new_tokens=exp.get("max_new_tokens", 64),
        num_repetitions=exp.get("num_repetitions", 1),
        token_buffer=exp.get("token_buffer", 10),
        batch_size=exp.get("batch_size", 8),
        model=ModelConfig(**model_raw),
        analysis_layers=LayerSpec(**raw.get("analysis_layers", {"fractions": []})),
        prompt_layers=LayerSpec(**raw.get("prompt_layers", {"fractions": []})),
        concept_vectors=ConceptVectorConfig(**cv_raw),
        concepts=raw.get("concepts", []),
        sentences=raw.get("sentences", []),
        prompt_conditions=prompt_conds,
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
