"""Analysis registry: map an ``analysis:`` step ``kind`` -> a callable.

An experiment's config declares which analyses to run in its ``analysis:`` block
(see src/config.py:AnalysisStep). Each step has a ``kind`` that must resolve to a
function registered here, plus params that are splatted in as kwargs. This is the
seam that lets a new experiment plug in its own analysis WITHOUT editing the
runner: standard analyses (controllability heatmaps, trace plots) register in
scripts/builtin_analyses.py; an experiment-specific analysis registers itself from
an ``analyze.py`` next to that experiment's config, which the runner imports
before dispatching.

Every registered function shares one keyword signature::

    fn(*, run_dir, results, model_name, cfg, **params) -> Any

  run_dir     Path      the run's output directory (write plots/CSVs under here)
  results     list      the in-memory list of per-trial result dicts
  model_name  str       short model name (for the concept-vector cache lookup)
  cfg         ExperimentConfig   the full typed config (for cache_dir, layers, ...)
  **params    the step's config params (ramp, sentence, metric, ...)

Functions should be BEST-EFFORT: the runner wraps each call in try/except so one
failing analysis never aborts the others or loses the already-saved run data.
"""

from typing import Any, Callable, Dict, List


# kind (str) -> analysis callable. Populated by @register at import time.
ANALYSIS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register ``fn`` under ``kind`` in ANALYSIS_REGISTRY.

    Re-registering the same ``kind`` overwrites the previous entry (last import
    wins), which is what lets an experiment's analyze.py override a built-in kind
    if it ever needs to.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        ANALYSIS_REGISTRY[kind] = fn
        return fn
    return deco


def run_analysis_steps(steps: List, *, run_dir, results, model_name, cfg,
                       verbose: bool = True) -> None:
    """Dispatch each AnalysisStep to its registered function (best-effort).

    Unknown kinds are reported and skipped; any exception inside an analysis is
    caught and reported so the remaining steps still run. Nothing here raises.
    """
    for step in steps:
        fn = ANALYSIS_REGISTRY.get(step.kind)
        if fn is None:
            print(f"  [analysis:{step.kind}] no registered analysis of this kind; "
                  f"skipping (known: {sorted(ANALYSIS_REGISTRY)})")
            continue
        if verbose:
            print(f"\nRunning analysis '{step.kind}'"
                  + (f" {step.params}" if step.params else "") + " ...")
        try:
            fn(run_dir=run_dir, results=results, model_name=model_name,
               cfg=cfg, **step.params)
        except Exception as e:  # noqa: BLE001 - deliberate fail-open per module docstring
            print(f"  [analysis:{step.kind}] skipped: {e}")
