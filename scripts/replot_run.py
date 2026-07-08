#!/usr/bin/env python3
"""Re-render a finished run's analyses in place (CPU-only, no model load).

Loads the run's ``results.pkl`` + its experiment config, registers the built-in
and experiment-specific analysis kinds, and dispatches the config's ``analysis:``
steps -- writing figures into ``<run_dir>/plots`` and REPLACING existing files.

Use after a run gains layers (the controllability heatmaps then span
the full depth automatically, since they read each trial's ``analysis_layers``),
or any time a plotting script changes and you want to refresh a run's figures.

    python scripts/replot_run.py --run-dir results/raw/<run> \
        --config experiments/main/config.yaml

Active sets default to the run's saved ``config.yaml`` (so it matches what was
actually generated -- e.g. layer_location excluded from the 07-04 run); override
with ``--set experiment.sets=[...]`` just like run_experiment.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.config import load_config, AnalysisStep
from src.utils.io import load_results, load_run_config
from src.analysis.registry import run_analysis_steps

# Mirrors run_experiment.DEFAULT_ANALYSIS -- inlined so replotting doesn't import
# the runner (and thus torch/transformers) just to draw figures on a CPU box.
DEFAULT_ANALYSIS = [
    AnalysisStep("controllability_heatmap", {"metrics": ["cos", "relnorm"]}),
    AnalysisStep("trace_plots", {}),
]


def _import_experiment_analyze(config_path):
    """Import an experiment's sibling analyze.py (registers its custom kinds)."""
    analyze_py = Path(config_path).resolve().parent / "analyze.py"
    if not analyze_py.exists():
        return
    spec = importlib.util.spec_from_file_location("experiment_analyze", analyze_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"Loaded experiment analyses from {analyze_py}")


def _parse_overrides(set_args):
    overrides = {}
    for s in set_args:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        overrides[k] = v
    return overrides


def main():
    ap = argparse.ArgumentParser(description="Re-render a run's analyses in place.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True, help="the experiment config the run used")
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help="override config values (e.g. experiment.sets=[intensity])")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config, overrides=_parse_overrides(args.sets))

    # Match the sets actually generated (from the run's saved config), unless the
    # user overrode experiment.sets on the CLI.
    if not any(s.split("=", 1)[0] == "experiment.sets" for s in args.sets):
        saved = load_run_config(run_dir)
        if saved.get("active_sets"):
            config.active_sets = list(saved["active_sets"])

    # Register built-in + experiment-specific analysis kinds.
    import builtin_analyses  # noqa: F401  (import registers the built-in kinds)
    _import_experiment_analyze(args.config)

    results, _ = load_results(run_dir / "results")
    layers = sorted({int(x) for r in results for x in (r.get("analysis_layers") or [])})
    print(f"Loaded {len(results)} trials; recorded layers = {layers}")
    print(f"active_sets = {config.active_sets}")

    steps = config.analysis or DEFAULT_ANALYSIS
    active, skipped = [], []
    for s in steps:
        (active if (s.set is None or s.set in config.active_sets) else skipped).append(s)
    for s in skipped:
        print(f"  [skip] {s.kind}: set '{s.set}' not active")

    run_analysis_steps(active, run_dir=run_dir, results=results,
                       model_name=config.model.name, cfg=config)
    print(f"\nRe-rendered {len(active)} analysis step(s) into {run_dir/'plots'}")


if __name__ == "__main__":
    main()
