"""Register the built-in analyses under the analysis registry.

Importing this module (done by run_experiment.py) registers the two standard
analyses so any experiment can request them by ``kind`` from its ``analysis:``
config block:

  controllability_heatmap  -> scripts/controllability_heatmap.py:generate_heatmaps
  trace_plots              -> scripts/plot_results.py:make_trace_plots

These adapters translate the registry's uniform (run_dir, results, model_name,
cfg, **params) signature into each script's own call, and pull the vector-cache
dir + extraction method off the config so a step only needs to name what DIFFERS
from the defaults (ramp, sentence, conditions, ...). Experiment-specific analyses
live in each experiment's own analyze.py, not here.

This module lives under scripts/ (not src/) because the two analyses it wraps are
scripts; run_experiment.py puts scripts/ on sys.path before importing it.
"""

from pathlib import Path

from src.analysis.registry import register
from controllability_heatmap import generate_heatmaps, DEFAULT_SENTENCE
from plot_results import make_trace_plots
from plot_layer_targeting import make_layer_targeting_plots
from plot1_concept import make_plot1_concepts


@register("controllability_heatmap")
def _controllability_heatmap(*, run_dir, results, model_name, cfg,
                             metrics=("cos", "relnorm"), sentence=DEFAULT_SENTENCE,
                             pos_conds=None, subdir=None, **kwargs):
    """Render the controllability heatmaps for one sentence, once per metric.

    `metrics` picks the channels (default: direction=cos + magnitude=relnorm).
    Remaining kwargs (ramp, ramp_name, pos_cond, neg_cond, baseline_cond, alpha)
    pass straight through to generate_heatmaps, so an experiment can point the
    ramp/poles at its own condition ids without any code change.

    `subdir` writes the output under plots/<subdir>/ instead of plots/ -- use it
    when a config runs this step more than once (e.g. two different ramps), since
    the heatmap filenames only encode the metric and would otherwise collide.

    `pos_conds` (a list) renders a SEPARATE heatmap set for EACH condition, each
    written to plots/[<subdir>/]<condition>/ so their engagement panels
    (condition - neutral) don't overwrite one another -- use it when every
    condition needs its own heatmap (e.g. the token-location / persistence
    conditions). It is mutually exclusive with a single `pos_cond` in kwargs.
    Without it, one set is written using the single pos_cond (default think_about).
    """
    cv = cfg.concept_vectors
    base_out = Path(run_dir) / "plots"
    if subdir:
        base_out = base_out / subdir

    def _render(metric, out_dir, pos_cond=None):
        extra = {} if pos_cond is None else {"pos_cond": pos_cond}
        generate_heatmaps(run_dir, model_name, vector_cache=cv.cache_dir,
                          method=cv.method, metric=metric, sentence=sentence,
                          results=results, out_dir=out_dir, **extra, **kwargs)

    if pos_conds:
        for cond in pos_conds:                      # one heatmap set per condition
            for metric in metrics:
                _render(metric, base_out / cond, pos_cond=cond)
    else:
        for metric in metrics:
            _render(metric, base_out)


@register("trace_plots")
def _trace_plots(*, run_dir, results, model_name, cfg,
                 sentence=DEFAULT_SENTENCE, layers=None, condition_ids=None, **kwargs):
    """Per-concept trace plots (plot1_cos / plot1_norms) for one sentence.

    `condition_ids` overrides which conditions become the per-line series (e.g. an
    experiment's own intensity ramp + its negative); default is the standard ramp.
    """
    make_trace_plots(results, Path(run_dir) / "plots", layers=layers,
                     sentence=sentence, condition_ids=condition_ids, **kwargs)


@register("plot1_concepts")
def _plot1_concepts(*, run_dir, results, model_name, cfg,
                    sentence=DEFAULT_SENTENCE, **params):
    """plot1_{concept}: the paper's Fig-1 family (2x2 cos/relnorm per concept).

    One figure per concept into plots/. Layers come from depth FRACTIONS
    (cos_frac=0.90, relnorm_frac=0.75 by default -> L55/L46 on gemma3-27b),
    resolved per model so the figures compare visually across models."""
    make_plot1_concepts(results, run_dir, sentence=sentence, **params)


@register("layer_targeting")
def _layer_targeting(*, run_dir, results, model_name, cfg,
                     baseline_run=None, baseline_conditions=None, **params):
    """Layer-targeting plots (7-12) for runs with layer-targeted prompts.

    Skips gracefully (prints a message, plots nothing) when the run has no
    `prompt_layer` data, so it is safe to request on any experiment.

    `baseline_run` (optional): a run dir to borrow the non-targeted baseline
    conditions from (e.g. the main run) so a layer_location run can be generated
    WITHOUT its controls. `baseline_conditions` overrides which ids to borrow.
    """
    from plot_layer_targeting import load_baseline_trials, BASELINE_CONDITIONS
    baseline_results = None
    if baseline_run:
        conds = tuple(baseline_conditions) if baseline_conditions else BASELINE_CONDITIONS
        baseline_results = load_baseline_trials(baseline_run, conds)
        print(f"  [layer_targeting] borrowed {len(baseline_results)} baseline trials "
              f"from {baseline_run}")
    make_layer_targeting_plots(results, Path(run_dir) / "plots",
                               baseline_results=baseline_results)
