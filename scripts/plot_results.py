#!/usr/bin/env python3
"""Per-concept, per-token trace plots for a completed run directory.

Two plots, for one sentence at one analysis layer, as a per-concept grid (one
subplot per concept, one line per condition = the intensity ramp think_intensity
1..4 + dont_think_about; ramp light->dark red, dont in gray):

  plot1_cos    y = cos(concept vector, residual)   (direction)
  plot1_norms  y = ||residual||                    (magnitude)

These are generated automatically at the end of run_experiment.py (via
make_trace_plots) for the default sentence + deepest layer, and can also be run
standalone. Reads only results.pkl -- no GPU / model load.

(Layer-targeting plots 7-12 live in scripts/plot_layer_targeting.py and are run
manually, only for runs that used layer-targeted prompts.)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_results


CAT = "The cat watched the bird through the window by the door near the garden."
# One line per condition: the intensity ramp (light->dark red) + the negative (gray).
_INTENSITY_IDS = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
_NEGATIVE_ID = "dont_think_about"


def _compliant(results):
    return [r for r in results if r["is_compliant"]]


def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot1_traces(results, out_dir: Path, analysis_layer: int, sentence_idx: int,
                 *, metric_key: str, metric_label: str, fname_tag: str,
                 condition_ids=None, draw_zero=False):
    """Per-concept grid of per-token traces for ONE sentence at ONE layer.

    Each subplot is a concept; each line is one condition, reading the trace from
    trial[metric_key][layer]. `metric_key` = "cosine_sim_anchored" (plot1_cos) or
    "norms_anchored" (plot1_norms). Returns the written path, or None if skipped.
    """
    if condition_ids is None:
        condition_ids = _INTENSITY_IDS + [_NEGATIVE_ID]
    compliant = _compliant(results)

    sentences = []
    for r in compliant:
        if r["sentence"] not in sentences:
            sentences.append(r["sentence"])
    if sentence_idx >= len(sentences):
        print(f"plot1_{fname_tag}: sentence_idx {sentence_idx} out of range")
        return None
    target_sent = sentences[sentence_idx]

    concepts = []
    for r in compliant:
        if r["concept"] and r["concept"] not in concepts:
            concepts.append(r["concept"])

    # lookup[(concept, condition_id)] -> trial (for this sentence)
    lookup = {}
    for r in compliant:
        if r["sentence"] == target_sent and r["condition_id"] in condition_ids:
            lookup[(r["concept"], r["condition_id"])] = r

    # Reds ramp for the positives (light->dark = low->high intensity); gray negative.
    pos_ids = [c for c in condition_ids if c != _NEGATIVE_ID]
    reds = plt.get_cmap("Reds")(np.linspace(0.35, 0.95, max(len(pos_ids), 1)))
    color_map = {cid: reds[i] for i, cid in enumerate(pos_ids)}
    color_map[_NEGATIVE_ID] = (0.5, 0.5, 0.5, 1.0)
    colors = [color_map[c] for c in condition_ids]

    n = len(concepts)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    any_data = False
    for i, concept in enumerate(concepts):
        ax = axes[i // ncols][i % ncols]
        toks_labeled = None
        for k, cid in enumerate(condition_ids):
            r = lookup.get((concept, cid))
            if r is None:
                continue
            trace = r[metric_key].get(str(analysis_layer)) or r[metric_key].get(analysis_layer)
            if not trace:
                continue
            any_data = True
            ax.plot(range(len(trace)), trace, marker="o", markersize=3,
                    linewidth=1, color=colors[k], label=cid)
            if toks_labeled is None:
                toks = r["anchored_token_strs"]
                toks_labeled = (["anchor"] + toks[1:]) if toks else toks
        if toks_labeled:
            ax.set_xticks(range(len(toks_labeled)))
            ax.set_xticklabels([t.strip() for t in toks_labeled], rotation=45,
                               ha="right", fontsize=7)
        ax.axvline(0.5, linestyle="--", alpha=0.3, color="gray")
        if draw_zero:
            ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.set_title(f"concept = {concept}", fontsize=10)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(metric_label)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if not any_data:
        plt.close(fig)
        print(f"plot1_{fname_tag}: no '{metric_key}' data at layer {analysis_layer}; skipped")
        return None

    handles = [Line2D([0], [0], color=colors[k], marker="o", markersize=4,
                      linewidth=1.5, label=cid) for k, cid in enumerate(condition_ids)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{metric_label} at layer {analysis_layer} | sentence: \"{target_sent}\"",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fname = f"plot1_{fname_tag}_L{analysis_layer}_s{sentence_idx}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot1_{fname_tag}: wrote {fname} (layer={analysis_layer}, s{sentence_idx}, "
          f"{len(concepts)} concepts)")
    return str(out_dir / fname)


def make_trace_plots(results, out_dir, *, layers=None, sentence=CAT, verbose=True):
    """Generate plot1_cos + plot1_norms for `sentence` at each of `layers`.

    Callable from run_experiment.py. Defaults to the deepest analysis layer.
    Returns the list of written paths (empty if the sentence isn't present).
    """
    out_dir = _ensure_dir(Path(out_dir))
    compliant = _compliant(results)
    sentences = []
    for r in compliant:
        if r["sentence"] not in sentences:
            sentences.append(r["sentence"])
    if sentence not in sentences:
        if verbose:
            print("  [skip trace plots] default sentence not in this run")
        return []
    sidx = sentences.index(sentence)
    if layers is None:
        al = sorted(int(x) for x in (results[0].get("analysis_layers") or []))
        layers = [al[-1]] if al else []

    paths = []
    for L in layers:
        p = plot1_traces(results, out_dir, L, sidx, metric_key="cosine_sim_anchored",
                         metric_label="cos(concept vec, residual)", fname_tag="cos",
                         draw_zero=True)
        if p:
            paths.append(p)
        p = plot1_traces(results, out_dir, L, sidx, metric_key="norms_anchored",
                         metric_label="‖residual‖", fname_tag="norms")
        if p:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Per-concept trace plots (plot1_cos, plot1_norms).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="analysis layers to plot (default: deepest recorded layer)")
    ap.add_argument("--sentence", default=CAT, help="sentence to plot (default: cat sentence)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    results, _ = load_results(run_dir / "results")
    print(f"Loaded {len(results)} trials")
    make_trace_plots(results, run_dir / "plots", layers=args.layers, sentence=args.sentence)


if __name__ == "__main__":
    main()
