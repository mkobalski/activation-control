#!/usr/bin/env python3
"""Layer-targeting plots (plots 7-12) for a completed run directory.

Only meaningful for runs that used LAYER-TARGETED prompts (`prompt_layers`
non-empty in the config, e.g. configs/experiment_layer_target_deep.yaml) -- the
`*_at_layer` conditions that name a specific layer in the prompt. This script is
NOT run automatically by run_experiment.py; invoke it manually on such a run:

    python scripts/plot_layer_targeting.py --run-dir results/raw/<RUN_DIR>

It asks: do prompts that tell the model to think "at layer L" actually move the
residual most at analysis_layer == L? Each plot compares a `*_at_layer` positive
against its non-targeted baseline as a per-token Δ (line panels, plots 7/8/10/11)
or a token-averaged Δ heatmap (plots 9/12), with the prompt_layer == analysis_layer
diagonal highlighted. Reads only results.pkl -- no GPU / model load.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_results


def _compliant(results):
    return [r for r in results if r["is_compliant"]]


def _mean_over_tokens(trace):
    arr = np.asarray(trace, dtype=np.float32)
    return float(arr.mean()) if arr.size else np.nan


def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mean_trace(compliant, condition_id, analysis_layer, metric_key,
                prompt_layer=None):
    """Shared reducer for the layer-targeting plots: gather every matching trial's
    per-token trace (`metric_key` dict at `analysis_layer`, optionally filtered to
    one `prompt_layer`), nan-pad to a common length, and nan-average per position.
    Returns None when no trial matches so callers can skip empty cells."""
    arrs = []
    for r in compliant:
        if r["condition_id"] != condition_id:
            continue
        if prompt_layer is not None and r.get("prompt_layer") != prompt_layer:
            continue
        d = r.get(metric_key) or {}
        trace = d.get(str(analysis_layer), d.get(analysis_layer))
        if not trace:
            continue
        arrs.append(np.asarray(trace, dtype=np.float32))
    if not arrs:
        return None
    max_len = max(len(a) for a in arrs)
    padded = np.full((len(arrs), max_len), np.nan, dtype=np.float32)
    for k, a in enumerate(arrs):
        padded[k, :len(a)] = a
    return np.nanmean(padded, axis=0)


def plot_layer_targeting(results, out_dir: Path, *,
                         positive_id: str, baseline_id: str,
                         metric_key: str, metric_label: str,
                         file_prefix: str,
                         prompt_layers=(8, 16, 25, 33),
                         analysis_layers=(8, 16, 25, 33)):
    """One figure: 4 panels (prompt_layer), 4 lines per panel (analysis_layer).
    Values are Δ = positive − baseline (per-token, matched at analysis_layer).
    Diagonal (analysis_layer == prompt_layer) is drawn thicker."""
    from matplotlib.lines import Line2D
    compliant = _compliant(results)

    baseline_traces = {
        aL: _mean_trace(compliant, baseline_id, aL, metric_key)
        for aL in analysis_layers
    }
    if all(v is None for v in baseline_traces.values()):
        print(f"{file_prefix}: no baseline data for {baseline_id}; skipping")
        return

    viridis = plt.get_cmap("viridis")
    n = max(len(analysis_layers) - 1, 1)
    colors = {aL: viridis(i / n) for i, aL in enumerate(analysis_layers)}

    fig, axes = plt.subplots(1, len(prompt_layers),
                             figsize=(4.2 * len(prompt_layers), 3.8),
                             sharey=True, squeeze=False)
    for j, pL in enumerate(prompt_layers):
        ax = axes[0][j]
        for aL in analysis_layers:
            pos = _mean_trace(compliant, positive_id, aL, metric_key,
                              prompt_layer=pL)
            base = baseline_traces[aL]
            if pos is None or base is None:
                continue
            m = min(len(pos), len(base))
            delta = pos[:m] - base[:m]
            is_diag = (aL == pL)
            ax.plot(range(m), delta, marker="o", markersize=3,
                    linewidth=2.4 if is_diag else 1.0,
                    color=colors[aL],
                    alpha=1.0 if is_diag else 0.7)
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.set_title(f"prompt_layer = {pL}")
        ax.set_xlabel("Token position")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"Δ {metric_label} (vs {baseline_id})")

    legend_handles = [
        Line2D([0], [0], color=colors[aL], marker="o", markersize=5,
               linewidth=2.0, label=f"analysis_layer = {aL}")
        for aL in analysis_layers
    ] + [
        Line2D([0], [0], color="k", lw=2.4,
               label="bold = diagonal (targeted)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f"{positive_id} − {baseline_id}: per-token Δ {metric_label}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fname = f"{file_prefix}_{positive_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{file_prefix}: wrote {fname}")


# ─── Plot 9: prompt_layer × analysis_layer heatmap of Δ (token-averaged) ────

def plot_layer_targeting_heatmap(results, out_dir: Path, *,
                                 positive_id: str, baseline_id: str,
                                 metric_key: str, metric_label: str,
                                 file_prefix: str,
                                 prompt_layers=(8, 16, 25, 33),
                                 analysis_layers=(8, 16, 25, 33)):
    """Heatmap of mean Δ (positive − baseline) across tokens, with
    rows = prompt_layer (asked), cols = analysis_layer (measured).
    If layer-targeting works, the diagonal should be brighter than
    off-diagonal cells."""
    compliant = _compliant(results)

    baseline_traces = {
        aL: _mean_trace(compliant, baseline_id, aL, metric_key)
        for aL in analysis_layers
    }
    if all(v is None for v in baseline_traces.values()):
        print(f"{file_prefix}: no baseline data for {baseline_id}; skipping")
        return

    mat = np.full((len(prompt_layers), len(analysis_layers)), np.nan,
                  dtype=np.float32)
    for i, pL in enumerate(prompt_layers):
        for j, aL in enumerate(analysis_layers):
            pos = _mean_trace(compliant, positive_id, aL, metric_key,
                              prompt_layer=pL)
            base = baseline_traces[aL]
            if pos is None or base is None:
                continue
            m = min(len(pos), len(base))
            if m == 0:
                continue
            mat[i, j] = float(np.nanmean(pos[:m] - base[:m]))

    if np.all(np.isnan(mat)):
        print(f"{file_prefix}: no data for {positive_id}; skipping")
        return

    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(analysis_layers),
                                    1.2 + 0.9 * len(prompt_layers)))
    vmax = float(np.nanmax(np.abs(mat)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(analysis_layers)))
    ax.set_xticklabels(analysis_layers)
    ax.set_yticks(range(len(prompt_layers)))
    ax.set_yticklabels(prompt_layers)
    ax.set_xlabel("analysis_layer (measured)")
    ax.set_ylabel("prompt_layer (asked)")

    for i, pL in enumerate(prompt_layers):
        for j, aL in enumerate(analysis_layers):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if abs(v) > 0.5 * vmax else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, color=txt_color)
            if pL == aL:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", lw=2.0))

    fig.colorbar(im, ax=ax, label=f"Mean Δ {metric_label}")
    ax.set_title(
        f"{positive_id} − {baseline_id}: mean Δ {metric_label}\n"
        f"(token-averaged; boxed cells = diagonal / targeted)",
        fontsize=11)
    fig.tight_layout()
    fname = f"{file_prefix}_{positive_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{file_prefix}: wrote {fname}")


# ─── Plot 9 (per-concept): grid of heatmaps, one per concept ───────────────

def plot_layer_targeting_heatmap_by_concept(
        results, out_dir: Path, *,
        positive_id: str, baseline_id: str,
        metric_key: str, metric_label: str,
        file_prefix: str,
        prompt_layers=(8, 16, 25, 33),
        analysis_layers=(8, 16, 25, 33)):
    """One heatmap per concept (prompt_layer × analysis_layer), shared color
    scale so concepts are directly comparable."""
    compliant = _compliant(results)
    concepts = []
    for r in compliant:
        if r["concept"] and r["concept"] not in concepts:
            concepts.append(r["concept"])
    if not concepts:
        print(f"{file_prefix}: no concepts found; skipping")
        return

    by_concept = {}
    for concept in concepts:
        sub = [r for r in compliant if r["concept"] == concept]
        base = {aL: _mean_trace(sub, baseline_id, aL, metric_key)
                for aL in analysis_layers}
        mat = np.full((len(prompt_layers), len(analysis_layers)),
                      np.nan, dtype=np.float32)
        for i, pL in enumerate(prompt_layers):
            for j, aL in enumerate(analysis_layers):
                pos = _mean_trace(sub, positive_id, aL, metric_key,
                                  prompt_layer=pL)
                b = base[aL]
                if pos is None or b is None:
                    continue
                m = min(len(pos), len(b))
                if m == 0:
                    continue
                mat[i, j] = float(np.nanmean(pos[:m] - b[:m]))
        by_concept[concept] = mat

    stacked = np.concatenate([m.ravel() for m in by_concept.values()])
    if np.all(np.isnan(stacked)):
        print(f"{file_prefix}: no data for {positive_id}; skipping")
        return
    vmax = float(np.nanmax(np.abs(stacked)))

    n = len(concepts)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    im = None
    for idx, concept in enumerate(concepts):
        ax = axes[idx // ncols][idx % ncols]
        mat = by_concept[concept]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(analysis_layers)))
        ax.set_xticklabels(analysis_layers, fontsize=8)
        ax.set_yticks(range(len(prompt_layers)))
        ax.set_yticklabels(prompt_layers, fontsize=8)
        ax.set_title(concept, fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel("prompt_layer")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("analysis_layer")
        for i, pL in enumerate(prompt_layers):
            for j, aL in enumerate(analysis_layers):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = "white" if abs(v) > 0.5 * vmax else "black"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=7, color=txt_color)
                if pL == aL:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="black", lw=1.5))

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"{positive_id} − {baseline_id}: mean Δ {metric_label} by concept\n"
        f"(token-averaged; boxed = diagonal / targeted)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 0.95, 0.95])
    cbar_ax = fig.add_axes([0.96, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"Mean Δ {metric_label}")
    fname = f"{file_prefix}_{positive_id}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{file_prefix}: wrote {fname}")


# ─── Plot 10: per-concept diagonal decomposition of plot9 ──────────────────

def plot10_per_concept_diagonal(results, out_dir: Path, *,
                                positive_id: str, baseline_id: str,
                                metric_key: str, metric_label: str,
                                file_prefix: str,
                                layers=(8, 16, 25, 33)):
    """One figure per concept. Subplot per layer L: baseline trace at
    analysis_layer=L overlaid with the targeted trace where
    prompt_layer == analysis_layer == L (the diagonal of plot9)."""
    from matplotlib.lines import Line2D

    compliant = _compliant(results)
    concepts = []
    for r in compliant:
        if r["concept"] and r["concept"] not in concepts:
            concepts.append(r["concept"])

    color_base = (0.5, 0.5, 0.5, 1.0)
    color_targ = plt.get_cmap("Reds")(0.75)

    for concept in concepts:
        sub = [r for r in compliant if r["concept"] == concept]
        baseline_traces = {L: _mean_trace(sub, baseline_id, L, metric_key)
                           for L in layers}
        targeted_traces = {L: _mean_trace(sub, positive_id, L, metric_key,
                                          prompt_layer=L)
                           for L in layers}

        if all(v is None for v in targeted_traces.values()):
            print(f"{file_prefix}: no data for {concept}; skipping")
            continue

        ymin, ymax = np.inf, -np.inf
        for tr in list(baseline_traces.values()) + list(targeted_traces.values()):
            if tr is None:
                continue
            ymin = min(ymin, float(np.nanmin(tr)))
            ymax = max(ymax, float(np.nanmax(tr)))
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01
        ylim = (ymin - pad, ymax + pad)

        fig, axes = plt.subplots(1, len(layers),
                                 figsize=(4.2 * len(layers), 3.6),
                                 sharey=True, squeeze=False)
        for j, L in enumerate(layers):
            ax = axes[0][j]
            base = baseline_traces[L]
            tgt = targeted_traces[L]
            if base is not None:
                ax.plot(range(len(base)), base, marker="o", markersize=3,
                        linewidth=1.2, color=color_base, label=baseline_id)
            if tgt is not None:
                ax.plot(range(len(tgt)), tgt, marker="o", markersize=3,
                        linewidth=1.2, color=color_targ,
                        label=f"{positive_id} @ L{L}")
            ax.axhline(0, color="k", lw=0.4, alpha=0.5)
            ax.set_title(f"layer {L}")
            ax.set_xlabel("Token position")
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"Mean {metric_label}")

        legend_handles = [
            Line2D([0], [0], color=color_base, marker="o",
                   markersize=4, linewidth=1.5, label=baseline_id),
            Line2D([0], [0], color=color_targ, marker="o",
                   markersize=4, linewidth=1.5,
                   label=f"{positive_id} (prompt_layer = analysis_layer)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(
            f"concept = {concept} | {positive_id} vs {baseline_id} (diagonal)",
            fontsize=12)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fname = f"{file_prefix}_{concept}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{file_prefix}: wrote {fname}")


# ─── Plot 11: per-concept diagonal, two targeted conditions overlaid ───────

def plot11_per_concept_two_targeted(results, out_dir: Path, *,
                                    positive_id_a: str, positive_id_b: str,
                                    metric_key: str, metric_label: str,
                                    file_prefix: str,
                                    layers=(8, 16, 25, 33)):
    """One figure per concept. Subplot per layer L overlays two
    layer-targeted conditions with prompt_layer = analysis_layer = L."""
    from matplotlib.lines import Line2D

    compliant = _compliant(results)
    concepts = []
    for r in compliant:
        if r["concept"] and r["concept"] not in concepts:
            concepts.append(r["concept"])

    color_a = plt.get_cmap("Reds")(0.55)
    color_b = plt.get_cmap("Reds")(0.9)

    for concept in concepts:
        sub = [r for r in compliant if r["concept"] == concept]
        traces_a = {L: _mean_trace(sub, positive_id_a, L, metric_key,
                                   prompt_layer=L) for L in layers}
        traces_b = {L: _mean_trace(sub, positive_id_b, L, metric_key,
                                   prompt_layer=L) for L in layers}

        if (all(v is None for v in traces_a.values())
                and all(v is None for v in traces_b.values())):
            print(f"{file_prefix}: no data for {concept}; skipping")
            continue

        ymin, ymax = np.inf, -np.inf
        for tr in list(traces_a.values()) + list(traces_b.values()):
            if tr is None:
                continue
            ymin = min(ymin, float(np.nanmin(tr)))
            ymax = max(ymax, float(np.nanmax(tr)))
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.01
        ylim = (ymin - pad, ymax + pad)

        fig, axes = plt.subplots(1, len(layers),
                                 figsize=(4.2 * len(layers), 3.6),
                                 sharey=True, squeeze=False)
        for j, L in enumerate(layers):
            ax = axes[0][j]
            ta = traces_a[L]
            tb = traces_b[L]
            if ta is not None:
                ax.plot(range(len(ta)), ta, marker="o", markersize=3,
                        linewidth=1.2, color=color_a, label=positive_id_a)
            if tb is not None:
                ax.plot(range(len(tb)), tb, marker="o", markersize=3,
                        linewidth=1.2, color=color_b, label=positive_id_b)
            ax.axhline(0, color="k", lw=0.4, alpha=0.5)
            ax.set_title(f"layer {L}")
            ax.set_xlabel("Token position")
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"Mean {metric_label}")

        legend_handles = [
            Line2D([0], [0], color=color_a, marker="o", markersize=4,
                   linewidth=1.5, label=f"{positive_id_a} (diag)"),
            Line2D([0], [0], color=color_b, marker="o", markersize=4,
                   linewidth=1.5, label=f"{positive_id_b} (diag)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(
            f"concept = {concept} | {positive_id_a} vs {positive_id_b} "
            f"(prompt_layer = analysis_layer)",
            fontsize=12)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fname = f"{file_prefix}_{concept}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{file_prefix}: wrote {fname}")


# ─── entry ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Layer-targeting plots (7-12).")
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    results, _ = load_results(run_dir / "results")
    out_dir = _ensure_dir(run_dir / "plots")

    actual_layers = tuple(results[0]["analysis_layers"]) if results else ()
    prompt_layers = tuple(sorted({
        r["prompt_layer"] for r in results if r.get("prompt_layer") is not None
    }))
    if not prompt_layers:
        print("This run has no layer-targeted prompt data (prompt_layers empty); "
              "nothing to plot. Use a config with prompt_layers set "
              "(e.g. experiment_layer_target_deep.yaml).")
        return
    print(f"Loaded {len(results)} trials. analysis_layers={actual_layers}, "
          f"prompt_layers={prompt_layers}. Writing to {out_dir}")

    # ---- Plots 7a/7b: concept-specific cosine, targeted vs non-targeted ----
    plot_layer_targeting(results, out_dir, positive_id="think_at_layer",
                         baseline_id="think_about", metric_key="cosine_sim_anchored",
                         metric_label="cos-sim", file_prefix="plot7a",
                         prompt_layers=prompt_layers, analysis_layers=actual_layers)
    plot_layer_targeting(results, out_dir, positive_id="think_intensely_at_layer",
                         baseline_id="think_intensely", metric_key="cosine_sim_anchored",
                         metric_label="cos-sim", file_prefix="plot7b",
                         prompt_layers=prompt_layers, analysis_layers=actual_layers)

    # ---- Plots 8a/8b: concept-less activation-norm layer-targeting ----
    plot_layer_targeting(results, out_dir, positive_id="ctrl_think_intensely_at_layer",
                         baseline_id="ctrl_think_intensely", metric_key="norms_anchored",
                         metric_label="‖residual‖", file_prefix="plot8a",
                         prompt_layers=prompt_layers, analysis_layers=actual_layers)
    plot_layer_targeting(results, out_dir, positive_id="ctrl_think_at_layer",
                         baseline_id="no_instruction", metric_key="norms_anchored",
                         metric_label="‖residual‖", file_prefix="plot8b",
                         prompt_layers=prompt_layers, analysis_layers=actual_layers)

    # ---- Plots 9a/9b: prompt_layer × analysis_layer Δ heatmap (+ by-concept) ----
    plot_layer_targeting_heatmap(results, out_dir, positive_id="think_at_layer",
                                 baseline_id="think_about", metric_key="cosine_sim_anchored",
                                 metric_label="cos-sim", file_prefix="plot9a",
                                 prompt_layers=prompt_layers, analysis_layers=actual_layers)
    plot_layer_targeting_heatmap(results, out_dir, positive_id="think_intensely_at_layer",
                                 baseline_id="think_intensely", metric_key="cosine_sim_anchored",
                                 metric_label="cos-sim", file_prefix="plot9b",
                                 prompt_layers=prompt_layers, analysis_layers=actual_layers)
    plot_layer_targeting_heatmap_by_concept(
        results, out_dir, positive_id="think_at_layer", baseline_id="think_about",
        metric_key="cosine_sim_anchored", metric_label="cos-sim",
        file_prefix="plot9a_by_concept",
        prompt_layers=prompt_layers, analysis_layers=actual_layers)
    plot_layer_targeting_heatmap_by_concept(
        results, out_dir, positive_id="think_intensely_at_layer", baseline_id="think_intensely",
        metric_key="cosine_sim_anchored", metric_label="cos-sim",
        file_prefix="plot9b_by_concept",
        prompt_layers=prompt_layers, analysis_layers=actual_layers)

    # ---- Plots 10a/10b: per-concept diagonal decomposition ----
    plot10_per_concept_diagonal(
        results, out_dir, positive_id="think_at_layer", baseline_id="think_about",
        metric_key="cosine_sim_anchored", metric_label="cos-sim",
        file_prefix="plot10a", layers=actual_layers)
    plot10_per_concept_diagonal(
        results, out_dir, positive_id="think_intensely_at_layer", baseline_id="think_intensely",
        metric_key="cosine_sim_anchored", metric_label="cos-sim",
        file_prefix="plot10b", layers=actual_layers)

    # ---- Plot 11: per-concept diagonal, two targeted variants overlaid ----
    plot11_per_concept_two_targeted(
        results, out_dir, positive_id_a="think_at_layer",
        positive_id_b="think_intensely_at_layer",
        metric_key="cosine_sim_anchored", metric_label="cos-sim",
        file_prefix="plot11", layers=actual_layers)

    # ---- Plot 12: activation-norm layer-targeting heatmap ----
    plot_layer_targeting_heatmap(
        results, out_dir, positive_id="ctrl_think_intensely_at_layer",
        baseline_id="ctrl_think_intensely", metric_key="norms_anchored",
        metric_label="‖residual‖", file_prefix="plot12",
        prompt_layers=prompt_layers, analysis_layers=actual_layers)


if __name__ == "__main__":
    main()
