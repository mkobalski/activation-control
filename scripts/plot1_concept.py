#!/usr/bin/env python3
"""plot1_{concept}: 2x2 per-concept modulation figure (paper styling).

Layout (one figure per concept):
                 left column                     right column
  top row     cosine  {dont, think, intensely}   cosine  {dont + intensity ramp}
  bottom row  relnorm {dont, think, intensely}   relnorm {dont + intensity ramp}

Styling:
  left  panels: dont_think_about  light gray
                think_about       medium gray
                think_intensely   the ramp's intensity-4/4 color (dark maroon)
  right panels: dont_think_about  light gray
                think_intensity_{1..4}_of_4 in the Reds ramp (light->dark)
                (NO think_about on the right)
  - each column has its own legend
  - y limits: per-row min-max (+pad); y tick marks only on the leftmost panels
  - y labels: "Cosine similarity" (top) / "Relative norm" (bottom)
  - x label "token"; token tick labels on the bottom row only
  - no subplot titles; suptitle = concept only (no sentence)
  - no top/right spines; no grid

Layers: cosine defaults to L55, relnorm to L46 (where each channel is cleanest).
Data comes straight from results.json (concept-bearing conditions only). CPU-only.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
REDS = plt.get_cmap("Reds")(np.linspace(0.40, 0.95, 4))
LIGHT_GRAY = (0.78, 0.78, 0.78)
MED_GRAY = (0.45, 0.45, 0.45)
INTENSE = REDS[3]                      # same color as intensity 4/4
LW, MS = 1.5, 3.5

LEFT = ["dont_think_about", "think_about", "think_intensely"]
RIGHT = ["dont_think_about", *RAMP]

STYLE = {
    "dont_think_about": (LIGHT_GRAY, "don't think about"),
    "think_about": (MED_GRAY, "think about"),
    "think_intensely": (INTENSE, "think intensely about"),
    **{c: (REDS[i], f"intensity {i+1}/4") for i, c in enumerate(RAMP)},
}


def _classify(s):
    t = s.strip().lower()
    if t in ("the", "a", "and", ",", ".", "hello"):
        return t
    if s.startswith("<") and s.endswith(">"):
        return "special"
    return "content"


def _trace(trial, layer, metric):
    """Sentence-span per-token readout from stored JSON traces."""
    if metric == "relnorm":
        d = trial.get("norms") or {}
        t = d.get(str(layer), d.get(layer))
        if not t:
            return None
        t = np.asarray(t, float)
        strs = (trial.get("anchored_token_strs") or [])[1:]
        content = [i for i, s in enumerate(strs) if _classify(s) == "content" and i < len(t)]
        if not content:
            return None
        return t / np.mean(t[content])
    d = trial.get("cosine_sim") or {}
    t = d.get(str(layer), d.get(layer))
    return np.asarray(t, float) if t else None


def _collect(rows, concept, sentence, layer, metric):
    out = {}
    conds = set(LEFT) | set(RIGHT)
    for r in rows:
        if (r.get("is_compliant") and r.get("concept") == concept
                and r["sentence"] == sentence and r["condition_id"] in conds):
            t = _trace(r, layer, metric)
            if t is not None:
                out[r["condition_id"]] = t
    return out


def render(rows, concept, sentence, *, cos_layer, relnorm_layer, out):
    data = {
        "cos": _collect(rows, concept, sentence, cos_layer, "cos"),
        "relnorm": _collect(rows, concept, sentence, relnorm_layer, "relnorm"),
    }
    if not data["cos"] or not data["relnorm"]:
        print(f"  [skip {concept}] missing traces (cos={len(data['cos'])}, "
              f"relnorm={len(data['relnorm'])})")
        return None

    tok_row = next(r for r in rows if r["sentence"] == sentence
                   and r.get("anchored_token_strs"))
    labels = [t.strip() or "␣" for t in tok_row["anchored_token_strs"][1:]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex="col")
    rowspec = [("cos", f"Cosine similarity (L{cos_layer})"),
               ("relnorm", f"Relative norm (L{relnorm_layer})")]
    for ri, (metric, ylab) in enumerate(rowspec):
        d = data[metric]
        # per-row min-max limits over every condition drawn in this row
        vals = [d[c] for c in set(LEFT) | set(RIGHT) if c in d]
        lo = min(float(np.min(v)) for v in vals)
        hi = max(float(np.max(v)) for v in vals)
        pad = 0.04 * (hi - lo) if hi > lo else 0.01
        for ci, conds in enumerate([LEFT, RIGHT]):
            ax = axes[ri][ci]
            for cond in conds:
                y = d.get(cond)
                if y is None:
                    continue
                color, _lab = STYLE[cond]
                ax.plot(range(len(y)), y, "-", color=color, lw=LW,
                        marker="o", markersize=MS)
            ax.set_ylim(lo - pad, hi + pad)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(False)
            if ci == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_xticks(range(len(labels)))
            if ri == 1:
                ax.set_xticklabels(labels, rotation=45, ha="right",
                                   fontsize=8, family="monospace")
                ax.set_xlabel("Token")

    # per-column legends: frameless, stacked vertically, in the upper-left of
    # each column's BOTTOM-row subplot. No suptitle — the concept is named in
    # the figure description (results/paper/Fig1.md), not on the figure.
    def _handles(conds):
        return [Line2D([0], [0], color=STYLE[c][0], lw=LW, marker="o",
                       markersize=MS, label=STYLE[c][1]) for c in conds]
    for ci, conds in enumerate([LEFT, RIGHT]):
        axes[1][ci].legend(handles=_handles(conds), loc="upper left",
                           ncol=1, frameon=False, fontsize=9,
                           handletextpad=0.5, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def make_plot1_concepts(results, run_dir, *, sentence, cos_frac=0.90,
                        relnorm_frac=0.75, cos_layer=None, relnorm_layer=None,
                        concepts=None, out_dir=None, verbose=True):
    """Pipeline entry: render plot1_{concept} for every concept in a run.

    Model-portable layer choice: `cos_frac` / `relnorm_frac` are relative depths
    (0.90 / 0.75 -> L55 / L46 on gemma3-27b's 62 layers), resolved against the
    run's n_layers and snapped to the nearest RECORDED layer -- so the same
    config produces comparable figures across models of different depths.
    Explicit `cos_layer` / `relnorm_layer` override the fractions.
    """
    from src.utils.io import load_run_config
    from src.utils.layers import fraction_to_layer

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded = sorted({int(x) for r in results for x in (r.get("analysis_layers") or [])})
    if not recorded:
        print("  [plot1_concepts] no recorded layers; skipping")
        return []
    n_layers = load_run_config(run_dir).get("n_layers") or (max(recorded) + 1)

    def _resolve(explicit, frac):
        if explicit is not None:
            return int(explicit)
        target = fraction_to_layer(frac, int(n_layers))
        return min(recorded, key=lambda L: abs(L - target))   # snap to recorded

    cL, rL = _resolve(cos_layer, cos_frac), _resolve(relnorm_layer, relnorm_frac)
    if verbose:
        print(f"  [plot1_concepts] cos layer L{cL} (frac {cos_frac}), "
              f"relnorm layer L{rL} (frac {relnorm_frac}); sentence: {sentence!r}")

    concepts = list(concepts) if concepts else \
        sorted({r["concept"] for r in results if r.get("concept")})
    paths = []
    for c in concepts:
        p = render(results, c, sentence, cos_layer=cL, relnorm_layer=rL,
                   out=out_dir / f"plot1_{c}.png")
        if p:
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="plot1_{concept}: 2x2 cos/relnorm modulation figure.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sentence", default="The bus was crowded, but I found a seat near the back.")
    ap.add_argument("--cos-layer", type=int, default=55)
    ap.add_argument("--relnorm-layer", type=int, default=46)
    ap.add_argument("--concepts", default=None,
                    help="comma list; default = all concepts present")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    with open(Path(args.run_dir) / "results.json") as f:
        rows = json.load(f)["results"]
    concepts = (args.concepts.split(",") if args.concepts
                else sorted({r["concept"] for r in rows if r.get("concept")}))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in concepts:
        render(rows, c, args.sentence, cos_layer=args.cos_layer,
               relnorm_layer=args.relnorm_layer, out=out_dir / f"plot1_{c}.png")


if __name__ == "__main__":
    main()
