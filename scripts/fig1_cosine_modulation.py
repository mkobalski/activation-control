#!/usr/bin/env python3
"""Figure 1 — concept-cosine modulation across tokens, for the paper.

Two panels, ONE concept / ONE layer / ONE sentence, RAW cosine similarity:
  (1a) think_about, dont_think_about, think_intensely
  (1b) think_about, dont_think_about, think_intensity_{1..4}_of_4

Styling:
  think_about       solid, black
  dont_think_about  solid, light gray
  think_intensely   solid, dark red                 (panel a)
  intensity ramp    solid, Reds light->dark 1..4    (panel b)

y = cos(concept vector, residual);  x = sentence tokens.

Data: every plotted condition is concept-bearing, so the per-token cosine traces
come straight from the run's results.json -- NO results.pkl needed. (The optional
--no-instruction line is the one thing that needs a pickle; it is off by default
and not part of the figure spec.)

CPU-only; no model load.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
FAN = ["dont_think_about", "think_about", *RAMP, "think_intensely"]  # for auto-select

# ---- styling -----------------------------------------------------------------
GRAY = (0.72, 0.72, 0.72)
BLUE = "#1f6fd6"
DARKRED = "#8c0d0d"
REDS = plt.get_cmap("Reds")(np.linspace(0.40, 0.95, 4))   # int1..int4


# Exhibit-A line weight: thin lines with visible round markers.
LW, MS = 1.5, 5


def _style(cond):
    """(color, linestyle, linewidth, label, zorder) for a condition id."""
    if cond == "no_instruction":
        return GRAY, "--", LW, "no instruction", 1
    if cond == "think_about":
        return "black", "-", LW, "think about", 5
    if cond == "dont_think_about":
        return GRAY, "-", LW, "don't think about", 2
    if cond == "think_intensely":
        return DARKRED, "-", LW, "think intensely", 4
    if cond in RAMP:
        i = RAMP.index(cond)
        return REDS[i], "-", LW, f"intensity {i+1}/4", 3
    return "gray", "-", LW, cond, 1


# ---- data --------------------------------------------------------------------

def _load_json(run_dir):
    with open(Path(run_dir) / "results.json") as f:
        return json.load(f)["results"]


def _classify(s):
    t = s.strip().lower()
    if t in ("the", "a", "and", ",", ".", "hello"):
        return t
    if s.startswith("<") and s.endswith(">"):
        return "special"
    return "content"


def _trace(trial, layer, anchored, metric="cos"):
    """Per-token readout from the stored JSON traces.

    cos      -> stored cosine_sim(_anchored)
    relnorm  -> stored norms(_anchored) / mean norm over the trial's CONTENT
                tokens (sentence-span classification, same as the heatmaps)
    """
    if metric == "relnorm":
        d = trial.get("norms_anchored" if anchored else "norms") or {}
        t = d.get(str(layer), d.get(layer))
        if not t:
            return None
        t = np.asarray(t, float)
        strs = (trial.get("anchored_token_strs") or [])[1:]      # sentence tokens
        classes = [_classify(s) for s in strs]
        # content positions within the sentence span; +1 if the trace is anchored
        content = [i for i, c in enumerate(classes) if c == "content"]
        pos = [i + 1 for i in content] if anchored else content
        pos = [i for i in pos if i < len(t)]
        if not pos:
            return None
        return t / np.mean(t[pos])
    d = trial.get("cosine_sim_anchored" if anchored else "cosine_sim") or {}
    t = d.get(str(layer), d.get(layer))
    return np.asarray(t, float) if t else None


def _concept_bearing_traces(rows, concept, sentence, layer, anchored, metric="cos"):
    """{condition_id: trace} for the compliant concept-bearing trials of this cell."""
    out = {}
    for r in rows:
        if (r.get("is_compliant") and r.get("concept") == concept
                and r["sentence"] == sentence and r["condition_id"] in (*FAN,)):
            t = _trace(r, layer, anchored, metric)
            if t is not None:
                out[r["condition_id"]] = t
    return out


def _no_instruction_trace(run_dir, concept, sentence, layer, anchored, vector_cache,
                          method):
    """Project the concept-less no_instruction residual onto the concept vector.

    Returns the per-token cosine trace (sentence-span, or anchored if requested),
    or None if the pickle / vector / trial isn't available.
    """
    from controllability_heatmap import load_vectors, get_acts
    from src.analysis.cosine import cosine_trace
    pkl = Path(run_dir) / "results.pkl"
    if not pkl.exists():
        print("  [no_instruction] no results.pkl -> skipping that line "
              "(re-run with a pickle present to include it)")
        return None
    import pickle
    with open(pkl, "rb") as f:
        rows = pickle.load(f)["results"]
    tr = next((r for r in rows if r.get("is_compliant")
               and r["condition_id"] == "no_instruction" and r["sentence"] == sentence), None)
    if tr is None:
        print("  [no_instruction] no compliant no_instruction trial for this sentence")
        return None
    vecs = load_vectors(vector_cache, _model_of(run_dir), [layer], method)
    if layer not in vecs:
        print(f"  [no_instruction] no concept vector cached at layer {layer}")
        return None
    concepts_L, V, _ = vecs[layer]
    if concept not in concepts_L:
        return None
    cv = V[concepts_L.index(concept)]
    key = "activations_anchored" if anchored else "activations"
    acts = tr.get(key) or {}
    a = acts.get(layer, acts.get(str(layer)))
    if a is None:
        return None
    return cosine_trace(cv, np.asarray(a, np.float32))


def _model_of(run_dir):
    from src.utils.io import load_run_config
    return load_run_config(run_dir).get("model", "gemma3_27b")


# ---- auto-select -------------------------------------------------------------

def auto_select(rows, anchored):
    """Return (concept, layer, sentence) with the largest mean per-token spread
    across the FAN conditions (skipping the anchor token when anchored)."""
    layers = sorted(int(x) for x in rows[0]["analysis_layers"])
    idx = defaultdict(dict)
    for r in rows:
        if not r.get("is_compliant") or r["condition_id"] not in FAN or r.get("concept") is None:
            continue
        for L in layers:
            t = _trace(r, L, anchored)
            if t is not None:
                idx[(r["concept"], r["sentence"], L)][r["condition_id"]] = t
    best = None
    for (concept, sent, L), d in idx.items():
        if not all(c in d for c in FAN):
            continue
        n = min(len(v) for v in d.values())
        if n < 3:
            continue
        M = np.vstack([d[c][:n] for c in FAN])
        spread = (M.max(0) - M.min(0))
        if anchored:
            spread = spread[1:]
        score = float(spread.mean())
        if best is None or score > best[0]:
            best = (score, concept, L, sent)
    return best[1:] if best else (None, None, None)


# ---- render ------------------------------------------------------------------

def render(run_dir, concept, layer, sentence, *, out, anchored=False,
           include_no_instruction=True, vector_cache="results/vector_cache",
           method="baseline", title=None, metric="cos"):
    rows = _load_json(run_dir)
    traces = _concept_bearing_traces(rows, concept, sentence, layer, anchored, metric)
    if not traces:
        sys.exit(f"No compliant traces for concept={concept} sentence={sentence!r} L={layer}")

    ni = None
    if include_no_instruction:
        ni = _no_instruction_trace(run_dir, concept, sentence, layer, anchored,
                                   vector_cache, method)

    # token labels from any trial of this sentence
    tok_row = next(r for r in rows if r["sentence"] == sentence and r.get("anchored_token_strs"))
    toks = tok_row["anchored_token_strs"]
    labels = toks if anchored else toks[1:]
    labels = [t.strip() or "␣" for t in labels]

    panels = [
        ("a", ["think_about", "dont_think_about", "think_intensely"]),
        ("b", ["think_about", "dont_think_about", *RAMP]),
    ]
    if include_no_instruction and ni is not None:
        panels = [(tag, ["no_instruction", *conds]) for tag, conds in panels]
    # Tight per-figure y-limits: min/max over every plotted series (both panels)
    # + a small pad, so the axis resolves THIS concept's dynamic range instead of
    # being anchored toward 0. Panels share the scale so (a) and (b) compare.
    all_y = [y for _, conds in panels for cond in conds
             for y in [ni if cond == "no_instruction" else traces.get(cond)]
             if y is not None]
    lo = min(float(np.min(y)) for y in all_y)
    hi = max(float(np.max(y)) for y in all_y)
    pad = 0.04 * (hi - lo) if hi > lo else 0.01

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (tag, conds) in zip(axes, panels):
        for cond in conds:
            y = ni if cond == "no_instruction" else traces.get(cond)
            if y is None:
                continue
            color, ls, lw, lab, z = _style(cond)
            ax.plot(range(len(y)), y, ls, color=color, lw=lw, zorder=z,
                    marker="o", markersize=MS, label=lab)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, family="monospace")
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25)
        ax.set_xlabel("token")
        ax.set_title(f"({tag}) " + ("intense verb" if tag == "a" else "1–4 intensity ramp"),
                     fontsize=11)
    axes[0].set_ylabel("relative norm  ‖r‖ / content-token mean" if metric == "relnorm"
                       else "cos(concept vector, residual)")

    # one merged legend below
    handles, seen = [], set()
    for _, conds in panels:
        for cond in conds:
            if cond in seen:
                continue
            seen.add(cond)
            color, ls, lw, lab, _z = _style(cond)
            handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw, marker="o",
                                  markersize=4, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    mtag = "relnorm" if metric == "relnorm" else "cos"
    fig.suptitle(title or f"concept = {concept}   ·   layer {layer}   ·   {mtag}   ·   \"{sentence}\"",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = Path(out)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Figure 1: concept-cosine modulation across tokens.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--concept", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--sentence", default=None)
    ap.add_argument("--auto", action="store_true", help="auto-pick concept/layer/sentence by max spread")
    ap.add_argument("--anchored", action="store_true", help="include the anchor token (default: sentence tokens only)")
    ap.add_argument("--no-instruction", dest="ni", action="store_true", help="include the no_instruction line (needs results.pkl)")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--metric", choices=["cos", "relnorm"], default="cos",
                    help="per-token readout: cosine (default) or relative norm")
    ap.add_argument("--out", default="fig1_cosine_modulation.png")
    args = ap.parse_args()

    if args.auto or not (args.concept and args.layer and args.sentence):
        rows = _load_json(args.run_dir)
        c, L, s = auto_select(rows, args.anchored)
        args.concept = args.concept or c
        args.layer = args.layer or L
        args.sentence = args.sentence or s
        print(f"auto-selected: concept={args.concept}  layer={args.layer}  sentence={args.sentence!r}")

    render(args.run_dir, args.concept, args.layer, args.sentence, out=args.out,
           anchored=args.anchored, include_no_instruction=args.ni,
           vector_cache=args.vector_cache, method=args.method, metric=args.metric)


if __name__ == "__main__":
    main()
