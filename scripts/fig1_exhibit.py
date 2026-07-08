#!/usr/bin/env python3
"""Fig 1 (Exhibit-A style): on-concept vs off-concept cosine across tokens.

Two panels for ONE probe concept, ONE layer, ONE sentence:
  (A) on-concept  : cos(c_probe, residual) when the PROMPT is the probe concept
  (B) off-concept : cos(c_probe, residual) when the PROMPT is a DIFFERENT concept
                    (a single named off-concept, or the mean over all others)

Both panels share the concept VECTOR (c_probe); only the prompted concept of the
trials differs. If the model is really steering *this* concept, panel A fans out
with the manipulation while panel B stays ~flat.

Lines: intensity ramp think_intensity_{1..4}_of_4 (Reds, light->dark) +
dont_think_about (gray) + think_about (black). y = cos(c_probe, residual).

Data: needs RAW residuals (results.pkl) + the cached concept vector, because the
off-concept panel projects OTHER concepts' residuals onto c_probe (not a stored
cosine). Pass --run-dir to read the run's pickle, or --subset to read a small
pre-extracted cache: {"toks": [...], "acts": {concept: {cond: (n_tok,d) array}}}.

CPU-only, no model load.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
CONDS = ["think_about", *RAMP, "dont_think_about"]

REDS = plt.get_cmap("Reds")(np.linspace(0.40, 0.95, 4))
GRAY = (0.5, 0.5, 0.5)


def _style(cond):
    if cond == "think_about":
        return "black", "-", 2.0, "think about"
    if cond == "dont_think_about":
        return GRAY, "-", 1.8, "dont_think_about"
    i = RAMP.index(cond)
    return REDS[i], "-", 1.8, cond


def _cos(vec, acts):
    from src.analysis.cosine import cosine_trace
    return cosine_trace(vec, np.asarray(acts, np.float32))


def _load_acts(args):
    """Return (toks, acts[concept][cond] -> ndarray) for the sentence + layer."""
    if args.subset:
        d = pickle.load(open(args.subset, "rb"))
        return d["toks"], d["acts"]
    # else pull from the full run pickle
    with open(Path(args.run_dir) / "results.pkl", "rb") as f:
        R = pickle.load(f)["results"]
    toks, acts = None, {}
    for r in R:
        if r["sentence"] != args.sentence or not r.get("is_compliant"):
            continue
        aa = r.get("activations_anchored") or {}
        a = aa.get(args.layer, aa.get(str(args.layer)))
        if a is None:
            continue
        acts.setdefault(r["concept"], {})[r["condition_id"]] = np.asarray(a, np.float32)
        if toks is None:
            toks = r["anchored_token_strs"]
    return toks, acts


def _probe_vector(args):
    from controllability_heatmap import load_vectors
    from src.utils.io import load_run_config
    model = args.model
    if not model and args.run_dir:
        model = load_run_config(args.run_dir).get("model", "gemma3_27b")
    model = model or "gemma3_27b"
    vecs = load_vectors(args.vector_cache, model, [args.layer], args.method)
    if args.layer not in vecs:
        sys.exit(f"no concept vector cached at layer {args.layer} for {model}")
    concepts_L, V, _ = vecs[args.layer]
    if args.probe_concept not in concepts_L:
        sys.exit(f"probe concept {args.probe_concept} not in vector cache")
    return V[concepts_L.index(args.probe_concept)]


def render(args):
    toks, acts = _load_acts(args)
    if not acts:
        sys.exit("no activations found for that sentence/layer")
    cv = _probe_vector(args)

    # panel A: probe-prompted trials; panel B: off-concept-prompted trials.
    on = acts.get(args.probe_concept, {})
    if args.off_concept.lower() in ("all", "mean", "avg"):
        # mean over every non-probe concept, per condition
        off = {}
        for cond in CONDS:
            arrs = [_cos(cv, d[cond]) for c, d in acts.items()
                    if c != args.probe_concept and cond in d]
            if arrs:
                m = min(len(a) for a in arrs)
                off[cond] = np.mean([a[:m] for a in arrs], axis=0)
        off_label = "mean over prompt ≠ %s" % args.probe_concept
    else:
        d = acts.get(args.off_concept, {})
        off = {cond: _cos(cv, d[cond]) for cond in CONDS if cond in d}
        off_label = "prompt = %s" % args.off_concept

    on_c = {cond: _cos(cv, on[cond]) for cond in CONDS if cond in on}

    labels = [(t.strip() or "anchor") for t in (toks or [])]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=False)
    for ax, panel, sub, ttl in [
        (axes[0], "A", on_c, f"(A) on-concept (prompt = {args.probe_concept}) — L{args.layer}"),
        (axes[1], "B", off, f"(B) off-concept (probe = {args.probe_concept}, {off_label}) — L{args.layer}"),
    ]:
        for cond in CONDS:
            y = sub.get(cond)
            if y is None:
                continue
            color, ls, lw, _lab = _style(cond)
            ax.plot(range(len(y)), y, ls, color=color, lw=lw, marker="o", markersize=3,
                    zorder=(5 if cond == "think_about" else 3))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, family="monospace")
        ax.grid(alpha=0.25)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel(f"cos(c_{args.probe_concept}, residual)")
    handles = []
    for cond in CONDS:
        color, ls, lw, lab = _style(cond)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw, marker="o",
                              markersize=4, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle(f"concept = {args.probe_concept}   ·   off-concept = {args.off_concept}"
                 f"   ·   L{args.layer}   ·   \"{args.sentence}\"", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Fig 1 Exhibit-A style (on/off concept cosine).")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--subset", default=None, help="pre-extracted subset pickle")
    ap.add_argument("--probe-concept", default="Milk")
    ap.add_argument("--off-concept", default="Lightning", help="a concept id, or 'all' for the mean")
    ap.add_argument("--layer", type=int, default=55)
    ap.add_argument("--sentence", default="The bus was crowded, but I found a seat near the back.")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="fig1_exhibit.png")
    args = ap.parse_args()
    render(args)


if __name__ == "__main__":
    main()
