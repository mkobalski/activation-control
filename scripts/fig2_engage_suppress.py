#!/usr/bin/env python3
"""[RETIRED as the paper's Fig 2 -- kept as the shared readout/baseline library
and to render the exploratory per-token heatmap on demand.]

Engagement / suppression heatmaps across depth (tokens x layers),
with the controllability-suite significance machinery.

2x2 grid for ONE sentence:
                 left                              right
  top row     cos ENGAGEMENT                    cos SUPPRESSION
              (think_about - no_instruction)    (dont_think_about - no_instruction)
  bottom row  relnorm ENGAGEMENT                relnorm SUPPRESSION

Axes per panel: y = sentence tokens (in order), x = ALL recorded layers.
Each subplot gets its OWN diverging RdBu color scale (red = above the neutral
baseline, blue = below), symmetric around 0.

Statistics (mirrors scripts/controllability_heatmap.py's engage/suppress panels):
  each cell's statistic is the MEAN over concepts of the per-concept delta
  (condition readout - neutral-baseline readout); its p-value comes from a
  per-cell SIGN-FLIP permutation null (B = 5000, two-sided); Benjamini-Hochberg
  FDR is applied across each panel's cells, and the delta value is printed ONLY
  in cells that survive (q < alpha). Cosine deltas are printed raw; relative-norm
  deltas x100.

Readouts:
  cos     cos(v_concept, residual); condition trials read from results.json,
          the concept-less no_instruction baseline is projected onto each
          concept vector from no_instruction_cache.pkl if present (falls back
          to scanning results.pkl; needs RAM > pickle size).
  relnorm ||r|| / trial's content-token mean norm; both sides from results.json
          (norms are stored for every trial). Concept-agnostic baseline.

CPU-only, no model load.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from controllability_heatmap import classify, bh_fdr, load_vectors  # noqa: E402

POS, NEG, BASE = "think_about", "dont_think_about", "no_instruction"
B = 5000                       # sign-flip permutations per cell (as in the suite)


# ---- data ---------------------------------------------------------------------

def _load_json(run_dir):
    """Parsed results.json rows, memoized ON DISK: parsing the ~263MB JSON takes
    several seconds and every figure script pays it, so the first call writes
    results_rows.cache.pkl beside it and later calls (any script, any process)
    load the pickle instead. Invalidated by results.json's mtime."""
    src = Path(run_dir) / "results.json"
    cache = Path(run_dir) / "results_rows.cache.pkl"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        with open(cache, "rb") as f:
            return pickle.load(f)
    with open(src) as f:
        rows = json.load(f)["results"]
    try:                                   # best-effort; atomic via tmp+rename
        tmp = cache.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    except OSError:
        pass
    return rows


def _baseline(run_dir, sentence):
    """no_instruction {layer: acts (n_tok,d)}, {layer: norms (n_tok,)} for one sentence."""
    cache = Path(run_dir) / "no_instruction_cache.pkl"
    if cache.exists():
        e = pickle.load(open(cache, "rb")).get(sentence)
        if e is None:
            sys.exit(f"no_instruction cache lacks {sentence!r}")
        return e["activations"], e["norms"]
    pkl = Path(run_dir) / "results.pkl"
    if not pkl.exists():
        return None, None
    print("  [fig2] cache missing; scanning full results.pkl (needs RAM > pkl size)...")
    with open(pkl, "rb") as f:
        R = pickle.load(f)["results"]
    for r in R:
        if (r["condition_id"] == BASE and r.get("is_compliant")
                and r["sentence"] == sentence):
            return ({int(k): np.asarray(v, np.float32) for k, v in r["activations"].items()},
                    {int(k): np.asarray(v, np.float32) for k, v in r["norms"].items()})
    return None, None


def _trace(r, key, L):
    d = r.get(key) or {}
    t = d.get(str(L), d.get(L))
    return np.asarray(t, np.float32) if t else None


def _relnorm(norm_vec, classes):
    content = [i for i, c in enumerate(classes) if c == "content" and i < len(norm_vec)]
    return norm_vec / np.mean(norm_vec[content]) if content else None


def build(run_dir, sentence, *, vector_cache, method, model):
    """Compute per-panel (M, P) matrices of mean deltas + sign-flip p-values.

    Returns (layers, labels, panels) with panels[(metric, kind)] = (M, P);
    the cos panels are None if the no_instruction baseline acts are unavailable.
    """
    rows = _load_json(run_dir)
    sub = [r for r in rows if r["sentence"] == sentence and r.get("is_compliant")]
    if not sub:
        sys.exit(f"no compliant trials for {sentence!r}")
    layers = sorted({int(x) for r in sub for x in (r.get("analysis_layers") or [])})
    toks = next(r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs"))[1:]
    labels = [t.strip() or "␣" for t in toks]
    classes = [classify(t) for t in toks]
    n_tok = len(labels)
    rng = np.random.default_rng(0)

    def _cond(cond):
        return {r["concept"]: r for r in sub
                if r["condition_id"] == cond and r.get("concept")}

    conds = {"engage": _cond(POS), "suppress": _cond(NEG)}
    concepts = sorted(set(conds["engage"]) & set(conds["suppress"]))

    base_acts, base_norms = _baseline(run_dir, sentence)
    vecs = load_vectors(vector_cache, model, layers, method) if base_acts else {}

    def _cell(deltas, M, P, ti, li):
        """Mean + two-sided sign-flip p (exactly as the suite's _delta_panel)."""
        dv = np.array([d for d in deltas if not np.isnan(d)])
        if len(dv) < 3:
            return
        obs = float(dv.mean())
        M[ti, li] = obs
        signs = rng.choice([-1.0, 1.0], size=(B, len(dv)))
        null = (signs * dv).mean(1)
        P[ti, li] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (B + 1)

    panels = {}
    for kind, cmap_rows in conds.items():
        # ---- cos channel: cond stored cosine - baseline projected cosine ----
        if base_acts:
            Mc = np.full((n_tok, len(layers)), np.nan)
            Pc = np.full((n_tok, len(layers)), np.nan)
            for li, L in enumerate(layers):
                if L not in vecs or L not in base_acts:
                    continue
                concepts_L, V, Vn = vecs[L]
                A = np.asarray(base_acts[L], np.float32)[:n_tok]
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                base_cos = An @ Vn.T                     # (n_tok, n_concepts)
                for ti in range(min(n_tok, A.shape[0])):
                    deltas = []
                    for c in concepts:
                        tr = _trace(cmap_rows[c], "cosine_sim", L)
                        if tr is None or ti >= len(tr) or c not in concepts_L:
                            continue
                        deltas.append(float(tr[ti]) - float(base_cos[ti, concepts_L.index(c)]))
                    _cell(deltas, Mc, Pc, ti, li)
            panels[("cos", kind)] = (Mc, Pc)
        else:
            panels[("cos", kind)] = None
        # ---- relnorm channel: cond relnorm - baseline relnorm ----
        Mr = np.full((n_tok, len(layers)), np.nan)
        Pr = np.full((n_tok, len(layers)), np.nan)
        for li, L in enumerate(layers):
            b = None
            if base_norms and L in base_norms:
                b = _relnorm(np.asarray(base_norms[L], np.float32)[:n_tok], classes)
            else:  # baseline norms also live in the json -> always available
                br = next((r for r in sub if r["condition_id"] == BASE), None)
                tr = _trace(br, "norms", L) if br else None
                b = _relnorm(tr[:n_tok], classes) if tr is not None else None
            if b is None:
                continue
            rel = {}
            for c in concepts:
                tr = _trace(cmap_rows[c], "norms", L)
                if tr is not None:
                    rel[c] = _relnorm(tr[:n_tok], classes)
            for ti in range(n_tok):
                deltas = [float(rel[c][ti]) - float(b[ti]) for c in concepts
                          if c in rel and rel[c] is not None
                          and ti < len(rel[c]) and ti < len(b)]
                _cell(deltas, Mr, Pr, ti, li)
        panels[("relnorm", kind)] = (Mr, Pr)
    return layers, labels, panels, len(concepts)


# ---- render --------------------------------------------------------------------

def render(run_dir, sentence, *, out, alpha=0.05, vector_cache="results/vector_cache",
           method="baseline", model="gemma3_27b", title=None):
    layers, labels, panels, n_concepts = build(
        run_dir, sentence, vector_cache=vector_cache, method=method, model=model)
    n_tok = len(labels)

    # rows = metric (own big centered header + shared symmetric colorbar);
    # cols = condition (big word; the subtraction formula is shown for cos only).
    row_defs = [
        ("cos", "Cosine similarity", "Δcos", 1.0, 3),
        ("relnorm", "Relative norm", "Δrelnorm", 1.0, 3),
    ]
    col_defs = [
        ("engage", "Engagement"),
        ("suppress", "Suppression"),
    ]

    fig = plt.figure(figsize=(3.0 + 0.42 * len(layers) * 2,
                              2 * (0.34 * n_tok) + 4.4),
                     layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.05)

    for ri, (metric, row_title, cbar_label, scale, dec) in enumerate(row_defs):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=17, fontweight="bold")
        sf_axes = sf.subplots(1, 2)

        # one shared, symmetric color scale across the row's two panels
        A = 0.0
        for kind, _ in col_defs:
            p = panels[(metric, kind)]
            if p is not None:
                a = np.nanmax(np.abs(p[0] * scale))
                if np.isfinite(a):
                    A = max(A, float(a))
        A = A if A > 0 else 1.0

        row_im = None
        for ci, (kind, word) in enumerate(col_defs):
            ax = sf_axes[ci]
            p = panels[(metric, kind)]
            if p is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "cos panel needs no_instruction_cache.pkl\n"
                        "(run the analysis where results.pkl can load, e.g. the GPU box)",
                        ha="center", va="center", fontsize=9, color="#666",
                        transform=ax.transAxes)
                continue
            M, P = p
            Q = bh_fdr(P)
            Md = M * scale
            im = ax.imshow(Md, aspect="auto", cmap="RdBu_r", vmin=-A, vmax=A)
            row_im = im
            ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
            if ci == 0:
                ax.set_yticks(range(n_tok))
                ax.set_yticklabels(labels, fontsize=8, family="monospace")
            else:
                ax.set_yticks(range(n_tok)); ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            if metric == "relnorm":
                ax.set_xlabel("Layer")
            ax.set_title(word, fontsize=14, pad=8)
            # numbers only in significant cells (suite convention)
            for ti in range(n_tok):
                for li in range(len(layers)):
                    v, q = M[ti, li], Q[ti, li]
                    if np.isnan(v) or np.isnan(q) or q >= alpha:
                        continue
                    d = v * scale
                    ax.text(li, ti, f"{d:.{dec}f}", ha="center", va="center", fontsize=5.5,
                            color="white" if abs(d) / A > 0.55 else "black")

        if row_im is not None:
            cb = sf.colorbar(row_im, ax=sf_axes, fraction=0.035, pad=0.02)
            cb.set_label(cbar_label, fontsize=9)

    fig.text(0.5, 0.006, f"numbers shown only where significant (BH-FDR q<{alpha:g}; "
             f"per-cell sign-flip null, B={B})", ha="center", fontsize=8, color="#444")
    # reserve a bottom strip so the footnote clears the 'Layer' axis labels
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 2: engagement/suppression heatmaps with significance.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sentence", default="The bus was crowded, but I found a seat near the back.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig2_engage_suppress.png")
    args = ap.parse_args()
    render(args.run_dir, args.sentence, out=args.out, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
