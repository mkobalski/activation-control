#!/usr/bin/env python3
"""[Figure RETIRED to results/paper/Exploratory analysis -- script kept for its
shared helpers and to render the exploratory figure on demand.]

Fig 3: engagement / suppression heatmaps by TOKEN POSITION, averaged across
sentences (the position-level generalization of Fig 2).

Same 2x2 layout, metrics, and significance machinery as
scripts/fig2_engage_suppress.py, but:
  * y = token position (1-indexed) instead of a single sentence's token strings;
  * each cell is averaged over ALL sentences in the set AND all concepts, so a
    row asks "what happens at the k-th generated token, on average";
  * positions are CLIPPED to the length of the SHORTEST sentence in the set, so
    every position row is populated by every sentence (no length bias where deep
    positions only contain the long sentences).

For a fixed (position p, layer L) cell and panel (metric x kind), the sampling
units are (sentence, concept) pairs; the per-unit delta is exactly Fig 2's
condition-minus-no_instruction delta evaluated at token p of that sentence. The
cell statistic is the mean over units, its p-value a two-sided sign-flip
permutation null (B=5000), and Benjamini-Hochberg FDR is applied per panel.

CPU-only, no model load. Reads condition traces from results.json and the
concept-free baseline from no_instruction_cache.pkl (loaded ONCE for all
sentences).
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from controllability_heatmap import classify, bh_fdr, load_vectors   # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm, POS, NEG, BASE, B  # noqa: E402


# ---- data ---------------------------------------------------------------------

def _load_baseline_cache(run_dir):
    """{sentence: {"activations": {L: (n_tok,d)}, "norms": {L: (n_tok,)}}} or None."""
    cache = Path(run_dir) / "no_instruction_cache.pkl"
    if not cache.exists():
        return None
    with open(cache, "rb") as f:
        return pickle.load(f)


def _cell(deltas, M, P, ti, li, rng):
    """Mean + two-sided sign-flip p over the finite unit deltas (Fig 2's rule)."""
    dv = np.array([d for d in deltas if not np.isnan(d)])
    if len(dv) < 3:
        return
    obs = float(dv.mean())
    M[ti, li] = obs
    signs = rng.choice([-1.0, 1.0], size=(B, len(dv)))
    null = (signs * dv).mean(1)
    P[ti, li] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (B + 1)


def build(run_dir, *, sentences=None, vector_cache, method, model):
    """Compute per-panel (M, P) matrices of position x layer, pooled over sentences.

    Returns (layers, n_pos, panels, meta) with panels[(metric, kind)] = (M, P)
    (cos panels None if the no_instruction baseline acts are unavailable).
    """
    rows = _load_json(run_dir)
    compliant = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in compliant:
        by_sent[r["sentence"]].append(r)
    sent_list = sentences if sentences else sorted(by_sent)

    layers = sorted({int(x) for r in compliant for x in (r.get("analysis_layers") or [])})
    cache = _load_baseline_cache(run_dir)

    # per-sentence record: tokens, classes, condition rows by concept, baseline
    recs = {}
    for s in sent_list:
        sub = by_sent.get(s, [])
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if not toks_row:
            continue
        toks = toks_row[1:]                                   # drop the anchor token
        classes = [classify(t) for t in toks]

        def _cond(cond):
            return {r["concept"]: r for r in sub
                    if r["condition_id"] == cond and r.get("concept")}
        engage, suppress = _cond(POS), _cond(NEG)
        concepts = sorted(set(engage) & set(suppress))
        if not concepts:
            continue

        base_acts = base_norms = None
        if cache is not None and s in cache:
            e = cache[s]
            base_acts = {int(k): np.asarray(v, np.float32) for k, v in e["activations"].items()}
            base_norms = {int(k): np.asarray(v, np.float32) for k, v in e["norms"].items()}
        # relnorm baseline can always fall back to the no_instruction json norms
        base_row = next((r for r in sub if r["condition_id"] == BASE), None)

        recs[s] = dict(n_tok=len(toks), classes=classes, engage=engage,
                       suppress=suppress, concepts=concepts,
                       base_acts=base_acts, base_norms=base_norms, base_row=base_row)

    if not recs:
        sys.exit("no usable sentences (need compliant think/don't + shared concepts)")

    # CLIP to the shortest sentence: every position row is present in every sentence
    n_pos = min(r["n_tok"] for r in recs.values())
    used = sorted(recs)
    short = min(used, key=lambda s: recs[s]["n_tok"])
    have_base_acts = any(recs[s]["base_acts"] for s in used)
    vecs = load_vectors(vector_cache, model, layers, method) if have_base_acts else {}
    rng = np.random.default_rng(0)

    # cache base cosine per (sentence, layer): (n_tok, n_concepts_L)
    base_cos_cache = {}
    if have_base_acts:
        for s in used:
            ba = recs[s]["base_acts"]
            if not ba:
                continue
            nt = recs[s]["n_tok"]
            for L in layers:
                if L not in vecs or L not in ba:
                    continue
                _, _, Vn = vecs[L]
                A = np.asarray(ba[L], np.float32)[:nt]
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                base_cos_cache[(s, L)] = An @ Vn.T

    panels = {}
    for kind in ("engage", "suppress"):
        # ---- cos channel ----
        if have_base_acts:
            Mc = np.full((n_pos, len(layers)), np.nan)
            Pc = np.full((n_pos, len(layers)), np.nan)
            for li, L in enumerate(layers):
                if L not in vecs:
                    continue
                concepts_L = vecs[L][0]
                units = []                                    # (n_unit, n_pos) delta rows
                for s in used:
                    bc = base_cos_cache.get((s, L))
                    if bc is None:
                        continue
                    cmap = recs[s][kind]
                    for c in recs[s]["concepts"]:
                        if c not in concepts_L:
                            continue
                        tr = _trace(cmap[c], "cosine_sim", L)
                        if tr is None:
                            continue
                        ci = concepts_L.index(c)
                        d = np.full(n_pos, np.nan)
                        m = min(n_pos, len(tr), bc.shape[0])
                        d[:m] = tr[:m] - bc[:m, ci]
                        units.append(d)
                if units:
                    U = np.vstack(units)
                    for p in range(n_pos):
                        _cell(U[:, p], Mc, Pc, p, li, rng)
            panels[("cos", kind)] = (Mc, Pc)
        else:
            panels[("cos", kind)] = None

        # ---- relnorm channel ----
        Mr = np.full((n_pos, len(layers)), np.nan)
        Pr = np.full((n_pos, len(layers)), np.nan)
        for li, L in enumerate(layers):
            units = []
            for s in used:
                rec = recs[s]
                nt, classes = rec["n_tok"], rec["classes"]
                bn = rec["base_norms"]
                if bn and L in bn:
                    b = _relnorm(np.asarray(bn[L], np.float32)[:nt], classes)
                else:
                    tr = _trace(rec["base_row"], "norms", L) if rec["base_row"] else None
                    b = _relnorm(np.asarray(tr, np.float32)[:nt], classes) if tr is not None else None
                if b is None:
                    continue
                cmap = rec[kind]
                for c in rec["concepts"]:
                    tr = _trace(cmap[c], "norms", L)
                    if tr is None:
                        continue
                    rel = _relnorm(np.asarray(tr, np.float32)[:nt], classes)
                    if rel is None:
                        continue
                    d = np.full(n_pos, np.nan)
                    m = min(n_pos, len(rel), len(b))
                    d[:m] = rel[:m] - b[:m]
                    units.append(d)
            if units:
                U = np.vstack(units)
                for p in range(n_pos):
                    _cell(U[:, p], Mr, Pr, p, li, rng)
        panels[("relnorm", kind)] = (Mr, Pr)

    meta = dict(n_sentences=len(used), n_pos=n_pos, shortest=short,
                short_len=recs[short]["n_tok"])
    return layers, n_pos, panels, meta


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, sentences=None, alpha=0.05,
           vector_cache="results/vector_cache", method="baseline", model="gemma3_27b"):
    layers, n_pos, panels, meta = build(
        run_dir, sentences=sentences, vector_cache=vector_cache, method=method, model=model)
    labels = [str(p) for p in range(1, n_pos + 1)]            # 1-indexed token position

    row_defs = [
        ("cos", "Cosine similarity", "Δcos", 1.0, 3),
        ("relnorm", "Relative norm", "Δrelnorm", 1.0, 3),
    ]
    col_defs = [("engage", "Engagement"), ("suppress", "Suppression")]

    fig = plt.figure(figsize=(3.0 + 0.42 * len(layers) * 2,
                              2 * (0.34 * 14) + 4.4),   # height pinned to Fig 2's 14-token reference
                     layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.05)

    for ri, (metric, row_title, cbar_label, scale, dec) in enumerate(row_defs):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=17, fontweight="bold")
        sf_axes = sf.subplots(1, 2)

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
            im = ax.imshow(M * scale, aspect="auto", cmap="RdBu_r", vmin=-A, vmax=A)
            row_im = im
            ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
            if ci == 0:
                ax.set_yticks(range(n_pos)); ax.set_yticklabels(labels, fontsize=8)
                ax.set_ylabel("Token position")
            else:
                ax.set_yticks(range(n_pos)); ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            if metric == "relnorm":
                ax.set_xlabel("Layer")
            ax.set_title(word, fontsize=14, pad=8)
            for ti in range(n_pos):
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

    fig.text(0.5, 0.006,
             f"averaged over {meta['n_sentences']} sentences × concepts; positions "
             f"clipped to the shortest sentence ({meta['short_len']} tokens).  "
             f"numbers shown only where significant (BH-FDR q<{alpha:g}; "
             f"per-cell sign-flip null over units, B={B})",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{meta['n_sentences']} sentences, {n_pos} positions, "
          f"shortest={meta['shortest']!r}]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(
        description="Fig 3: engagement/suppression by token position, averaged over sentences.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sentences", default=None,
                    help="optional '||'-separated subset of sentences (default: all compliant)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig3_position_engage_suppress.png")
    args = ap.parse_args()
    sents = args.sentences.split("||") if args.sentences else None
    render(args.run_dir, out=args.out, sentences=sents, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
