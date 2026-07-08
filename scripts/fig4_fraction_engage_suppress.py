#!/usr/bin/env python3
"""Fig 4: engagement / suppression heatmaps by FRACTIONAL sentence position,
averaged across sentences (no clipping).

Same 2x2 layout, metrics, and significance machinery as Fig 2 / Fig 3, but the
y-axis is "how far into the sentence" as a fraction in [0, 1] rather than an
absolute token index. Each generated token i of an n-token sentence is placed at
fractional position f = i / (n - 1)  (first token -> 0.0, last token -> 1.0), and
f is binned into `--n-bins` equal-width bins over [0, 1] (upper limit inclusive).
Because position is normalized per sentence, EVERY sentence contributes across
the whole axis -- no sentences or tokens are dropped (contrast Fig 3, which clips
to the shortest sentence).

Sampling unit = (sentence, concept) pair, exactly as Fig 3: within a bin a unit's
value is the MEAN of that unit's Fig-2 deltas over whatever tokens of that
sentence land in the bin (NaN if none), so correlated within-sentence tokens are
collapsed to one value per unit before pooling. The cell statistic is the mean
over units; its p-value a two-sided sign-flip permutation null (B=5000); BH-FDR
per panel.

CPU-only, no model load.
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
from fig3_position_engage_suppress import _load_baseline_cache, _cell  # noqa: E402


def _bins_for(n_tok, n_bins):
    """Bin index in [0, n_bins-1] for each of the n_tok tokens (f = i/(n-1))."""
    if n_tok <= 1:
        return np.zeros(max(n_tok, 0), dtype=int)
    f = np.arange(n_tok) / (n_tok - 1)                     # 0.0 .. 1.0 inclusive
    return np.minimum((f * n_bins).astype(int), n_bins - 1)


def _bin_means(delta_vec, bin_idx, n_bins):
    """Average delta_vec into n_bins by bin_idx; NaN for empty bins."""
    out = np.full(n_bins, np.nan)
    m = min(len(delta_vec), len(bin_idx))
    for b in range(n_bins):
        sel = [delta_vec[i] for i in range(m) if bin_idx[i] == b and not np.isnan(delta_vec[i])]
        if sel:
            out[b] = float(np.mean(sel))
    return out


def build(run_dir, *, sentences=None, n_bins=10, vector_cache, method, model):
    """Per-panel (M, P) matrices of fractional-position-bin x layer, pooled over
    (sentence, concept) units. Returns (layers, n_bins, panels, meta)."""
    rows = _load_json(run_dir)
    compliant = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in compliant:
        by_sent[r["sentence"]].append(r)
    sent_list = sentences if sentences else sorted(by_sent)

    layers = sorted({int(x) for r in compliant for x in (r.get("analysis_layers") or [])})
    cache = _load_baseline_cache(run_dir)

    recs = {}
    for s in sent_list:
        sub = by_sent.get(s, [])
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if not toks_row:
            continue
        toks = toks_row[1:]
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
        base_row = next((r for r in sub if r["condition_id"] == BASE), None)

        recs[s] = dict(n_tok=len(toks), classes=classes, engage=engage, suppress=suppress,
                       concepts=concepts, base_acts=base_acts, base_norms=base_norms,
                       base_row=base_row, bins=_bins_for(len(toks), n_bins))

    if not recs:
        sys.exit("no usable sentences (need compliant think/don't + shared concepts)")

    used = sorted(recs)
    have_base_acts = any(recs[s]["base_acts"] for s in used)
    vecs = load_vectors(vector_cache, model, layers, method) if have_base_acts else {}
    rng = np.random.default_rng(0)

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
        # ---- cos ----
        if have_base_acts:
            Mc = np.full((n_bins, len(layers)), np.nan)
            Pc = np.full((n_bins, len(layers)), np.nan)
            for li, L in enumerate(layers):
                if L not in vecs:
                    continue
                concepts_L = vecs[L][0]
                units = []                                    # (n_unit, n_bins) bin-averaged
                for s in used:
                    bc = base_cos_cache.get((s, L))
                    if bc is None:
                        continue
                    rec = recs[s]
                    cmap = rec[kind]
                    for c in rec["concepts"]:
                        if c not in concepts_L:
                            continue
                        tr = _trace(cmap[c], "cosine_sim", L)
                        if tr is None:
                            continue
                        ci = concepts_L.index(c)
                        m = min(len(tr), bc.shape[0], rec["n_tok"])
                        dv = np.asarray(tr[:m], np.float32) - bc[:m, ci]
                        units.append(_bin_means(dv, rec["bins"], n_bins))
                if units:
                    U = np.vstack(units)
                    for b in range(n_bins):
                        _cell(U[:, b], Mc, Pc, b, li, rng)
            panels[("cos", kind)] = (Mc, Pc)
        else:
            panels[("cos", kind)] = None

        # ---- relnorm ----
        Mr = np.full((n_bins, len(layers)), np.nan)
        Pr = np.full((n_bins, len(layers)), np.nan)
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
                    m = min(len(rel), len(b))
                    dv = rel[:m] - b[:m]
                    units.append(_bin_means(dv, rec["bins"], n_bins))
            if units:
                U = np.vstack(units)
                for b in range(n_bins):
                    _cell(U[:, b], Mr, Pr, b, li, rng)
        panels[("relnorm", kind)] = (Mr, Pr)

    meta = dict(n_sentences=len(used), n_bins=n_bins)
    return layers, n_bins, panels, meta


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, sentences=None, n_bins=10, alpha=0.05,
           vector_cache="results/vector_cache", method="baseline", model="gemma3_27b"):
    layers, n_bins, panels, meta = build(
        run_dir, sentences=sentences, n_bins=n_bins,
        vector_cache=vector_cache, method=method, model=model)
    # y ticks sit BETWEEN cells (at bin edges), labeled by fraction 0.0..1.0;
    # edge k is at imshow data coord k-0.5 (start 0.0 at top -> end 1.0 at bottom)
    edges = [k - 0.5 for k in range(n_bins + 1)]
    edge_labels = [f"{k / n_bins:.1f}" for k in range(n_bins + 1)]

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
                ax.set_yticks(edges); ax.set_yticklabels(edge_labels, fontsize=8)
                ax.set_ylabel("Position (fraction of sentence)")
            else:
                ax.set_yticks(edges); ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False)
            if metric == "relnorm":
                ax.set_xlabel("Layer")
            ax.set_title(word, fontsize=14, pad=8)
            for ti in range(n_bins):
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
             f"averaged over {meta['n_sentences']} sentences × concepts; y = token's "
             f"fractional position in its sentence (i/(n−1)), {n_bins} bins, no clipping.  "
             f"numbers shown only where significant (BH-FDR q<{alpha:g}; "
             f"per-cell sign-flip null over units, B={B})",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{meta['n_sentences']} sentences, {n_bins} fractional bins]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(
        description="Fig 4: engagement/suppression by fractional sentence position.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sentences", default=None,
                    help="optional '||'-separated subset of sentences (default: all compliant)")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig4_fraction_engage_suppress.png")
    args = ap.parse_args()
    sents = args.sentences.split("||") if args.sentences else None
    render(args.run_dir, out=args.out, sentences=sents, n_bins=args.n_bins, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
