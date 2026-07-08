#!/usr/bin/env python3
"""Fig 8: positional targeting of concept engagement (token_location conditions).

Line plots of the readout vs. fractional sentence position, 2x2:
                 Beginning of sentence            End of sentence
  Cosine @L61    engage@begin / suppress@begin     engage@end / suppress@end
  Relnorm @L43   (+ think-everywhere, neutral refs) (same)

x = fraction of sentence progression f = i/(n-1), binned; y = the readout
(cosine to the concept vector @ layer 61, or relative norm @ layer 43), averaged
over 50 sentences x 10 concepts. Lines:
  * left column  : loc_beginning (engage only at the beginning) and
                   loc_not_beginning (suppress at the beginning)
  * right column : loc_end and loc_not_end
  * both columns : think_about (engage everywhere) and no_instruction (neutral),
                   as position-independent references.

Statistics (cross-sentence convention of Figs 3-7): sampling unit = (sentence,
concept) pair, collapsed to one value per fractional bin (mean over its tokens in
the bin). Shaded band = 95% bootstrap CI over units (B=2000). Significance: each
location condition vs. no_instruction, PAIRED per unit at each bin, two-sided
sign-flip permutation (B=5000) with BH-FDR across bins; significant bins get a
filled marker on the line.

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

from controllability_heatmap import classify, load_vectors, bh_fdr       # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm            # noqa: E402
from fig4_fraction_engage_suppress import _bins_for, _bin_means          # noqa: E402

COS_L, RN_L = 61, 43
BASE = "no_instruction"
THINK = "think_about"
# column -> (positive cond, negative cond, column title)
COLUMNS = [
    ("beginning", "loc_beginning", "loc_not_beginning", "Beginning of sentence"),
    ("end", "loc_end", "loc_not_end", "End of sentence"),
]
ROWS = [("cos", COS_L, "Cosine similarity (layer 61)", "cosine to concept vector"),
        ("relnorm", RN_L, "Relative norm (layer 43)", "‖r‖ / content-mean")]


def build(run_dir, *, n_bins=10, x_mode="fraction", vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0,
          cos_delta=True):
    """x_mode='fraction': 10 fractional-position bins over [0,1].
       x_mode='position': absolute token position, clipped to the shortest sentence."""
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [RN_L, COS_L], method)
    loc_conds = [c for _, p, n, _ in COLUMNS for c in (p, n)]
    wanted = set(loc_conds) | {THINK}

    # x-axis columns: fractional bins, or absolute token position clipped to the
    # shortest usable sentence (so every position is present in every sentence).
    usable_lens = [len(tr) - 1 for s, sub in by_sent.items()
                   if (tr := next((r["anchored_token_strs"] for r in sub
                                   if r.get("anchored_token_strs")), None)) is not None
                   and cache.get(s) is not None]
    n_x = (min(usable_lens) if usable_lens else n_bins) if x_mode == "position" else n_bins

    # units: list of dicts keyed (metric, cond) -> per-column vector (len n_x)
    units = []
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        # fraction -> bin index per token; position -> each token is its own column
        # (tokens at position >= n_x are dropped by _bin_means), clipping to shortest.
        bins = _bins_for(n_tok, n_x) if x_mode == "fraction" else np.arange(n_tok)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        # baselines at each layer
        base_cos = {}
        acos = np.asarray(ent["activations"][COS_L], np.float32)[:n_tok]
        acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
        base_cos_all = acos @ vecs[COS_L][2].T if COS_L in vecs else None       # (n_tok, nC)
        base_rn = _relnorm(np.asarray(ent["norms"][RN_L], np.float32)[:n_tok], classes)

        def readout(row, metric):
            L = COS_L if metric == "cos" else RN_L
            if metric == "cos":
                tr = _trace(row, "cosine_sim", L)
                return np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
            tr = _trace(row, "norms", L)
            return _relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None

        concepts_cosL = vecs[COS_L][0] if COS_L in vecs else []
        for c in sorted(concepts):
            u = {}
            base_cos_c = (base_cos_all[:, concepts_cosL.index(c)]
                          if (base_cos_all is not None and c in concepts_cosL) else None)
            # baseline lines: cos -> 0 in delta mode (else raw base cos); relnorm raw
            if base_cos_c is not None:
                u[("cos", BASE)] = _bin_means(
                    np.zeros_like(base_cos_c) if cos_delta else base_cos_c, bins, n_x)
            if base_rn is not None:
                u[("relnorm", BASE)] = _bin_means(base_rn, bins, n_x)
            # think + location condition lines
            for cond in wanted:
                row = byc.get(cond, {}).get(c)
                if row is None:
                    continue
                vcos = readout(row, "cos")
                if vcos is not None and (not cos_delta or base_cos_c is not None):
                    cv = (vcos - base_cos_c) if cos_delta else vcos     # Δcos vs no_instruction
                    u[("cos", cond)] = _bin_means(cv, bins, n_x)
                vrn = readout(row, "relnorm")
                if vrn is not None:
                    u[("relnorm", cond)] = _bin_means(vrn, bins, n_x)
            if u:
                units.append(u)

    rng = np.random.default_rng(seed)

    def line_stats(metric, cond):
        rowsU = [u[(metric, cond)] for u in units if (metric, cond) in u]
        if not rowsU:
            return None
        U = np.vstack(rowsU)
        mean = np.nanmean(U, axis=0)
        idx = rng.integers(0, U.shape[0], size=(n_boot, U.shape[0]))
        boot = np.nanmean(U[idx], axis=1)
        return dict(mean=mean, lo=np.nanpercentile(boot, 2.5, axis=0),
                    hi=np.nanpercentile(boot, 97.5, axis=0), n=U.shape[0])

    def paired_q(metric, cond, ref=BASE):
        """BH-FDR q per bin for cond vs ref, paired over units (sign-flip null)."""
        diffs = [u[(metric, cond)] - u[(metric, ref)]
                 for u in units if (metric, cond) in u and (metric, ref) in u]
        p = np.full(n_x, np.nan)
        if diffs:
            D = np.vstack(diffs)
            for b in range(n_x):
                dv = D[:, b]; dv = dv[~np.isnan(dv)]
                if len(dv) >= 3:
                    obs = float(dv.mean())
                    signs = rng.integers(0, 2, size=(n_perm, len(dv))) * 2.0 - 1.0
                    null = (signs * dv).mean(1)
                    p[b] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)
        return bh_fdr(p)

    stats = {}
    for metric, _, _, _ in ROWS:
        for cond in [BASE, THINK, *loc_conds]:
            stats[(metric, cond)] = line_stats(metric, cond)
        for _, pos, neg, _ in COLUMNS:
            stats[("q", metric, pos)] = paired_q(metric, pos)
            stats[("q", metric, neg)] = paired_q(metric, neg)
    return stats, n_x, len(units)


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, alpha=0.05, cos_delta=True, x_mode="fraction", **kw):
    stats, n_x, n_units = build(run_dir, cos_delta=cos_delta, x_mode=x_mode, **kw)
    position = (x_mode == "position")
    xc = np.arange(1, n_x + 1) if position else (np.arange(n_x) + 0.5) / n_x
    xlabel = "Token position" if position else "Fraction of sentence progression"

    style = {                                    # cond -> (color, linestyle, label suffix)
        BASE: ("#7f8c8d", ":", "No instruction"),
        THINK: ("#27ae60", "--", "Think (everywhere)"),
    }
    pos_c, neg_c = "#c0392b", "#2471a3"
    loc_conds_all = [c for _, p, n, _ in COLUMNS for c in (p, n)]

    fig = plt.figure(figsize=(12.5, 8.6), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.05)

    for ri, (metric, L, row_title, ylab) in enumerate(ROWS):
        delta = (metric == "cos" and cos_delta)
        if delta:
            row_title = "Δ Cosine similarity (layer 61)"
            ylab = "Δ cosine  (condition − no instruction)"
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=15, fontweight="bold")
        axes = sf.subplots(1, 2, sharey=True)
        for ci, (colkey, pos, neg, coltitle) in enumerate(COLUMNS):
            ax = axes[ci]
            # reference lines (no significance markers). In Δcos mode the
            # no_instruction line is identically 0 -> use the axis baseline instead.
            refs = [THINK] if delta else [THINK, BASE]
            for cond in refs:
                st = stats[(metric, cond)]
                col, ls, lab = style[cond]
                ax.plot(xc, st["mean"], color=col, ls=ls, lw=1.6, label=lab, zorder=2)
                ax.fill_between(xc, st["lo"], st["hi"], color=col, alpha=0.12, zorder=1)
            # location conditions with per-bin significance markers vs no_instruction
            region = "beginning" if colkey == "beginning" else "end"
            for cond, col, verb in ((pos, pos_c, "Think"), (neg, neg_c, "Don't think")):
                st = stats[(metric, cond)]
                if st is None:
                    continue
                q = stats[("q", metric, cond)]
                sig = q < alpha
                ax.plot(xc, st["mean"], color=col, ls="-", lw=2.0,
                        label=f"{verb} @ {region}", zorder=3)
                ax.fill_between(xc, st["lo"], st["hi"], color=col, alpha=0.16, zorder=2)
                if np.any(sig):
                    ax.plot(xc[sig], st["mean"][sig], "o", color=col, ms=5,
                            mec="black", mew=0.5, zorder=4)
            if ri == 0:
                ax.set_title(coltitle, fontsize=13)
            if ri == len(ROWS) - 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.axhline(1.0 if metric == "relnorm" else 0.0, color="#bbb", lw=0.7, zorder=0)
            if position:
                ax.set_xlim(0.5, n_x + 0.5); ax.set_xticks(range(1, n_x + 1))
            else:
                ax.set_xlim(0, 1)
            ax.legend(fontsize=8, framealpha=0.9, loc="best")
            # relnorm row: rescale y to the data's min-max (padded) instead of
            # anchoring at 0, since relnorm lives near 1.0 (sharey covers both cols)
            if metric == "relnorm":
                band = np.concatenate(
                    [stats[(metric, cc)][k][~np.isnan(stats[(metric, cc)][k])]
                     for cc in ([BASE, THINK] + loc_conds_all) if stats.get((metric, cc))
                     for k in ("lo", "hi")])
                pad = 0.06 * (band.max() - band.min())
                ax.set_ylim(band.min() - pad, band.max() + pad)

    xdesc = (f"token position (clipped to the shortest sentence, {n_x} tokens)"
             if position else "fractional sentence position")
    fig.text(0.5, 0.005, f"avg over {n_units} (sentence × concept) units by {xdesc}; bands = 95% bootstrap "
             f"CI (B=2000); filled markers = location condition differs from No instruction there "
             f"(paired sign-flip, BH-FDR q<{alpha:g}, B=5000).  cos @L61, relnorm @L43.",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{n_units} units, {n_x} x-cols, mode={x_mode}]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 8: positional targeting (token_location) line plots.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--cos-raw", action="store_true",
                    help="plot raw cosine on the top row (default: Δcos vs no_instruction)")
    ap.add_argument("--x-mode", choices=["fraction", "position"], default="fraction",
                    help="x-axis: fractional progression (default) or absolute token position (clipped)")
    ap.add_argument("--out", default="fig8_location_position.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, n_bins=args.n_bins, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model,
           cos_delta=not args.cos_raw, x_mode=args.x_mode)


if __name__ == "__main__":
    main()
