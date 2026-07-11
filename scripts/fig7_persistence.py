#!/usr/bin/env python3
"""Fig 7: temporal persistence of concept engagement (persistence conditions).

Cosine-only (Δcos vs no_instruction @ layer 61). Each persistence condition is
compared against `persist_throughout` (sustained engagement, the natural
reference), plotted as two lines: throughout in GREY, the other condition in RED.

Fig 7a (two subplots, x = fractional sentence position):
    left  : throughout  vs  persist_first_half   ("first half" = beginning)
    right : throughout  vs  persist_once          ("once mid-sentence")
Fig 7b (separate, x = absolute token position, clipped to shortest sentence):
    throughout  vs  persist_after_fourth  ("...starting after the fourth word")
    — absolute position because that instruction keys off a fixed word count.

Measure: Δcos_{s,c}(t) = cos(v_c, r_cond) − cos(v_c, r_no_instruction) @L61,
differenced within each (sentence, concept) unit. Bands = 95% bootstrap CI over
units (B=2000). Filled markers = the condition differs from `throughout` at that
bin (paired sign-flip permutation over units, BH-FDR across bins, B=5000).

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

COS_L = 61
BASE, THROUGH = "no_instruction", "persist_throughout"
GREY, RED = "#7f8c8d", "#c0392b"
YLAB = "Δ cosine  (cond − no instruction, layer 61)"
# Fig 7a panels (fractional x): (condition, label)
A_PANELS = [("persist_first_half", "First half (beginning)"),
            ("persist_once", "Once mid-sentence")]
B_COND = ("persist_after_fourth", "After the 4th word")     # Fig 7b (position x)


def build(run_dir, *, n_bins=10, vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [COS_L], method)
    frac_conds = [THROUGH] + [c for c, _ in A_PANELS]          # fractional-x conditions
    pos_conds = [THROUGH, B_COND[0]]                           # position-x conditions
    wanted = set(frac_conds) | set(pos_conds)

    usable_lens = [len(tr) - 1 for s, sub in by_sent.items()
                   if (tr := next((r["anchored_token_strs"] for r in sub
                                   if r.get("anchored_token_strs")), None)) is not None
                   and cache.get(s) is not None]
    n_pos = min(usable_lens) if usable_lens else n_bins

    units = []                       # list of dicts keyed (x_mode, cond) -> Δcos vector
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        bins_frac, bins_pos = _bins_for(n_tok, n_bins), np.arange(n_tok)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        acos = np.asarray(ent["activations"][COS_L], np.float32)[:n_tok]
        acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
        base_cos_all = acos @ vecs[COS_L][2].T if COS_L in vecs else None
        concepts_cosL = vecs[COS_L][0] if COS_L in vecs else []

        for c in sorted(concepts):
            if base_cos_all is None or c not in concepts_cosL:
                continue
            base_cos_c = base_cos_all[:, concepts_cosL.index(c)]
            u = {}
            for cond in wanted:
                row = byc.get(cond, {}).get(c)
                if row is None:
                    continue
                tr = _trace(row, "cosine_sim", COS_L)
                if tr is None:
                    continue
                dcos = np.asarray(tr, np.float32)[:n_tok] - base_cos_c
                if cond in frac_conds:
                    u[("fraction", cond)] = _bin_means(dcos, bins_frac, n_bins)
                if cond in pos_conds:
                    u[("position", cond)] = _bin_means(dcos, bins_pos, n_pos)
            if u:
                units.append(u)

    rng = np.random.default_rng(seed)

    def line_stats(xm, cond):
        rowsU = [u[(xm, cond)] for u in units if (xm, cond) in u]
        if not rowsU:
            return None
        U = np.vstack(rowsU)
        idx = rng.integers(0, U.shape[0], size=(n_boot, U.shape[0]))
        boot = np.nanmean(U[idx], axis=1)
        return dict(mean=np.nanmean(U, axis=0), lo=np.nanpercentile(boot, 2.5, axis=0),
                    hi=np.nanpercentile(boot, 97.5, axis=0), n=U.shape[0])

    def paired_q(xm, cond, n_x, ref=THROUGH):
        """cond vs throughout, paired per unit, sign-flip null, FDR across bins."""
        diffs = [u[(xm, cond)] - u[(xm, ref)]
                 for u in units if (xm, cond) in u and (xm, ref) in u]
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
    for cond in frac_conds:
        stats[("fraction", cond)] = line_stats("fraction", cond)
        if cond != THROUGH:
            stats[("q", "fraction", cond)] = paired_q("fraction", cond, n_bins)
    for cond in pos_conds:
        stats[("position", cond)] = line_stats("position", cond)
        if cond != THROUGH:
            stats[("q", "position", cond)] = paired_q("position", cond, n_pos)
    return stats, n_bins, n_pos, len(units)


# ---- render --------------------------------------------------------------------

def _panel(ax, xc, st_through, st_cond, q, cond_label, alpha):
    """Grey throughout + red condition, CI bands, red markers where cond≠throughout."""
    ax.plot(xc, st_through["mean"], color=GREY, lw=2.0, label="Throughout", zorder=2)
    ax.fill_between(xc, st_through["lo"], st_through["hi"], color=GREY, alpha=0.18, zorder=1)
    ax.plot(xc, st_cond["mean"], color=RED, lw=2.0, label=cond_label, zorder=3)
    ax.fill_between(xc, st_cond["lo"], st_cond["hi"], color=RED, alpha=0.16, zorder=2)
    if q is not None and np.any(q < alpha):
        sig = q < alpha
        ax.plot(xc[sig], st_cond["mean"][sig], "o", color=RED, ms=5, mec="black", mew=0.5, zorder=4)
    ax.axhline(0, color="#bbb", lw=0.7, zorder=0)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")


def render(run_dir, *, out_a, out_b, alpha=0.05, **kw):
    stats, n_bins, n_pos, n_units = build(run_dir, **kw)
    xc_frac = (np.arange(n_bins) + 0.5) / n_bins
    xc_pos = np.arange(1, n_pos + 1)
    foot = (f"avg over {n_units} (sentence × concept) units; bands = 95% bootstrap CI (B=2000); "
            f"red markers = condition differs from Throughout there (paired sign-flip, BH-FDR q<{alpha:g}, "
            f"B=5000).  Δcos @ layer 61.")

    # ---- Fig 7a ----
    figa, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, (cond, lab) in zip(axes, A_PANELS):
        _panel(ax, xc_frac, stats[("fraction", THROUGH)], stats[("fraction", cond)],
               stats.get(("q", "fraction", cond)), lab, alpha)
        ax.set_title(f"Throughout  vs  {lab}", fontsize=12)
        ax.set_xlabel("Fraction of sentence progression")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel(YLAB, fontsize=10)
    figa.text(0.5, 0.005, foot, ha="center", fontsize=8, color="#444")
    figa.get_layout_engine().set(rect=(0, 0.03, 1, 1))
    figa.savefig(out_a, dpi=160, bbox_inches="tight"); plt.close(figa)
    print(f"wrote {out_a}")

    # ---- Fig 7b ----
    figb, ax = plt.subplots(1, 1, figsize=(6.6, 4.8), layout="constrained")
    _panel(ax, xc_pos, stats[("position", THROUGH)], stats[("position", B_COND[0])],
           stats.get(("q", "position", B_COND[0])), B_COND[1], alpha)
    ax.set_title(f"Throughout  vs  {B_COND[1]}", fontsize=12)
    ax.set_xlabel("Token position"); ax.set_ylabel(YLAB, fontsize=10)
    ax.set_xlim(0.5, n_pos + 0.5); ax.set_xticks(range(1, n_pos + 1))
    figb.text(0.5, 0.005, f"clipped to the shortest sentence ({n_pos} tokens).  " + foot,
              ha="center", fontsize=7.5, color="#444")
    figb.get_layout_engine().set(rect=(0, 0.04, 1, 1))
    figb.savefig(out_b, dpi=160, bbox_inches="tight"); plt.close(figb)
    print(f"wrote {out_b}  [{n_units} units, {n_bins} frac bins, {n_pos} positions]")
    return str(out_a), str(out_b)


def main():
    ap = argparse.ArgumentParser(description="Fig 7a/10b: temporal persistence (cosine).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out-a", default="fig10a_persistence.png")
    ap.add_argument("--out-b", default="fig10b_persistence.png")
    args = ap.parse_args()
    render(args.run_dir, out_a=args.out_a, out_b=args.out_b, n_bins=args.n_bins, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
