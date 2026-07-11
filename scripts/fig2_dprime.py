#!/usr/bin/env python3
"""Fig 2: engagement/suppression depth profiles as SDT sensitivity d'.

(Promoted 2026-07-10; replaces the AUROC version, retired to
results/paper/"Exploratory analysis"/Retired_Fig2_auroc_depth_profile.*)

Same design as the retired AUROC version (fig2_depth_profile.py) — engagement (red) / suppression (blue) vs
no_instruction, x = depth % — but y is the signal-detection sensitivity:

    d'_c(L) = [ mean_s cond_c,s(L) − mean_s base_c,s(L) ] / SD_s( base_c,s(L) )

computed PER CONCEPT c across the 50 sentences s (the within-concept baseline SD;
pooling concepts would let the between-concept offsets dominate the denominator
and collapse the score), then averaged over the 10 concepts. cond/base values are
the trial's token-mean readout (cosine to the concept vector, or relative norm —
for relnorm the baseline is concept-agnostic but enters each concept's d'
identically).

Interpretation: how many baseline-widths the instruction displaces the readout —
graded (does not saturate like AUROC), dimensionless, cross-model comparable
(the 1/sqrt(d) scale hits numerator and denominator equally). CAVEAT: generation
is deterministic, so sigma_baseline is across-SENTENCE (content) variability, not
trial noise — d' is control signal relative to natural content-driven variation.

Statistics: bands = 95% TWO-WAY cluster bootstrap (B=2000): each replicate
resamples BOTH the 50 sentences and the 10 concepts with replacement (one shared
sentence draw across concepts — they share sentences — and one concept-weight
draw), recomputing every concept's d' (weighted mean/SD form) and the
concept-weighted average. The claim the band supports is "controllable in
GENERAL" — generalization over sentences AND concepts; conditioning on the 10
concepts (sentence-only bootstrap) gives several-fold thinner bands. Ringed
markers = the numerator differs from 0 (sentence-clustered sign-flip on the
per-sentence paired deltas, shared flips across concepts, sigma held at its
observed value; B=5000, two-sided), BH-FDR across layers per curve.

CPU-only; in the driver as Fig 2.
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

from controllability_heatmap import classify, bh_fdr, load_vectors        # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm, POS, NEG   # noqa: E402

CONDS = [(POS, "Think about vs No instruction", "#c0392b"),
         (NEG, "Don't think about vs No instruction", "#2471a3")]
ROWS = [("cos", "Cosine similarity", "d′  (mean over concepts)"),
        ("relnorm", "Relative norm", "d′  (mean over concepts)")]
N_LAYERS_TOTAL = 62


def dprime_stats(per_concept, S_global, n_L, rng, n_boot, n_perm):
    """per_concept: list of (X, B, idx) with X, B (S_c x n_L) cond/baseline
    matrices over that concept's sentences and idx = their indices into the
    GLOBAL sentence list. Returns the concept-averaged d' per layer, a TWO-WAY
    cluster-bootstrap CI (sentences AND concepts resampled; sentence draws and
    sign-flips shared across concepts, which share sentences), and sign-flip q
    (numerator vs 0)."""
    nC = len(per_concept)
    # shared resamples: one sentence-multiplicity draw + one concept draw per
    # replicate; one sentence flip set for the permutation null
    W = rng.multinomial(S_global, np.full(S_global, 1.0 / S_global),
                        size=n_boot).astype(float)
    Mc = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=n_boot).astype(float)
    E = rng.integers(0, 2, size=(n_perm, S_global)) * 2.0 - 1.0
    obs = np.full((nC, n_L), np.nan)
    boot = np.full((nC, n_boot, n_L), np.nan)
    null = np.full((nC, n_perm, n_L), np.nan)
    for ci, (X, B, idx) in enumerate(per_concept):
        if X.shape[0] < 3:
            continue
        Wc, Ec = W[:, idx], E[:, idx]
        for li in range(n_L):
            x, b = X[:, li], B[:, li]
            ok = ~(np.isnan(x) | np.isnan(b))
            if ok.sum() < 3:
                continue
            x, b = x[ok], b[ok]
            sd = b.std(ddof=1)
            if sd <= 0:
                continue
            obs[ci, li] = (x.mean() - b.mean()) / sd
            # sentence-cluster part: weighted mean/SD within each replicate
            w = Wc[:, ok]
            n = w.sum(1)
            good = n > 1
            mB = np.divide(w @ b, n, out=np.full(n_boot, np.nan), where=good)
            mX = np.divide(w @ x, n, out=np.full(n_boot, np.nan), where=good)
            eB2 = np.divide(w @ (b * b), n, out=np.full(n_boot, np.nan), where=good)
            var = (eB2 - mB ** 2) * np.divide(n, n - 1, out=np.ones(n_boot), where=good)
            sd_b = np.sqrt(np.clip(var, 1e-24, None))
            boot[ci, :, li] = (mX - mB) / sd_b
            # sign-flip null on the paired numerator (sigma fixed at observed)
            d = x - b
            null[ci, :, li] = (Ec[:, ok] @ d) / ok.sum() / sd
    dp = np.nanmean(obs, axis=0)
    # concept part of the bootstrap: weight each concept's replicate d' by the
    # concept-multiplicity draw (nan-aware renormalization)
    wts = Mc.T[:, :, None]                                  # (nC, n_boot, 1)
    okb = ~np.isnan(boot)
    num = np.nansum(np.where(okb, boot, 0.0) * wts, axis=0)
    den = (wts * okb).sum(axis=0)
    bavg = np.divide(num, den, out=np.full((n_boot, n_L), np.nan), where=den > 0)
    lo = np.nanpercentile(bavg, 2.5, axis=0)
    hi = np.nanpercentile(bavg, 97.5, axis=0)
    navg = np.nanmean(null, axis=0)
    p = np.full(n_L, np.nan)
    for li in range(n_L):
        if not np.isnan(dp[li]):
            p[li] = (1 + int((np.abs(navg[:, li]) >= abs(dp[li]) - 1e-15).sum())) \
                / (n_perm + 1)
    return dict(dp=dp, lo=lo, hi=hi, q=bh_fdr(p),
                per_concept=obs, n_concepts=int(np.sum(~np.isnan(obs).all(1))))


def build(run_dir, *, vector_cache="results/vector_cache", method="baseline",
          model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, layers, method)
    conds = [c for c, _, _ in CONDS]
    concepts_L = vecs[layers[0]][0] if layers[0] in vecs else []

    # vals[(metric, cond)][concept][sentence] = per-layer vector (token-mean);
    # bases[metric][concept][sentence] likewise (relnorm duplicated per concept).
    vals = {(m, c): defaultdict(dict) for m, _, _ in ROWS for c in conds}
    bases = {m: defaultdict(dict) for m, _, _ in ROWS}

    for s, sub in by_sent.items():
        ent = cache.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if ent is None or toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]

        base_cos, base_rel = {}, {}
        for L in layers:
            A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            base_cos[L] = An @ vecs[L][2].T if L in vecs else None
            rl = _relnorm(np.asarray(ent["norms"][L], np.float32)[:n_tok], classes)
            base_rel[L] = float(np.nanmean(rl)) if rl is not None else np.nan

        byc = defaultdict(dict)
        for r in sub:
            if r["condition_id"] in conds and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r

        for c in concepts_L:
            ci = concepts_L.index(c)
            bc = np.array([float(np.nanmean(base_cos[L][:, ci]))
                           if base_cos[L] is not None else np.nan for L in layers])
            br = np.array([base_rel[L] for L in layers])
            bases["cos"][c][s] = bc
            bases["relnorm"][c][s] = br
            for cond in conds:
                r_ = byc.get(cond, {}).get(c)
                if r_ is None:
                    continue
                xc = np.full(len(layers), np.nan); xr = np.full(len(layers), np.nan)
                for li, L in enumerate(layers):
                    tr = _trace(r_, "cosine_sim", L)
                    if tr is not None:
                        xc[li] = float(np.nanmean(np.asarray(tr, np.float32)[:n_tok]))
                    nr = _trace(r_, "norms", L)
                    if nr is not None:
                        rl = _relnorm(np.asarray(nr, np.float32)[:n_tok], classes)
                        if rl is not None:
                            xr[li] = float(np.nanmean(rl))
                vals[("cos", cond)][c][s] = xc
                vals[("relnorm", cond)][c][s] = xr

    rng = np.random.default_rng(seed)
    all_sents = sorted({s for m in bases for c in bases[m] for s in bases[m][c]})
    s_idx = {s: i for i, s in enumerate(all_sents)}
    stats = {}
    for (metric, cond), by_c in vals.items():
        per_concept = []
        for c in concepts_L:
            ss = sorted(set(by_c.get(c, {})) & set(bases[metric][c]))
            if len(ss) < 3:
                continue
            X = np.vstack([by_c[c][s] for s in ss])
            B = np.vstack([bases[metric][c][s] for s in ss])
            per_concept.append((X, B, np.array([s_idx[s] for s in ss])))
        stats[(metric, cond)] = dprime_stats(per_concept, len(all_sents), len(layers),
                                             rng, n_boot, n_perm)
    return layers, stats


def render(run_dir, *, out, alpha=0.05, **kw):
    layers, stats = build(run_dir, **kw)
    x = np.arange(len(layers))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                             layout="constrained")
    depth_pct = [f"{100 * L / N_LAYERS_TOTAL:.0f}" for L in layers]
    for ax, (metric, title, ylab) in zip(axes, ROWS):
        for cond, label, color in CONDS:
            st = stats[(metric, cond)]
            ax.plot(x, st["dp"], "-o", color=color, lw=2.0, ms=4.5,
                    label=label, zorder=3)
            ax.fill_between(x, st["lo"], st["hi"], color=color, alpha=0.15, zorder=2)
            sig = st["q"] < alpha
            if np.any(sig):
                ax.plot(x[sig], st["dp"][sig], "o", color=color, ms=8,
                        mfc=color, mec="black", mew=0.9, zorder=4)
        ax.axhline(0.0, color="#888", lw=0.9, ls="--", zorder=1)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(depth_pct)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=10, labelspacing=0.4,
                   bbox_to_anchor=(0.02, 0.90))
    axes[1].set_xlabel("Depth (%)", fontsize=11)
    # no on-figure fine print: measure/statistics definitions live in Fig2.md
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{stats[('cos', POS)]['n_concepts']} concepts]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 2: d' depth profiles (condition vs no_instruction).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig2_dprime.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha, vector_cache=args.vector_cache,
           method=args.method, model=args.model)


if __name__ == "__main__":
    main()
