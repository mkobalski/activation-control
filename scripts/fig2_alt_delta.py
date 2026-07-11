#!/usr/bin/env python3
"""Fig 2_alt (DRAFT, comparison only): Fig 2 with raw effect sizes instead of AUROC.

Same design as fig2_depth_profile.py — engagement (think_about, red) and
suppression (dont_think_about, blue) vs no_instruction, x = depth % — but y is
the MEAN CHANGE in the readout itself:

    top    : mean over units of  Δcos     = token-mean cos(cond) − token-mean cos(baseline)
    bottom : mean over units of  Δrelnorm = token-mean relnorm(cond) − relnorm(baseline)

Δ is differenced within each (sentence, concept) unit (cancels the per-concept
cosine offsets — the same role pairing plays in Fig 2's paired AUROC). NOTE:
native units — NOT cross-model comparable (raw Δcos carries the 1/sqrt(d)
width factor; see MEASURES.md §6); that is exactly the trade being compared.

Statistics (sentence-clustered, as the whole suite):
  band = 95% cluster bootstrap over the 50 sentences (B=2000);
  ringed marker = mean Δ ≠ 0, sentence-clustered sign-flip (all of a
  sentence's units flip together; B=5000, two-sided), BH-FDR across layers.

CPU-only. Not in the driver — draft for comparison with Fig 2.
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
ROWS = [("cos", "Cosine similarity", "Δ cosine  (cond − no instruction)"),
        ("relnorm", "Relative norm", "Δ relnorm  (cond − no instruction)")]
N_LAYERS_TOTAL = 62


def delta_stats(by_s, n_L, rng, n_boot, n_perm):
    """Mean per-unit Δ per layer + sentence-cluster bootstrap CI + clustered
    sign-flip p (vs 0), from {sentence: (X, B)} unit-x-layer matrices."""
    sents = sorted(by_s)
    S = len(sents)
    Dsum = np.zeros((S, n_L)); Dcnt = np.zeros((S, n_L))
    n_units = 0
    for si, s in enumerate(sents):
        X, B = by_s[s]
        D = X - B
        ok = ~np.isnan(D)
        Dsum[si] = np.where(ok, D, 0.0).sum(0)
        Dcnt[si] = ok.sum(0)
        n_units += X.shape[0]
    tot = Dcnt.sum(0)
    mean = np.divide(Dsum.sum(0), tot, out=np.full(n_L, np.nan), where=tot > 0)
    # cluster bootstrap (multiplicity-vector form)
    Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
    den = Mm @ Dcnt
    boots = np.divide(Mm @ Dsum, den, out=np.full_like(den, np.nan), where=den > 0)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    # sentence-clustered sign-flip vs 0
    signs = rng.integers(0, 2, size=(n_perm, S)) * 2.0 - 1.0
    null = np.divide(signs @ Dsum, tot[None, :],
                     out=np.full((n_perm, n_L), np.nan), where=tot[None, :] > 0)
    p = np.full(n_L, np.nan)
    for li in range(n_L):
        if tot[li] >= 10 and not np.isnan(mean[li]):
            p[li] = (1 + int((np.abs(null[:, li]) >= abs(mean[li]) - 1e-15).sum())) \
                / (n_perm + 1)
    return dict(mean=mean, lo=lo, hi=hi, q=bh_fdr(p), n_units=n_units, n_sent=S)


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

    pairs = {(m, c): {} for m, _, _ in ROWS for c in conds}
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
        concepts_L = vecs[layers[0]][0] if layers[0] in vecs else []

        byc = defaultdict(dict)
        for r in sub:
            if r["condition_id"] in conds and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r

        for cond in conds:
            Xc, Bc, Xr, Br = [], [], [], []
            for c, r_ in byc[cond].items():
                if c not in concepts_L:
                    continue
                ci = concepts_L.index(c)
                xc = np.full(len(layers), np.nan); bc = np.full(len(layers), np.nan)
                xr = np.full(len(layers), np.nan); br = np.full(len(layers), np.nan)
                for li, L in enumerate(layers):
                    tr = _trace(r_, "cosine_sim", L)
                    if tr is not None and base_cos[L] is not None:
                        v = np.asarray(tr, np.float32)[:n_tok]
                        m = min(len(v), n_tok)
                        xc[li] = float(np.nanmean(v[:m]))
                        bc[li] = float(np.nanmean(base_cos[L][:m, ci]))
                    nr = _trace(r_, "norms", L)
                    if nr is not None:
                        rl = _relnorm(np.asarray(nr, np.float32)[:n_tok], classes)
                        if rl is not None:
                            xr[li] = float(np.nanmean(rl))
                            br[li] = base_rel[L]
                Xc.append(xc); Bc.append(bc); Xr.append(xr); Br.append(br)
            if Xc:
                pairs[("cos", cond)][s] = (np.vstack(Xc), np.vstack(Bc))
                pairs[("relnorm", cond)][s] = (np.vstack(Xr), np.vstack(Br))

    rng = np.random.default_rng(seed)
    stats = {key: delta_stats(by_s, len(layers), rng, n_boot, n_perm)
             for key, by_s in pairs.items()}
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
            ax.plot(x, st["mean"], "-o", color=color, lw=2.0, ms=4.5,
                    label=label, zorder=3)
            ax.fill_between(x, st["lo"], st["hi"], color=color, alpha=0.15, zorder=2)
            sig = st["q"] < alpha
            if np.any(sig):
                ax.plot(x[sig], st["mean"][sig], "o", color=color, ms=8,
                        mfc=color, mec="black", mew=0.9, zorder=4)
        ax.axhline(0.0, color="#888", lw=0.9, ls="--", zorder=1)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(depth_pct)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[1].legend(loc="upper right", frameon=False, fontsize=10, labelspacing=0.4)
    axes[1].set_xlabel("Depth (%)", fontsize=11)
    fig.text(0.5, 0.005, "DRAFT (comparison with Fig 2): mean per-unit Δ vs no_instruction in NATIVE units "
             "(not cross-model comparable);  bands = 95% cluster bootstrap over the 50 sentences (B=2000); "
             "ringed = Δ≠0 (sentence-clustered sign-flip, BH-FDR across layers, B=5000).",
             ha="center", fontsize=7.5, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st0 = stats[("cos", POS)]
    print(f"wrote {out}  [{st0['n_units']} units, {st0['n_sent']} sentences]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 2_alt (draft): mean-Δ depth profiles (condition vs no_instruction).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig2_alt_delta.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha, vector_cache=args.vector_cache,
           method=args.method, model=args.model)


if __name__ == "__main__":
    main()
