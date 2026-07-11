#!/usr/bin/env python3
"""Fig 3: endpoint-gain depth profiles as SDT sensitivity d'.

(Promoted 2026-07-10; replaces the AUROC version, retired to
results/paper/"Exploratory analysis"/Retired_Fig3_gain_auroc.*. The
adjacent-step AUROC companion Fig3_aux is kept.)

Same design as the retired AUROC version (fig3_gain_auroc.py) — lexical step (think_about -> think_intensely,
green) and numeric ramp endpoints (intensity 1/4 -> 4/4, yellow), x = depth % —
but y is the sensitivity of the HIGH endpoint against the LOW endpoint:

    d'_c(L) = [ mean_s last_c,s(L) − mean_s first_c,s(L) ] / SD_s( first_c,s(L) )

per concept c across the 50 sentences (the LOW endpoint plays the baseline role:
its within-concept across-sentence SD is the denominator), averaged over the 10
concepts. Values are the trial's token-mean readout (cosine / relative norm).
Dimensionless, graded (no AUROC saturation), cross-model comparable; the usual
caveat: SD is across-sentence (content) variability, not trial noise.

Statistics as fig2_dprime.py: 95% TWO-WAY cluster bootstrap (sentences AND
concepts resampled per replicate, shared sentence draws across concepts;
B=2000) — supports "the dial works in general"; ringed markers = numerator ≠ 0
(sentence-clustered sign-flip, shared flips, sigma fixed; B=5000, two-sided),
BH-FDR across layers per curve.

CPU-only; in the driver as Fig 3.
"""

import argparse
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

from controllability_heatmap import classify                              # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm             # noqa: E402
from fig2_dprime import dprime_stats                                      # noqa: E402

CONTRASTS = [
    ("lex", "think_about", "think_intensely",
     "Think about → Think intensely about", "#1e8449"),
    ("ramp", "think_intensity_1_of_4", "think_intensity_4_of_4",
     "Think at intensity {1→4} about", "#d4a017"),
]
ROWS = [("cos", "Cosine similarity", "d′  (mean over concepts)"),
        ("relnorm", "Relative norm", "d′  (mean over concepts)")]
N_LAYERS_TOTAL = 62


def build(run_dir, *, n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    wanted = {c for _, a, b, _, _ in CONTRASTS for c in (a, b)}

    # vals[(metric, key)][concept][sentence] = (last_vec, first_vec) per-layer
    vals = {(m, k): defaultdict(dict) for m, _, _ in ROWS for k, *_ in CONTRASTS}
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]

        byc = defaultdict(dict)
        for r in sub:
            if r["condition_id"] in wanted and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r

        def unit_vec(row, metric):
            out = np.full(len(layers), np.nan)
            for li, L in enumerate(layers):
                if metric == "cos":
                    tr = _trace(row, "cosine_sim", L)
                    if tr is not None:
                        out[li] = float(np.nanmean(np.asarray(tr, np.float32)[:n_tok]))
                else:
                    tr = _trace(row, "norms", L)
                    if tr is not None:
                        rl = _relnorm(np.asarray(tr, np.float32)[:n_tok], classes)
                        if rl is not None:
                            out[li] = float(np.nanmean(rl))
            return out

        for key, c_first, c_last, _, _ in CONTRASTS:
            common = sorted(set(byc.get(c_first, {})) & set(byc.get(c_last, {})))
            for c in common:
                for metric, _, _ in ROWS:
                    vals[(metric, key)][c][s] = (unit_vec(byc[c_last][c], metric),
                                                 unit_vec(byc[c_first][c], metric))

    rng = np.random.default_rng(seed)
    all_sents = sorted({s for v in vals.values() for c in v for s in v[c]})
    s_idx = {s: i for i, s in enumerate(all_sents)}
    stats = {}
    for (metric, key), by_c in vals.items():
        per_concept = []
        for c in sorted(by_c):
            ss = sorted(by_c[c])
            if len(ss) < 3:
                continue
            X = np.vstack([by_c[c][s][0] for s in ss])       # high endpoint
            B = np.vstack([by_c[c][s][1] for s in ss])       # low endpoint (baseline role)
            per_concept.append((X, B, np.array([s_idx[s] for s in ss])))
        stats[(metric, key)] = dprime_stats(per_concept, len(all_sents), len(layers),
                                            rng, n_boot, n_perm)
    return layers, stats


def render(run_dir, *, out, alpha=0.05, **kw):
    layers, stats = build(run_dir, **kw)
    x = np.arange(len(layers))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                             layout="constrained")
    depth_pct = [f"{100 * L / N_LAYERS_TOTAL:.0f}" for L in layers]
    for ax, (metric, title, ylab) in zip(axes, ROWS):
        for key, _, _, label, color in CONTRASTS:
            st = stats[(metric, key)]
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
    # no on-figure fine print: measure/statistics definitions live in Fig3.md
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{stats[('cos', 'ramp')]['n_concepts']} concepts]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 3: endpoint-gain d' depth profiles.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig3_dprime.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha)


if __name__ == "__main__":
    main()
