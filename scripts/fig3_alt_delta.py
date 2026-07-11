#!/usr/bin/env python3
"""Fig 3_alt (DRAFT, comparison only): Fig 3 with raw endpoint gains instead of AUROC.

Same design as fig3_gain_auroc.py — lexical step (think_about -> think_intensely,
green) and numeric ramp endpoints (intensity 1/4 -> 4/4, yellow), x = depth % —
but y is the MEAN GAIN in the readout itself (the retired Fig 6's measure):

    top    : mean over units of  Δcos     = token-mean cos(last) − cos(first)
    bottom : mean over units of  Δrelnorm = token-mean relnorm(last) − relnorm(first)

Δ is within-unit (same sentence & concept at both endpoints — concept offsets
cancel). NOTE: native units — NOT cross-model comparable (raw Δcos carries the
1/sqrt(d) width factor; see MEASURES.md §6); that is the trade being compared.

Statistics as fig2_alt_delta.py: 95% sentence-cluster bootstrap bands (B=2000);
ringed marker = Δ≠0 by sentence-clustered sign-flip (B=5000, two-sided) with
BH-FDR across layers per curve.

CPU-only. Not in the driver — draft for comparison with Fig 3.
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
from fig2_alt_delta import delta_stats                                    # noqa: E402

CONTRASTS = [
    ("lex", "think_about", "think_intensely",
     "Think about → Think intensely about", "#1e8449"),
    ("ramp", "think_intensity_1_of_4", "think_intensity_4_of_4",
     "Think at intensity {1→4} about", "#d4a017"),
]
ROWS = [("cos", "Cosine similarity", "Δ cosine  (last − first)"),
        ("relnorm", "Relative norm", "Δ relnorm  (last − first)")]
N_LAYERS_TOTAL = 62


def build(run_dir, *, n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    wanted = {c for _, a, b, _, _ in CONTRASTS for c in (a, b)}

    pairs = {(m, k): {} for m, _, _ in ROWS for k, *_ in CONTRASTS}
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
            if not common:
                continue
            for metric, _, _ in ROWS:
                X = np.vstack([unit_vec(byc[c_last][c], metric) for c in common])
                B = np.vstack([unit_vec(byc[c_first][c], metric) for c in common])
                pairs[(metric, key)][s] = (X, B)

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
        for key, _, _, label, color in CONTRASTS:
            st = stats[(metric, key)]
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
    fig.text(0.5, 0.005, "DRAFT (comparison with Fig 3): mean per-unit endpoint gain (last − first) in NATIVE units "
             "(not cross-model comparable);  bands = 95% cluster bootstrap over the 50 sentences (B=2000); "
             "ringed = Δ≠0 (sentence-clustered sign-flip, BH-FDR across layers, B=5000).",
             ha="center", fontsize=7.5, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st0 = stats[("cos", "ramp")]
    print(f"wrote {out}  [{st0['n_units']} units, {st0['n_sent']} sentences]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 3_alt (draft): mean endpoint-gain depth profiles.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig3_alt_delta.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha)


if __name__ == "__main__":
    main()
