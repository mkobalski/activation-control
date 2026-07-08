#!/usr/bin/env python3
"""Fig 6: ramp GAIN (magnitude change from lowest to highest instructed
intensity), by fractional sentence position, averaged across sentences.

Identical to Fig 5 in every way EXCEPT the per-unit statistic: instead of the
rank (signed Spearman over levels), the cell shows the GAIN measure of the
earlier runs (see scripts/controllability_heatmap.py, the "gain_*" heatmaps):

    per (concept, sentence, token, layer):  g = readout(last level) − readout(first level)
    Gain(cell) = mean over units of g          (in the readout's native units)

So Fig 6 keeps magnitude (Δcos / Δrelnorm) where Fig 5 keeps only the sign/order.
Both endpoints must be present for a unit to contribute (the old code required the
full ramp; gain only uses first/last, which is what we require here).

Columns (as Fig 5):
  * left  — "think about" (first) vs "think intensely" (last): gain = x_intensely − x_think
  * right — the intensity ramp: gain = x_{4/4} − x_{1/4}

Statistics adjusted for cross-sentence averaging exactly as Fig 3–5: unit =
(sentence, concept) pair; within a fractional bin a unit is collapsed to the mean
of its per-token gain; the cell statistic is the mean over units; the null is a
two-sided sign-flip permutation over units (B=5000); BH-FDR per panel. (This is
the same sign-flip null the 05-05 gain heatmaps already used per concept, here
generalized to (sentence, concept) units.)

CPU-only, no model load, no baseline needed.
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

from controllability_heatmap import classify, bh_fdr                    # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm           # noqa: E402
from fig3_position_engage_suppress import _cell                         # noqa: E402
from fig4_fraction_engage_suppress import _bins_for, _bin_means         # noqa: E402

# columns: (key, big word, [first_cond, last_cond]); gain = readout(last) - readout(first)
COLS = [
    ("think_ramp", "Think → Think intensely", ["think_about", "think_intensely"]),
    ("intensity_ramp", "Intensity ramp (1→4)",
     ["think_intensity_1_of_4", "think_intensity_4_of_4"]),
]


def build(run_dir, *, sentences=None, n_bins=10, metric_rows=("cos", "relnorm")):
    """Per-panel (M, P) of fractional-position-bin x layer for each
    (metric, column). Returns (layers, n_bins, panels, meta)."""
    rows = _load_json(run_dir)
    compliant = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in compliant:
        by_sent[r["sentence"]].append(r)
    sent_list = sentences if sentences else sorted(by_sent)

    layers = sorted({int(x) for r in compliant for x in (r.get("analysis_layers") or [])})
    wanted = {c for _, _, ends in COLS for c in ends}

    recs = {}
    for s in sent_list:
        sub = by_sent.get(s, [])
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if not toks_row:
            continue
        toks = toks_row[1:]
        classes = [classify(t) for t in toks]
        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)
        if not concepts:
            continue
        recs[s] = dict(n_tok=len(toks), classes=classes, bins=_bins_for(len(toks), n_bins),
                       byc=byc, concepts=sorted(concepts))

    if not recs:
        sys.exit("no usable sentences (need the intensity/think conditions)")
    used = sorted(recs)
    rng = np.random.default_rng(0)

    def _readout(row, metric, L, classes, n_tok):
        if metric == "cos":
            tr = _trace(row, "cosine_sim", L)
            return np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
        tr = _trace(row, "norms", L)
        if tr is None:
            return None
        return _relnorm(np.asarray(tr, np.float32)[:n_tok], classes)

    panels = {}
    for metric in metric_rows:
        for col_key, _word, (c_first, c_last) in COLS:
            M = np.full((n_bins, len(layers)), np.nan)
            P = np.full((n_bins, len(layers)), np.nan)
            for li, L in enumerate(layers):
                units = []
                for s in used:
                    rec = recs[s]
                    nt, classes, bins = rec["n_tok"], rec["classes"], rec["bins"]
                    for c in rec["concepts"]:
                        r0 = rec["byc"].get(c_first, {}).get(c)
                        r1 = rec["byc"].get(c_last, {}).get(c)
                        if r0 is None or r1 is None:
                            continue
                        x0 = _readout(r0, metric, L, classes, nt)
                        x1 = _readout(r1, metric, L, classes, nt)
                        if x0 is None or x1 is None:
                            continue
                        m = min(len(x0), len(x1))
                        gain = np.full(nt, np.nan)
                        gain[:m] = np.asarray(x1[:m], float) - np.asarray(x0[:m], float)
                        units.append(_bin_means(gain, bins, n_bins))
                if units:
                    U = np.vstack(units)
                    for b in range(n_bins):
                        _cell(U[:, b], M, P, b, li, rng)
            panels[(metric, col_key)] = (M, P)

    meta = dict(n_sentences=len(used), n_bins=n_bins)
    return layers, n_bins, panels, meta


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, sentences=None, n_bins=10, alpha=0.05):
    layers, n_bins, panels, meta = build(run_dir, sentences=sentences, n_bins=n_bins)
    edges = [k - 0.5 for k in range(n_bins + 1)]
    edge_labels = [f"{k / n_bins:.1f}" for k in range(n_bins + 1)]

    # (metric, row header, colorbar label, decimals)
    row_defs = [
        ("cos", "Cosine similarity", "Δcos (gain)", 3),
        ("relnorm", "Relative norm", "Δrelnorm (gain)", 3),
    ]

    fig = plt.figure(figsize=(3.0 + 0.42 * len(layers) * 2,
                              2 * (0.34 * 14) + 4.4),   # height pinned to Fig 2's 14-token reference
                     layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.05)

    for ri, (metric, row_title, cbar_label, dec) in enumerate(row_defs):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=17, fontweight="bold")
        sf_axes = sf.subplots(1, 2)

        A = 0.0
        for col_key, _, _ in COLS:
            M = panels[(metric, col_key)][0]
            a = np.nanmax(np.abs(M))
            if np.isfinite(a):
                A = max(A, float(a))
        A = A if A > 0 else 1.0

        row_im = None
        for ci, (col_key, word, _ends) in enumerate(COLS):
            ax = sf_axes[ci]
            M, P = panels[(metric, col_key)]
            Q = bh_fdr(P)
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-A, vmax=A)
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
                    ax.text(li, ti, f"{v:.{dec}f}", ha="center", va="center", fontsize=5.5,
                            color="white" if abs(v) / A > 0.55 else "black")

        if row_im is not None:
            cb = sf.colorbar(row_im, ax=sf_axes, fraction=0.035, pad=0.02)
            cb.set_label(cbar_label, fontsize=9)

    fig.text(0.5, 0.006,
             f"GAIN = mean over (sentence × concept) units of [readout(last level) − "
             f"readout(first level)]; averaged over {meta['n_sentences']} sentences; y = "
             f"fractional sentence position, {n_bins} bins.  numbers shown only where "
             f"significant (BH-FDR q<{alpha:g}; per-cell sign-flip null over units, B=5000)",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.022, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  [{meta['n_sentences']} sentences, {n_bins} fractional bins]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(
        description="Fig 6: ramp gain (last − first readout) by fractional position.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sentences", default=None,
                    help="optional '||'-separated subset of sentences (default: all compliant)")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig6_gain_intensity.png")
    args = ap.parse_args()
    sents = args.sentences.split("||") if args.sentences else None
    render(args.run_dir, out=args.out, sentences=sents, n_bins=args.n_bins, alpha=args.alpha)


if __name__ == "__main__":
    main()
