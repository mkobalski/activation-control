#!/usr/bin/env python3
"""[RETIRED as the paper's Fig 3 2026-07-10 -- replaced by fig3_dprime.py; kept
runnable, and still powers Fig3_aux (adjacent-step resolution).]

Gain as AUROC — is the TOP of each intensity contrast
discriminable from its BOTTOM, per depth?

Companion to Fig 4 (rank): where Fig 4 asks "does the readout ORDER with
intensity (all levels)", this asks "how separable are the endpoints":

    green  : think_about            ->  think_intensely           (lexical)
    yellow : think_intensity_1_of_4 ->  think_intensity_4_of_4    (numeric)

Same two-panel format as Figs 2/4 (x = depth %):
    top    : Cosine similarity  -> PAIRED AUROC  = P(last level's token-mean
             cosine > the SAME unit's first level)  [pairing cancels the
             per-concept geometric offsets]
    bottom : Relative norm      -> POOLED AUROC over all last-vs-first value
             pairs (concept-agnostic readout)

One value per (sentence, concept) unit per side per layer = the trial's
token-mean readout. Bands = 95% cluster bootstrap over the 50 sentences
(B=2000); ringed markers = AUROC != 0.5 (paired: per-unit swap; pooled:
within-sentence swap; B=5000, two-sided), BH-FDR across the 20 layers per
curve. 0.5 = chance; >0.5 = the higher-intensity endpoint reads HIGHER.
Definitions in results/paper/Fig3.md. CPU-only, no model load.
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

from controllability_heatmap import classify, bh_fdr                      # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm             # noqa: E402

# (key, first condition, last condition, label, color)
CONTRASTS = [
    ("lex", "think_about", "think_intensely",
     "Think about → Think intensely about", "#1e8449"),
    ("ramp", "think_intensity_1_of_4", "think_intensity_4_of_4",
     "Think at intensity {1→4} about", "#d4a017"),
]
ROWS = [("cos", "Cosine similarity", "Paired AUROC"),
        ("relnorm", "Relative norm", "Pooled AUROC")]
N_LAYERS_TOTAL = 62


def build(run_dir, *, contrasts=CONTRASTS, n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    wanted = {c for _, a, b, _, _ in contrasts for c in (a, b)}

    # pairs[(metric, key)][sentence] = (X_last, X_first): unit x layer matrices
    pairs = {(m, k): {} for m, _, _ in ROWS for k, *_ in contrasts}
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

        for key, c_first, c_last, _, _ in contrasts:
            common = sorted(set(byc.get(c_first, {})) & set(byc.get(c_last, {})))
            if not common:
                continue
            for metric, _, _ in ROWS:
                X = np.vstack([unit_vec(byc[c_last][c], metric) for c in common])
                B = np.vstack([unit_vec(byc[c_first][c], metric) for c in common])
                pairs[(metric, key)][s] = (X, B)

    rng = np.random.default_rng(seed)
    stats = {}
    for mkey, by_s in pairs.items():
        metric = mkey[0]
        sents = sorted(by_s)
        S = len(sents)
        n_L = len(layers)
        auc = np.full(n_L, np.nan); lo = np.full(n_L, np.nan)
        hi = np.full(n_L, np.nan); p = np.full(n_L, np.nan)

        for li in range(n_L):
            xs, bs = [], []
            for s in sents:
                xv, bv = by_s[s][0][:, li], by_s[s][1][:, li]
                ok = ~(np.isnan(xv) | np.isnan(bv))
                xs.append(xv[ok]); bs.append(bv[ok])
            k = np.array([len(v) for v in xs], float)
            if k.sum() < 10:
                continue

            if metric == "cos":
                # paired: per-unit win, cluster bootstrap, per-unit swap null
                wins_s = [np.where(x > b, 1.0, np.where(x == b, 0.5, 0.0))
                          for x, b in zip(xs, bs)]
                Wsum = np.array([w.sum() for w in wins_s])
                auc[li] = Wsum.sum() / k.sum()
                Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
                nn = Mm @ k
                boots = np.divide(Mm @ Wsum, nn, out=np.full(n_boot, np.nan), where=nn > 0)
                lo[li], hi[li] = np.nanpercentile(boots, [2.5, 97.5])
                w_all = np.concatenate(wins_s)
                F = rng.integers(0, 2, size=(n_perm, len(w_all)))
                null = (np.where(F == 1, 1.0 - w_all, w_all)).mean(1)
                obs_dev = abs(auc[li] - 0.5)
                p[li] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)
            else:
                # pooled: win-count matrices, cluster bootstrap, sentence-swap null
                def wins(A_blocks, B_blocks):
                    W = np.empty((S, S))
                    for i, a in enumerate(A_blocks):
                        for j, b in enumerate(B_blocks):
                            if len(a) == 0 or len(b) == 0:
                                W[i, j] = 0.0
                            else:
                                d = a[:, None] - b[None, :]
                                W[i, j] = (d > 0).sum() + 0.5 * (d == 0).sum()
                    return W
                C_xb = wins(xs, bs); C_xx = wins(xs, xs)
                C_bb = wins(bs, bs); C_bx = wins(bs, xs)
                n_tot = k.sum()
                auc[li] = C_xb.sum() / (n_tot * n_tot)
                Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
                U = ((Mm @ C_xb) * Mm).sum(1)
                nn = Mm @ k
                boots = np.divide(U, nn * nn, out=np.full(n_boot, np.nan), where=nn > 0)
                lo[li], hi[li] = np.nanpercentile(boots, [2.5, 97.5])
                F = rng.integers(0, 2, size=(n_perm, S)).astype(float)
                G = 1.0 - F
                U0 = ((G @ C_xb) * G).sum(1) + ((G @ C_xx) * F).sum(1) \
                   + ((F @ C_bb) * G).sum(1) + ((F @ C_bx) * F).sum(1)
                null = U0 / (n_tot * n_tot)
                obs_dev = abs(auc[li] - 0.5)
                p[li] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)

        stats[mkey] = dict(auc=auc, lo=lo, hi=hi, q=bh_fdr(p),
                           n_units=int(sum(v[0].shape[0] for v in by_s.values())),
                           n_sent=S)
    return layers, stats


def render(run_dir, *, out, alpha=0.05, contrasts=CONTRASTS, footnote=None, **kw):
    layers, stats = build(run_dir, contrasts=contrasts, **kw)
    x = np.arange(len(layers))
    depth_pct = [f"{100 * L / N_LAYERS_TOTAL:.0f}" for L in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                             layout="constrained")
    for ax, (metric, title, ylab) in zip(axes, ROWS):
        for key, _a, _b, label, color in contrasts:
            st = stats[(metric, key)]
            ax.plot(x, st["auc"], "-o", color=color, lw=2.0, ms=4.5,
                    label=label, zorder=3)
            ax.fill_between(x, st["lo"], st["hi"], color=color, alpha=0.15, zorder=2)
            sig = st["q"] < alpha
            if np.any(sig):
                ax.plot(x[sig], st["auc"][sig], "o", color=color, ms=8,
                        mfc=color, mec="black", mew=0.9, zorder=4)
        ax.axhline(0.5, color="#888", lw=0.9, ls="--", zorder=1)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(depth_pct)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), frameon=False,
                   fontsize=10, labelspacing=0.4)
    axes[1].set_xlabel("Depth (%)", fontsize=11)

    if footnote:
        fig.text(0.5, 0.005, footnote, ha="center", fontsize=7.5, color="#444")
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    counts = ", ".join(f"{k} n={stats[('cos', k)]['n_units']}" for k, *_ in contrasts)
    print(f"wrote {out}  [{counts}; {stats[('cos', contrasts[0][0])]['n_sent']} sentences]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 3: endpoint-gain AUROC depth profiles.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig3_gain_auroc.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha)


if __name__ == "__main__":
    main()
