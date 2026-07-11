#!/usr/bin/env python3
"""Appendix Fig A1: AUROC by fractional sentence position — the positional
companion of
Fig 2 (fig2_depth_profile.py).

Same two-panel design and statistics as Fig 2, but x = the token's fractional
position in its sentence (f = i/(n-1), binned into --n-bins equal bins over
[0,1]) instead of depth; the readout layer is FIXED per channel at its depth-
profile peak (cos: layer 61 ~ 98% depth; relnorm: layer 43 ~ 69%; both
overridable).

    top    : Cosine similarity  -> PAIRED AUROC  (red think_about, blue
             dont_think_about, each vs no_instruction; pairing cancels the
             per-concept geometric offsets)
    bottom : Relative norm      -> POOLED AUROC  (concept-agnostic readout)

Per (sentence, concept) unit and bin: condition value = mean readout over the
unit's tokens in that bin; baseline value = the same bin of the sentence's
no_instruction trial (projected onto the unit's concept vector for cosine; its
relnorm for the norm channel). Bands = 95% cluster bootstrap over the 50
sentences (B=2000); ringed markers = AUROC != 0.5 (paired: per-unit swap;
pooled: within-sentence swap; B=5000, two-sided), BH-FDR across bins per curve.
Full definitions in results/paper/FigA1.md. CPU-only, no model load.
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
from fig4_fraction_engage_suppress import _bins_for, _bin_means           # noqa: E402

CONDS = [(POS, "Think about vs No instruction", "#c0392b"),
         (NEG, "Don't think about vs No instruction", "#2471a3")]


def build(run_dir, *, cos_layer=61, relnorm_layer=43, n_bins=10,
          vector_cache="results/vector_cache", method="baseline",
          model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [cos_layer, relnorm_layer], method)
    conds = [c for c, _, _ in CONDS]

    # pairs[(metric, cond)][sentence] = (X, B): unit x bin matrices
    pairs = {(m, c): {} for m in ("cos", "relnorm") for c in conds}
    for s, sub in by_sent.items():
        ent = cache.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if ent is None or toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        bins = _bins_for(n_tok, n_bins)

        A = np.asarray(ent["activations"][cos_layer], np.float32)[:n_tok]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        base_cos_tok = An @ vecs[cos_layer][2].T if cos_layer in vecs else None
        brl = _relnorm(np.asarray(ent["norms"][relnorm_layer], np.float32)[:n_tok], classes)
        base_rel_bin = _bin_means(brl, bins, n_bins) if brl is not None else None
        concepts_L = vecs[cos_layer][0] if cos_layer in vecs else []

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
                tr = _trace(r_, "cosine_sim", cos_layer)
                if tr is not None and base_cos_tok is not None:
                    v = np.asarray(tr, np.float32)[:n_tok]
                    m = min(len(v), n_tok)
                    Xc.append(_bin_means(v[:m], bins[:m], n_bins))
                    Bc.append(_bin_means(base_cos_tok[:m, ci], bins[:m], n_bins))
                nr = _trace(r_, "norms", relnorm_layer)
                if nr is not None and base_rel_bin is not None:
                    rl = _relnorm(np.asarray(nr, np.float32)[:n_tok], classes)
                    if rl is not None:
                        Xr.append(_bin_means(rl, bins, n_bins))
                        Br.append(base_rel_bin)
            if Xc:
                pairs[("cos", cond)][s] = (np.vstack(Xc), np.vstack(Bc))
            if Xr:
                pairs[("relnorm", cond)][s] = (np.vstack(Xr), np.vstack(Br))

    rng = np.random.default_rng(seed)
    stats = {}
    for key, by_s in pairs.items():
        metric = key[0]
        sents = sorted(by_s)
        S = len(sents)
        auc = np.full(n_bins, np.nan)
        lo = np.full(n_bins, np.nan); hi = np.full(n_bins, np.nan)
        p = np.full(n_bins, np.nan)

        if metric == "cos":
            # ---- PAIRED AUROC per bin ----
            for bi in range(n_bins):
                wins_s, ks = [], []
                for s in sents:
                    xv, bv = by_s[s][0][:, bi], by_s[s][1][:, bi]
                    ok = ~(np.isnan(xv) | np.isnan(bv))
                    w = np.where(xv[ok] > bv[ok], 1.0,
                                 np.where(xv[ok] == bv[ok], 0.5, 0.0))
                    wins_s.append(w); ks.append(len(w))
                k = np.array(ks, float)
                if k.sum() < 10:
                    continue
                Wsum = np.array([w.sum() for w in wins_s])
                auc[bi] = Wsum.sum() / k.sum()
                Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
                nn = Mm @ k
                boots = np.divide(Mm @ Wsum, nn, out=np.full(n_boot, np.nan), where=nn > 0)
                lo[bi], hi[bi] = np.nanpercentile(boots, [2.5, 97.5])
                w_all = np.concatenate(wins_s)
                F = rng.integers(0, 2, size=(n_perm, len(w_all)))
                null = (np.where(F == 1, 1.0 - w_all, w_all)).mean(1)
                obs_dev = abs(auc[bi] - 0.5)
                p[bi] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)
        else:
            # ---- POOLED AUROC per bin (win-count matrices; cluster stats) ----
            for bi in range(n_bins):
                xs, bs = [], []
                for s in sents:
                    xv, bv = by_s[s][0][:, bi], by_s[s][1][:, bi]
                    ok = ~(np.isnan(xv) | np.isnan(bv))
                    xs.append(xv[ok]); bs.append(bv[ok])
                k = np.array([len(v) for v in xs], float)
                if k.sum() < 10:
                    continue

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
                auc[bi] = C_xb.sum() / (n_tot * n_tot)
                Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
                U = ((Mm @ C_xb) * Mm).sum(1)
                nn = Mm @ k
                boots = np.divide(U, nn * nn, out=np.full(n_boot, np.nan), where=nn > 0)
                lo[bi], hi[bi] = np.nanpercentile(boots, [2.5, 97.5])
                F = rng.integers(0, 2, size=(n_perm, S)).astype(float)
                G = 1.0 - F
                U0 = ((G @ C_xb) * G).sum(1) + ((G @ C_xx) * F).sum(1) \
                   + ((F @ C_bb) * G).sum(1) + ((F @ C_bx) * F).sum(1)
                null = U0 / (n_tot * n_tot)
                obs_dev = abs(auc[bi] - 0.5)
                p[bi] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)

        stats[key] = dict(auc=auc, lo=lo, hi=hi, q=bh_fdr(p),
                          n_units=int(sum(len(v[0]) for v in by_s.values())), n_sent=S)
    return stats


def render(run_dir, *, out, cos_layer=61, relnorm_layer=43, n_bins=10, alpha=0.05, **kw):
    stats = build(run_dir, cos_layer=cos_layer, relnorm_layer=relnorm_layer,
                  n_bins=n_bins, **kw)
    x = (np.arange(n_bins) + 0.5) / n_bins
    rows = [("cos", "Cosine similarity", "Paired AUROC"),
            ("relnorm", "Relative norm", "Pooled AUROC")]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                             layout="constrained")
    for ax, (metric, title, ylab) in zip(axes, rows):
        for cond, label, color in CONDS:
            st = stats[(metric, cond)]
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
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.1))
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # legend in the TOP panel's lower-right corner — its curves sit near
    # ceiling, leaving that region empty (unlike the bottom panel)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 0.06), frameon=False,
                   fontsize=10, labelspacing=0.4)   # lifted clear of the 0.5 line
    axes[1].set_xlabel("Position (fraction of sentence)", fontsize=11)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st0 = stats[("cos", POS)]
    print(f"wrote {out}  [{st0['n_units']} units, {st0['n_sent']} sentences; "
          f"cos@L{cos_layer}, relnorm@L{relnorm_layer}]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Appendix Fig A1: AUROC by fractional sentence position.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cos-layer", type=int, default=61)
    ap.add_argument("--relnorm-layer", type=int, default=43)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="figA1_position_auroc.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, cos_layer=args.cos_layer,
           relnorm_layer=args.relnorm_layer, n_bins=args.n_bins, alpha=args.alpha,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
