#!/usr/bin/env python3
"""[RETIRED as the paper's Fig 2 2026-07-10 -- replaced by fig2_dprime.py; kept runnable.]

AUROC depth profiles — how discriminable is each instruction from
no_instruction, per layer?

Two stacked subplots (top: cosine readout; bottom: relative norm). x = the 20
recorded analysis layers; y = AUROC for separating condition trials from
no_instruction trials by the readout:
    red  : think_about       vs no_instruction   (Engagement)
    blue : dont_think_about  vs no_instruction   (Suppression)
AUROC = P(condition value > baseline value) + 0.5·P(tie); 0.5 = chance,
1.0 = perfectly separable, <0.5 = condition BELOW baseline. Rank-based, so
scale-free and cross-model comparable (cf. MEASURES.md §6).

Samples: one value per (sentence, concept) unit per side = the token-mean
readout of that trial; the baseline side is the same sentence's no_instruction
trial (projected onto the unit's concept vector for cosine; its relnorm for the
norm channel — concept-agnostic, so duplicated across a sentence's concepts).

Statistics — clustered at the SENTENCE level, because a sentence's single
no_instruction trial is shared by its ~10 concept units (units are not
independent):
  * band  = 95% cluster bootstrap CI (resample the 50 sentences with
            replacement, keep all their units; B=2000);
  * ringed marker = AUROC ≠ 0.5 by a within-sentence swap permutation (each
            sentence's condition/baseline value sets swapped with p=1/2;
            B=5000, two-sided), BH-FDR across the 20 layers per curve.

CPU-only, no model load; reads results.json + no_instruction_cache.pkl +
the concept-vector cache.
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

from controllability_heatmap import classify, bh_fdr, load_vectors, rankdata  # noqa: E402
from fig2_engage_suppress import _load_json, _trace, _relnorm, POS, NEG       # noqa: E402

CONDS = [(POS, "Think about vs No instruction", "#c0392b"),
         (NEG, "Don't think about vs No instruction", "#2471a3")]
# cos -> PAIRED AUROC (per-unit baseline cancels the concept offsets);
# relnorm -> POOLED AUROC (concept-agnostic readout; pooled separability).
# Definitions live in results/paper/Fig2.md (kept off the axes).
ROWS = [("cos", "Cosine similarity", "Paired AUROC"),
        ("relnorm", "Relative norm", "Pooled AUROC")]
N_LAYERS_TOTAL = 62                     # gemma3-27b; x labeled as depth %


def _auroc(x, y):
    """AUROC = P(x > y) + 0.5 P(x == y), via the rank-sum identity (ties shared)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.nan
    r = rankdata(np.concatenate([x, y]))
    return (r[:n].sum() - n * (n + 1) / 2) / (n * m)


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

    # per sentence s: pairs[(metric, cond)][s] = (X, B) arrays over its concepts,
    # each entry a per-layer vector (token-mean readout).
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
    stats = {}
    for key, by_s in pairs.items():
        metric = key[0]
        sents = sorted(by_s)
        S = len(sents)
        auc = np.full(len(layers), np.nan)
        lo = np.full(len(layers), np.nan); hi = np.full(len(layers), np.nan)
        p = np.full(len(layers), np.nan)

        if metric == "cos":
            # ---- PAIRED AUROC: P(x_u > b_u) over units (win = 1 / 0.5 / 0) ----
            for li in range(len(layers)):
                wins_s, ks = [], []
                for s in sents:
                    xv, bv = by_s[s][0][:, li], by_s[s][1][:, li]
                    ok = ~(np.isnan(xv) | np.isnan(bv))
                    w = np.where(xv[ok] > bv[ok], 1.0,
                                 np.where(xv[ok] == bv[ok], 0.5, 0.0))
                    wins_s.append(w); ks.append(len(w))
                k = np.array(ks, float)
                if k.sum() < 10:
                    continue
                Wsum = np.array([w.sum() for w in wins_s])
                auc[li] = Wsum.sum() / k.sum()
                # cluster bootstrap over sentences
                Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
                nn = Mm @ k
                boots = np.divide(Mm @ Wsum, nn, out=np.full(n_boot, np.nan), where=nn > 0)
                lo[li], hi[li] = np.nanpercentile(boots, [2.5, 97.5])
                # per-unit swap permutation: flipping a unit maps win w -> 1 - w
                w_all = np.concatenate(wins_s)
                F = rng.integers(0, 2, size=(n_perm, len(w_all)))
                null = (np.where(F == 1, 1.0 - w_all, w_all)).mean(1)
                obs_dev = abs(auc[li] - 0.5)
                p[li] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)
            stats[key] = dict(auc=auc, lo=lo, hi=hi, q=bh_fdr(p),
                              n_units=int(sum(len(v[0]) for v in by_s.values())), n_sent=S)
            continue

        # ---- POOLED AUROC (relnorm): full cross-comparison, cluster stats ----
        for li in range(len(layers)):
            # per-sentence value blocks, unit-paired NaN drop (keeps sides equal)
            xs, bs = [], []
            for s in sents:
                xv, bv = by_s[s][0][:, li], by_s[s][1][:, li]
                ok = ~(np.isnan(xv) | np.isnan(bv))
                xs.append(xv[ok]); bs.append(bv[ok])
            k = np.array([len(v) for v in xs], float)      # units per sentence
            if k.sum() < 10:
                continue

            def wins(A_blocks, B_blocks):
                """W[s,t] = #(a > b) + 0.5·#(a == b) between blocks (SxS)."""
                W = np.empty((S, S))
                for i, a in enumerate(A_blocks):
                    for j, b in enumerate(B_blocks):
                        if len(a) == 0 or len(b) == 0:
                            W[i, j] = 0.0
                        else:
                            d = a[:, None] - b[None, :]
                            W[i, j] = (d > 0).sum() + 0.5 * (d == 0).sum()
                return W

            C_xb = wins(xs, bs)                     # cond beats base
            C_xx = wins(xs, xs)                     # cond beats cond (for swaps)
            C_bb = wins(bs, bs)
            C_bx = wins(bs, xs)

            n_tot = k.sum()
            auc[li] = C_xb.sum() / (n_tot * n_tot)

            # ---- cluster bootstrap: multiplicity vector m over sentences ----
            Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
            U = ((Mm @ C_xb) * Mm).sum(1)                   # m^T C m per resample
            nn = Mm @ k
            boots = np.divide(U, nn * nn, out=np.full(n_boot, np.nan), where=nn > 0)
            lo[li], hi[li] = np.nanpercentile(boots, [2.5, 97.5])

            # ---- within-sentence swap permutation (vectorized over flips) ----
            F = rng.integers(0, 2, size=(n_perm, S)).astype(float)   # 1 = swapped
            G = 1.0 - F
            U0 = ((G @ C_xb) * G).sum(1) + ((G @ C_xx) * F).sum(1) \
               + ((F @ C_bb) * G).sum(1) + ((F @ C_bx) * F).sum(1)
            null = U0 / (n_tot * n_tot)
            obs_dev = abs(auc[li] - 0.5)
            p[li] = (1 + int((np.abs(null - 0.5) >= obs_dev - 1e-15).sum())) / (n_perm + 1)
        stats[key] = dict(auc=auc, lo=lo, hi=hi, q=bh_fdr(p),
                          n_units=int(sum(len(v[0]) for v in by_s.values())), n_sent=S)
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
            ax.plot(x, st["auc"], "-o", color=color, lw=2.0, ms=4.5,
                    label=label, zorder=3)
            ax.fill_between(x, st["lo"], st["hi"], color=color, alpha=0.15, zorder=2)
            sig = st["q"] < alpha
            if np.any(sig):
                ax.plot(x[sig], st["auc"][sig], "o", color=color, ms=8,
                        mfc=color, mec="black", mew=0.9, zorder=4)
        ax.axhline(0.5, color="#888", lw=0.9, ls="--", zorder=1)
        # panel title inside, upper-left (definitions live in Fig2a.md)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(depth_pct)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # single legend for both panels: upper-right of the BOTTOM subplot
    axes[1].legend(loc="upper right", frameon=False, fontsize=10, labelspacing=0.4)
    axes[1].set_xlabel("Depth (%)", fontsize=11)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st0 = stats[("cos", POS)]
    print(f"wrote {out}  [{st0['n_units']} units, {st0['n_sent']} sentences]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 2: AUROC depth profiles (condition vs no_instruction).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig2_depth_profile.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha, vector_cache=args.vector_cache,
           method=args.method, model=args.model)


if __name__ == "__main__":
    main()
