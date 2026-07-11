#!/usr/bin/env python3
"""Fig 5: engagement & suppression by POS category, in SDT sensitivity d'.

(Promoted 2026-07-10; replaces the max-normalized version, retired to
results/paper/"Exploratory analysis"/Retired_Fig5a_pos_categories.* — d' is
in battery units, so no max-normalization is needed.)

Same 2x2 design as the retired max-normalized version (fig5_pos_categories.py) — Engagement / Suppression
columns; cosine @L55 (top) and relative norm @L46 (bottom) rows; x = 9 UPOS
categories — but every bar is the SDT sensitivity:

    d'_{c,g} = [ mean_s cond_{c,s,g} − mean_s base_{c,s,g} ] / SD_s( base_{c,s,g} )

per concept c across sentences s, where the per-sentence value is the mean
readout over the sentence's tokens in category g; averaged over the 10 concepts.
Within-concept baseline SD (pooling would be offset-dominated). Both rows are in
the SAME units (baseline-widths) — d' is dimensionless and cross-model
comparable, so cosine and relnorm panels can be read against each other, unlike
Fig 5a's mixed units. CAVEAT: sigma_baseline = across-sentence (content)
variability, not trial noise (deterministic generation).

Statistics: error bars = 95% TWO-WAY cluster bootstrap (B=2000): each replicate
resamples BOTH the 50 sentences and the 10 concepts (shared sentence draw across
concepts), recomputing every concept's d' and the concept-weighted average —
supports "controllable in general", not just for these 10 concepts.
solid/faded = numerator ≠ 0 (sentence-clustered sign-flip on the paired
per-sentence deltas, shared flips across concepts, sigma fixed; B=5000,
two-sided), BH-FDR across the 9 categories per panel.

CPU-only; in the driver as Fig 5.
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
from fig5_pos_categories import CATS, _pos_by_sentence, _token_upos, GREEN  # noqa: E402

COLS = [("Engagement", POS), ("Suppression", NEG)]
ROWS = [("cos", 55, "d′  (cosine, mean over concepts)"),
        ("relnorm", 46, "d′  (relnorm, mean over concepts)")]


def dprime_bars(per_concept, S_global, nG, rng, n_boot, n_perm):
    """per_concept: list of (X, B, idx), each X/B (S_c x nG) cond/baseline
    per-sentence category means, idx = global sentence indices. Returns
    concept-averaged d' per category + TWO-WAY cluster-bootstrap CI (sentences
    AND concepts resampled; shared sentence draws/flips across concepts) + q."""
    nC = len(per_concept)
    W = rng.multinomial(S_global, np.full(S_global, 1.0 / S_global),
                        size=n_boot).astype(float)
    Mc = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=n_boot).astype(float)
    E = rng.integers(0, 2, size=(n_perm, S_global)) * 2.0 - 1.0
    obs = np.full((nC, nG), np.nan)
    boot = np.full((nC, n_boot, nG), np.nan)
    null = np.full((nC, n_perm, nG), np.nan)
    for ci, (X, B, idx) in enumerate(per_concept):
        if X.shape[0] < 3:
            continue
        Wc, Ec = W[:, idx], E[:, idx]
        for g in range(nG):
            x, b = X[:, g], B[:, g]
            ok = ~(np.isnan(x) | np.isnan(b))
            if ok.sum() < 3:
                continue
            x, b = x[ok], b[ok]
            sd = b.std(ddof=1)
            if sd <= 0:
                continue
            obs[ci, g] = (x.mean() - b.mean()) / sd
            w = Wc[:, ok]
            n = w.sum(1)
            good = n > 1
            mB = np.divide(w @ b, n, out=np.full(n_boot, np.nan), where=good)
            mX = np.divide(w @ x, n, out=np.full(n_boot, np.nan), where=good)
            eB2 = np.divide(w @ (b * b), n, out=np.full(n_boot, np.nan), where=good)
            var = (eB2 - mB ** 2) * np.divide(n, n - 1, out=np.ones(n_boot), where=good)
            boot[ci, :, g] = (mX - mB) / np.sqrt(np.clip(var, 1e-24, None))
            d = x - b
            null[ci, :, g] = (Ec[:, ok] @ d) / ok.sum() / sd
    dp = np.nanmean(obs, axis=0)
    wts = Mc.T[:, :, None]                                  # (nC, n_boot, 1)
    okb = ~np.isnan(boot)
    num = np.nansum(np.where(okb, boot, 0.0) * wts, axis=0)
    den = (wts * okb).sum(axis=0)
    bavg = np.divide(num, den, out=np.full((n_boot, nG), np.nan), where=den > 0)
    lo = np.nanpercentile(bavg, 2.5, axis=0)
    hi = np.nanpercentile(bavg, 97.5, axis=0)
    navg = np.nanmean(null, axis=0)
    p = np.full(nG, np.nan)
    for g in range(nG):
        if not np.isnan(dp[g]):
            p[g] = (1 + int((np.abs(navg[:, g]) >= abs(dp[g]) - 1e-15).sum())) / (n_perm + 1)
    return dict(dp=dp, lo=lo, hi=hi, q=bh_fdr(p))


def build(run_dir, *, pos_path="pos_tags.json", vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    pos_words = _pos_by_sentence(pos_path)
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    layers = [L for _, L, _ in ROWS]
    vecs = load_vectors(vector_cache, model, layers, method)
    conds = [cid for _, cid in COLS]
    nG = len(CATS)

    def cat_means(v, cat_of):
        out = np.full(nG, np.nan)
        for g in range(nG):
            sel = v[cat_of == g]
            sel = sel[~np.isnan(sel)]
            if len(sel):
                out[g] = float(sel.mean())
        return out

    # vals[(metric, cond)][concept][sentence] / bases[metric][concept][sentence]
    # = per-category vector of that sentence's token-mean readout
    vals = {(m, cid): defaultdict(dict) for m, _, _ in ROWS for cid in conds}
    bases = {m: defaultdict(dict) for m, _, _ in ROWS}
    cat_tokens = np.zeros(nG, int)

    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if words is None or toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        cat_of = np.array([CATS.index(u) if u in CATS else -1
                           for u in _token_upos(s, toks, words)])
        for u in cat_of:
            if u >= 0:
                cat_tokens[u] += 1

        base_tok = {}
        for metric, L, _ in ROWS:
            if metric == "cos":
                A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                base_tok[metric] = An @ vecs[L][2].T if L in vecs else None
            else:
                base_tok[metric] = _relnorm(np.asarray(ent["norms"][L], np.float32)[:n_tok], classes)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            if r["condition_id"] in conds and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r
                concepts.add(r["concept"])

        for metric, L, _ in ROWS:
            concepts_L = vecs[L][0] if L in vecs else []
            for c in sorted(concepts):
                # baseline per-category vector for this (concept, sentence)
                if metric == "cos":
                    if base_tok["cos"] is None or c not in concepts_L:
                        continue
                    bvec = cat_means(np.asarray(base_tok["cos"][:, concepts_L.index(c)], float), cat_of)
                else:
                    if base_tok["relnorm"] is None:
                        continue
                    bvec = cat_means(np.asarray(base_tok["relnorm"], float), cat_of)
                bases[metric][c][s] = bvec
                for cid in conds:
                    r_ = byc.get(cid, {}).get(c)
                    if r_ is None:
                        continue
                    if metric == "cos":
                        tr = _trace(r_, "cosine_sim", L)
                        v = np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
                    else:
                        tr = _trace(r_, "norms", L)
                        v = _relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None
                    if v is not None:
                        vals[(metric, cid)][c][s] = cat_means(np.asarray(v, float), cat_of)

    rng = np.random.default_rng(seed)
    all_sents = sorted({s for m in bases for c in bases[m] for s in bases[m][c]})
    s_idx = {s: i for i, s in enumerate(all_sents)}
    stats = {}
    for (metric, cid), by_c in vals.items():
        per_concept = []
        for c in sorted(by_c):
            ss = sorted(set(by_c[c]) & set(bases[metric][c]))
            if len(ss) < 3:
                continue
            per_concept.append((np.vstack([by_c[c][s] for s in ss]),
                                np.vstack([bases[metric][c][s] for s in ss]),
                                np.array([s_idx[s] for s in ss])))
        stats[(metric, cid)] = dprime_bars(per_concept, len(all_sents), nG,
                                           rng, n_boot, n_perm)
    return stats, cat_tokens


def render(run_dir, *, out, alpha=0.05, **kw):
    stats, cat_tokens = build(run_dir, **kw)
    xlabels = [f"{c}\n(n={cat_tokens[i]})" for i, c in enumerate(CATS)]
    x = np.arange(len(CATS))

    fig = plt.figure(figsize=(8.4, 8.6), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)
    for ri, (metric, L, ylab) in enumerate(ROWS):
        axes = subfigs[ri].subplots(1, len(COLS))
        for ci, (col, cid) in enumerate(COLS):
            ax = axes[ci]
            st = stats[(metric, cid)]
            dp, lo, hi = st["dp"], st["lo"], st["hi"]
            sig = st["q"] < alpha
            for j in range(len(CATS)):
                ax.bar(x[j], dp[j], color=GREEN, edgecolor="black", linewidth=0.6,
                       alpha=0.95 if sig[j] else 0.3, zorder=2)
            yerr = np.vstack([np.clip(dp - lo, 0, None), np.clip(hi - dp, 0, None)])
            ax.errorbar(x, dp, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5, zorder=3)
            ax.axhline(0, color="#555", linewidth=0.8, zorder=1)
            ax.text(0.97, 0.96, col, transform=ax.transAxes, ha="right", va="top",
                    fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            ax.margins(x=0.02)
        # shared y span within each row (both panels are in the same d' units)
        lims = [ax.get_ylim() for ax in axes]
        lo_r, hi_r = min(l for l, _ in lims), max(h for _, h in lims)
        for ax in axes:
            ax.set_ylim(lo_r, hi_r)
    # no on-figure fine print: definitions live in Fig5.md
        fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 5: d' by POS category.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pos-path", default="pos_tags.json")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--out", default="fig5_dprime.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, pos_path=args.pos_path,
           vector_cache=args.vector_cache, method=args.method, model=args.model)


if __name__ == "__main__":
    main()
