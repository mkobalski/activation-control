#!/usr/bin/env python3
"""Fig 9: token-TYPE targeting (loc_punctuation, loc_adjectives) by POS category.

Do "think about {concept} only on punctuation / only on adjectives" actually
concentrate concept engagement on the targeted token type? 2x2 grid:

              Only on punctuation        Only on adjectives
  Δcos @L55   bars/cat                    bars/cat
  Δrelnorm@L46 bars/cat                   bars/cat

Each panel: for every UPOS category, two grouped bars, both as Δ-from-
no_instruction (per (sentence, concept) unit):
    * the location condition  (loc_punctuation / loc_adjectives)
    * think_about             (engage everywhere, the reference)
The targeted category (PUNCT / ADJ) is highlighted. The question is whether the
location bar exceeds the think bar ON the targeted category.

Measures (per token, at the row's layer):
    Δcos     = cos(v_c, r_cond) − cos(v_c, r_no_instruction)        @L55
    Δrelnorm = ‖r_cond‖/content-mean − ‖r_no_instr‖/content-mean    @L46
both differenced within the (sentence, concept) unit (cancels the concept offset).

Statistics: unit = (sentence, concept), collapsed to one Δ per category (mean over
its tokens in the category). Bars = mean over units; error bars = 95% bootstrap CI
(B=2000, resampling units). Significance star = the location condition differs from
think_about on that category, paired per unit (two-sided sign-flip, B=5000), with
BH-FDR across the 9 categories. POS tags from pos_tags.json (char-span aligned).

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
from fig7_pos_categories import CATS, _pos_by_sentence, _token_upos      # noqa: E402

COS_L, RN_L = 55, 46
BASE, THINK = "no_instruction", "think_about"
# column: (condition, targeted UPOS, title)
COLUMNS = [("loc_punctuation", "PUNCT", "Only on punctuation"),
           ("loc_adjectives", "ADJ", "Only on adjectives")]
ROWS = [("cos", COS_L, "Δ Cosine similarity (layer 55)", "Δcos  (cond − no instr)"),
        ("relnorm", RN_L, "Δ Relative norm (layer 46)", "Δrelnorm  (cond − no instr)")]


def build(run_dir, *, pos_path="pos_tags.json", vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    pos_words = _pos_by_sentence(pos_path)
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [RN_L, COS_L], method)
    conds = [c for c, _, _ in COLUMNS] + [THINK]

    # per (metric, cond) -> {(sentence, concept): vector over CATS}; keyed by unit
    # so the loc-vs-think paired test aligns on the same (sentence, concept).
    panel_units = {(m, c): {} for m, _, _, _ in ROWS for c in conds}

    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if words is None or toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        cat_of = np.array([CATS.index(u) if u in CATS else -1 for u in _token_upos(s, toks, words)])

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in conds and c:
                byc[cond][c] = r
                concepts.add(c)

        acos = np.asarray(ent["activations"][COS_L], np.float32)[:n_tok]
        acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
        base_cos_all = acos @ vecs[COS_L][2].T if COS_L in vecs else None
        base_rn = _relnorm(np.asarray(ent["norms"][RN_L], np.float32)[:n_tok], classes)
        concepts_cosL = vecs[COS_L][0] if COS_L in vecs else []

        def delta(row, metric, base):
            L = COS_L if metric == "cos" else RN_L
            if metric == "cos":
                tr = _trace(row, "cosine_sim", L)
                v = np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
            else:
                tr = _trace(row, "norms", L)
                v = _relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None
            return (v - base) if (v is not None and base is not None) else None

        for c in sorted(concepts):
            base_cos_c = (base_cos_all[:, concepts_cosL.index(c)]
                          if (base_cos_all is not None and c in concepts_cosL) else None)
            for cond in conds:
                row = byc.get(cond, {}).get(c)
                if row is None:
                    continue
                for metric, base in (("cos", base_cos_c), ("relnorm", base_rn)):
                    d = delta(row, metric, base)
                    if d is None:
                        continue
                    vec = np.full(len(CATS), np.nan)
                    for gi in range(len(CATS)):
                        sel = d[(cat_of == gi)]; sel = sel[~np.isnan(sel)]
                        if len(sel):
                            vec[gi] = float(sel.mean())
                    panel_units[(metric, cond)][(s, c)] = vec

    rng = np.random.default_rng(seed)

    def agg(metric, cond):
        vals = list(panel_units[(metric, cond)].values())
        U = np.vstack(vals) if vals else np.full((0, len(CATS)), np.nan)
        mean = np.nanmean(U, axis=0) if U.shape[0] else np.full(len(CATS), np.nan)
        lo = np.full(len(CATS), np.nan); hi = np.full(len(CATS), np.nan)
        if U.shape[0] > 1:
            idx = rng.integers(0, U.shape[0], size=(n_boot, U.shape[0]))
            boot = np.nanmean(U[idx], axis=1)
            lo, hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
        return dict(mean=mean, lo=lo, hi=hi, U=U, n=np.sum(~np.isnan(U), axis=0))

    def paired_q_vs_think(metric, cond):
        """loc vs think, paired per (sentence, concept) unit per category (sign-flip), FDR across cats."""
        da, dt = panel_units[(metric, cond)], panel_units[(metric, THINK)]
        keys = sorted(set(da) & set(dt))                 # units having both conditions
        p = np.full(len(CATS), np.nan)
        if keys:
            D = np.vstack([da[k] - dt[k] for k in keys])
            for gi in range(len(CATS)):
                dv = D[:, gi]; dv = dv[~np.isnan(dv)]
                if len(dv) >= 3:
                    obs = float(dv.mean())
                    signs = rng.integers(0, 2, size=(n_perm, len(dv))) * 2.0 - 1.0
                    null = (signs * dv).mean(1)
                    p[gi] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)
        return bh_fdr(p)

    stats = {}
    for metric, _, _, _ in ROWS:
        for cond in conds:
            stats[(metric, cond)] = agg(metric, cond)
        for cond, _, _ in COLUMNS:
            stats[("q_vs_think", metric, cond)] = paired_q_vs_think(metric, cond)
    cat_tokens = np.zeros(len(CATS), int)
    for s, sub in by_sent.items():
        words = pos_words.get(s)
        tr = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if words and tr:
            for u in _token_upos(s, tr[1:], words):
                if u in CATS:
                    cat_tokens[CATS.index(u)] += 1
    return stats, cat_tokens


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, alpha=0.05, **kw):
    stats, cat_tokens = build(run_dir, **kw)
    x = np.arange(len(CATS))
    xlabels = [f"{c}\n(n={cat_tokens[i]})" for i, c in enumerate(CATS)]
    w = 0.4
    loc_c, think_c = "#c0392b", "#95a5a6"

    fig = plt.figure(figsize=(7.6 * len(COLUMNS), 8.8), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)

    for ri, (metric, L, row_title, ylab) in enumerate(ROWS):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=15, fontweight="bold")
        axes = sf.subplots(1, len(COLUMNS), sharey=True)
        for ci, (cond, target, coltitle) in enumerate(COLUMNS):
            ax = axes[ci]
            ti = CATS.index(target)
            ax.axvspan(ti - 0.5, ti + 0.5, color="#f7dc6f", alpha=0.35, zorder=0)  # target highlight
            for series, cser, off, lab in ((cond, loc_c, -w / 2, "Location-targeted"),
                                           (THINK, think_c, w / 2, "Think (everywhere)")):
                st = stats[(metric, series)]
                yerr = np.vstack([np.clip(st["mean"] - st["lo"], 0, None),
                                  np.clip(st["hi"] - st["mean"], 0, None)])
                ax.bar(x + off, st["mean"], width=w, color=cser, edgecolor="black",
                       linewidth=0.5, label=lab, zorder=2)
                ax.errorbar(x + off, st["mean"], yerr=yerr, fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=2, zorder=3)
            # stars where loc differs from think (paired, FDR across categories)
            q = stats[("q_vs_think", metric, cond)]
            for gi in range(len(CATS)):
                if q[gi] < alpha:
                    top = np.nanmax([stats[(metric, cond)]["hi"][gi], stats[(metric, THINK)]["hi"][gi], 0])
                    ax.text(gi, top, "*", ha="center", va="bottom", fontsize=13, color="black")
            ax.axhline(0, color="#555", lw=0.8, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
            ax.get_xticklabels()[ti].set_fontweight("bold")
            if ri == 0:
                ax.set_title(coltitle, fontsize=13)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
                ax.legend(fontsize=8, framealpha=0.9, loc="best")
            ax.margins(x=0.02)

    fig.text(0.5, 0.005, "Δ vs no_instruction, avg over (sentence × concept) units; error bars = 95% "
             "bootstrap CI (B=2000).  ★ = location condition differs from Think-everywhere on that "
             "category (paired sign-flip, BH-FDR q<0.05 across categories, B=5000).  highlighted "
             "category = the instruction's target.  Δcos @L55, Δrelnorm @L46.",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 9: token-type targeting (punctuation / adjectives) by POS.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pos-path", default="pos_tags.json")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="fig9_type_targeting.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, pos_path=args.pos_path, vector_cache=args.vector_cache,
           method=args.method, model=args.model, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
