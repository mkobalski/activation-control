#!/usr/bin/env python3
"""Fig 7: the four controllability metrics as a function of TOKEN CATEGORY (POS).

2x4 grid of bar charts:
              Engagement     Suppression     Rank            Gain
  cos @L55    bars/cat        bars/cat        bars/cat        bars/cat
  relnorm@L46 bars/cat        bars/cat        bars/cat        bars/cat

x of each panel = word-level UPOS category (from pos_tags.json, spaCy
en_core_web_sm), restricted to the well-populated tags. y = the metric value;
error bars = 95% bootstrap CI.

Per-token metrics (at the channel's peak layer L: cos->55, relnorm->46):
  Engagement  = readout(think_about)      − readout(no_instruction)
  Suppression = readout(dont_think_about) − readout(no_instruction)
  Rank        = signed Spearman(intensity level 1..4, readout)          in [-1,1]
  Gain        = readout(think_intensity_4_of_4) − readout(think_intensity_1_of_4)
where readout = cos(v_concept, residual) (cos row) or ||r||/content-mean (relnorm row).
Rank/Gain use the canonical 4-level intensity ramp (as the 05-05 runs).

Statistics (matches the cross-sentence convention of Figs 3-6): the sampling unit
is the (sentence, concept) pair; a unit is collapsed to ONE value per category =
the mean of its per-token metric over that sentence's tokens in the category (so
correlated within-sentence tokens are not double-counted). A category's bar is the
mean over units; the 95% CI is a percentile bootstrap resampling the units
(B=2000). Bars whose CI excludes 0 are drawn solid (significant), else faded.

Token->category alignment is by CHARACTER SPAN: each model token inherits the
UPOS of the spaCy word covering its characters, so word-piece splits inherit their
word's tag (all 594 tokens of the 50 sentences tag cleanly). MODEL-INDEPENDENT.

CPU-only, no model load.
"""

import argparse
import json
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
from fig2_engage_suppress import _load_json, _trace, _relnorm, POS, NEG, BASE  # noqa: E402
from fig5_rank_intensity import _signed_spearman                         # noqa: E402

RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
CATS = ["NOUN", "VERB", "DET", "PUNCT", "ADP", "PRON", "ADJ", "ADV", "CCONJ"]
# (metric_key, column title); panels are (row_metric x these)
COLS = ["Engagement", "Suppression", "Rank", "Gain"]
# rows: (readout metric, layer, row header, y-label)
ROWS = [("cos", 55, "Cosine similarity (layer 55)", "Δcos / ρ"),
        ("relnorm", 46, "Relative norm (layer 46)", "Δrelnorm / ρ")]


def _pos_by_sentence(pos_path):
    entries = json.load(open(pos_path))["entries"]
    return {e["text"]: e["words"] for e in entries}


def _token_upos(sentence, toks, words):
    """UPOS per model token via char-span overlap with the spaCy words. None if unmatched."""
    out = []
    cur = 0
    for t in toks:
        ts = t.strip()
        if not ts:
            out.append(None); continue
        j = sentence.find(ts, cur)
        if j < 0:
            j = cur
        a, b = j, j + len(ts)
        cur = b
        tag = None
        for w in words:                      # word whose char span overlaps the token
            if a < w["end"] and b > w["start"]:
                tag = w["upos"]; break
        out.append(tag)
    return out


def build(run_dir, *, pos_path="pos_tags.json", vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    pos_words = _pos_by_sentence(pos_path)
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    layers = [46, 55]
    vecs = load_vectors(vector_cache, model, layers, method)
    wanted = {POS, NEG, BASE, *RAMP}

    # per (metric,column) -> list of unit vectors over CATS (NaN where absent)
    panel_units = {(m, c): [] for m, _, _, _ in ROWS for c in COLS}

    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if words is None or toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        upos = _token_upos(s, toks, words)
        cat_of = np.array([CATS.index(u) if u in CATS else -1 for u in upos])

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        # per-metric baselines at each layer
        base_cos = {}                        # {L: (n_tok, n_concepts_L)}
        base_rn = {}                         # {L: (n_tok,)}
        ent = cache.get(s)
        for L in layers:
            if ent and L in vecs:
                a = np.asarray(ent["activations"][int(L)], np.float32)[:n_tok]
                an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
                base_cos[L] = an @ vecs[L][2].T
            if ent:
                base_rn[L] = _relnorm(np.asarray(ent["norms"][int(L)], np.float32)[:n_tok], classes)

        def readout(cond, c, metric, L):
            row = byc.get(cond, {}).get(c)
            if row is None:
                return None
            if metric == "cos":
                tr = _trace(row, "cosine_sim", L)
                return np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
            tr = _trace(row, "norms", L)
            return _relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None

        for c in sorted(concepts):
            for metric, L, _, _ in ROWS:
                # baseline readout for engagement/suppression
                if metric == "cos":
                    concepts_L = vecs[L][0] if L in vecs else []
                    b = base_cos[L][:, concepts_L.index(c)] if (L in base_cos and c in concepts_L) else None
                else:
                    b = base_rn.get(L)
                think = readout(POS, c, metric, L)
                dont = readout(NEG, c, metric, L)
                ramp = [readout(rc, c, metric, L) for rc in RAMP]

                metvec = {}
                metvec["Engagement"] = (think - b) if (think is not None and b is not None) else None
                metvec["Suppression"] = (dont - b) if (dont is not None and b is not None) else None
                # gain: last - first of the ramp
                metvec["Gain"] = (ramp[-1] - ramp[0]) if (ramp[-1] is not None and ramp[0] is not None) else None
                # rank: per-token signed spearman over present ramp levels
                present = [(lev, v) for lev, v in enumerate(ramp) if v is not None]
                if len(present) >= 3:
                    rk = np.full(n_tok, np.nan)
                    for ti in range(n_tok):
                        lv = [lev for lev, v in present if not np.isnan(v[ti])]
                        vv = [v[ti] for lev, v in present if not np.isnan(v[ti])]
                        if len(lv) >= 3:
                            rk[ti] = _signed_spearman(lv, vv)
                    metvec["Rank"] = rk
                else:
                    metvec["Rank"] = None

                # collapse this unit to one value per category
                for col in COLS:
                    v = metvec[col]
                    row_vec = np.full(len(CATS), np.nan)
                    if v is not None:
                        for gi in range(len(CATS)):
                            sel = v[(cat_of == gi)]
                            sel = sel[~np.isnan(sel)]
                            if len(sel):
                                row_vec[gi] = float(sel.mean())
                    panel_units[(metric, col)].append(row_vec)

    # aggregate per panel per category:
    #   mean, 95% bootstrap CI (units), sign-flip p, then BH-FDR across the 9
    #   categories within the panel (matches the heatmaps' per-panel FDR).
    rng = np.random.default_rng(seed)
    stats = {}
    for key, unit_list in panel_units.items():
        U = np.vstack(unit_list) if unit_list else np.full((0, len(CATS)), np.nan)
        mean = np.nanmean(U, axis=0)
        n = np.sum(~np.isnan(U), axis=0)
        lo = np.full(len(CATS), np.nan); hi = np.full(len(CATS), np.nan)
        p = np.full(len(CATS), np.nan)
        if U.shape[0] > 1:
            idx = rng.integers(0, U.shape[0], size=(n_boot, U.shape[0]))
            boot = np.nanmean(U[idx], axis=1)                 # (n_boot, n_cats)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
        for gi in range(len(CATS)):                           # two-sided sign-flip null
            dv = U[:, gi]; dv = dv[~np.isnan(dv)]
            if len(dv) >= 3:
                obs = float(dv.mean())
                signs = rng.integers(0, 2, size=(n_perm, len(dv))) * 2.0 - 1.0
                null = (signs * dv).mean(1)
                p[gi] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)
        q = bh_fdr(p)                                         # BH-FDR over the categories
        stats[key] = dict(mean=mean, lo=lo, hi=hi, n=n, p=p, q=q)
    # category token counts (for x labels) from the last sentence loop is not global;
    # recompute simple totals across sentences
    cat_tokens = np.zeros(len(CATS), int)
    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if words is None or toks_row is None:
            continue
        for u in _token_upos(s, toks_row[1:], words):
            if u in CATS:
                cat_tokens[CATS.index(u)] += 1
    return stats, cat_tokens


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, alpha=0.05, **kw):
    stats, cat_tokens = build(run_dir, **kw)
    xlabels = [f"{c}\n(n={cat_tokens[i]})" for i, c in enumerate(CATS)]
    x = np.arange(len(CATS))

    fig = plt.figure(figsize=(4.2 * len(COLS), 8.6), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)

    for ri, (metric, L, row_title, ylab) in enumerate(ROWS):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=15, fontweight="bold")
        axes = sf.subplots(1, len(COLS))
        for ci, col in enumerate(COLS):
            ax = axes[ci]
            st = stats[(metric, col)]
            mean, lo, hi = st["mean"], st["lo"], st["hi"]
            sig = st["q"] < alpha                             # BH-FDR across categories
            colors = ["#c0392b" if m >= 0 else "#2471a3" for m in mean]
            yerr = np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])
            for j in range(len(CATS)):
                ax.bar(x[j], mean[j], color=colors[j], edgecolor="black", linewidth=0.6,
                       alpha=0.95 if sig[j] else 0.3, zorder=2)
            ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5, zorder=3)
            ax.axhline(0, color="#555", linewidth=0.8, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
            ax.set_title(col, fontsize=12)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=10)
            # tight y-limits: pin baseline at 0 for all-positive panels, pad only
            # the populated side(s) so there is no dead space below 0 (e.g. cos Rank)
            vals = np.concatenate([a[~np.isnan(a)] for a in (mean, lo, hi)]) \
                if np.any(~np.isnan(mean)) else np.array([0.0])
            lower = min(0.0, float(vals.min())); upper = max(0.0, float(vals.max()))
            span = (upper - lower) or 1.0
            ax.set_ylim(lower - (0.06 * span if lower < 0 else 0.0),
                        upper + (0.06 * span if upper > 0 else 0.0))
            ax.margins(x=0.02)

    fig.text(0.5, 0.005, "bars = mean over (sentence × concept) units; error bars = 95% bootstrap CI "
             "(B=2000, resampling units); solid = significant at BH-FDR q<0.05 (per-panel sign-flip null, "
             "B=5000), faded = not.  cos row @ layer 55, relnorm row @ layer 46.  Rank/Gain over the 4-level "
             "intensity ramp.",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 7: metrics by POS token category (bar plots + 95% CI).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pos-path", default="pos_tags.json")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="fig7_pos_categories.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, pos_path=args.pos_path, vector_cache=args.vector_cache,
           method=args.method, model=args.model, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
