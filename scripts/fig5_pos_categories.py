#!/usr/bin/env python3
"""[RETIRED as the paper's Fig 5 2026-07-10 -- replaced by fig5_dprime.py; kept
runnable (renders the retired Fig5a/5b pair) and still the POS-alignment
library for fig9/fig11.]

Controllability metrics as a function of TOKEN CATEGORY (POS).

Fig 5a — 2x2 grid of bar charts:
              Engagement     Suppression
  cos @L55    bars/cat        bars/cat
  relnorm@L46 bars/cat        bars/cat

Fig 5b — 1x2: the Rank (signed Spearman) panels, cos @L55 and relnorm @L46.
All bars are drawn in a single green regardless of sign.

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
# (metric_key, column title); build() computes all four, render splits them:
# Fig 5a = COLS_A (2x3), Fig 5b = the two Rank panels
COLS = ["Engagement", "Suppression", "Rank", "Gain"]
COLS_A = ["Engagement", "Suppression"]
# rows: (readout metric, layer, row header, y-label)
ROWS = [("cos", 55, "Cosine similarity (layer 55)",
         "Change in cosine similarity (normalized to max value)"),
        ("relnorm", 46, "Relative norm (layer 46)", "Δrelnorm")]
GREEN = "#1e8449"                                   # all bars, regardless of sign


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

    # per (metric,column) -> list of (sentence, unit vector over CATS); the
    # sentence key drives the cluster bootstrap / clustered sign-flip below
    # (a sentence's ~10 concept-units share its single no_instruction trial,
    # so units are NOT independent across concepts).
    panel_units = {(m, c): [] for m, _, _, _ in ROWS for c in COLS}
    # for the cosine-row PROFILES (Engagement/Suppression): per-sentence
    # per-token sums over units
    PROFILE_COLS = ("Engagement", "Suppression")
    tok_acc = {col: {} for col in PROFILE_COLS}     # col -> {s: [tok_sums, n_units]}
    sent_lab = {}                                   # s -> cat index per token (-1 = other)

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
        sent_lab[s] = cat_of

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
                    panel_units[(metric, col)].append((s, row_vec))
                    # token-resolution accumulation for the cosine profiles
                    if metric == "cos" and col in tok_acc and v is not None:
                        acc = tok_acc[col].setdefault(s, [np.zeros(n_tok), 0])
                        acc[0] += np.nan_to_num(np.asarray(v, float)[:n_tok])
                        acc[1] += 1

    # aggregate per panel per category — CLUSTERED AT THE SENTENCE LEVEL:
    #   mean over units; 95% CI = cluster bootstrap (resample the 50 sentences
    #   with replacement, keeping each sentence's unit block; B=n_boot);
    #   significance = sentence-clustered sign-flip (all of a sentence's unit
    #   values flip together; B=n_perm, two-sided), then BH-FDR across the 9
    #   categories within the panel.
    rng = np.random.default_rng(seed)
    stats = {}
    for key, unit_list in panel_units.items():
        sents = sorted({s for s, _ in unit_list})
        s_idx = {s: i for i, s in enumerate(sents)}
        S = len(sents)
        sums = np.zeros((S, len(CATS)))                       # per-sentence nansums
        cnts = np.zeros((S, len(CATS)))
        for s, vec in unit_list:
            i = s_idx[s]
            ok = ~np.isnan(vec)
            sums[i, ok] += vec[ok]
            cnts[i, ok] += 1.0
        tot_cnt = cnts.sum(0)
        n = tot_cnt.astype(int)
        mean = np.divide(sums.sum(0), tot_cnt,
                         out=np.full(len(CATS), np.nan), where=tot_cnt > 0)
        lo = np.full(len(CATS), np.nan); hi = np.full(len(CATS), np.nan)
        p = np.full(len(CATS), np.nan)
        if S > 1:
            # cluster bootstrap over sentences (multiplicity-vector form)
            Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
            den = Mm @ cnts
            boot = np.divide(Mm @ sums, den, out=np.full_like(den, np.nan), where=den > 0)
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
            # sentence-clustered sign-flip null
            signs = rng.integers(0, 2, size=(n_perm, S)) * 2.0 - 1.0
            null = np.divide(signs @ sums, tot_cnt[None, :],
                             out=np.full((n_perm, len(CATS)), np.nan),
                             where=tot_cnt[None, :] > 0)
            for gi in range(len(CATS)):
                if tot_cnt[gi] >= 3 and not np.isnan(mean[gi]):
                    p[gi] = (1 + int((np.abs(null[:, gi]) >= abs(mean[gi]) - 1e-15).sum())) \
                        / (n_perm + 1)
        q = bh_fdr(p)                                         # BH-FDR over the categories
        stats[key] = dict(mean=mean, lo=lo, hi=hi, n=n, p=p, q=q, profile=False)

    # ---- cosine-row PROFILES: unit-sum normalization of the per-token means ----
    # profile_g = mean_g / sum_g' mean_g'  (all-positive panels; [0,1], sums to 1;
    # dimensionless -> cross-model comparable, the 1/sqrt(d) cosine scale cancels).
    # CI: recompute the WHOLE profile inside each sentence-cluster bootstrap
    # replicate (handles the numerator/denominator coupling). Significance: the
    # null is the UNIFORM profile 1/9 -> within-sentence category-label
    # permutation (each sentence's token->label assignment shuffled, label
    # multiset preserved), two-sided add-one p, BH-FDR across categories.
    for col in PROFILE_COLS:
        sents_p = sorted(tok_acc[col])
        Sp = len(sents_p)
        nC = len(CATS)
        sums = np.zeros((Sp, nC)); cnts = np.zeros((Sp, nC))
        for si, s in enumerate(sents_p):
            tok_sums, n_units = tok_acc[col][s]
            lab = sent_lab[s][:len(tok_sums)]
            for gi in range(nC):
                n_gs = int((lab == gi).sum())
                if n_gs:
                    sums[si, gi] = tok_sums[lab == gi].sum() / n_gs
                    cnts[si, gi] = n_units
        tot_cnt = cnts.sum(0)
        mean = np.divide(sums.sum(0), tot_cnt,
                         out=np.full(nC, np.nan), where=tot_cnt > 0)
        profile = mean / np.nanmax(mean)          # MAX-normalized: 1 = strongest cat
        # joint cluster bootstrap (each replicate normalized by ITS OWN max)
        Mm = rng.multinomial(Sp, np.full(Sp, 1.0 / Sp), size=n_boot).astype(float)
        den = Mm @ cnts
        bmean = np.divide(Mm @ sums, den, out=np.full_like(den, np.nan), where=den > 0)
        bprof = bmean / np.nanmax(bmean, axis=1, keepdims=True)
        lo = np.nanpercentile(bprof, 2.5, axis=0)
        hi = np.nanpercentile(bprof, 97.5, axis=0)
        # significance: IS THE CATEGORY'S RESPONSE DIFFERENT FROM ZERO —
        # the sentence-clustered sign-flip vs 0 already computed on the raw
        # per-unit means in the main loop (normalization-invariant: the max is
        # positive, so mean_g != 0 <=> rel_g != 0). Keep those p/q.
        prior = stats[("cos", col)]
        stats[("cos", col)] = dict(mean=profile, lo=lo, hi=hi,
                                   n=tot_cnt.astype(int),
                                   p=prior["p"], q=prior["q"],
                                   profile=True)
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

def _draw_panel(ax, series, xlabels, *, title=None, corner=None, ylab=None,
                alpha=0.05, legend=False):
    """series = list of (stats dict, color, label-or-None); grouped if len > 1.
    title -> centered above the axes; corner -> inside the upper-right corner."""
    x = np.arange(len(CATS))
    w = 0.8 / len(series)
    all_vals = [np.array([0.0])]
    for si, (st, color, label) in enumerate(series):
        off = (si - (len(series) - 1) / 2) * w
        mean, lo, hi = st["mean"], st["lo"], st["hi"]
        sig = st["q"] < alpha                             # BH-FDR across categories
        for j in range(len(CATS)):
            ax.bar(x[j] + off, mean[j], width=w, color=color, edgecolor="black",
                   linewidth=0.6, alpha=0.95 if sig[j] else 0.3, zorder=2,
                   label=label if j == 0 else None)
        yerr = np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])
        ax.errorbar(x + off, mean, yerr=yerr, fmt="none", ecolor="black",
                    elinewidth=0.9, capsize=2.5 if len(series) == 1 else 1.5, zorder=3)
        all_vals += [a[~np.isnan(a)] for a in (mean, lo, hi)]
    ax.axhline(0, color="#555", linewidth=0.8, zorder=1)
    if title:
        ax.set_title(title, fontsize=12)
    if corner:
        ax.text(0.97, 0.96, corner, transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    if ylab:
        ax.set_ylabel(ylab, fontsize=10)
    if legend:
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    # tight y-limits: pin baseline at 0 for all-positive panels, pad only
    # the populated side(s) so there is no dead space on an empty side
    vals = np.concatenate(all_vals)
    lower = min(0.0, float(vals.min())); upper = max(0.0, float(vals.max()))
    span = (upper - lower) or 1.0
    ax.set_ylim(lower - (0.06 * span if lower < 0 else 0.0),
                upper + (0.06 * span if upper > 0 else 0.0))
    ax.margins(x=0.02)


def render(run_dir, *, out_a, out_b, alpha=0.05, **kw):
    stats, cat_tokens = build(run_dir, **kw)
    xlabels = [f"{c}\n(n={cat_tokens[i]})" for i, c in enumerate(CATS)]

    # ---- Fig 5a: Engagement / Suppression (2x2) ----
    # no row titles, no on-figure fine print: layer identities and the
    # normalization/statistics description live in the caption (Fig5a.md)
    fig = plt.figure(figsize=(4.2 * len(COLS_A), 8.6), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)
    for ri, (metric, L, row_title, ylab) in enumerate(ROWS):
        axes = subfigs[ri].subplots(1, len(COLS_A))
        for ci, col in enumerate(COLS_A):
            _draw_panel(axes[ci], [(stats[(metric, col)], GREEN, None)], xlabels,
                        corner=col, ylab=ylab if ci == 0 else None, alpha=alpha)
        if metric == "relnorm":                    # bottom row: shared y span
            lims = [ax.get_ylim() for ax in axes]
            lo, hi = min(l for l, _ in lims), max(h for _, h in lims)
            for ax in axes:
                ax.set_ylim(lo, hi)
    fig.savefig(out_a, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_a}")

    # ---- Fig 5b: the two Rank panels (1x2) ----
    fig = plt.figure(figsize=(9.0, 5.0), layout="constrained")
    axes = fig.subplots(1, 2)
    for ci, (metric, L, row_title, _) in enumerate(ROWS):
        _draw_panel(axes[ci], [(stats[(metric, "Rank")], GREEN, None)], xlabels,
                    title=row_title,
                    ylab="Rank (signed Spearman ρ)" if ci == 0 else None, alpha=alpha)
    fig.text(0.5, 0.005, "Rank = per-token signed Spearman ρ over the 4-level intensity ramp, collapsed per "
             "(sentence, concept) unit by category.\n"
             "solid/faded = ρ differs from ZERO (sentence-clustered sign-flip, BH-FDR q<0.05 across categories, "
             "B=5000); error bars = 95% cluster bootstrap over the 50 sentences (B=2000).",
             ha="center", fontsize=7.5, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 1))
    fig.savefig(out_b, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_b}")
    return str(out_a), str(out_b)


def main():
    ap = argparse.ArgumentParser(description="Fig 5a/7b: metrics by POS token category (bar plots + 95% CI).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pos-path", default="pos_tags.json")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-a", default="fig5a_pos_categories.png")
    ap.add_argument("--out-b", default="fig5b_pos_rank.png")
    args = ap.parse_args()
    render(args.run_dir, out_a=args.out_a, out_b=args.out_b, pos_path=args.pos_path,
           vector_cache=args.vector_cache, method=args.method, model=args.model,
           n_boot=args.n_boot)


if __name__ == "__main__":
    main()
