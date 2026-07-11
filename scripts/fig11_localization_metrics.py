#!/usr/bin/env python3
"""Fig 11: localization metrics for positional/type targeting (token_location).

Turns the qualitative "did targeting work" of Figs 8-9 into four scalar metrics
per condition, on both channels, each compared against think_about (engage
everywhere) to isolate the instruction's contribution from the model's intrinsic
positional/type bias.

Per trial (sentence s, concept c, condition κ), token i, fractional position
f_i = i/(n-1); signal s_i = Δ vs no_instruction (Δcos @L61 or Δrelnorm @L43),
differenced within the (s,c) unit. Target mask M (T = target tokens, O = rest):
  loc_beginning : f <= 1/3      loc_end : f >= 2/3
  loc_punctuation : UPOS==PUNCT loc_adjectives : UPOS==ADJ

Metrics:
  Inside gain   G_in  = mean_{i in T} s_i
  Outside leak  G_out = mean_{i in O} s_i
  Localization  SI    = (G_in - G_out) / (|G_in| + |G_out|)   in [-1,1]
                        (bounded form of the ratio R = G_in/G_out; R printed too)
  CoM error     Σ_i p̂_i · d_i,  p̂ = normalized max(0,s_i),
                d_i = min_{j in T} |f_i - f_j|  (frac. distance to nearest target)

Each metric is computed for the location condition AND for think_about on the same
(condition-specific) mask. Bars = mean over (sentence, concept) units; error bars =
95% bootstrap CI (B=2000). ★ = loc differs from think (paired sign-flip over units,
B=5000, BH-FDR across the 4 conditions in the panel).

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
from fig5_pos_categories import _pos_by_sentence, _token_upos            # noqa: E402

COS_L, RN_L = 61, 43
BASE, THINK = "no_instruction", "think_about"
# (condition, label, mask-kind)
CONDS = [("loc_beginning", "Beginning", "begin"),
         ("loc_end", "End", "end"),
         ("loc_punctuation", "Punctuation", "PUNCT"),
         ("loc_adjectives", "Adjectives", "ADJ")]
CHANNELS = [("cos", COS_L, "Δ Cosine similarity (layer 61)"),
            ("relnorm", RN_L, "Δ Relative norm (layer 43)")]
METRICS = ["G_in", "G_out", "SI", "CoM"]
METRIC_LABEL = {"G_in": "Inside-target gain", "G_out": "Outside leakage",
                "SI": "Localization (selectivity)", "CoM": "Center-of-mass error"}


def _mask(kind, f, upos):
    if kind == "begin":
        return f <= 1 / 3 + 1e-9
    if kind == "end":
        return f >= 2 / 3 - 1e-9
    return np.array([u == kind for u in upos])           # PUNCT / ADJ


def _dist_to_target(f, T):
    """Fractional distance from each token to the nearest target token (0 if in T)."""
    tf = f[T]
    return np.array([float(np.min(np.abs(fi - tf))) for fi in f])


def _metrics(sig, T, O, d):
    """Return dict of the four metrics for signal `sig` (per-token, may be NaN)."""
    v = ~np.isnan(sig)
    Ti, Oi = T & v, O & v
    if Ti.sum() < 1 or Oi.sum() < 1:
        return None
    g_in = float(sig[Ti].mean()); g_out = float(sig[Oi].mean())
    si = (g_in - g_out) / (abs(g_in) + abs(g_out) + 1e-12)
    r = g_in / g_out if abs(g_out) > 1e-9 else np.nan
    p = np.clip(np.where(v, sig, 0.0), 0, None)
    com = float((p / p.sum() * d).sum()) if p.sum() > 1e-12 else np.nan
    return dict(G_in=g_in, G_out=g_out, SI=si, CoM=com, R=r)


def build(run_dir, *, pos_path="pos_tags.json", vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    pos_words = _pos_by_sentence(pos_path)
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [COS_L], method)
    wanted = {c for c, _, _ in CONDS} | {THINK}

    # store[(channel, cond, series, metric)] = {(s,c): value}
    store = defaultdict(lambda: defaultdict(dict))

    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if words is None or toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n = len(toks)
        if n < 3:
            continue
        classes = [classify(t) for t in toks]
        upos = _token_upos(s, toks, words)
        f = np.arange(n) / (n - 1)
        masks = {}
        for _, _, kind in CONDS:
            T = _mask(kind, f, upos)
            if T.sum() >= 1 and (~T).sum() >= 1:
                masks[kind] = (T, ~T, _dist_to_target(f, T))

        acos = np.asarray(ent["activations"][COS_L], np.float32)[:n]
        acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
        base_cos_all = acos @ vecs[COS_L][2].T if COS_L in vecs else None
        base_rn = _relnorm(np.asarray(ent["norms"][RN_L], np.float32)[:n], classes)
        concepts_cosL = vecs[COS_L][0] if COS_L in vecs else []

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        def signal(row, channel, base_cos_c):
            if channel == "cos":
                tr = _trace(row, "cosine_sim", COS_L)
                v = np.asarray(tr, np.float32)[:n] if tr is not None else None
                return (v - base_cos_c) if (v is not None and base_cos_c is not None) else None
            tr = _trace(row, "norms", RN_L)
            v = _relnorm(np.asarray(tr, np.float32)[:n], classes) if tr is not None else None
            return (v - base_rn) if (v is not None and base_rn is not None) else None

        for c in sorted(concepts):
            base_cos_c = (base_cos_all[:, concepts_cosL.index(c)]
                          if (base_cos_all is not None and c in concepts_cosL) else None)
            for channel, _, _ in CHANNELS:
                think_sig = signal(byc.get(THINK, {}).get(c), channel, base_cos_c) \
                    if THINK in byc and c in byc[THINK] else None
                for cond, _, kind in CONDS:
                    if kind not in masks:
                        continue
                    T, O, d = masks[kind]
                    loc_sig = signal(byc.get(cond, {}).get(c), channel, base_cos_c) \
                        if cond in byc and c in byc[cond] else None
                    if loc_sig is not None:
                        m = _metrics(loc_sig, T, O, d)
                        if m:
                            for mk in METRICS + ["R"]:
                                store[(channel, cond, "loc", mk)][(s, c)] = m[mk]
                    if think_sig is not None:
                        m = _metrics(think_sig, T, O, d)
                        if m:
                            for mk in METRICS + ["R"]:
                                store[(channel, cond, "think", mk)][(s, c)] = m[mk]

    rng = np.random.default_rng(seed)

    def agg(key):
        vals = np.array([v for v in store[key].values() if not np.isnan(v)])
        if len(vals) < 2:
            return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=len(vals))
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        boot = vals[idx].mean(1)
        return dict(mean=float(vals.mean()), lo=float(np.percentile(boot, 2.5)),
                    hi=float(np.percentile(boot, 97.5)), n=len(vals))

    def paired_p(channel, cond, metric):
        da, dt = store[(channel, cond, "loc", metric)], store[(channel, cond, "think", metric)]
        keys = [k for k in (set(da) & set(dt)) if not (np.isnan(da[k]) or np.isnan(dt[k]))]
        if len(keys) < 3:
            return np.nan
        dv = np.array([da[k] - dt[k] for k in keys]); obs = dv.mean()
        signs = rng.integers(0, 2, size=(n_perm, len(dv))) * 2.0 - 1.0
        null = (signs * dv).mean(1)
        return (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)

    stats = {}
    for channel, _, _ in CHANNELS:
        for metric in METRICS:
            ps = []
            for cond, _, _ in CONDS:
                stats[(channel, cond, "loc", metric)] = agg((channel, cond, "loc", metric))
                stats[(channel, cond, "think", metric)] = agg((channel, cond, "think", metric))
                ps.append(paired_p(channel, cond, metric))
            q = bh_fdr(np.array(ps))
            for (cond, _, _), qi in zip(CONDS, q):
                stats[("q", channel, cond, metric)] = qi
    # console summary of the raw localization ratio R (per the user's original metric)
    print("Localization ratio R = inside/outside (mean over units):")
    for channel, _, _ in CHANNELS:
        for cond, lab, _ in CONDS:
            print(f"  {channel:7s} {lab:11s} loc R={agg((channel, cond, 'loc', 'R'))['mean']:.2f} "
                  f"think R={agg((channel, cond, 'think', 'R'))['mean']:.2f}")
    return stats


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, alpha=0.05, **kw):
    stats = build(run_dir, **kw)
    labels = [lab for _, lab, _ in CONDS]
    x = np.arange(len(CONDS)); w = 0.4
    loc_c, think_c = "#c0392b", "#95a5a6"

    fig = plt.figure(figsize=(4.6 * len(METRICS), 8.8), layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=0.06)

    for ri, (channel, L, row_title) in enumerate(CHANNELS):
        sf = subfigs[ri]
        sf.suptitle(row_title, fontsize=14, fontweight="bold")
        axes = sf.subplots(1, len(METRICS))
        for ci, metric in enumerate(METRICS):
            ax = axes[ci]
            for series, cser, off, lab in (("loc", loc_c, -w / 2, "Location-targeted"),
                                           ("think", think_c, w / 2, "Think (everywhere)")):
                mean = np.array([stats[(channel, c, series, metric)]["mean"] for c, _, _ in CONDS])
                lo = np.array([stats[(channel, c, series, metric)]["lo"] for c, _, _ in CONDS])
                hi = np.array([stats[(channel, c, series, metric)]["hi"] for c, _, _ in CONDS])
                yerr = np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])
                ax.bar(x + off, mean, width=w, color=cser, edgecolor="black", linewidth=0.5,
                       label=lab, zorder=2)
                ax.errorbar(x + off, mean, yerr=yerr, fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=2, zorder=3)
            for j, (cond, _, _) in enumerate(CONDS):                # significance stars
                q = stats.get(("q", channel, cond, metric), np.nan)
                if not np.isnan(q) and q < alpha:
                    tops = [stats[(channel, cond, s, metric)]["hi"] for s in ("loc", "think")]
                    ax.text(j, np.nanmax(tops + [0]), "*", ha="center", va="bottom", fontsize=13)
            ax.axhline(0, color="#555", lw=0.8, zorder=1)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
            if ri == 0:
                ax.set_title(METRIC_LABEL[metric], fontsize=12)
            if metric == "CoM":
                ax.set_ylabel("fractional distance (lower = better)", fontsize=8)
            if ci == 0:
                ax.legend(fontsize=8, framealpha=0.9, loc="best")
            ax.margins(x=0.04)

    fig.text(0.5, 0.005, "Δ vs no_instruction, avg over (sentence × concept) units; error bars = 95% "
             "bootstrap CI (B=2000).  ★ = location condition differs from Think-everywhere on that "
             "metric (paired sign-flip, BH-FDR q<0.05 across conditions, B=5000).  masks: beginning/end = "
             "fractional thirds, punctuation/adjectives = UPOS.  Δcos @L61, Δrelnorm @L43.",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 11: localization metrics for token_location targeting.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--pos-path", default="pos_tags.json")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="fig11_localization_metrics.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, pos_path=args.pos_path, vector_cache=args.vector_cache,
           method=args.method, model=args.model, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
