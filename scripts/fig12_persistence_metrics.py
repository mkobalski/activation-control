#!/usr/bin/env python3
"""Fig 12: temporal-control metrics for the persistence conditions.

Frames each persistence instruction as an intended on-window [τ_on, τ_off] with
off-regions before (B) and after (A), and scores how well the actual engagement
profile matches it. Signal s_i = Δcos vs no_instruction @L61 (differenced within
the (sentence, concept) unit — the channel where persistence shows, Fig 7).

Intended windows (fractional position f_i = i/(n-1); after_fourth by token index):
  throughout   : W = all
  first_half   : W = f<=1/2 ; A = f>1/2
  once         : B = f<0.4 ; W = 0.4<=f<=0.6 ; A = f>0.6            (pulse ~0.5)
  after_fourth : B = tokens 1-4 ; W = tokens 5.. ; τ_on = f of token 5

Metrics:
  Persistence     mean_{i∈W} s_i           (level held across the requested window)
  Leakage before  mean_{i∈B} s_i           (engagement before the window; ~0 ideal)
  Leakage after   mean_{i∈A} s_i           (engagement after the window; ~0 ideal)
  Rebound         max_{i∈A}(s_i − s(τ_off)) clipped ≥0   (improper return after "stop")
  Onset error     detected_rise − τ_on     (half-max crossing on the avg profile)
  Offset error    detected_fall − τ_off

Level metrics: per unit, mean + 95% bootstrap CI, vs think_about (which is always
on, so its leakage is the no-gating reference); ★ = loc differs from think (paired
sign-flip, B=5000, BH-FDR across the conditions in the panel). Edge errors:
detected on the bootstrapped average Δcos profile (10 fractional bins); the CI is
the bootstrap spread of (detected − requested). CPU-only, no model load.
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
from fig4_fraction_engage_suppress import _bins_for, _bin_means          # noqa: E402

COS_L = 61
THROUGH, FIRST, ONCE, AFTER = ("persist_throughout", "persist_first_half",
                               "persist_once", "persist_after_fourth")
THINK, BASE = "think_about", "no_instruction"
LABEL = {THROUGH: "Throughout", FIRST: "First half", ONCE: "Once", AFTER: "After 4th"}
N_BINS = 10

# level-metric panels: (metric key, title, applicable conditions)
LEVEL_PANELS = [
    ("persistence", "Persistence score", [THROUGH, FIRST, ONCE, AFTER]),
    ("leak_before", "Leakage before", [ONCE, AFTER]),
    ("leak_after", "Leakage after", [FIRST, ONCE]),
    ("rebound", "Rebound after suppression", [FIRST, ONCE]),
]
# edge-error panels: (metric, title, [(condition, requested-edge-getter)])
EDGE_PANELS = [("onset", "Onset error", [ONCE, AFTER]),
               ("offset", "Offset error", [FIRST, ONCE])]


def _windows(cond, n, f):
    """Return (B, W, A, tau_off, tau_on) masks (bool arrays or None) + edges (frac)."""
    idx = np.arange(n)
    if cond == THROUGH:
        return None, np.ones(n, bool), None, None, None
    if cond == FIRST:
        return None, f <= 0.5 + 1e-9, f > 0.5 + 1e-9, 0.5, None
    if cond == ONCE:
        return f < 0.4, (f >= 0.4) & (f <= 0.6), f > 0.6, 0.5, 0.5
    if cond == AFTER:
        return idx < 4, idx >= 4, None, None, 4.0 / (n - 1)
    return None, None, None, None, None


def _level(sig, B, W, A, tau_off, f):
    v = ~np.isnan(sig)
    out = {}
    if W is not None and (W & v).sum() >= 1:
        out["persistence"] = float(sig[W & v].mean())
    if B is not None and (B & v).sum() >= 1:
        out["leak_before"] = float(sig[B & v].mean())
    if A is not None and (A & v).sum() >= 1:
        out["leak_after"] = float(sig[A & v].mean())
        if tau_off is not None:
            j = int(np.argmin(np.abs(f - tau_off)))
            if v[j]:
                out["rebound"] = max(0.0, float(sig[A & v].max() - sig[j]))
    return out


def _detect(profile):
    """Half-max rising/falling crossings on a rectified profile -> (onset_f, offset_f)."""
    p = np.clip(profile, 0, None)
    if not np.isfinite(p).all() or p.max() - p.min() <= 1e-12:
        return np.nan, np.nan
    above = np.where(p >= p.min() + 0.5 * (p.max() - p.min()))[0]
    if len(above) == 0:
        return np.nan, np.nan
    return (above[0] + 0.5) / len(p), (above[-1] + 0.5) / len(p)


def build(run_dir, *, metric="cos", layer=None, vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", n_boot=2000, n_perm=5000, seed=0):
    """metric: 'cos' (default, layer 61) or 'relnorm' (layer 43 — the norm
    channel's peak). Same windows/detection either way."""
    L_SIG = layer if layer is not None else (COS_L if metric == "cos" else 43)
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    cache = pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, [L_SIG], method)
    conds = [THROUGH, FIRST, ONCE, AFTER]
    wanted = set(conds) | {THINK}

    lvl = defaultdict(lambda: defaultdict(dict))       # (metric,cond,series)->{(s,c):val}
    prof = defaultdict(dict)                            # cond -> {(s,c): binned Δcos}
    f5 = {}                                             # (s,c)->frac pos of token 5 (AFTER)

    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n = len(toks)
        if n < 4:
            continue
        classes = [classify(t) for t in toks]
        f = np.arange(n) / (n - 1)
        bins = _bins_for(n, N_BINS)
        if metric == "cos":
            acos = np.asarray(ent["activations"][L_SIG], np.float32)[:n]
            acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
            base_cos_all = acos @ vecs[L_SIG][2].T if L_SIG in vecs else None
            base_rn = None
        else:
            base_cos_all = None
            base_rn = _relnorm(np.asarray(ent["norms"][L_SIG], np.float32)[:n], classes)
        concepts_cosL = vecs[L_SIG][0] if L_SIG in vecs else []

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        def dcos(row, base_c):
            if metric == "cos":
                tr = _trace(row, "cosine_sim", L_SIG)
                v = np.asarray(tr, np.float32)[:n] if tr is not None else None
            else:
                tr = _trace(row, "norms", L_SIG)
                v = _relnorm(np.asarray(tr, np.float32)[:n], classes) if tr is not None else None
            return (v - base_c) if (v is not None and base_c is not None) else None

        for c in sorted(concepts):
            if metric == "cos":
                if base_cos_all is None or c not in concepts_cosL:
                    continue
                base_c = base_cos_all[:, concepts_cosL.index(c)]
            else:
                if base_rn is None:
                    continue
                base_c = base_rn
            think_sig = dcos(byc.get(THINK, {}).get(c), base_c) if c in byc.get(THINK, {}) else None
            for cond in conds:
                B, W, A, toff, ton = _windows(cond, n, f)
                loc_sig = dcos(byc.get(cond, {}).get(c), base_c) if c in byc.get(cond, {}) else None
                if loc_sig is not None:
                    for mk, val in _level(loc_sig, B, W, A, toff, f).items():
                        lvl[(mk, cond, "loc")][(s, c)] = val
                    prof[cond][(s, c)] = _bin_means(loc_sig, bins, N_BINS)
                    if cond == AFTER:
                        f5[(s, c)] = 4.0 / (n - 1)
                if think_sig is not None:
                    for mk, val in _level(think_sig, B, W, A, toff, f).items():
                        lvl[(mk, cond, "think")][(s, c)] = val

    rng = np.random.default_rng(seed)

    def agg(d):
        vals = np.array([v for v in d.values() if not np.isnan(v)])
        if len(vals) < 2:
            return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=len(vals))
        boot = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
        return dict(mean=float(vals.mean()), lo=float(np.percentile(boot, 2.5)),
                    hi=float(np.percentile(boot, 97.5)), n=len(vals))

    def paired_p(metric, cond):
        da, dt = lvl[(metric, cond, "loc")], lvl[(metric, cond, "think")]
        keys = [k for k in (set(da) & set(dt)) if not (np.isnan(da[k]) or np.isnan(dt[k]))]
        if len(keys) < 3:
            return np.nan
        dv = np.array([da[k] - dt[k] for k in keys]); obs = dv.mean()
        signs = rng.integers(0, 2, size=(n_perm, len(dv))) * 2.0 - 1.0
        null = (signs * dv).mean(1)
        return (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)

    stats = {}
    for metric, _, cond_list in LEVEL_PANELS:
        ps = []
        for cond in cond_list:
            stats[(metric, cond, "loc")] = agg(lvl[(metric, cond, "loc")])
            stats[(metric, cond, "think")] = agg(lvl[(metric, cond, "think")])
            ps.append(paired_p(metric, cond))
        for cond, q in zip(cond_list, bh_fdr(np.array(ps))):
            stats[("q", metric, cond)] = q

    # edge errors: bootstrap the average profile, detect, error = detected − requested
    req_on = {ONCE: 0.5, AFTER: float(np.mean(list(f5.values()))) if f5 else np.nan}
    req_off = {FIRST: 0.5, ONCE: 0.5}
    for metric, _, cond_list in EDGE_PANELS:
        for cond in cond_list:
            keys = list(prof[cond])
            if len(keys) < 3:
                stats[("edge", metric, cond)] = dict(mean=np.nan, lo=np.nan, hi=np.nan,
                                                     req=np.nan, detected=np.nan)
                continue
            P = np.vstack([prof[cond][k] for k in keys])
            on0, off0 = _detect(np.nanmean(P, 0))
            det0 = on0 if metric == "onset" else off0
            req = (req_on if metric == "onset" else req_off).get(cond, np.nan)
            errs = []
            for _ in range(n_boot):
                bi = rng.integers(0, len(keys), size=len(keys))
                on, off = _detect(np.nanmean(P[bi], 0))
                d = on if metric == "onset" else off
                if not np.isnan(d):
                    errs.append(d - req)
            errs = np.array(errs)
            stats[("edge", metric, cond)] = dict(
                mean=float(det0 - req) if not np.isnan(det0) else np.nan,
                lo=float(np.percentile(errs, 2.5)) if len(errs) else np.nan,
                hi=float(np.percentile(errs, 97.5)) if len(errs) else np.nan,
                req=req, detected=det0)
    return stats


# ---- render --------------------------------------------------------------------

def render(run_dir, *, out, alpha=0.05, **kw):
    stats = build(run_dir, **kw)
    loc_c, think_c = "#c0392b", "#95a5a6"

    fig = plt.figure(figsize=(16, 9), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.06)
    top.suptitle("Level metrics  (Δcos @L61 vs no_instruction;  red = condition, grey = think-everywhere)",
                 fontsize=13, fontweight="bold")
    bot.suptitle("Edge-timing error  (detected − requested edge, fractional position; 0 = perfect)",
                 fontsize=13, fontweight="bold")

    axes_t = top.subplots(1, len(LEVEL_PANELS))
    for ax, (metric, title, cond_list) in zip(axes_t, LEVEL_PANELS):
        x = np.arange(len(cond_list)); w = 0.4
        for series, cser, off, lab in (("loc", loc_c, -w / 2, "Location"),
                                       ("think", think_c, w / 2, "Think")):
            mean = np.array([stats[(metric, c, series)]["mean"] for c in cond_list])
            lo = np.array([stats[(metric, c, series)]["lo"] for c in cond_list])
            hi = np.array([stats[(metric, c, series)]["hi"] for c in cond_list])
            ax.bar(x + off, mean, width=w, color=cser, edgecolor="black", linewidth=0.5, label=lab)
            ax.errorbar(x + off, mean, yerr=np.vstack([np.clip(mean - lo, 0, None),
                        np.clip(hi - mean, 0, None)]), fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=2)
        for j, cond in enumerate(cond_list):
            q = stats.get(("q", metric, cond), np.nan)
            if not np.isnan(q) and q < alpha:
                tops = [stats[(metric, cond, s)]["hi"] for s in ("loc", "think")]
                ax.text(j, np.nanmax(tops + [0]), "*", ha="center", va="bottom", fontsize=13)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in cond_list], fontsize=9, rotation=20, ha="right")
        ax.set_title(title, fontsize=12)
        ax.margins(x=0.15)
    axes_t[0].set_ylabel("Δcos", fontsize=10); axes_t[0].legend(fontsize=8, loc="best")

    axes_b = bot.subplots(1, len(EDGE_PANELS))
    for ax, (metric, title, cond_list) in zip(axes_b, EDGE_PANELS):
        x = np.arange(len(cond_list))
        mean = np.array([stats[("edge", metric, c)]["mean"] for c in cond_list])
        lo = np.array([stats[("edge", metric, c)]["lo"] for c in cond_list])
        hi = np.array([stats[("edge", metric, c)]["hi"] for c in cond_list])
        ax.errorbar(x, mean, yerr=np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)]),
                    fmt="o", color=loc_c, ms=8, capsize=4, elinewidth=1.2, mec="black")
        ax.axhline(0, color="#555", lw=1.0, ls="--", label="perfect")
        for j, c in enumerate(cond_list):
            req = stats[("edge", metric, c)]["req"]; det = stats[("edge", metric, c)]["detected"]
            ax.annotate(f"req {req:.2f}\ndet {det:.2f}", (j, mean[j]), fontsize=7,
                        xytext=(8, 0), textcoords="offset points", va="center", color="#555")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in cond_list], fontsize=9)
        ax.set_title(title, fontsize=12); ax.set_ylabel("error (fractional)", fontsize=9)
        ax.set_xlim(-0.5, len(cond_list) - 0.5); ax.margins(x=0.2)

    fig.text(0.5, 0.005, "avg over (sentence × concept) units; level bars = 95% bootstrap CI (B=2000), "
             "★ = loc differs from think (paired sign-flip, BH-FDR q<0.05, B=5000); edge errors detected by "
             "half-max crossing on the bootstrapped mean profile (10 frac bins), CI over B=2000.  after_fourth "
             "onset requested = mean fractional position of token 5.",
             ha="center", fontsize=8, color="#444")
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 12: persistence temporal-control metrics.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--method", default="baseline")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="fig12_persistence_metrics.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, vector_cache=args.vector_cache,
           method=args.method, model=args.model, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
