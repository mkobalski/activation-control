#!/usr/bin/env python3
"""Fig 4: intensity-rank depth profiles — does the readout track instructed
intensity, and where in depth?

Same two-panel format and dimensions as Fig 2 (fig2_depth_profile.py): top =
Cosine similarity readout, bottom = Relative norm readout; x = depth %.
Each panel shows the mean signed Spearman rank correlation (the retired Fig 5's
measure) between instructed intensity level and the readout, for two contrasts:

    green  : think_about -> think_intensely            (2 levels; rho = sign)
    yellow : think_intensity_{1..4}_of_4               (4 levels; classic rank)

Per (sentence, concept) unit and layer: rho is computed PER TOKEN over the
levels present (2-level needs both; 4-level needs >=3), then averaged over the
unit's tokens -> one value per unit. No no_instruction baseline is involved
(rank is a within-condition ordering). Curve = mean over the ~500 units;
band = 95% cluster bootstrap over the 50 sentences (B=2000); ringed marker =
mean rho != 0 by per-unit sign-flip permutation (valid because the signed
Spearman is sign-symmetric under H0; B=5000, two-sided), BH-FDR across the 20
layers per curve. Full definitions in results/paper/Fig4.md. CPU-only.
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
from fig5_rank_intensity import _signed_spearman                          # noqa: E402

# (key, ordered condition levels, min levels per token, label, color)
CONTRASTS = [
    ("lex", ["think_about", "think_intensely"], 2,
     "Think about → Think intensely about", "#1e8449"),          # green
    ("ramp", [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)], 3,
     "Think at intensity {1→4} about", "#d4a017"),               # yellow/gold
]
ROWS = [("cos", "Cosine similarity"), ("relnorm", "Relative norm")]
N_LAYERS_TOTAL = 62


def build(run_dir, *, n_boot=2000, n_perm=5000, seed=0):
    rows = _load_json(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    wanted = {c for _, levels, _, _, _ in CONTRASTS for c in levels}

    # units[(metric, key)][sentence] = list of per-layer unit values (token-mean rho)
    units = {(m, k): defaultdict(list) for m, _ in ROWS for k, *_ in CONTRASTS}
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            if r["condition_id"] in wanted and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r
                concepts.add(r["concept"])

        def readout(row, metric, L):
            if metric == "cos":
                tr = _trace(row, "cosine_sim", L)
                return np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
            tr = _trace(row, "norms", L)
            return _relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None

        for key, levels, min_lv, _, _ in CONTRASTS:
            for c in sorted(concepts):
                rows_c = {lev: byc[cond].get(c) for lev, cond in enumerate(levels)
                          if c in byc.get(cond, {})}
                if len(rows_c) < min_lv:
                    continue
                for metric, _ in ROWS:
                    per_layer = np.full(len(layers), np.nan)
                    for li, L in enumerate(layers):
                        reads = {lev: readout(r_, metric, L) for lev, r_ in rows_c.items()}
                        reads = {lev: v for lev, v in reads.items() if v is not None}
                        if len(reads) < min_lv:
                            continue
                        rhos = []
                        for ti in range(n_tok):
                            lv = [lev for lev, v in reads.items()
                                  if ti < len(v) and not np.isnan(v[ti])]
                            vv = [reads[lev][ti] for lev in lv]
                            if len(lv) >= min_lv:
                                rho = _signed_spearman(lv, vv)
                                if not np.isnan(rho):
                                    rhos.append(rho)
                        if rhos:
                            per_layer[li] = float(np.mean(rhos))
                    units[(metric, key)][s].append(per_layer)

    rng = np.random.default_rng(seed)
    stats = {}
    for mkey, by_s in units.items():
        sents = sorted(by_s)
        S = len(sents)
        blocks = [np.vstack(by_s[s]) for s in sents]        # per sentence: (k_s, n_layers)
        U = np.vstack(blocks)
        mean = np.nanmean(U, axis=0)
        # cluster bootstrap over sentences
        k = np.array([b.shape[0] for b in blocks], float)
        sums = np.vstack([np.nansum(b, axis=0) for b in blocks])       # (S, n_layers)
        cnts = np.vstack([np.sum(~np.isnan(b), axis=0) for b in blocks]).astype(float)
        Mm = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float)
        num = Mm @ sums
        den = Mm @ cnts
        boots = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        lo = np.nanpercentile(boots, 2.5, axis=0)
        hi = np.nanpercentile(boots, 97.5, axis=0)
        # per-unit sign-flip permutation vs 0, per layer
        p = np.full(len(mean), np.nan)
        for li in range(len(mean)):
            dv = U[:, li]; dv = dv[~np.isnan(dv)]
            if len(dv) >= 3:
                obs = float(dv.mean())
                signs = rng.choice([-1.0, 1.0], size=(n_perm, len(dv)))
                null = (signs * dv).mean(1)
                p[li] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1)
        stats[mkey] = dict(mean=mean, lo=lo, hi=hi, q=bh_fdr(p),
                           n_units=U.shape[0], n_sent=S)
    return sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])}), stats


def render(run_dir, *, out, alpha=0.05, **kw):
    layers, stats = build(run_dir, **kw)
    x = np.arange(len(layers))
    depth_pct = [f"{100 * L / N_LAYERS_TOTAL:.0f}" for L in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                             layout="constrained")
    for ax, (metric, title) in zip(axes, ROWS):
        for key, _levels, _m, label, color in CONTRASTS:
            st = stats[(metric, key)]
            ax.plot(x, st["mean"], "-o", color=color, lw=2.0, ms=4.5,
                    label=label, zorder=3)
            ax.fill_between(x, st["lo"], st["hi"], color=color, alpha=0.15, zorder=2)
            sig = st["q"] < alpha
            if np.any(sig):
                ax.plot(x[sig], st["mean"][sig], "o", color=color, ms=8,
                        mfc=color, mec="black", mew=0.9, zorder=4)
        ax.axhline(0, color="#888", lw=0.9, ls="--", zorder=1)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean Spearman ρ", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(depth_pct)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), frameon=False,
                   fontsize=10, labelspacing=0.4)
    axes[1].set_xlabel("Depth (%)", fontsize=11)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    st0 = stats[("cos", "ramp")]
    print(f"wrote {out}  [ramp n={st0['n_units']} units, "
          f"lex n={stats[('cos','lex')]['n_units']}; {st0['n_sent']} sentences]")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Fig 4: intensity-rank depth profiles.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig4_rank_depth.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha)


if __name__ == "__main__":
    main()
