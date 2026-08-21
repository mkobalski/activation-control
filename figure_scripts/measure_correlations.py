#!/usr/bin/env python3
"""Measure_Correlations — how independent the six measures are (Figure 15).

Lower-triangular Spearman correlation matrix over the panel. For each model the six
scores are read on the projection channel at the same layers the benchmark uses,
giving a 25 x 6 matrix; the 15 pairwise rank correlations are what this plots.

Rank correlation rather than Pearson because the measures are on different scales
(five d', one rho) and a few models sit far from the rest on some axes. At n = 25 an
|rho| below 0.40 is not different from zero at p < 0.05, so cells at or above that
threshold are drawn bold and the largest is boxed.

Reads ONLY results/SCORES_<model>.json -- no raw, no model load, no recompute of any
measure. scipy is deliberately not a dependency: the ranking and the participation
ratio are a few lines of numpy below.

AAAI conventions: TrueType, 300 dpi, no title, no top/right spines, .pdf + .png,
caption material to a side-car .md (never on the figure).
"""
import argparse
import json
from pathlib import Path

from paths import AC_DATA, out                                       # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                # noqa: E402
from matplotlib.patches import Rectangle                             # noqa: E402

from roster import ROSTER                                            # noqa: E402  (single-source roster)

# The six measures that enter S, in the battery's canonical report order. Kept as a
# literal rather than imported from aggregate_scalar so this figure has no dependency
# on the scoring layer -- it reads the frozen SCORES like every other paper figure.
MEASURES = [("engage", "Engage"), ("suppress", "Suppress"), ("dial_rank", "Dial rank"),
            ("temporal_control", "Temporal control"), ("coverage", "Coverage"),
            ("layer_targeting", "Layer targeting")]

# At n = 25, |rho| >= 0.40 is the two-sided p < 0.05 threshold.
SIG = 0.40

GREY_TEXT = "#52514e"
DASH_GREY = "#8a8983"
VALUE_TEXT = "#0b0b0b"
# Diverging blue -> warm grey -> red, linear in rho from the neutral centre. The
# centre doubles as the fill for the empty diagonal, so a zero correlation and "not
# applicable" read the same weight.
NEUTRAL = (0.9373, 0.9333, 0.9176)
CMAP = LinearSegmentedColormap.from_list(
    "rho", [(0.1873, 0.4904, 0.8700), NEUTRAL, (0.8910, 0.2858, 0.2828)])
EMPTY_FILL = NEUTRAL


def score_matrix(data_root):
    """(n_models x 6) projection-channel scores, and the models that supplied them."""
    rows, used = [], []
    for name, _, _ in ROSTER:
        p = Path(data_root) / f"SCORES_{name}.json"
        if not p.exists():
            continue
        meas = json.load(open(p))["measures"]
        vals = [((meas.get(k, {}).get("channels", {}) or {}).get("proj") or {}).get("score")
                for k, _ in MEASURES]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        rows.append(vals)
        used.append(name)
    return np.asarray(rows, dtype=float), used


def _rankdata(a):
    """Ranks with ties averaged (scipy.stats.rankdata's 'average', in six lines)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    srt, i = a[order], 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman_matrix(X):
    """Spearman rho = Pearson on the per-column ranks."""
    R = np.column_stack([_rankdata(X[:, k]) for k in range(X.shape[1])])
    return np.corrcoef(R, rowvar=False)


def participation_ratio(C):
    """Effective dimensionality: (sum lambda)^2 / sum lambda^2 over the eigenvalues.

    6.0 would mean six fully independent axes, 1.0 a single shared factor.
    """
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0.0, None)
    return float(lam.sum() ** 2 / np.sum(lam ** 2))


def render(C, out_path, pdf=False):
    n = len(MEASURES)
    labels = [lab for _, lab in MEASURES]
    fig, ax = plt.subplots(figsize=(6.64, 5.38))

    biggest = max(((abs(C[i, j]), i, j) for i in range(n) for j in range(i)),
                  default=(0, None, None))[1:]
    for i in range(n):
        for j in range(i + 1):
            if i == j:                                   # diagonal: no self-correlation
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=EMPTY_FILL, lw=0))
                ax.text(j, i, "—", ha="center", va="center", color=DASH_GREY, fontsize=9)
                continue
            rho = C[i, j]
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   facecolor=CMAP((rho + 1) / 2), lw=0))
            strong = abs(rho) >= SIG
            ax.text(j, i, f"{rho:+.2f}", ha="center", va="center", color=VALUE_TEXT,
                    fontsize=10.5 if strong else 9.5,
                    fontweight="bold" if strong else "normal")
            if (i, j) == biggest:                        # the one pair worth naming
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.8))

    ax.set_xlim(-.6, n - .4); ax.set_ylim(n - .4, -.6)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5, color=GREY_TEXT)
    ax.set_yticklabels(labels, fontsize=9.5, color=GREY_TEXT)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.04, shrink=0.55, aspect=18,
                      anchor=(0.0, 0.72), ticks=[-1, -.5, 0, .5, 1])
    cb.ax.set_yticklabels(["−1", "−0.5", "0", "+0.5", "+1"], fontsize=8,
                          color=GREY_TEXT)
    cb.ax.set_title("Spearman ρ", fontsize=8.5, color=GREY_TEXT, pad=6, loc="center")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, color=GREY_TEXT)

    fig.tight_layout()
    exts = [out_path] + ([str(Path(out_path).with_suffix(".pdf"))] if pdf else [])
    for e in exts:
        fig.savefig(e, bbox_inches="tight")
        print(f"wrote {e}")
    plt.close(fig)


def write_md(out_md, C, used, pr):
    n = len(MEASURES)
    pairs = sorted(((abs(C[i, j]), C[i, j], MEASURES[i][1], MEASURES[j][1])
                    for i in range(n) for j in range(i)), reverse=True)
    L = [f"# {Path(out_md).stem} — caption material\n",
         "*Auto-emitted; not on the figure.* Spearman rank correlation between the six "
         f"measures across the {len(used)} panel models, each read on the projection "
         "channel at the layers the benchmark uses. Rank correlation because the measures "
         "have different scales and a few models are far from the others on some measures. "
         f"At $n = {len(used)}$, $|\\rho|$ below {SIG:.2f} is not different from zero at "
         f"$p < 0.05$; {sum(1 for a, *_ in pairs if a >= SIG)} of the 15 pairs are above "
         "that level (bold). The largest, boxed, is "
         f"{pairs[0][2]} with {pairs[0][3]} at $\\rho = {pairs[0][1]:+.2f}$ — expected, "
         "since Coverage is an Engage contrast restricted to a model's weakest token "
         "category. The correlation matrix has an effective dimensionality of "
         f"{pr:.2f} of a possible {n} (participation ratio of its eigenvalues).\n",
         "Values ($\\rho$, descending by magnitude):\n"]
    L += [f"- **{a} × {b}**: {r:+.2f}" for _, r, a, b in pairs]
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Measure-correlation matrix (Fig 15).")
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Measure_Correlations.png"))
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    X, used = score_matrix(args.data_root)
    if len(used) < 3:
        raise SystemExit(f"need at least 3 models with all six proj scores; got {len(used)}")
    C = spearman_matrix(X)
    pr = participation_ratio(C)
    print(f"{len(used)} models x {len(MEASURES)} measures; effective dimensionality {pr:.2f}")
    render(C, args.out, pdf=args.pdf)
    write_md(str(Path(args.out).with_suffix(".md")), C, used, pr)


if __name__ == "__main__":
    main()
