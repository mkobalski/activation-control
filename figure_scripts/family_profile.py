#!/usr/bin/env python3
"""Per-measure controllability profile figure (style, projection channel).

Half-width (single column). Three models spanning the controllability range --
strong / mid / weak, from distinct families -- profiled across the measures that
feed the controllability score S. Top panel: the five d'-scale measures
(Engage, Temporal control, Coverage, Suppress, Layer targeting), each row one
line with the three models' projection d' and 95% CI. Bottom strip: Dial rank,
which is a Spearman rho in [-1,1] and so cannot share the d' axis -- shown on its
own rho axis.

Reads ONLY results/SCORES_<model>.json (the frozen scoring chain). The dots/CIs
are the per-measure MARGINAL two-way (sentence x concept) cluster bootstraps
stored there -- NOT the joint scalar CI -- so a model whose joint SCALAR_CI is
stale (purged main run) is still fine here.

AAAI conventions: no title, brief axis labels, no top/right spines, 300 dpi,
.pdf + .png, caption material to a side-car .md (never on the figure).
"""
import argparse
import json
import os
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths
import roster                                                        # single-source roster

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.gridspec import GridSpec                            # noqa: E402
from model_family_colors import family_color                       # canonical palette

# Three representatives spanning the S range, distinct families (strong/mid/weak).
# Chosen 2026-07-23; swap here to re-cut the figure.
TRIPLE = ["llama_8b", "qwen_72b", "gptoss_20b_low"]
DETAIL = {m: roster.DETAIL_ALL[m] for m in TRIPLE}   # labels from the single-source roster

# d'-scale measures (share the d' axis), ordered later by mean d'. Two-word names
# are stacked on two lines. Dial rank is handled separately (rho scale).
# dial_resolution dropped 2026-08-08: removed from S, so it no longer belongs in a
# profile of the scored measures. It is still computed and kept in SCORES_<model>.json
# as a diagnostic -- it is simply not plotted. Five d' measures remain, plus dial rank.
DPRIME = [("engage", "Engage"), ("temporal_control", "Temporal\nControl"),
          ("coverage", "Coverage"),
          ("suppress", "Suppress"), ("layer_targeting", "Layer\nTargeting")]
FS = 6.5
DODGE = np.linspace(-0.12, 0.12, len(TRIPLE))   # tiny offset: reads as one line, colors separable


def cell(data_root, model, measure):
    """(score, lo, hi) for one measure's proj channel, or (None, None, None)."""
    p = Path(data_root) / f"SCORES_{model}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found (need SCORES_<model>.json for {model})")
    c = (json.load(open(p))["measures"].get(measure, {}).get("channels", {}) or {}).get("proj")
    return (c.get("score"), c.get("lo"), c.get("hi")) if c else (None, None, None)


def render(data_root, out):
    means = [(k, lab, np.nanmean([cell(data_root, m, k)[0] for m in TRIPLE]))
             for k, lab in DPRIME]
    means.sort(key=lambda r: r[2])                       # ascending -> low measures at bottom
    rows = [(k, l) for k, l, _ in means]

    fig = plt.figure(figsize=(3.34, 1.76))               # AAAI single column; height capped at Fig 1
    gs = GridSpec(2, 1, height_ratios=[6, 0.5], hspace=0.95)
    axT = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

    for yi, (k, lab) in enumerate(rows):
        for m, off in zip(TRIPLE, DODGE):
            s, lo, hi = cell(data_root, m, k)
            if s is None:
                continue
            axT.errorbar(s, yi + off, xerr=[[max(s - lo, 0)], [max(hi - s, 0)]], fmt="o",
                         ms=5, color=family_color(m), ecolor=family_color(m),
                         elinewidth=1.0, capsize=1.8, mec="white", mew=0.4, zorder=3)
    axT.axvline(0, color="#888", lw=0.7, ls="--")
    axT.set_yticks(range(len(rows))); axT.set_yticklabels([l for _, l in rows], fontsize=FS)
    axT.set_ylim(-0.45, len(rows) - 0.55); axT.set_xlim(-1.5, 18.5)
    axT.set_xticks([0, 5, 10, 15]); axT.tick_params(labelsize=FS - 0.5)
    axT.set_xlabel("d′", fontsize=FS, labelpad=1.5)
    axT.grid(axis="x", alpha=0.22, lw=0.5)
    axT.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=family_color(m), mec="white",
                      ms=5, label=DETAIL.get(m, m)) for m in TRIPLE]
    axT.legend(handles=handles, loc="lower right", fontsize=FS - 0.5, frameon=False,
               handletextpad=0.3, borderaxespad=0.4, labelspacing=0.3)

    for m, off in zip(TRIPLE, DODGE):
        s, lo, hi = cell(data_root, m, "dial_rank")
        if s is None:
            continue
        axB.errorbar(s, off, xerr=[[max(s - lo, 0)], [max(hi - s, 0)]], fmt="o", ms=5,
                     color=family_color(m), ecolor=family_color(m), elinewidth=1.0,
                     capsize=1.8, mec="white", mew=0.4, zorder=3)
    axB.axvline(0.5, color="#888", lw=0.7, ls="--")
    axB.set_yticks([0]); axB.set_yticklabels(["Dial Rank"], fontsize=FS); axB.set_ylim(-0.3, 0.3)
    axB.set_xlim(0, 1.02); axB.set_xticks([0, 0.5, 1.0]); axB.tick_params(labelsize=FS - 0.5)
    axB.set_xlabel("ρ", fontsize=FS, labelpad=1.5)
    axB.grid(axis="x", alpha=0.22, lw=0.5)
    axB.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.28, right=0.97, top=0.98, bottom=0.14)
    for ext in (out, str(Path(out).with_suffix(".png" if out.endswith(".pdf") else ".pdf"))):
        fig.savefig(ext, bbox_inches="tight")
        print(f"wrote {ext}")
    plt.close(fig)
    return rows


def write_md(data_root, rows, out_md):
    order = [DETAIL.get(m, m) for m in TRIPLE]
    L = [f"# {Path(out_md).stem} — caption material\n",
         "*Auto-emitted; not on the figure.* Half-width (single column). Per-measure "
         "**projection $d'$** with 95% CI for three models spanning the controllability "
         f"range: **{order[0]}** (strong), **{order[1]}** (mid), **{order[2]}** (weak). "
         "Top panel: the five $d'$-scale measures that feed the controllability score "
         "$S$ (Engage, Temporal control, Coverage, Suppress, Layer targeting), "
         "ordered by mean $d'$; each model is one dot per measure "
         "(small vertical offset only to separate colors). Bottom strip: **Dial rank** "
         "(Spearman $\\rho\\in[-1,1]$, a different scale, shown on its own $\\rho$ axis; "
         "dashed line = chance). CIs are the per-measure **marginal** two-way (sentence "
         "$\\times$ concept) cluster bootstraps ($B=2000$). The three share a profile "
         "shape — strong engagement, an ordered dial, end-dominant temporal control — "
         "while all three collapse toward zero on Suppression and the designed-null "
         "Layer targeting, the axes where even the strongest model exerts little control.\n",
         "Values (proj $d'$ [95% CI]; Dial rank in $\\rho$):\n"]
    keys = [k for k, _ in DPRIME] + ["dial_rank"]
    for m in TRIPLE:
        parts = []
        for k in keys:
            s, lo, hi = cell(data_root, m, k)
            if s is not None:
                parts.append(f"{k} {s:.2f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"- **{DETAIL.get(m, m)}**: " + "; ".join(parts))
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Per-measure controllability profile figure ().")
    ap.add_argument("--data-root", default=AC_DATA,
                    help="dir holding the SCORES_<model>.json files (default: $AC_DATA)")
    ap.add_argument("--out", default=out("Family_Profile.pdf"))
    args = ap.parse_args()
    if not args.data_root:
        raise SystemExit("set --data-root (or $AC_DATA) to the dir holding SCORES_*.json")
    rows = render(args.data_root, args.out)
    write_md(args.data_root, rows, str(Path(args.out).with_suffix(".md")))


if __name__ == "__main__":
    main()
