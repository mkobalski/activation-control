#!/usr/bin/env python3
"""Lexical_Dial — Depth_Dial's dial-RANK panel, for the LEXICAL intensifier, three models.

The Dial-rank curve of Depth_Dial (mean signed Spearman rho vs network depth) computed on
the LEXICAL manipulation (think about -> think intensely about, J=2) instead of the numeric
ramp, for three models spanning the lexical-dial range, high -> medium -> low (top to bottom):
  (a) GLM 4.7 Flash         -- lexical intensifier SUCCEEDS (rank CI above 0)
  (b) Gemma 3 27B           -- ambiguous (rank CI straddles 0)
  (c) Mistral Small 3.1 24B -- lexical intensifier REVERSES (rank CI below 0)
The three are similar in size (24-31B) to isolate lexical behavior from scale. The Dial-rank
axis is fixed to [-1, 1] and shared across panels for comparability.

Only Dial RANK is shown: for a J=2 lexical dial the (genuine, single adjacent-pair) Dial
resolution d' is a monotone function of the same win-rate as rank, so it merely duplicates the
rank curve -- rank alone carries the signal. (This is also why S scores the dial from the
numeric ramp, whose 1v2/2v3/3v4 resolution is genuinely distinct from rank; see Depth_Dial.)
Reads the frozen rank_lexical curve from PROFILES_<model>.json.
"""
import argparse
import os
import sys
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engage_suppress as es                                         # frozen PROFILES loader
import think_intensity as ti                                         # dial colours

# high -> medium -> low lexical-dial performer (top to bottom); all ~24-31B (size-matched).
MODELS = ["glm47_flash", "gemma3_27b", "mistral_small_31_24b"]


def render(data_root, out):
    es.use_data_root(data_root)
    fig, axes = plt.subplots(3, 1, figsize=(3.34, 4.8), sharex=True)         # a/b/c stacked
    fig.subplots_adjust(left=0.155, right=0.96, top=0.965, bottom=0.075, hspace=0.32)
    depth = None
    for i, m in enumerate(MODELS):
        axL = axes[i]
        depth, r_m, r_lo, r_hi = es.load_profile(m, "rank_lexical", ch="proj")
        xi = np.arange(len(depth))
        axL.set_ylim(-1.05, 1.05)                                        # shared rank axis
        axL.axhline(0, color="#bbb", lw=0.6, ls="--", zorder=0)
        axL.fill_between(xi, r_lo, r_hi, color=ti.ORANGE, alpha=0.12, lw=0, zorder=1)
        axL.plot(xi, r_m, color=ti.ORANGE, lw=1.3, marker="o", ms=2.2, label="Dial Rank", zorder=4)
        axL.set_yticks([-1, -0.5, 0, 0.5, 1])
        axL.tick_params(axis="y", labelsize=6)
        axL.spines[["top", "right"]].set_visible(False)
        axL.text(-0.01, 1.04, f"({'abc'[i]}) {es.DETAIL[m]}", transform=axL.transAxes,
                 fontsize=7, fontweight="bold", ha="left", va="bottom")
    want = [10, 30, 50, 70, 90]                                          # depth axis on shared bottom
    tix = [depth.index(v) for v in want if v in depth]
    axes[-1].set_xticks(tix); axes[-1].set_xticklabels([str(depth[j]) for j in tix], fontsize=6)
    axes[-1].set_xlabel("Depth (%)", fontsize=7, labelpad=1.5)
    axes[1].set_ylabel("Dial rank  $\\rho$", fontsize=7.5)               # centred on the middle panel

    svg = str(Path(out).with_suffix(".svg"))
    fig.savefig(svg, bbox_inches="tight"); print(f"wrote {svg}")
    es._savefig(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Lexical-intensifier dial (rank+resolution vs depth), three models.")
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Lexical_Dial.png"))
    args = ap.parse_args()
    render(args.data_root, args.out)
    Path(str(Path(args.out).with_suffix(".md"))).write_text(
        f"# {Path(args.out).stem} — caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width column, three panels stacked vertically "
        "(a/b/c, top to bottom). The **Dial rank** curve of "
        "Figure 4 (mean signed Spearman $\\rho$ vs network depth, orange) for the **lexical** "
        "intensifier (think about $\\to$ think intensely about) instead of the numeric ramp, for "
        "three **size-matched** models (24--31B), high to low: "
        "**(a)** GLM 4.7 Flash, where it **succeeds** (rank $\\approx+0.59$, CI above zero); "
        "**(b)** Gemma 3 27B, **ambiguous** (rank $\\approx+0.37$, CI straddles zero); "
        "**(c)** Mistral Small 3.1 24B, where the intensifier **reverses** (rank $\\approx-0.90$, CI "
        "below zero -- ``think intensely'' makes the concept weaker). The Dial-rank axis is fixed to $[-1,1]$ "
        "and shared across panels. Bands are 95% two-way (sentence $\\times$ concept) "
        "cluster-bootstrap CIs. Only Dial rank is shown: for a $J{=}2$ lexical dial the genuine "
        "single adjacent-pair Dial resolution $d'$ is a monotone function of the same win-rate as "
        "rank and simply duplicates this curve. The cross-model inconsistency of the lexical "
        "intensifier is why $S$ scores the dial from the numeric ramp alone.\n")
    print("wrote", str(Path(args.out).with_suffix(".md")))


if __name__ == "__main__":
    main()
