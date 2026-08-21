#!/usr/bin/env python3
"""Channels_Depth — engage/suppress d' vs depth, per readout channel (Figure 12).

The Figure-5a depth profile (engage red / suppress blue sensitivity d' vs network depth,
focal gemma3_27b) stacked in one column for the three readout channels, on a SHARED d' axis:
  (a) Cosine        (direction)   -- d' of cos(residual, unit concept vector)
  (b) Relative norm (magnitude)   -- d' of ||r|| / mean content-token norm
  (c) Projection                  -- d' of ||r||*cos  (identical to Figure 5a)
Engagement is carried by the cosine (direction) channel; the relative-norm (magnitude)
channel is small. BOTH curves are conventional d' = (instructed - baseline)/sigma, so
positive = the readout ROSE under the instruction (suppress is negated from the stored
sign-flipped score). Suppression fails to drive the concept below baseline and mildly
rebounds, so it sits at/above zero rather than dipping negative.

Reads the frozen per-channel curves from PROFILES_<model>.json via engage_suppress.
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

CHANNELS = [("cos", "Cosine  (direction)"),
            ("relnorm", "Relative norm  (magnitude)"),
            ("proj", "Projection")]


def render(data_root, focal, out):
    es.use_data_root(data_root)
    fig, axes = plt.subplots(3, 1, figsize=(3.34, 5.2), sharex=True, sharey=True)  # a/b/c stacked
    fig.subplots_adjust(left=0.135, right=0.965, top=0.965, bottom=0.075, hspace=0.28)
    depth = None
    for i, (ax, (ch, chlab)) in enumerate(zip(axes, CHANNELS)):
        depth, me, le, he = es.load_profile(focal, "engage", ch=ch)
        _, ms0, ls0, hs0 = es.load_profile(focal, "suppress", ch=ch)
        # CONVENTIONAL d' = (instructed - baseline)/sigma for both: the stored suppress is the
        # sign-FLIPPED score (-raw); negate it (and swap the band) so positive = readout ROSE.
        ms = -np.asarray(ms0, float); ls, hs = -np.asarray(hs0, float), -np.asarray(ls0, float)
        xi = np.arange(len(depth))
        for mean, lo, hi, col, lab in ((me, le, he, es.ENGAGE_C, "Engage"),
                                       (ms, ls, hs, es.SUPPRESS_C, "Suppress")):
            ax.plot(xi, mean, color=col, lw=1.3, marker="o", ms=2.2, label=lab, zorder=3)
            ax.fill_between(xi, lo, hi, color=col, alpha=0.15, zorder=2)
        ax.axhline(0, color="#888", lw=0.7, zorder=1)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.01, 1.04, f"({'abc'[i]}) {chlab}", transform=ax.transAxes, fontsize=7.5,
                fontweight="bold", ha="left", va="bottom")
    want = [10, 30, 50, 70, 90]                                          # depth axis on shared bottom
    tix = [depth.index(v) for v in want if v in depth]
    axes[-1].set_xticks(tix); axes[-1].set_xticklabels([str(depth[j]) for j in tix], fontsize=6)
    axes[-1].set_xlabel("Depth (%)", fontsize=7.5, labelpad=1.5)
    axes[1].set_ylabel("$d'$", fontsize=9)                              # centred on the middle panel
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.3,
                   labelspacing=0.25, borderaxespad=0.3)

    svg = str(Path(out).with_suffix(".svg"))
    fig.savefig(svg, bbox_inches="tight"); print(f"wrote {svg}")
    es._savefig(fig, out)


def main():
    ap = argparse.ArgumentParser(description="Per-channel engage/suppress d'-vs-depth decomposition of Fig 4a.")
    ap.add_argument("--model", default="gemma3_27b")
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Channels_Depth.png"))
    args = ap.parse_args()
    render(args.data_root, args.model, args.out)
    Path(str(Path(args.out).with_suffix(".md"))).write_text(
        f"# {Path(args.out).stem} — caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width column, three panels stacked vertically "
        "(a/b/c, top to bottom), shared $d'$ axis. The "
        f"Figure-4a depth profile (**{args.model}**: engage/suppress sensitivity $d'$ vs network "
        "depth, averaged over 50 sentences x 10 concepts) shown for the three readout channels: "
        "**(a)** cosine cos(r, ĉ) (direction), **(b)** relative norm ‖r‖ / mean content-token norm "
        "(magnitude), **(c)** projection ‖r‖·cos (identical to Figure 4a). Engagement (red) is "
        "carried by the cosine channel (peak $d'\\approx 8.5$) while the relative-norm channel is "
        "small ($\\approx 1.4$); the projection ($\\approx 6.1$) tracks the cosine. Both are "
        "conventional $d'=(\\text{instructed}-\\text{baseline})/\\sigma$ (positive = readout rose "
        "under the instruction); suppress (blue) fails to push the concept below baseline and "
        "mildly rebounds, so it sits at/above zero. "
        "Bands are 95% two-way (sentence x concept) cluster-bootstrap CIs.\n")
    print("wrote", str(Path(args.out).with_suffix(".md")))


if __name__ == "__main__":
    main()
