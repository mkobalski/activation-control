#!/usr/bin/env python3
"""Depth_Dial — half-width horizontal two-panel figure (Figure 5).

One AAAI single-column row, two panels (gemma3_27b focal, `--model` tunable):
  (a) engage / suppress sensitivity d' vs network depth.
  (b) numeric-intensity dial: Dial rank (Spearman rho, orange) vs depth.

Both panels are frozen PROFILES_<model>.json curves. Depth data + styling come from the
sibling engage_suppress (es); the dial colour and the axis-limit helper come from
think_intensity (ti) so nothing duplicates or drifts.

AAAI-compliant (TrueType, 300 dpi, no top/right spines, no gridlines, no titles).
"""
import argparse
import os
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.gridspec import GridSpec                                 # noqa: E402

import engage_suppress as es                                         # depth data + styling
import think_intensity as ti                                         # dial colours + _aligned_limits
from roster import FOCAL                                              # single-source focal model


def render(focal, out):
    # engage/suppress depth curves straight from PROFILES_<model>.json (frozen); no run
    # dir / raw access -- the old es.focal_profiles only touched raw to enumerate layers,
    # which this figure never used.
    depth, mean_e, lo_e, hi_e = es.load_profile(focal, "engage")
    _, mean_s, lo_s, hi_s = es.load_profile(focal, "suppress")
    xi = np.arange(len(depth))
    # dial curves (numeric ramp 1->4)
    xr, r_m, r_lo, r_hi = es.load_profile(focal, "rank_numeric")
    xr = np.asarray(xr, float)
    rb, ra = max(0, -np.nanmin(r_lo)), np.nanmax(r_hi)
    (rl, rh), = ti._aligned_limits([(rb, ra)])

    fig = plt.figure(figsize=(3.34, 1.5))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.06], wspace=0.5)
    axA = fig.add_subplot(gs[0]); axC = fig.add_subplot(gs[1])

    # ---- (a) engage / suppress d' vs depth ----
    # CONVENTIONAL d' = (instructed - baseline)/sigma for BOTH: positive = concept readout
    # rose under the instruction. Engage is stored raw (+). The stored `suppress` curve is the
    # sign-FLIPPED score (-raw, "success-positive"); negate it here (and swap the band) so it,
    # too, reads as raw d'. Suppression mildly backfires, so it now sits >=0, overlaying engage
    # -- instead of dipping below zero, which misleads readers into thinking it "worked".
    ms = -np.asarray(mean_s, float)
    sl, sh = -np.asarray(hi_s, float), -np.asarray(lo_s, float)  # negate+swap
    for mean, lo, hi, col, lab in (
            (mean_e, lo_e, hi_e, es.ENGAGE_C, "Engage"),
            (ms, sl, sh, es.SUPPRESS_C, "Suppress")):
        axA.plot(xi, mean, color=col, lw=1.2, marker="o", ms=1.9, label=lab)
        axA.fill_between(xi, lo, hi, color=col, alpha=0.15)
    axA.axhline(0, color="#888", lw=0.6)
    want = [50, 100]
    tix = [depth.index(v) for v in want if v in depth]
    axA.set_xticks(tix); axA.set_xticklabels([str(depth[i]) for i in tix], fontsize=6.5)
    axA.set_xlabel("Depth (%)", fontsize=6.5, labelpad=1.0)
    axA.set_ylabel("$d'$", fontsize=6.5, labelpad=1.0)
    axA.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.0,
               labelspacing=0.18, borderaxespad=0.2)
    axA.tick_params(axis="y", labelsize=6.5); es._nospine(axA)

    # ---- (b) dial rank vs depth ----
    # The Dial resolution curve (and its twin right-hand axis) was dropped 2026-08-08:
    # the measure is no longer part of S, so plotting it beside Dial rank implied a
    # scored quantity. Single axis now -- no twinx, no aligned-zero machinery needed.
    axC.set_ylim(rl, rh)
    axC.axhline(0, color="#bbb", lw=0.6, ls="--", zorder=0)
    axC.fill_between(xr, r_lo, r_hi, color=ti.ORANGE, alpha=0.12, lw=0, zorder=1)
    axC.plot(xr, r_m, color=ti.ORANGE, lw=1.2, marker="o", ms=1.9, label="Dial Rank", zorder=4)
    axC.set_xticks([50, 100])
    axC.set_yticks([t for t in np.round(np.arange(-2.0, 2.001, 0.2), 1) if rl - 1e-9 <= t <= rh + 1e-9])
    axC.set_xlabel("Depth (%)", fontsize=6.5, labelpad=1.0)
    axC.set_ylabel("Dial Rank  $\\rho$", fontsize=6.5, labelpad=1.0)
    axC.tick_params(axis="y", labelsize=6.5)
    axC.tick_params(axis="x", labelsize=6.5)
    axC.spines[["top", "right"]].set_visible(False)
    axC.legend(frameon=False, fontsize=6.5, loc="upper left",
               handlelength=1.1, labelspacing=0.22)

    for ax, lab in ((axA, "a"), (axC, "b")):
        ax.text(-0.02, 1.06, f"({lab})", transform=ax.transAxes, fontsize=6.5,
                fontweight="bold", ha="right", va="bottom")

    fig.subplots_adjust(left=0.115, right=0.9, top=0.9, bottom=0.2)
    svg = str(Path(out).with_suffix(".svg"))
    fig.savefig(svg, bbox_inches="tight"); print(f"wrote {svg}")
    es._savefig(fig, out)


def write_md(focal, out_md):
    body = (
        f"# {Path(out_md).stem} — caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width, two panels, projection channel; focal "
        f"model **{focal}**. **(a)** Engage (red) & suppress (blue) sensitivity $d'$ vs network "
        "depth, both as conventional $d'=(\\text{instructed}-\\text{baseline})/\\sigma$ (positive = "
        "concept readout rose under the instruction). Engage rises steeply; suppression fails to "
        "drive the concept below baseline and mildly rebounds, so both curves sit $\\geq 0$. "
        "**(b)** Numeric intensity ramp (1->4) dial: **Dial rank** (mean signed Spearman "
        "$\\rho$; orange) vs depth, dashed line = chance. "
        "Bands are 95% two-way (sentence x concept) cluster-bootstrap CIs.\n")
    open(out_md, "w").write(body)
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Half-width depth + dial figure (Fig 5).")
    ap.add_argument("--model", default=FOCAL, help="focal model (default: roster.FOCAL / $AC_FOCAL)")
    ap.add_argument("--out", default=out("Depth_Dial.png"))
    ap.add_argument("--data-root", default=AC_DATA,
                    help="data dir holding raw/ + PROFILES_ (default: $AC_DATA)")
    args = ap.parse_args()
    es.use_data_root(args.data_root)
    render(args.model, args.out)
    write_md(args.model, str(Path(args.out).with_suffix(".md")))


if __name__ == "__main__":
    main()
