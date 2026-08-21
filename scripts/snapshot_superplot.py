#!/usr/bin/env python3
"""Experiment C figure: Olmo 3 7B controllability across training snapshots.

Separate from superplot.py's cross-model panels ON PURPOSE — the snapshot lane
(scripts/olmo_snapshot_lane.sh) is a within-model training-time curve, not part
of the retained model panel; the roster in figure_scripts/roster.py excludes the
snapshot short names, and this script mirrors that separation.

Reads the same precomputed JSONs (SCORES_/SCALAR_CI_/ONSET_OFFSET_WORD_) as
superplot.py, same channel convention (default: proj). One line-plot row per
MAIN measure plus the controllability scalar S, x = training stage in temporal
order, 95% CI whiskers, per-stage compliance noted in the x labels.

Output: results/olmo3_7b_snapshot_comparison.png / .pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
import matplotlib.pyplot as plt                                          # noqa: E402

from model_family_colors import family_color                             # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Temporal order of the six points. Compliance (main run, batched greedy) is
# stamped from the run logs of 2026-07-23 (snapshots) / the retained run dir's
# metrics (final Instruct, 2026-07-22).
STAGES = [
    ("olmo3_7b_s1_700k",  "Stage 1\n700k steps",        74.7),
    ("olmo3_7b_s1_final", "Stage 1\nfinal (1.41M)",     82.6),
    ("olmo3_7b_base",     "Base\n(post mid-train)",     99.5),
    ("olmo3_7b_sft",      "SFT",                        90.0),
    ("olmo3_7b_dpo",      "DPO",                        41.2),
    ("olmo3_7b",          "Instruct\n(RLVR)",           68.4),
]

ROWS = [("engage", "Engage  $d'$"),
        ("dial_rank", r"Dial rank  $\rho$"),
        ("dial_resolution", "Dial resolution  $d'$"),
        ("temporal_control", "Temporal control"),
        ("coverage", "Coverage  $d'$"),
        ("scalar", r"Controllability  $S$")]


def load_points(data_root, channel="proj"):
    """{model: {metric: (score, lo, hi) or None}} for the six snapshot points."""
    root = Path(data_root)
    pts = {}
    for name, _, _ in STAGES:
        sp = root / f"SCORES_{name}.json"
        if not sp.exists():
            print(f"  [warn] missing {sp.name}; point will be skipped")
            pts[name] = {}
            continue
        meas = json.load(open(sp))["measures"]
        row = {}
        for key, _ in ROWS:
            if key == "scalar":
                continue
            ch = (meas.get(key, {}).get("channels", {}) or {}).get(channel)
            row[key] = ((ch.get("score"), ch.get("lo"), ch.get("hi"))
                        if ch and ch.get("score") is not None else None)
        cp = root / f"SCALAR_CI_{name}.json"
        if cp.exists():
            d = json.load(open(cp))
            row["scalar"] = (d.get("scalar"), d.get("ci_lo"), d.get("ci_hi"))
        else:
            row["scalar"] = None
        pts[name] = row
    return pts


def render(pts, out_stem, channel):
    col = family_color("Olmo")
    x = np.arange(len(STAGES), dtype=float)
    n = len(ROWS)
    fig, axes = plt.subplots(n, 1, figsize=(6.2, 1.35 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    plt.subplots_adjust(left=0.16, right=0.97, top=1 - 0.4 / (1.35 * n),
                        bottom=0.16, hspace=0.25)
    for ax, (key, ylab) in zip(axes, ROWS):
        xs, ys, los, his = [], [], [], []
        for xi, (name, _, _) in zip(x, STAGES):
            t = pts.get(name, {}).get(key)
            if t is None or t[0] is None:
                continue
            xs.append(xi); ys.append(t[0])
            los.append(t[0] - t[1] if t[1] is not None else 0)
            his.append(t[2] - t[0] if t[2] is not None else 0)
        ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=4.5, zorder=3)
        ax.errorbar(xs, ys, yerr=[np.maximum(los, 0), np.maximum(his, 0)],
                    fmt="none", ecolor="#222", elinewidth=0.8, capsize=2, zorder=2)
        ax.axhline(0, color="#888", lw=0.6)
        # Low-compliance points (DPO, 41%) can have near-degenerate bootstrap CIs
        # Clip the axis to the data + well-behaved CIs so one exploding
        # whisker can't flatten the row; the clipped whisker runs off-axis.
        if ys:
            span = (max(ys) - min(ys)) or max(abs(max(ys)), 1e-6)
            sane_lo = [y - l for y, l in zip(ys, los) if l <= 3 * span]
            sane_hi = [y + h for y, h in zip(ys, his) if h <= 3 * span]
            lo_lim = min([min(ys)] + sane_lo + [0]) - 0.08 * span
            hi_lim = max([max(ys)] + sane_hi) + 0.08 * span
            ax.set_ylim(lo_lim, hi_lim)
        # visually separate pretraining-only points from post-training
        ax.axvline(2.5, color="#bbb", lw=0.8, ls=":")
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.06)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(1.0, 1.14, "pre-training", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color="#666", style="italic")
    axes[0].text(4.0, 1.14, "post-training", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8, color="#666", style="italic")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{lab}\n[{c:.0f}%]" for _, lab, c in STAGES], fontsize=7.5)
    axes[-1].set_xlabel("Training stage   [main-run compliance]", fontsize=8.5)
    fig.text(0.16, 0.005, "CI whiskers exceeding 3× the row's score range are clipped "
             "(low-compliance bootstrap degeneracy; affects DPO).",
             fontsize=6.5, color="#666", style="italic")
    fig.suptitle(f"Olmo 3 7B — controllability across training  (channel: {channel})",
                 fontsize=11, fontweight="bold", y=0.998)
    for ext in ("png", "pdf"):
        out = f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Olmo 3 7B training-snapshot comparison figure (Experiment C).")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--channel", default="proj")
    args = ap.parse_args()
    pts = load_points(args.data_root, args.channel)
    render(pts, str(Path(args.data_root) / "olmo3_7b_snapshot_comparison"), args.channel)


if __name__ == "__main__":
    main()
