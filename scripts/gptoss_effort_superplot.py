#!/usr/bin/env python3
"""gpt-oss-20b effort-sweep figure (Experiment B): controllability vs reasoning effort.

Separate from superplot.py's cross-model panels ON PURPOSE — low/medium/high are
the SAME weights at different harmony reasoning-effort levels (registry note), so
they read as a within-model dose curve, not three panel entries; superplot's
MODELS list carries only gptoss_20b_low.

Same JSON sources and channel convention as superplot.py / snapshot_superplot.py.
Points render for whichever efforts have SCORES_<name>.json — run it now with
low+medium and re-run when high lands; nothing else to update. Compliance for
the x labels is read live from the newest retained main run dir's results.json
(falls back to an unlabeled tick if the raw dir is absent).

Pod-local (/local/ac-gpt20) per the shared-volume rules; rendered output goes to
the shared results/ store: results/gptoss_20b_effort_comparison.png / .pdf
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

EFFORTS = [("gptoss_20b_low", "Low"),
           ("gptoss_20b_medium", "Medium"),
           ("gptoss_20b_high", "High")]

ROWS = [("engage", "Engage  $d'$"),
        ("dial_rank", r"Dial rank  $\rho$"),
        ("dial_resolution", "Dial resolution  $d'$"),
        ("temporal_control", "Temporal control"),
        ("coverage", "Coverage  $d'$"),
        ("scalar", r"Controllability  $S$")]


def live_compliance(root, name):
    """Compliance rate from the newest retained main run dir, or None."""
    dirs = sorted((root / "raw").glob(f"*_{name}_activation_control"), reverse=True)
    for d in dirs:
        rj = d / "results.json"
        if rj.exists():
            try:
                return json.load(open(rj))["metrics"]["compliance_rate"]
            except Exception:
                continue
    return None


def load_points(data_root, channel="proj"):
    root = Path(data_root)
    pts, comp = {}, {}
    for name, _ in EFFORTS:
        comp[name] = live_compliance(root, name)
        sp = root / f"SCORES_{name}.json"
        if not sp.exists():
            print(f"  [note] no {sp.name} yet; point skipped")
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
        row["scalar"] = None
        if cp.exists():
            d = json.load(open(cp))
            row["scalar"] = (d.get("scalar"), d.get("ci_lo"), d.get("ci_hi"))
        pts[name] = row
    return pts, comp


def render(pts, comp, out_stem, channel):
    col = family_color("GPT-OSS")
    x = np.arange(len(EFFORTS), dtype=float)
    n = len(ROWS)
    fig, axes = plt.subplots(n, 1, figsize=(4.6, 1.35 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    plt.subplots_adjust(left=0.20, right=0.96, top=1 - 0.4 / (1.35 * n),
                        bottom=0.10, hspace=0.25)
    for ax, (key, ylab) in zip(axes, ROWS):
        xs, ys, los, his = [], [], [], []
        for xi, (name, _) in zip(x, EFFORTS):
            t = pts.get(name, {}).get(key)
            if t is None or t[0] is None:
                continue
            xs.append(xi); ys.append(t[0])
            los.append(t[0] - t[1] if t[1] is not None else 0)
            his.append(t[2] - t[0] if t[2] is not None else 0)
        ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=5, zorder=3)
        ax.errorbar(xs, ys, yerr=[np.maximum(los, 0), np.maximum(his, 0)],
                    fmt="none", ecolor="#222", elinewidth=0.8, capsize=2, zorder=2)
        ax.axhline(0, color="#888", lw=0.6)
        if ys:  # same degenerate-CI clipping rule as snapshot_superplot.py
            span = (max(ys) - min(ys)) or max(abs(max(ys)), 1e-6)
            sane_lo = [y - l for y, l in zip(ys, los) if l <= 3 * span]
            sane_hi = [y + h for y, h in zip(ys, his) if h <= 3 * span]
            ax.set_ylim(min([min(ys)] + sane_lo + [0]) - 0.15 * span,
                        max([max(ys)] + sane_hi) + 0.15 * span)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.10)
        ax.spines[["top", "right"]].set_visible(False)
    labels = []
    for name, lab in EFFORTS:
        c = comp.get(name)
        labels.append(f"{lab}\n[{100*c:.0f}%]" if c is not None else lab)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[-1].set_xlabel("Harmony reasoning effort   [main-run compliance]", fontsize=8.5)
    fig.suptitle(f"gpt-oss-20b — controllability vs reasoning effort  (channel: {channel})",
                 fontsize=10.5, fontweight="bold", y=0.998)
    for ext in ("png", "pdf"):
        out = f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="gpt-oss-20b reasoning-effort comparison figure (Experiment B).")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--channel", default="proj")
    args = ap.parse_args()
    pts, comp = load_points(args.data_root, args.channel)
    render(pts, comp, str(Path(args.data_root) / "gptoss_20b_effort_comparison"), args.channel)


if __name__ == "__main__":
    main()
