#!/usr/bin/env python3
"""Experiment C combined figure: Olmo 7B AND 3.1-32B controllability across training.

Overlays the two training-time curves on shared STAGE axes (the families have
different pretraining step counts, so align by stage, not absolute step):
    S1 early -> S1 final -> Base -> SFT -> DPO -> Instruct

Measure rows come from SCORES_<model>.json (identical scorer for every point).
The controllability scalar S row is special: the shared scalar_ci.py was changed
2026-07-23 to fold layer_targeting into S (7-component def), and the two INSTRUCT
anchors were re-scored under it — but the snapshots' raw is pruned and cannot be
recomputed, so they are locked to the OLD 5-component S. For a consistent curve
this script reads the anchors' OLD-def S from a private recompute
(results_private/SCALAR_CI_<anchor>_oldS.json) instead of the shared file.

Pod-local (/local/ac-gpt20); output -> shared results/olmo_snapshot_comparison_7b_32b.{png,pdf}
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
import matplotlib.pyplot as plt                                          # noqa: E402

from model_family_colors import family_color, family_shades              # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE = PROJECT_ROOT / "results_private"

STAGE_ORDER = ["Stage 1\nearly", "Stage 1\nfinal", "Base\n(post mid-train)",
               "SFT", "DPO", "Instruct"]

# (label, size, [model short name per stage], anchor_short_name)
SERIES = [
    ("Olmo 3 7B", 7,
     ["olmo3_7b_s1_700k", "olmo3_7b_s1_final", "olmo3_7b_base",
      "olmo3_7b_sft", "olmo3_7b_dpo", "olmo3_7b"], "olmo3_7b"),
    ("Olmo 3.1 32B", 32,
     ["olmo31_32b_s1_328k", "olmo31_32b_s1_final", "olmo31_32b_base",
      "olmo31_32b_sft", "olmo31_32b_dpo", "olmo31_32b"], "olmo31_32b"),
]

ROWS = [("engage", "Engage  $d'$"),
        ("dial_rank", r"Dial rank  $\rho$"),
        ("dial_resolution", "Dial resolution  $d'$"),
        ("temporal_control", "Temporal control"),
        ("coverage", "Coverage  $d'$"),
        ("scalar", r"Controllability  $S$")]

# New-def scalar S (2026-07-23 canonical: 7-component geomean incl. layer_targeting).
# Reconstructed from the stored per-measure POINT scores in SCORES_<model>.json —
# validated to reproduce the anchors' shared-file S exactly. Snapshots' raw is
# pruned so the joint bootstrap CI is unavailable; the S row is POINTS ONLY.
_EPS = 1e-6
_DREF = {"engage": 8.0, "suppress": 3.0, "dial_resolution": 3.0,
         "temporal_control": 5.0, "coverage": 1.5, "layer_targeting": 5.0}
_KEPT = ["engage", "suppress", "dial_rank", "dial_resolution",
         "temporal_control", "coverage", "layer_targeting"]


def _new_scalar(scores_path, channel="proj"):
    """New-def S point from stored per-measure scores, or None if any is missing."""
    if not scores_path.exists():
        return None
    meas = json.load(open(scores_path))["measures"]
    logs = []
    for k in _KEPT:
        s = (meas.get(k, {}).get("channels", {}) or {}).get(channel, {}).get("score")
        if s is None:
            return None
        p = (s + 1) / 2 if k == "dial_rank" else 0.5 + s / (2 * _DREF[k])
        logs.append(np.log(np.clip(p, _EPS, 1 - _EPS)))
    return float(np.clip(2 * np.exp(np.mean(logs)) - 1, 0.0, 1.0))


def load_series(root, models, anchor, channel="proj"):
    """[{metric: (score,lo,hi)}] per stage. Anchor's scalar comes from the private
    OLD-def recompute so it matches the snapshots' 5-component S."""
    root = Path(root)
    out = []
    for name in models:
        row = {}
        sp = root / f"SCORES_{name}.json"
        if sp.exists():
            meas = json.load(open(sp))["measures"]
            for key, _ in ROWS:
                if key == "scalar":
                    continue
                ch = (meas.get(key, {}).get("channels", {}) or {}).get(channel)
                row[key] = ((ch.get("score"), ch.get("lo"), ch.get("hi"))
                            if ch and ch.get("score") is not None else None)
        # scalar: NEW-def point (7-component), reconstructed from SCORES for every
        # point so the whole curve is one consistent definition. No CI — the joint
        # bootstrap needs per-trial raw, pruned for snapshots (lo=hi=None).
        s = _new_scalar(root / f"SCORES_{name}.json", channel)
        row["scalar"] = (s, None, None) if s is not None else None
        out.append(row)
    return out


def render(series_data, out_stem, channel):
    shades = family_shades("Olmo", 2)  # size-graded: 7B lighter, 32B darker
    markers = ["o", "s"]
    x = np.arange(len(STAGE_ORDER), dtype=float)
    n = len(ROWS)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 1.4 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    plt.subplots_adjust(left=0.16, right=0.97, top=1 - 0.5 / (1.4 * n),
                        bottom=0.14, hspace=0.25)
    for ax, (key, ylab) in zip(axes, ROWS):
        allys = []
        for (label, size, models, anchor), data, sh, mk in zip(
                SERIES, series_data, shades, markers):
            xs, ys, los, his = [], [], [], []
            for xi, row in zip(x, data):
                t = row.get(key)
                if t is None or t[0] is None:
                    continue
                xs.append(xi); ys.append(t[0])
                los.append(t[0] - t[1] if t[1] is not None else 0)
                his.append(t[2] - t[0] if t[2] is not None else 0)
            ax.plot(xs, ys, "-", color=sh, lw=1.6, marker=mk, ms=5,
                    label=label, zorder=3)
            if key != "scalar":  # S row is point-only (no joint CI without raw)
                ax.errorbar(xs, ys, yerr=[np.maximum(los, 0), np.maximum(his, 0)],
                            fmt="none", ecolor="#222", elinewidth=0.7, capsize=2, zorder=2)
            allys += ys
        ax.axhline(0, color="#888", lw=0.6)
        ax.axvline(2.5, color="#bbb", lw=0.8, ls=":")  # pre/post-training divide
        # clip degenerate CIs (low-compliance bootstrap; 7B DPO, 32B S1-early)
        if allys:
            span = (max(allys) - min(allys)) or max(abs(max(allys)), 1e-6)
            lo_lim, hi_lim = min(allys + [0]), max(allys)
            for (_, _, _, _), data in zip(SERIES, series_data):
                t = data and None
            # re-walk to gather sane CI extents
            sane = []
            for data in series_data:
                for row in data:
                    t = row.get(key)
                    if not t or t[0] is None:
                        continue
                    if t[1] is not None and (t[0] - t[1]) <= 3 * span:
                        sane.append(t[1])
                    if t[2] is not None and (t[2] - t[0]) <= 3 * span:
                        sane.append(t[2])
            if sane:
                lo_lim = min([lo_lim] + sane); hi_lim = max([hi_lim] + sane)
            ax.set_ylim(lo_lim - 0.1 * span, hi_lim + 0.1 * span)
        ax.set_ylabel(ylab, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.margins(x=0.06)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    axes[0].text(1.0, 1.02, "pre-training", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.5, color="#888", style="italic")
    axes[0].text(4.0, 1.02, "post-training", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.5, color="#888", style="italic")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(STAGE_ORDER, fontsize=7.5)
    axes[-1].set_xlabel("Training stage  (7B S1: 700k/1.41M · 32B S1: 328k/656k steps)",
                        fontsize=8)
    fig.suptitle(f"Olmo controllability across training — 7B vs 3.1-32B  (channel: {channel})",
                 fontsize=11, fontweight="bold", y=0.999)
    fig.text(0.16, 0.004, "Measure rows: 95% cluster-bootstrap CIs. Controllability S: "
             "canonical 7-component def (POINT estimates, no error bars — joint bootstrap "
             "CI needs per-trial raw, pruned for snapshots).",
             fontsize=6.3, color="#666", style="italic")
    for ext in ("png", "pdf"):
        out = f"{out_stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Combined Olmo 7B + 3.1-32B training-snapshot figure.")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "results"))
    ap.add_argument("--channel", default="proj")
    args = ap.parse_args()
    series_data = [load_series(args.data_root, models, anchor, args.channel)
                   for (_, _, models, anchor) in SERIES]
    render(series_data, str(Path(args.data_root) / "olmo_snapshot_comparison_7b_32b"),
           args.channel)


if __name__ == "__main__":
    main()
