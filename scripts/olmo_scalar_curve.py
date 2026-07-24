#!/usr/bin/env python3
"""Single-panel Experiment C figure: Controllability S across training, 7B vs 3.1-32B.

Just the canonical 7-component S (point estimates), two Olmo curves aligned by
stage. S reconstructed from stored per-measure scores in SCORES_<model>.json
(validated to reproduce the shared-file S). No error bars — snapshots' raw is
pruned, so no joint bootstrap CI. Output: results/olmo_scalar_curve_7b_32b.{png,pdf}
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
import matplotlib.pyplot as plt                                          # noqa: E402
from model_family_colors import family_shades                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results"
STAGES = ["Stage 1\nearly", "Stage 1\nfinal", "Base\n(mid-train)", "SFT", "DPO", "Instruct"]
SERIES = [
    ("Olmo 3 7B", ["olmo3_7b_s1_700k", "olmo3_7b_s1_final", "olmo3_7b_base",
                   "olmo3_7b_sft", "olmo3_7b_dpo", "olmo3_7b"]),
    ("Olmo 3.1 32B", ["olmo31_32b_s1_328k", "olmo31_32b_s1_final", "olmo31_32b_base",
                      "olmo31_32b_sft", "olmo31_32b_dpo", "olmo31_32b"]),
]
_EPS = 1e-6
# MUST match aggregate_scalar.D_REF (engage/coverage recalibrated 2026-07-24).
_DREF = {"engage": 16.4, "suppress": 3.0, "dial_resolution": 3.0,
         "temporal_control": 5.0, "coverage": 4.11, "layer_targeting": 5.0}
_KEPT = ["engage", "suppress", "dial_rank", "dial_resolution",
         "temporal_control", "coverage", "layer_targeting"]


def new_scalar(model, channel="proj"):
    p = ROOT / f"SCORES_{model}.json"
    if not p.exists():
        return None
    meas = json.load(open(p))["measures"]
    logs = []
    for k in _KEPT:
        s = (meas.get(k, {}).get("channels", {}) or {}).get(channel, {}).get("score")
        if s is None:
            return None
        pk = (s + 1) / 2 if k == "dial_rank" else 0.5 + s / (2 * _DREF[k])
        logs.append(np.log(np.clip(pk, _EPS, 1 - _EPS)))
    return float(np.clip(2 * np.exp(np.mean(logs)) - 1, 0.0, 1.0))


def main():
    shades = family_shades("Olmo", 2)
    markers = ["o", "s"]
    x = np.arange(len(STAGES), dtype=float)
    fig, ax = plt.subplots(figsize=(3.34, 2.7))                       # AAAI single column
    for (label, models), sh, mk in zip(SERIES, shades, markers):
        ys = [new_scalar(m) for m in models]
        ax.plot(x, ys, "-", color=sh, lw=1.6, marker=mk, ms=5, label=label, zorder=3)
    ax.axvline(2.5, color="#bbb", lw=0.9, ls=":")
    ax.text(1.0, 0.96, "pre-training", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5, color="#aaa", style="italic")
    ax.text(4.0, 0.96, "post-training", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5, color="#aaa", style="italic")
    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=6.0)
    ax.set_ylabel("Controllability score", fontsize=8)
    ax.set_xlabel("Training stage", fontsize=8)
    ax.set_ylim(0, None)
    ax.margins(x=0.05)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    for ext in ("png", "pdf"):
        out = ROOT / f"olmo_scalar_curve_7b_32b.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
