#!/usr/bin/env python3
"""Cross-model comparison superplot -- the repo's shipped at-a-glance artifact.

One bar-plot row per metric (projection channel) plus a final row for the
controllability scalar S, across all models with a SCORES_<model>.json present.
Family-grouped, within-family shade darkening with size, 95% CI whiskers -- same
convention as the paper's Engage_Suppress_AcrossModels figure.

Reads ONLY the precomputed JSONs (SCORES_<model>.json + SCALAR_CI_<model>.json),
so it is fast, deterministic, and auto-scales to whatever models are present. The
post-run orchestrator regenerates it after each run. Rendered output is gitignored
(regenerated on demand); this script is the tracked, shippable part.

Two figures, kept apart so the excluded measures don't read as if they were on the
same "higher = better, ~0 = chance" scale as the rest:
  - model_comparison.png              : the 6 scalar measures + the scalar S
  - null_measures_model_comparison.png: layer_targeting (designed null ≈ 0) and
                                        onset/offset error (↓-is-better, [0,1])
Both regenerate on every run. (dial_resolution_pool and token_group are omitted --
a robustness variant and a near-universal failure.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.transforms as mtransforms                             # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [("gemma2_9b", "Gemma", 9), ("gemma3_27b", "Gemma", 27), ("gemma4_31b", "Gemma", 31),
          ("qwen36_27b", "Qwen", 27), ("qwen_72b", "Qwen", 72), ("qwen35_122b_a10b", "Qwen", 122),
          ("llama33_70b", "Llama", 70), ("gptoss_120b_low", "GPT-OSS", 120)]
FAMILY_ORDER = ["Gemma", "Qwen", "Llama", "GPT-OSS"]
FAMILY_CMAP = {"Gemma": "Blues", "Qwen": "Purples", "Llama": "Greens", "GPT-OSS": "Oranges"}
DETAIL = {"gemma2_9b": "Gemma 2 9B", "gemma3_27b": "Gemma 3 27B", "gemma4_31b": "Gemma 4 31B",
          "qwen36_27b": "Qwen 3.6 27B", "qwen_72b": "Qwen 2.5 72B", "qwen35_122b_a10b": "Qwen 3.5 122B",
          "llama33_70b": "Llama 3.3 70B", "gptoss_120b_low": "GPT-OSS 120B"}

# (SCORES key, row label). Two figures, kept apart on purpose:
#  - MAIN_ROWS: the six ↑-is-better measures that feed the scalar; the S row is
#    appended when rendering -> model_comparison.png
#  - NULL_ROWS: the two measures EXCLUDED from the scalar, which don't share the
#    others' "higher = better, ~0 = chance" reading (layer targeting is a designed
#    null ≈ 0; onset/offset is a ↓-is-better error) -> null_measures_model_comparison.png
MAIN_ROWS = [("engage", "Engage  $d'$"),
             ("dial_rank", r"Dial rank  $\rho$"),
             ("dial_resolution", "Dial resolution  $d'$"),
             ("temporal_control", "Temporal control"),
             ("coverage", "Coverage  $d'$")]
NULL_ROWS = [("suppress", "Suppress / rebound  $d'$"),
             ("layer_targeting", r"Layer targeting  ($\approx$0)"),
             ("onset_offset_error", "Onset/offset err.  ($\\downarrow$)")]
SCALAR_ROW = ("scalar", r"Controllability  $S$")


def load_bars(data_root, channel="proj"):
    """{model: {metric: (score,lo,hi) or None, 'scalar': (S,lo,hi), fam, size}}."""
    root = Path(data_root)
    bars = {}
    for name, fam, size in MODELS:
        sp = root / f"SCORES_{name}.json"
        if not sp.exists():
            continue
        meas = json.load(open(sp))["measures"]
        row = {"fam": fam, "size": size}
        for key, _ in MAIN_ROWS + NULL_ROWS:
            ch = (meas.get(key, {}).get("channels", {}) or {}).get(channel)
            row[key] = ((ch.get("score"), ch.get("lo"), ch.get("hi"))
                        if ch and ch.get("score") is not None else None)
        cp = root / f"SCALAR_CI_{name}.json"
        if cp.exists():
            d = json.load(open(cp))
            row["scalar"] = (d.get("scalar"), d.get("ci_lo"), d.get("ci_hi"))
        else:
            row["scalar"] = None
        bars[name] = row
    return bars


def _layout(bars):
    """x positions grouped by family then size; colors shade with size; family
    label anchors. Returns (pos, colors, labels, order, fam_pos)."""
    pos, colors, labels, order, fam_pos = [], [], [], [], {}
    x = 0.0
    for fam in FAMILY_ORDER:
        mem = sorted([m for m in bars if bars[m]["fam"] == fam], key=lambda m: bars[m]["size"])
        if not mem:
            continue
        shades = np.linspace(0.5, 0.9, len(mem)) if len(mem) > 1 else [0.72]
        fam_pos[fam] = []
        for m, sh in zip(mem, shades):
            pos.append(x); colors.append(plt.get_cmap(FAMILY_CMAP[fam])(sh))
            labels.append(DETAIL.get(m, m)); order.append(m); fam_pos[fam].append(x); x += 1
        x += 0.9
    return pos, colors, labels, order, fam_pos


def _bar_row(ax, bars, order, pos, colors, getter, ylab):
    for xp, m, col in zip(pos, order, colors):
        t = getter(bars[m])
        if t is None or t[0] is None:
            continue
        sc, lo, hi = t
        ax.bar(xp, sc, width=0.82, color=col, edgecolor="black", linewidth=0.4, zorder=2)
        if lo is not None and hi is not None:                # onset/offset has no aggregate CI
            ax.errorbar(xp, sc, yerr=[[max(sc - lo, 0)], [max(hi - sc, 0)]], fmt="none",
                        ecolor="#222", elinewidth=0.8, capsize=2, zorder=3)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel(ylab, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.margins(x=0.015)
    ax.spines[["top", "right"]].set_visible(False)


def render(bars, rows, out, channel, title):
    """Render one stacked bar-grid (one row per (key, label) in `rows`) to `out`."""
    pos, colors, labels, order, fam_pos = _layout(bars)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7.4, 1.35 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    plt.subplots_adjust(left=0.13, right=0.98, top=1 - 0.35 / (1.35 * n), bottom=0.11, hspace=0.22)
    for ax, (key, ylab) in zip(axes, rows):
        _bar_row(ax, bars, order, pos, colors, lambda r, k=key: r.get(k), ylab)
    trans = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    for fam, xps in fam_pos.items():
        axes[0].text(float(np.mean(xps)), 1.06, fam, transform=trans, ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color=plt.get_cmap(FAMILY_CMAP[fam])(0.78))
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    fig.suptitle(f"{title}  (channel: {channel})", fontsize=11, fontweight="bold", y=0.998)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Cross-model comparison superplots from the precomputed JSONs.")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "results"),
                    help="dir holding SCORES_<model>.json + SCALAR_CI_<model>.json")
    ap.add_argument("--channel", default="proj", help="readout channel (default: proj)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "model_comparison.png"),
                    help="scalar-measures + S figure")
    ap.add_argument("--out-null", default=str(PROJECT_ROOT / "results" / "null_measures_model_comparison.png"),
                    help="excluded (null) measures figure")
    args = ap.parse_args()
    bars = load_bars(args.data_root, args.channel)
    if not bars:
        raise SystemExit(f"no SCORES_*.json in {args.data_root}; run compute_scores.py first")
    print(f"[superplot] {len(bars)} model(s)")
    render(bars, MAIN_ROWS + [SCALAR_ROW], args.out, args.channel,
           "Controllability battery across models")
    render(bars, NULL_ROWS, args.out_null, args.channel,
           "Excluded (null) measures across models")


if __name__ == "__main__":
    main()
