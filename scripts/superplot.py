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

Three figures, kept apart so measures on different scales don't read as if they were
on the others' "higher = better, ~0 = chance" scale:
  - model_comparison.png                    : the scalar measures + the scalar S
  - null_measures_model_comparison.png      : suppress/rebound and layer_targeting
                                              (designed null ≈ 0)
  - degenerate_measures_model_comparison.png: word-based onset/offset error -- the
                                              combined ↓-is-better aggregate, then the
                                              SIGNED onset and offset edges with their
                                              (often degenerate) per-edge CIs
All regenerate on every run. (dial_resolution_pool and token_group are omitted -- a
robustness variant and a near-universal failure.)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
import matplotlib.pyplot as plt                                          # noqa: E402

# Canonical model-family palette. This file is a VERBATIM copy of the paper repo's
# activation-controllability/figure_scripts/model_family_colors.py (the two repos are
# independent, so it is duplicated rather than imported across them); `diff` the two
# to check they are in sync. A .tex twin defines the same colors for in-text use.
from model_family_colors import family_color, family_shades              # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = [("gemma2_9b", "Gemma", 9), ("gemma4_12b", "Gemma", 12), ("gemma3_27b", "Gemma", 27),
          ("gemma4_31b", "Gemma", 31),
          ("qwen35_4b", "Qwen", 4), ("qwen35_9b", "Qwen", 9), ("qwen36_27b", "Qwen", 27),
          ("qwen_72b", "Qwen", 72), ("qwen35_122b_a10b", "Qwen", 122),
          ("qwen3_235b_a22b_2507", "Qwen", 235), ("qwen35_397b_a17b", "Qwen", 397),
          ("qwen3_coder_480b", "Qwen", 480),
          ("llama_8b", "Llama", 8), ("llama33_70b", "Llama", 70),
          ("llama4_scout", "Llama", 109), ("llama4_maverick", "Llama", 400),
          ("gptoss_20b_low", "GPT-OSS", 20), ("gptoss_120b_low", "GPT-OSS", 120),
          ("olmo3_7b", "Olmo", 7), ("olmo31_32b", "Olmo", 32),
          ("mistral_small_31_24b", "Mistral", 24), ("mistral_small_4", "Mistral", 119),
          ("glm47_flash", "GLM", 31), ("glm46v", "GLM", 106), ("glm52", "GLM", 745)]
# ministral3_3b (Ministral 3 3B) is intentionally OMITTED: at 43.6% compliance its
# across-sentence baseline σ is near-degenerate, so the Engage/Coverage d′ bootstrap
# CIs explode (~1e12) and dominate those rows' axes. Left out of the comparison like
# gemma3_270m (37% compliance); its SCORES_/SCALAR_CI_ JSONs are still produced.
FAMILY_ORDER = ["Gemma", "Qwen", "Llama", "GPT-OSS", "Olmo", "Mistral", "GLM"]
DETAIL = {"gemma2_9b": "Gemma 2 9B", "gemma4_12b": "Gemma 4 12B", "gemma3_27b": "Gemma 3 27B",
          "gemma4_31b": "Gemma 4 31B",
          "qwen35_4b": "Qwen 3.5 4B", "qwen35_9b": "Qwen 3.5 9B", "qwen36_27b": "Qwen 3.6 27B",
          "qwen_72b": "Qwen 2.5 72B", "qwen35_122b_a10b": "Qwen 3.5 122B",
          "qwen3_235b_a22b_2507": "Qwen 3 235B", "qwen35_397b_a17b": "Qwen 3.5 397B",
          "qwen3_coder_480b": "Qwen 3 Coder 480B",
          "llama_8b": "Llama 3.1 8B", "llama33_70b": "Llama 3.3 70B",
          "llama4_scout": "Llama 4 Scout 109B", "llama4_maverick": "Llama 4 Maverick 400B",
          "gptoss_20b_low": "GPT-OSS 20B", "gptoss_120b_low": "GPT-OSS 120B",
          "olmo3_7b": "Olmo 3 7B", "olmo31_32b": "Olmo 3.1 32B",
          "mistral_small_31_24b": "Mistral Small 3.1 24B",
          "mistral_small_4": "Mistral Small 4 119B",
          "glm47_flash": "GLM 4.7 Flash 31B", "glm46v": "GLM 4.6V 106B",
          "glm52": "GLM 5.2 745B (FP8)"}

# Model release dates (YYYY-MM) for within-family ordering. Same-month ties break by size.
RELEASE = {"gemma2_9b": "2024-06", "gemma3_27b": "2025-03", "gemma4_31b": "2026-03", "gemma4_12b": "2026-06",
           "qwen_72b": "2024-09", "qwen3_235b_a22b_2507": "2025-07", "qwen3_coder_480b": "2025-07",
           "qwen35_122b_a10b": "2026-02", "qwen35_397b_a17b": "2026-02", "qwen35_4b": "2026-03",
           "qwen35_9b": "2026-03", "qwen36_27b": "2026-04",
           "llama_8b": "2024-07", "llama33_70b": "2024-12", "llama4_scout": "2025-04", "llama4_maverick": "2025-04",
           "gptoss_20b_low": "2025-08", "gptoss_120b_low": "2025-08",
           "olmo3_7b": "2025-11", "olmo31_32b": "2025-12",
           "mistral_small_31_24b": "2025-03", "mistral_small_4": "2026-03",
           "glm47_flash": "2026-01", "glm46v": "2025-12", "glm52": "2026-06"}

# (SCORES key, row label). Three figures, kept apart on purpose:
#  - MAIN_ROWS: every measure that FEEDS the scalar S (as of 2026-07-23 this includes
#    suppress and layer_targeting, folded back in); the S row is appended when
#    rendering -> model_comparison.png. suppress is shown at its raw signed d' (a
#    below-baseline outlier and rebounders read directly); layer_targeting ≈ 0.
#  - NULL_ROWS: token_group -- the one NULL measure excluded from S (near-universal
#    failure: models cannot steer the concept onto a target token type). The other
#    measure excluded from S, onset/offset error, is a timing diagnostic and gets
#    its own figure below -> null_measures_model_comparison.png
#  - DEGENERATE_ROWS: the word-based onset/offset error (combined aggregate + the
#    separate signed onset & offset edges) -> degenerate_measures_model_comparison.png
MAIN_ROWS = [("engage", "Engage"),
             ("suppress", "Suppress"),
             ("dial_rank", "Dial rank"),
             ("dial_resolution", "Dial resolution"),
             ("temporal_control", "Temporal control"),
             ("coverage", "Coverage"),
             ("layer_targeting", "Layer targeting")]
NULL_ROWS = [("token_group", r"Token group  ($\approx$0)")]
# DEGENERATE_ROWS -> its own figure (degenerate_measures_model_comparison.png): the
# word-based onset/offset error. Row 1 = combined aggregate (↓-is-better, no CI);
# rows 2-3 = the SIGNED per-edge errors (0 = on-time) with their per-edge CIs, which
# are often near-degenerate because the 10-bin half-max quantizes the detected edge.
DEGENERATE_ROWS = [("onset_offset_error", "Onset/offset  ($\\downarrow$)"),
                   ("oo_onset", "Onset error"),
                   ("oo_offset", "Offset error")]
SCALAR_ROW = ("scalar", r"Controllability  $S$")


def load_bars(data_root, channel="proj"):
    """{model: {metric: (score,lo,hi) or None, 'scalar': (S,lo,hi), fam, size}}.

    Reads the flat `results/` layout written by compute_scores.py / scalar_ci.py:
    SCORES_<model>.json, ONSET_OFFSET_WORD_<model>.json, SCALAR_CI_<model>.json."""
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
            if ch and ch.get("score") is not None:
                lo, hi = ch.get("lo"), ch.get("hi")
                if lo is not None and hi is not None and (abs(lo) > 1e3 or abs(hi) > 1e3):
                    lo = hi = None   # degenerate bootstrap CI (near-0 baseline sigma) -> point estimate
                row[key] = (ch.get("score"), lo, hi)
            else:
                row[key] = None
        # Onset/offset error: source the WORD-based supersession
        # (ONSET_OFFSET_WORD_<model>.json), not the frozen 4th-TOKEN value in SCORES.
        # The onset gate is scored against the actual 4th-WORD boundary; offset gate
        # unchanged. See README / METRICS (2026-07-22). Falls back to SCORES if absent.
        wp = root / f"ONSET_OFFSET_WORD_{name}.json"
        if wp.exists():
            wc = (json.load(open(wp)).get("channels", {}) or {}).get(channel)
            if wc and wc.get("score") is not None:
                row["onset_offset_error"] = (wc.get("score"), wc.get("lo"), wc.get("hi"))
                edg = {e["edge"]: e for e in wc.get("edges", [])}
                for key, nm in (("oo_onset", "onset"), ("oo_offset", "offset")):
                    e = edg.get(nm)
                    row[key] = ((e["mean"], e.get("lo"), e.get("hi"))
                                if e and e.get("mean") is not None else None)
            else:
                row["onset_offset_error"] = row["oo_onset"] = row["oo_offset"] = None
        cp = root / f"SCALAR_CI_{name}.json"
        row["scalar"], row["point"] = None, False
        if cp.exists():
            d = json.load(open(cp))
            row["scalar"] = (d.get("scalar"), d.get("ci_lo"), d.get("ci_hi"))
            # Models scored from externally supplied recordings that were not retained:
            # S is a point estimate, the joint bootstrap cannot be re-run. Marked in the
            # figure so a bar with no error bar never reads as a tight CI.
            row["point"] = bool(d.get("point_estimate"))
        bars[name] = row
    return bars


def _layout(bars):
    """x positions grouped by family then release date (size breaks ties); colors
    shade with size. Returns (pos, colors, labels, order)."""
    pos, colors, labels, order = [], [], [], []
    x = 0.0
    for fam in FAMILY_ORDER:
        mem = sorted([m for m in bars if bars[m]["fam"] == fam],
                     key=lambda m: (RELEASE.get(m, "9999-99"), bars[m]["size"]))
        if not mem:
            continue
        # single-model families take the full-strength color; family_shades(fam, 1)
        # returns the lightest tint, which would read as washed-out beside the
        # largest member of a multi-model family.
        shades = family_shades(fam, len(mem)) if len(mem) > 1 else [family_color(fam)]
        for m, sh in zip(mem, shades):
            pos.append(x); colors.append(sh)
            labels.append(DETAIL.get(m, m)); order.append(m); x += 1
        x += 0.9
    return pos, colors, labels, order


def _bar_row(ax, bars, order, pos, colors, getter, ylab, mark_point=False):
    for xp, m, col in zip(pos, order, colors):
        t = getter(bars[m])
        if t is None or t[0] is None:
            continue
        sc, lo, hi = t
        ax.bar(xp, sc, width=0.82, color=col, edgecolor="black", linewidth=0.4, zorder=2)
        if lo is not None and hi is not None:                # onset/offset has no aggregate CI
            ax.errorbar(xp, sc, yerr=[[max(sc - lo, 0)], [max(hi - sc, 0)]], fmt="none",
                        ecolor="#222", elinewidth=0.8, capsize=2, zorder=3)
        elif mark_point and bars[m].get("point"):
            # point-estimate S (raw not retained -> no joint bootstrap): open marker on the
            # bar top so a missing error bar never reads as a tight CI.
            ax.plot(xp, sc, marker="o", mfc="white", mec="#222", mew=0.9, ms=4.0, zorder=4)
    ax.axhline(0, color="#888", lw=0.6)
    # Horizontal row-header label (rotation=0): vertical labels would overflow the short
    # one-page rows and collide with neighbours; horizontal decouples label length from row
    # height. Right-aligned just left of the y-tick numbers.
    ax.set_ylabel(ylab, fontsize=9, rotation=0, ha="right", va="center", labelpad=6)
    ax.tick_params(labelsize=7)
    ax.margins(x=0.015)
    ax.spines[["top", "right"]].set_visible(False)


def render(bars, rows, out, channel, title):
    """Render one stacked bar-grid (one row per (key, label) in `rows`) to `out`."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pos, colors, labels, order = _layout(bars)
    n = len(rows)
    # Compact per-row height so the full grid + caption fits ONE AAAI page (US Letter, ~9"
    # text height). Width = AAAI text width (7.0") so \includegraphics[width=\textwidth] adds
    # no rescaling and in-figure font sizes are preserved. When `title` is empty the suptitle
    # is dropped (title lives in the caption) and the top margin shrinks accordingly.
    ROW_H, BOT_IN = 0.78, 0.95
    TOP_IN = 0.45 if title else 0.12
    figh = ROW_H * n + TOP_IN + BOT_IN
    fig, axes = plt.subplots(n, 1, figsize=(7.0, figh), sharex=True, squeeze=False)
    axes = axes[:, 0]
    plt.subplots_adjust(left=0.20, right=0.985, top=1 - TOP_IN / figh, bottom=BOT_IN / figh, hspace=0.22)
    for ax, (key, ylab) in zip(axes, rows):
        _bar_row(ax, bars, order, pos, colors, lambda r, k=key: r.get(k), ylab,
                 mark_point=(key == "scalar"))
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    # Family is read off the x-label COLOR (as in Figure 1) rather than colored family
    # headers across the top: tint each model label with its family color.
    for lab, m in zip(axes[-1].get_xticklabels(), order):
        lab.set_color(family_color(bars[m]["fam"]))
    if title:
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
    ap.add_argument("--out-degenerate",
                    default=str(PROJECT_ROOT / "results" / "degenerate_measures_model_comparison.png"),
                    help="degenerate onset/offset error figure (combined + separate onset/offset)")
    args = ap.parse_args()
    bars = load_bars(args.data_root, args.channel)
    if not bars:
        raise SystemExit(f"no SCORES_*.json in {args.data_root}; run compute_scores.py first")
    print(f"[superplot] {len(bars)} model(s)")
    render(bars, MAIN_ROWS + [SCALAR_ROW], args.out, args.channel,
           "Controllability battery across models")
    render(bars, NULL_ROWS, args.out_null, args.channel,
           "Excluded (null) measures across models")
    render(bars, DEGENERATE_ROWS, args.out_degenerate, args.channel,
           "Onset/offset error (word) — combined ↓, then signed onset & offset (0 = on-time)")


if __name__ == "__main__":
    main()
