#!/usr/bin/env python3
"""Temporal_control — projection-channel AAAI figure (focal panels + across-models).

Self-contained like the other paper scripts (engage_suppress / scalar): it
reads the frozen scoring artifacts (--data-root / $AC_DATA) and imports only the
shared model_family_colors palette -- NO dependency on scripts/explore.py or the
scoring layer.

Row (a): positional targeting for the focal model (gemma3_27b) at the targeting depth
  (90% -> layer 55). Three panels -- Think at beginning / Think once mid-sentence /
  Think at end -- each the concept's per-position projection Δ vs the no-instruction
  baseline (‖r‖·cos − baseline), against Generic think, with the commanded target
  region shaded. Bands = 95% two-way (sentence × concept) cluster bootstrap, B=2000.
  This illustrative profile is precomputed by scripts/figdata.py into
  FIGDATA_<model>.json (read here); the SCORE it illustrates is frozen in row (b).
Row (b): Temporal control score across models (FROZEN, from SCORES_<model>.json),
  family-grouped, within-family shade darkening with size, 95% CI whiskers.

Temporal control pools the standardized in/out (located − generic) contrast over the
three targeting conditions loc_beginning / persist_once / loc_end; persist_once is the
'mid' span. It is on a d'-like (baseline-SD-standardized) scale but is a contrast of
contrasts, so -- following the battery -- it is labeled without a $d'$ symbol.

AAAI conventions: TrueType (no Type-3), 300 dpi, capitalized labels/legends, no titles,
no top/right spines, no gridlines. Caption material -> Temporal_control.md.
"""
import argparse
import json
import os
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.transforms as mtransforms                             # noqa: E402

# Shared palette (same module the other paper figures in this directory use).
from model_family_colors import family_color, family_shades             # noqa: E402

# ------------------------------- focal / conditions ------------------------------
from roster import FOCAL                                                  # noqa: E402  (single-source focal)
CONCEPTS_OK = None                                                        # (all concepts in cache)
PROJ_F_LOC = 0.90                                                         # targeting depth fraction
GENERIC = "think_about"
# (condition_id, capitalized label, shaded target span) — left -> right.
PANELS = [("loc_beginning", "Think at beginning",      (0.0, 1 / 3)),
          ("persist_once",  "Think once mid-sentence", (1 / 3, 2 / 3)),
          ("loc_end",       "Think at end",            (2 / 3, 1.0))]
MED_GRAY = (0.45, 0.45, 0.45)
ENGAGE_C = "#c0392b"
SPAN_Y = "#f0d000"
N_BOOT = 2000

# across-model roster (row b): the single canonical source (roster.py), shared with
# every other paper figure and superplot's x-axis.
from roster import MODELS, FAMILY_ORDER, DETAIL                           # noqa: E402  (single-source roster)

DATA_ROOT = None
RAW = None
VC = None


def use_data_root(root):
    global DATA_ROOT, RAW, VC
    if not root:
        raise SystemExit("set --data-root (or $AC_DATA) to the results dir "
                         "(holds raw/, vector_cache/, SCORES_*.json)")
    root = Path(root)
    if not (root / "raw").is_dir() and (root / "results" / "raw").is_dir():
        root = root / "results"
    DATA_ROOT, RAW, VC = root, root / "raw", root / "vector_cache"


# ------------------------------- data layer (frozen) -----------------------------
# The focal per-position profiles are precomputed by scripts/figdata.py (the scoring
# layer, which reads raw) into FIGDATA_<model>.json. This figure just reads them, like
# every other paper figure -- no results.json / vector_cache access at figure time.

def load_positional(model):
    """{cond: (centers, mean, lo, hi)} and the targeting layer, from
    FIGDATA_<model>.json (frozen; identical to the old inline position_profile)."""
    p = Path(DATA_ROOT) / f"FIGDATA_{model}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run scripts/figdata.py (or postprocess.py) first")
    d = json.load(open(p))

    def arr(a):
        return np.array([np.nan if v is None else v for v in a], float)
    prof = {cond: (arr(v["centers"]), arr(v["mean"]), arr(v["lo"]), arr(v["hi"]))
            for cond, v in d["profiles"].items()}
    return prof, int(d["targeting_layer"])


def load_temporal(model, ch="proj"):
    p = Path(DATA_ROOT) / f"SCORES_{model}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    c = (json.load(open(p))["measures"].get("temporal_control", {}).get("channels", {}) or {}).get(ch)
    if not c or c.get("score") is None:
        return None
    return c["score"], c["lo"], c["hi"]


# ------------------------------- render ------------------------------------------

def _nospine(ax):
    ax.spines[["top", "right"]].set_visible(False)


CAP = 3000.0        # y-axis clip; the end-target projection runs past this (drawn dashed)


def _draw_trace(ax, x, y, color, label):
    """Projection trace: solid where <= CAP, dashed on segments exceeding CAP so the
    clipped over-range portion reads as continuing off the top of the axis."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    for i in range(len(x) - 1):
        over = max(y[i], y[i + 1]) > CAP
        ax.plot(x[i:i + 2], y[i:i + 2], color=color, lw=1.4, ls="--" if over else "-",
                zorder=3, label=label if i == 0 else None)
    ax.plot(x[y <= CAP], y[y <= CAP], color=color, marker="o", ms=2.3, ls="none", zorder=4)


def render(model, out, pdf=False):
    prof, L = load_positional(model)
    gc, gm, glo, ghi = prof[GENERIC]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.9), sharey=True)   # full-width, side by side
    plt.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.24, wspace=0.09)
    for ax, (cond, lab, span) in zip(axes, PANELS):
        c, m, lo, hi = prof[cond]
        ax.axvspan(span[0], span[1], color="#fcefb4", alpha=0.6, lw=0, zorder=0)
        ax.axhline(0, color="#bbb", lw=0.6, zorder=1)
        ax.fill_between(gc, glo, np.minimum(ghi, CAP), color=MED_GRAY, alpha=0.15, lw=0, zorder=2)
        ax.fill_between(c, lo, np.minimum(np.asarray(hi), CAP), color=ENGAGE_C, alpha=0.15, lw=0, zorder=2)
        _draw_trace(ax, gc, gm, MED_GRAY, "Generic think")
        _draw_trace(ax, c, m, ENGAGE_C, lab)
        ax.set_ylim(0, CAP); ax.set_xlim(0, 1)
        ax.set_yticks([0, 1000, 2000, 3000]); ax.set_xticks([0, 0.5, 1.0])
        ax.legend(frameon=False, fontsize=7.0, loc="upper left", handlelength=1.2,
                  labelspacing=0.2, borderaxespad=0.3)
        ax.tick_params(labelsize=7.0); _nospine(ax)
    axes[0].set_ylabel(r"Projection $\Delta$ vs baseline", fontsize=7.5)
    fig.supxlabel("Fraction of transcribed sentence", fontsize=7.5, y=0.02)

    exts = [out] + ([str(Path(out).with_suffix(".pdf"))] if pdf else [])
    for ext in exts:
        fig.savefig(ext, bbox_inches="tight")
        print(f"wrote {ext}")
    plt.close(fig)
    return L


def write_md(out_md, L):
    body = (
        "# Temporal_control — caption material\n\n"
        "*Auto-emitted; not on the figure.* Full width, three side-by-side panels. Projection "
        f"channel. Focal model **{FOCAL}**: per-position projection Δ vs the no-instruction "
        "baseline across the transcribed sentence, for the three region-targeting instructions "
        "(left→right: think at beginning / once mid-sentence / at end; red) against generic "
        "*think about* (gray); the instructed region is shaded. Pooled over 50 sentences × 10 "
        f"concepts at the targeting depth (90% depth; layer {L}); bands = 95% two-way (sentence "
        f"× concept) cluster bootstrap, B={N_BOOT}. Each y-axis is clipped at {int(CAP)}; the "
        "end-target projection runs beyond this and is drawn dashed. The single temporal-control "
        "score, and its cross-model spread, appear in the controllability profile figure.\n")
    open(out_md, "w").write(body)
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Temporal_control.png"))
    ap.add_argument("--pdf", action="store_true", help="also emit .pdf (promotion step)")
    args = ap.parse_args()
    use_data_root(args.data_root)
    L = render(FOCAL, args.out, pdf=args.pdf)
    print(f"[temporal_control] focal = {FOCAL} (targeting layer {L})")
    write_md(str(Path(args.out).with_suffix(".md")), L)


if __name__ == "__main__":
    main()
