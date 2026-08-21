#!/usr/bin/env python3
"""Controllability scalar figure (style, projection channel).

Single full-width panel: the controllability scalar S in [0,1] per model, grouped
by family then size, within-family shade darkening with size, whiskers = the 95%
JOINT two-way (sentence x concept) cluster-bootstrap CI.

Reads ONLY the emitted JSON — results/SCALAR_CI_<model>.json (scalar + ci_lo/hi,
written by scripts/scalar_ci.py --channels projection). No run artifacts, no model
load, no recompute; this is a pure rendering step, so the bars cannot drift from
the numbers the battery reported.

AAAI conventions: no title, brief axis label, no gridlines, no top/right spines,
300 dpi, .pdf + .png, caption material emitted to a side-car .md (never on the
figure).
"""
import argparse
import json
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import Patch                                # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
import matplotlib.transforms as mtransforms                         # noqa: E402
from model_family_colors import family_color, family_shades         # canonical palette

import os

# Roster / family order / labels from the single canonical source (roster.py). scalar
# shows the FULL panel (retained-raw MODELS + the MoE extension), so it uses ROSTER.
from roster import ROSTER as MODELS, FAMILY_ORDER, DETAIL_ALL as DETAIL, RELEASE   # noqa: E402  (single-source roster)


def _nospine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _savefig(fig, out):
    for ext in (out, str(Path(out).with_suffix(".png" if out.endswith(".pdf") else ".pdf"))):
        fig.savefig(ext, bbox_inches="tight")
        print(f"wrote {ext}")
    plt.close(fig)


# Pretty names for the measures that actually enter S. Which ones those ARE is read
# off each SCALAR_CI file's `components` at load time (never hardcoded) -- the battery
# dropped suppress from S on 2026-07-17, and a hardcoded list silently went stale.
MEASURE_NAME = {"engage": "engage", "suppress": "suppress", "dial_rank": "dial rank",
                "dial_resolution": "dial resolution", "temporal_control": "temporal control",
                "coverage": "coverage", "layer_targeting": "layer targeting"}
MEASURE_ORDER = ["engage", "suppress", "dial_rank", "dial_resolution",
                 "temporal_control", "coverage", "layer_targeting"]
# Why a measure that COULD enter S doesn't (mirrors aggregate_scalar.py's rationale).
# suppress and layer_targeting were folded IN on 2026-07-23; dial_resolution was
# dropped on 2026-08-08.
DROP_REASON = {"dial_resolution": "redundant with dial rank"}


def kept_measures(d):
    """Measures contributing to S for one model, from its CI file's `components`
    keys ("<measure>|<channel>"), in canonical report order."""
    ms = {k.split("|")[0] for k in (d.get("components") or {})}
    return [m for m in MEASURE_ORDER if m in ms] + sorted(ms - set(MEASURE_ORDER))


def load_bars(ci_dir, channels):
    """{model: dict(fam, size, S, lo, hi, n_boot, kept)} from SCALAR_CI_<model>.json."""
    bars = {}
    for m, fam, size in MODELS:
        p = Path(ci_dir) / f"SCALAR_CI_{m}.json"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        d = json.load(open(p))
        cs = d.get("channel_set")
        if cs != channels:
            raise SystemExit(
                f"{p} holds channel_set={cs!r}, expected {channels!r}. Re-run "
                f"scripts/scalar_ci.py --channels {channels} for this model.")
        lo, hi = d.get("ci_lo"), d.get("ci_hi")
        point = d.get("point_estimate") or lo is None or hi is None   # no raw run -> no CI
        bars[m] = dict(fam=fam, size=size, S=d["scalar"],
                       lo=(d["scalar"] if point else lo), hi=(d["scalar"] if point else hi),
                       point=point, n_boot=d.get("n_boot"), kept=kept_measures(d))
    if not bars:
        raise SystemExit(f"no SCALAR_CI_*.json found in {ci_dir}")
    return bars




def render(bars, out):
    # Grouped by family; WITHIN family ordered by release date (generation), ties by
    # size. Shade darkens with recency (oldest = lightest). Family names sit on top.
    order, color, xs = [], {}, []
    x = 0.0
    GAP = 1.2                                       # blank space between families
    # families ordered by their TOP performer, descending (so Llama leads). The
    # battery panel (superplot.py) uses the same rule keyed on the same S, so the two
    # figures share one x-ordering. NB this makes the axis a function of the scores:
    # if S changes, families can reorder -- say so in the caption.
    present = [f for f in FAMILY_ORDER if any(bars[m]["fam"] == f for m in bars)]
    fam_order = sorted(present, reverse=True,
                       key=lambda f: max(bars[m]["S"] for m in bars if bars[m]["fam"] == f))
    for fam in fam_order:
        mem = sorted([m for m in bars if bars[m]["fam"] == fam],
                     key=lambda m: (RELEASE.get(m, "9999-99"), bars[m]["size"]))
        if not mem:
            continue
        # single-model families take the full-strength color; family_shades(fam, 1)
        # would hand back the lightest tint (see engage_suppress.family_bar_colors)
        shades = family_shades(fam, len(mem)) if len(mem) > 1 else [family_color(fam)]
        for m, sh in zip(mem, shades):
            color[m] = sh
            order.append(m)
            xs.append(x)
            x += 1
        x += GAP
    xlab = [DETAIL.get(m, m) for m in order]

    fig, ax = plt.subplots(figsize=(7.5, 1.95))                # full-width, extra-short
    plt.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.44)
    for xp, m in zip(xs, order):
        b = bars[m]
        ax.bar(xp, b["S"], width=0.82, color=color[m], edgecolor="black", linewidth=0.5, zorder=2)
        if b["point"]:   # point estimate (no raw run -> no bootstrap CI):
            # open marker at the bar top, no whisker, so it is never read as a tight CI
            ax.plot(xp, b["S"], marker="o", mfc="white", mec="#222", mew=0.9, ms=4.2, zorder=3)
            continue
        yerr = [[max(b["S"] - b["lo"], 0)], [max(b["hi"] - b["S"], 0)]]
        if b["fam"] == "GPT-OSS":   # dark bars only: white halo so the CI stays visible
            ax.errorbar(xp, b["S"], yerr=yerr, fmt="none", ecolor="white",
                        elinewidth=2.3, capsize=3.1, capthick=2.3, zorder=2.9)
        ax.errorbar(xp, b["S"], yerr=yerr, fmt="none", ecolor="#222",
                    elinewidth=0.9, capsize=2.5, zorder=3)
    ax.set_ylabel("Controllability $S$", fontsize=10)
    ax.set_ylim(0, 0.7)
    ax.set_yticks(np.arange(0, 0.71, 0.1))
    ax.tick_params(labelsize=8)
    ax.set_xlim(xs[0] - (GAP + 0.6), xs[-1] + (GAP + 0.6))   # leading/trailing gap ~ a family gap
    _nospine(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlab, rotation=30, ha="right", fontsize=7.5)
    # tint each x label with its family color (family reads off the gaps + hue)
    for lab, m in zip(ax.get_xticklabels(), order):
        lab.set_color(family_color(bars[m]["fam"]))
    _savefig(fig, out)
    return order


NUMWORD = {4: "four", 5: "five", 6: "six", 7: "seven"}


def write_md(bars, order, out_md, channels):
    nb = next((bars[m]["n_boot"] for m in order if bars[m]["n_boot"]), 2000)
    kept = bars[order[0]]["kept"]                       # identical across models
    odd = [m for m in order if bars[m]["kept"] != kept]
    if odd:
        raise SystemExit(f"models disagree on which measures enter S: {odd} differ from "
                         f"{order[0]} ({kept}). Re-run scripts/scalar_ci.py so all "
                         f"SCALAR_CI_*.json share one measure set.")
    n_kept = NUMWORD.get(len(kept), str(len(kept)))
    kept_txt = ", ".join(MEASURE_NAME.get(m, m) for m in kept)
    # Excluded = whatever the CI file dropped (each with its stock reason), then the
    # two measures the battery never scores into S at all.
    excl = [f"{MEASURE_NAME.get(m, m)} ({DROP_REASON.get(m, 'excluded')})"
            for m in MEASURE_ORDER if m not in kept]
    excl += ["onset/offset timing (a coarse ↓-error)", "token group (near-universal failure)"]
    excl_txt = (", ".join(excl[:-1]) + f", and {excl[-1]}") if len(excl) > 1 else excl[0]
    excl_txt = excl_txt[0].upper() + excl_txt[1:]
    L = [f"# {Path(out_md).stem} — caption material\n",
         f"*Auto-emitted; not on the figure.* Full-width. Controllability score "
         f"**S ∈ [0, 1]** (0 = at-chance / no control, 1 = perfect) on the "
         f"**{channels}** channel, per model, grouped by family (families ordered by "
         f"their top performer, descending) and within family by release date "
         f"(generation); bar color denotes family and shade darkens with recency. "
         f"S is the measure-equal weighted "
         f"**geometric** mean of {n_kept} measures ({kept_txt}), each mapped to a "
         f"probability p (0.5 = chance) by a fixed pre-registered link, then rescaled "
         f"S = clip(2G − 1, 0, 1); it is absolute (a model's S does not depend on "
         f"the panel) and conjunctive (one weak axis drags it down). {excl_txt} "
         f"are excluded by design. Whiskers = 95% **joint** two-way (sentence × "
         f"concept) cluster bootstrap, B={nb} — one shared resample recomputes all "
         f"{n_kept} measures per replicate, so the interval carries the cross-measure "
         f"covariance (the per-measure CIs must not be pooled component-wise).\n",
         "Values (S [95% CI]):\n"]
    for m in order:
        b = bars[m]
        L.append(f"- **{DETAIL.get(m, m)}**: {b['S']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]")
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Controllability scalar bar figure ().")
    ap.add_argument("--data-root", default=AC_DATA,
                    help="data dir holding the SCALAR_CI_<model>.json files (default: $AC_DATA)")
    ap.add_argument("--channels", default="projection",
                    help="expected channel_set in the CI JSONs (guard against mixing)")
    ap.add_argument("--out", default=out("Scalar.pdf"))
    args = ap.parse_args()
    if not args.data_root:
        raise SystemExit("set --data-root (or $AC_DATA) to the dir holding SCALAR_CI_*.json")

    bars = load_bars(args.data_root, args.channels)
    order = render(bars, args.out)
    write_md(bars, order, str(Path(args.out).with_suffix(".md")), args.channels)


if __name__ == "__main__":
    main()
