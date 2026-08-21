#!/usr/bin/env python3
"""Temporal_precision — projection-channel AAAI figure (focal panels + onset/offset).

Companion to Temporal_control. Row (a) illustrates the two *precision* instructions
for the focal model (gemma3_27b); rows (b) show the word-based onset & offset timing
error across models. Reuses the sibling temporal_control for roster / palette and the
frozen FIGDATA loader; the focal profiles themselves are precomputed by
scripts/figdata.py into FIGDATA_<model>.json and just read here -- no raw access.

Row (a) — focal gemma3_27b, per-word/-position projection Δ vs the no-instruction
  baseline, each instruction (red) vs generic *think about* (gray), commanded 'on'
  region shaded:
    · First half     — fractional x (position in sentence), shade [0, 0.5]
    · After 4th word — WORD-index x (word count is tokenizer-invariant), shade word 5+
  Bands = 95% two-way (sentence × concept) cluster bootstrap, B=2000. Illustrative,
  precomputed into FIGDATA_<model>.json by scripts/figdata.py.
Rows (b) — onset error (after-4th gate) and offset error (first-half gate) across
  models, SIGNED (0 = on-time; <0 = fires early, >0 = late), each bar with its per-edge
  95% CI, from the FROZEN word-based ONSET_OFFSET_WORD_<model>.json. Onset CIs are often
  near-degenerate (the 10-bin half-max quantizes the detected edge).

AAAI conventions: TrueType, 300 dpi, capitalized labels/legends, no titles, no top/right
spines. Caption material -> Temporal_precision.md.
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
import temporal_control as tc                                        # noqa: E402  (roster/palette/loaders)
import scalar as sa                                                  # noqa: E402  (RELEASE dates + family order rule)

# The full panel = the 20 retained-raw models + the 5 large MoE models (roster.ROSTER).
# The MoE models have no raw runs, but this figure's edge-timing bars read the frozen
# ONSET_OFFSET_WORD_<model>.json, which exists for all 25. (token_coverage stays on the
# 20-model roster.MODELS -- it needs raw POS readouts the MoE models lack.)
from roster import ROSTER, DETAIL_ALL as DETAIL                          # noqa: E402  (single-source roster)


def fam_order_by_S(data_root, roster, present):
    """Families ordered by their TOP performer on the controllability scalar S,
    descending -- the same rule Figure 2 (scalar) and the battery panel
    (superplot.py) use, so every cross-model figure shares one family order.
    Falls back to the fixed FAMILY_ORDER for any family with no S available."""
    S = {}
    for m in present:
        p = Path(data_root) / f"SCALAR_CI_{m}.json"
        if p.exists():
            v = json.load(open(p)).get("scalar")
            if v is not None:
                S[m] = v

    def top(fam):
        vs = [S[m] for m, f, _ in roster if f == fam and m in S]
        return max(vs) if vs else float("-inf")

    fams = [f for f in tc.FAMILY_ORDER if any(f == fm for m, fm, _ in roster if m in present)]
    return sorted(fams, key=top, reverse=True)

from roster import FOCAL                                                  # noqa: E402  (single-source focal)
GENERIC = tc.GENERIC
MED_GRAY, ENGAGE_C, SPAN_Y = tc.MED_GRAY, tc.ENGAGE_C, tc.SPAN_Y


# --------------------------- focal profiles (frozen) -----------------------------
# The focal fractional + per-word profiles are precomputed by scripts/figdata.py
# (the scoring layer) into FIGDATA_<model>.json; read here, no raw access at figure time.

def load_focal(model):
    """(prof_fraction, prof_word, targeting_layer, min_word_count) from FIGDATA.

    prof_fraction[cond] = (centers, mean, lo, hi) over 10 position bins (shares the
    fractional-profile block with temporal_control); prof_word[cond] = (word_idx,
    mean, lo, hi). Identical to the old inline position_profile / word_profile."""
    prof_all, L = tc.load_positional(model)
    prof_fh = {k: prof_all[k] for k in (GENERIC, "persist_first_half") if k in prof_all}

    p = Path(tc.DATA_ROOT) / f"FIGDATA_{model}.json"
    w = json.load(open(p))["word_profiles"]

    def arr(a):
        return np.array([np.nan if v is None else v for v in a], float)
    prof_a4 = {cond: (arr(v["x"]), arr(v["mean"]), arr(v["lo"]), arr(v["hi"]))
               for cond, v in w["conds"].items()}
    return prof_fh, prof_a4, L, int(w["min_word_count"])


def load_edges(model, ch="proj"):
    p = Path(tc.DATA_ROOT) / f"ONSET_OFFSET_WORD_{model}.json"
    if not p.exists():
        return None, None
    c = json.load(open(p))["channels"].get(ch)
    if not c:
        return None, None
    e = {x["edge"]: x for x in c["edges"]}
    return e.get("onset"), e.get("offset")


# ------------------------------- render ------------------------------------------

def _focal_panel(ax, prof, cond, lab, span, xlabel, ylabel=False):
    for cc, color, cl in ((GENERIC, MED_GRAY, "Generic think"), (cond, ENGAGE_C, lab)):
        if cc not in prof:
            continue
        cx, m, lo, hi = prof[cc]
        ax.plot(cx, m, color=color, marker="o", ms=2.6, lw=1.4, label=cl)
        ax.fill_between(cx, lo, hi, color=color, alpha=0.13)
    ax.axvspan(span[0], span[1], color=SPAN_Y, alpha=0.12)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.3, labelspacing=0.25)
    ax.tick_params(labelsize=6.5)
    tc._nospine(ax)
    if ylabel:
        ax.set_ylabel(r"Projection $\Delta$ vs baseline", fontsize=8)


def _bar_layout(models_present):
    pos, xcol, xlab, order, fam_pos = [], [], [], [], {}
    x = 0.0
    size_of = {m: s for m, f, s in ROSTER}
    # Family order and within-family order aligned with Figure 2 / the battery panel:
    # families by top performer on S, then within a family by release date (size breaks ties).
    for fam in fam_order_by_S(tc.DATA_ROOT, ROSTER, models_present):
        mem = sorted([m for m, f, s in ROSTER if f == fam and m in models_present],
                     key=lambda mm: (sa.RELEASE.get(mm, "9999-99"), size_of[mm]))
        if not mem:
            continue
        shades = tc.family_shades(fam, len(mem)) if len(mem) > 1 else [tc.family_color(fam)]
        fam_pos[fam] = []
        for m, sh in zip(mem, shades):
            pos.append(x); xcol.append(sh); xlab.append(DETAIL.get(m, m))
            order.append(m); fam_pos[fam].append(x); x += 1
        x += 0.9
    return pos, xcol, xlab, order, fam_pos


def _bar_panel(ax, edges, which, pos, xcol, order, ylab):
    for xp, m, col in zip(pos, order, xcol):
        e = edges[m][0] if which == "onset" else edges[m][1]
        if not e or e.get("mean") is None:
            continue
        sc, lo, hi = e["mean"], e.get("lo"), e.get("hi")
        ax.bar(xp, sc, width=0.82, color=col, edgecolor="black", linewidth=0.5, zorder=2)
        if lo is not None and hi is not None:
            yerr = [[max(sc - lo, 0)], [max(hi - sc, 0)]]
            if m.startswith("gptoss"):   # dark bars only: white halo so the CI stays visible
                ax.errorbar(xp, sc, yerr=yerr, fmt="none", ecolor="white",
                            elinewidth=2.1, capsize=2.8, capthick=2.1, zorder=2.9)
            ax.errorbar(xp, sc, yerr=yerr, fmt="none", ecolor="#222",
                        elinewidth=0.9, capsize=2.2, zorder=3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.margins(x=0.01)
    tc._nospine(ax)


def render(model, out, pdf=False):
    prof_fh, prof_a4, L, mw = load_focal(model)
    edges = {m: e for m, _, _ in ROSTER if any(e := load_edges(m))}
    pos, xcol, xlab, order, fam_pos = _bar_layout(edges)

    fig = plt.figure(figsize=(7.2, 6.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 2.0], hspace=0.42,
                             left=0.10, right=0.98, top=0.95, bottom=0.16)
    gs_a = outer[0].subgridspec(1, 2, wspace=0.22)
    ax_fh = fig.add_subplot(gs_a[0, 0])
    ax_a4 = fig.add_subplot(gs_a[0, 1], sharey=ax_fh)
    gs_b = outer[1].subgridspec(2, 1, hspace=0.12)          # tight gap within panel (b)
    ax_on = fig.add_subplot(gs_b[0])
    ax_off = fig.add_subplot(gs_b[1], sharex=ax_on)

    _focal_panel(ax_fh, prof_fh, "persist_first_half", "First half", (0.0, 0.5),
                 "Fraction of transcribed sentence", ylabel=True)
    _focal_panel(ax_a4, prof_a4, "persist_after_fourth", "After 4th word",
                 (4.5, mw + 0.5), "Word number from sentence start")
    ax_a4.set_xlim(0.5, mw + 0.5); ax_a4.set_xticks(range(1, mw + 1))
    ax_a4.tick_params(labelleft=False)

    _bar_panel(ax_on, edges, "onset", pos, xcol, order, "Onset error")
    _bar_panel(ax_off, edges, "offset", pos, xcol, order, "Offset error")
    ax_on.tick_params(labelbottom=False)
    ax_off.set_xticks(pos); ax_off.set_xticklabels(xlab, rotation=30, ha="right", fontsize=6.5)
    # Family reads off the x-label color (as in Figure 1), not colored family headers.
    fam_of = {m: f for m, f, _ in ROSTER}
    for lab, m in zip(ax_off.get_xticklabels(), order):
        lab.set_color(tc.family_color(fam_of[m]))

    ax_fh.text(-0.30, 1.05, "(a)", transform=ax_fh.transAxes, fontsize=11,
               fontweight="bold", ha="left", va="bottom")
    ax_on.text(-0.075, 1.10, "(b)", transform=ax_on.transAxes, fontsize=11,
               fontweight="bold", ha="left", va="bottom")

    exts = [out, str(Path(out).with_suffix(".svg"))] + ([str(Path(out).with_suffix(".pdf"))] if pdf else [])
    for ext in exts:
        fig.savefig(ext, bbox_inches="tight")
        print(f"wrote {ext}")
    plt.close(fig)
    return L, mw, edges, order


def write_md(out_md, L, mw, edges, order):
    lines = [
        "# Temporal_precision — caption material\n",
        "*Auto-emitted; not on the figure.* Projection channel (‖r‖·cos). Companion to "
        "Temporal_control: temporal control asks *where* the concept lands, this figure "
        "asks how precisely a model turns it **on and off in time** during the transcription.\n",
        "**(a)** Focal model **{f}** at the targeting depth (90% depth; layer {L}): per-position "
        "projection Δ vs the no-instruction baseline for two precision instructions (red) against "
        "generic *think about* (gray), commanded region shaded. *First half* — think about the "
        "concept only during the first half of the sentence (fractional position; shade [0, 0.5]). "
        "*After 4th word* — start thinking about it after the fourth word; plotted on **word index** "
        "(word count is tokenizer-invariant, so the boundary is one clean line at word 4→5), clipped "
        "at the shortest sentence ({mw} words). Pooled over 50 sentences × 10 concepts; bands = 95% "
        "two-way (sentence × concept) cluster bootstrap, B=2000.\n".format(f=FOCAL, L=L, mw=mw),
        "**(b)** Onset error (*after 4th word* gate) and offset error (*first half* gate) across "
        "models, from the word-based `ONSET_OFFSET_WORD_<model>.json`: the SIGNED edge-timing error "
        "(detected − requested, fractional position; 0 = on-time, <0 = edge fires earlier than "
        "commanded, >0 = later), each bar with its per-edge 95% bootstrap CI (B=2000). Grouped by "
        "family, ordered by size. Onset CIs are often near-degenerate because the 10-bin half-max "
        "crossing quantizes the detected edge. Values (onset [95% CI] / offset [95% CI]):\n",
    ]

    def fmt(e):
        if not e or e.get("mean") is None:
            return "n/a"
        lo = "%.2f" % e["lo"] if e.get("lo") is not None else "n/a"
        hi = "%.2f" % e["hi"] if e.get("hi") is not None else "n/a"
        return f"{e['mean']:.2f} [{lo}, {hi}]"

    for m in order:
        on, off = edges[m]
        lines.append(f"- **{DETAIL.get(m, m)}**: onset {fmt(on)} / offset {fmt(off)}")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Temporal_precision.png"))
    ap.add_argument("--pdf", action="store_true", help="also emit .pdf (promotion step)")
    args = ap.parse_args()
    tc.use_data_root(args.data_root)
    print(f"[temporal_precision] focal = {FOCAL}")
    L, mw, edges, order = render(FOCAL, args.out, pdf=args.pdf)
    write_md(str(Path(args.out).with_suffix(".md")), L, mw, edges, order)


if __name__ == "__main__":
    main()
