#!/usr/bin/env python3
"""Token_Targeting — token-group control across models (single panel).

Figure 18 (token-specific targeting). Instructed to place the concept on a specific
token TYPE -- *think about {concept} on the punctuation* (`loc_punctuation`) or *on the
adjectives* (`loc_adjectives`) -- does it concentrate there? The token_group measure is
the standardized in-region − out-of-region contrast (relative to generic *think about*),
pooled over those two instructions; higher = the concept lands on the commanded token
type, near-zero/negative = no token-specific targeting.

A retired measure that was computed all along (it is in every SCORES_<model>.json) but
never plotted. Fully self-contained: reads the frozen SCORES (the same value the battery
computed), roster/palette from the sibling temporal_control. Companion to
Token_Coverage.

AAAI conventions: TrueType, 300 dpi, capitalized labels, no titles, no top/right spines.
Caption material -> Token_Targeting.md.
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
import scalar as sa                                                  # noqa: E402  (RELEASE dates)
from temporal_precision import fam_order_by_S                        # noqa: E402  (shared family-order rule)

# Full panel = 20 retained-raw models + 5 large MoE models (roster.ROSTER). token_group
# is a MARGINAL per-measure two-way cluster-bootstrap CI that does not need raw, so the
# MoE models carry real whiskers like every other model; their values are read from the
# stored SCORES in results/.
from roster import ROSTER, DETAIL_ALL as DETAIL                          # noqa: E402  (single-source roster)


def load_token_group(model, ch="proj"):
    p = Path(tc.DATA_ROOT) / f"SCORES_{model}.json"
    if not p.exists():
        return None
    c = (json.load(open(p))["measures"].get("token_group", {}).get("channels", {}) or {}).get(ch)
    return (c["score"], c["lo"], c["hi"]) if c and c.get("score") is not None else None


def render(out, pdf=False):
    tg = {m: load_token_group(m) for m, _, _ in ROSTER}
    tg = {m: v for m, v in tg.items() if v}

    pos, xcol, xlab, order, fam_pos = [], [], [], [], {}
    x = 0.0
    size_of = {m: s for m, f, s in ROSTER}
    # Family order and within-family order aligned with Figure 2 / the battery panel:
    # families by top performer on S, then within a family by release date (size breaks ties).
    for fam in fam_order_by_S(tc.DATA_ROOT, ROSTER, set(tg)):
        mem = sorted([m for m, f, s in ROSTER if f == fam and m in tg],
                     key=lambda mm: (sa.RELEASE.get(mm, "9999-99"), size_of[mm]))
        if not mem:
            continue
        shades = tc.family_shades(fam, len(mem)) if len(mem) > 1 else [tc.family_color(fam)]
        fam_pos[fam] = []
        for m, sh in zip(mem, shades):
            pos.append(x); xcol.append(sh); xlab.append(DETAIL.get(m, m))
            order.append(m); fam_pos[fam].append(x); x += 1
        x += 0.9

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    plt.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.30)
    for xp, m, col in zip(pos, order, xcol):
        sc, lo, hi = tg[m]
        ax.bar(xp, sc, width=0.82, color=col, edgecolor="black", linewidth=0.5, zorder=2)
        ax.errorbar(xp, sc, yerr=[[max(sc - lo, 0)], [max(hi - sc, 0)]], fmt="none",
                    ecolor="#222", elinewidth=0.9, capsize=2.2, zorder=3)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylabel("Token targeting", fontsize=10)
    ax.tick_params(labelsize=8); ax.margins(x=0.01); tc._nospine(ax)
    ax.set_xticks(pos); ax.set_xticklabels(xlab, rotation=30, ha="right", fontsize=8)
    # Family reads off the x-label color (as in Figure 1), not colored family headers.
    fam_of = {m: f for m, f, _ in ROSTER}
    for lab, m in zip(ax.get_xticklabels(), order):
        lab.set_color(tc.family_color(fam_of[m]))

    exts = [out, str(Path(out).with_suffix(".svg"))] + ([str(Path(out).with_suffix(".pdf"))] if pdf else [])
    for ext in exts:
        fig.savefig(ext, bbox_inches="tight"); print(f"wrote {ext}")
    plt.close(fig)
    return tg, order


def write_md(out_md, tg, order):
    L = ["# Token_Targeting — caption material\n",
         "*Auto-emitted; not on the figure.* Projection channel (‖r‖·cos). Token targeting "
         "across models (frozen, `SCORES`): the standardized in-region − out-of-region contrast "
         "(relative to generic *think about*), pooled over the two token-type instructions "
         "*think on the punctuation* (`loc_punctuation`) and *think on the adjectives* "
         "(`loc_adjectives`). Higher = the concept concentrates on the commanded token type; "
         "near zero or negative = no token-specific targeting. Grouped by family, families "
         "ordered by their top performer on the controllability scalar S and models within a "
         "family by release date; whiskers = 95% two-way (sentence × concept) cluster bootstrap, "
         "B=2000. A retired "
         "measure, computed but not previously plotted. Values (token targeting; [95% CI]):\n"]
    for m in order:
        t = tg[m]
        L.append(f"- **{DETAIL.get(m, m)}**: {t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]")
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Token_Targeting.png"))
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    tc.use_data_root(args.data_root)
    tg, order = render(args.out, pdf=args.pdf)
    write_md(str(Path(args.out).with_suffix(".md")), tg, order)


if __name__ == "__main__":
    main()
