#!/usr/bin/env python3
"""Training — half-width single-panel figure (Figure 9).

One AAAI single-column panel: controllability S across Olmo training stages
(Olmo 3 7B & 3.1-32B), point estimates only (snapshot raw was pruned -> no joint
bootstrap CI). The S reconstruction and the stage/series definitions are reused
verbatim from scripts/olmo_scalar_curve.py so the numbers cannot drift.

AAAI conventions: TrueType, 300 dpi, no titles, no top/right spines. Caption -> .md.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.gridspec import GridSpec                                 # noqa: E402

AC = AC_ROOT
sys.path.insert(0, str(AC / "scripts"))
import olmo_scalar_curve as osc                                          # noqa: E402  (S reconstruction)
from model_family_colors import family_color, family_shades             # noqa: E402

# Shorter stage labels for the narrow half-width panel (osc.STAGES is 2-line + wide).
STAGE_LABELS = ["Stage 1 early", "Stage 1 final", "Base", "SFT", "DPO", "Instruct"]


def load_S_ci(data_root, model):
    d = json.load(open(Path(data_root) / f"SCALAR_CI_{model}.json"))
    return d["scalar"], d.get("ci_lo"), d.get("ci_hi")


def render(data_root, out):
    osc.ROOT = Path(data_root)                                           # point S reconstruction at our data
    # Single panel: the Olmo training curve. The GPT-OSS reasoning-effort panel was
    # dropped 2026-08-08 -- that result is not being reported, and the corresponding
    # paragraph is commented out in the manuscript. Panel letters go with it, since
    # there is only one panel left. The GPT-OSS bar code is retained below under
    # `if False:` so the panel can be restored without rewriting it.
    fig = plt.figure(figsize=(2.30, 2.05))
    axA = fig.add_subplot(1, 1, 1)

    # ---- Olmo training curve ----
    x = np.arange(len(STAGE_LABELS), dtype=float)
    shades = family_shades("Olmo", 2)
    for (label, models), sh, mk in zip(osc.SERIES, shades, ("o", "s")):
        ys = [osc.new_scalar(m) for m in models]
        axA.plot(x, ys, "-", color=sh, lw=1.2, marker=mk, ms=2.8, label=label, zorder=3)
    axA.axvline(2.5, color="#bbb", lw=0.8, ls=":")                       # pre-/post-training split
    axA.set_xticks(x)
    axA.set_xticklabels(STAGE_LABELS, rotation=40, ha="right", fontsize=6.0)
    axA.set_ylabel("Controllability $S$", fontsize=7)
    axA.set_ylim(0, None); axA.margins(x=0.06)
    axA.tick_params(axis="y", labelsize=6.5)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(fontsize=6.5, frameon=False, loc="upper left", handlelength=1.2,
               labelspacing=0.2, borderaxespad=0.2)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.26)
    for ext in (out, str(Path(out).with_suffix(".pdf"))):
        fig.savefig(ext, bbox_inches="tight"); print(f"wrote {ext}")
    plt.close(fig)


def write_md(data_root, out_md):
    osc.ROOT = Path(data_root)
    vals = {lab: [osc.new_scalar(m) for m in models] for lab, models in osc.SERIES}
    body = (
        "# Training — caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width, single panel, projection channel. "
        "Controllability score $S$ across Olmo training stages (early/final pre-training, "
        "base mid-training, SFT, DPO, Instruct) for Olmo 3 7B and Olmo 3.1 32B; point estimates "
        "only (snapshot raw pruned -> no joint CI); dotted line splits pre- vs post-training.\n\n"
        + "".join(
            f"- {lab}: " + ", ".join(f"{s}={v:.3f}" for s, v in zip(STAGE_LABELS, vs))
            + f"  (peak at {STAGE_LABELS[vs.index(max(vs))]}; "
            + ("monotonically rising)" if all(b >= a for a, b in zip(vs, vs[1:]))
               else "NOT monotonic -- ends below peak)") + "\n"
            for lab, vs in vals.items())
        + "\nThe GPT-OSS reasoning-effort panel was dropped 2026-08-08 and is not reported.\n")
    open(out_md, "w").write(body)
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Half-width Olmo-training + GPT-OSS-reasoning figure (Fig 7).")
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Training.png"))
    args = ap.parse_args()
    render(args.data_root, args.out)
    write_md(args.data_root, str(Path(args.out).with_suffix(".md")))


if __name__ == "__main__":
    main()
