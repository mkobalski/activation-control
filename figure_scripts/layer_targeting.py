#!/usr/bin/env python3
"""Layer_targeting — focal layer-targeting heatmap + across-models null bar.

Figure 14 (layer-targeted control). Instructed to think about the concept only at a
single target layer, does the concept concentrate there? It does not -- layer targeting
is the battery's cleanest designed null.

  (a) Focal gemma3_27b: heatmap of the standardized concept projection Δ vs the
      no-instruction baseline, as a function of (target layer instructed × analysis
      layer read), column-demeaned within each analysis layer so a working targeter
      would light up the diagonal. It is flat -- no diagonal.
  (b) Layer targeting across models (FROZEN, SCORES): the mean diagonal contrast; every
      model sits at ≈ 0.

Row (a) reads the (target × analysis) grid frozen in FIGDATA_<model>.json (precomputed
from the LT run by scripts/figdata.py); row (b) reads the frozen SCORES. Roster/palette
from the sibling temporal_control. No raw access at figure time.

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
import matplotlib.transforms as mtransforms                             # noqa: E402

AC = AC_ROOT
sys.path.insert(0, str(AC)); sys.path.insert(0, str(AC / "scripts"))
import temporal_control as tc                                         # noqa: E402  (roster/palette)
from roster import FOCAL                                                  # noqa: E402  (single-source focal)


def load_grid(model):
    """(targets, analysis, grid) from FIGDATA_<model>.json -- the (target × analysis)
    column-demeaned standardized projection Δ grid, precomputed by scripts/figdata.py
    (the scoring layer, which reads the LT run). Identical to the old inline layer_grid;
    the figure no longer touches raw."""
    p = Path(tc.DATA_ROOT) / f"FIGDATA_{model}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run scripts/figdata.py (or postprocess.py) first")
    lt = json.load(open(p)).get("layer_targeting")
    if not lt:
        raise SystemExit(f"{p} has no layer_targeting grid -- the model's LT run was not scored")
    targets = [int(t) for t in lt["targets"]]
    analysis = [int(a) for a in lt["analysis"]]
    grid = np.array([[np.nan if v is None else v for v in row] for row in lt["grid"]], float)
    return targets, analysis, grid


def load_layer_targeting(model, ch="proj"):
    p = Path(tc.DATA_ROOT) / f"SCORES_{model}.json"
    if not p.exists():
        return None
    c = (json.load(open(p))["measures"].get("layer_targeting", {}).get("channels", {}) or {}).get(ch)
    return (c["score"], c["lo"], c["hi"]) if c and c.get("score") is not None else None


def render(model, out, pdf=False, n_total=62):
    targets, analysis, grid = load_grid(model)
    # clip the analysis (column) axis to the swept target range so both axes share the
    # same layers -> a SQUARE heatmap centred on where targeting could show up.
    mn = min(targets)
    keep = [i for i, L in enumerate(analysis) if L >= mn]
    analysis = [analysis[i] for i in keep]
    grid = grid[:, keep]
    diag = np.nanmean([grid[ti, analysis.index(T)] for ti, T in enumerate(targets) if T in analysis])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    vmax = float(np.nanmax(np.abs(grid)))
    im = ax.imshow(grid, aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    for ti, T in enumerate(targets):                       # boxed diagonal (analysis == target)
        if T in analysis:
            ax.add_patch(plt.Rectangle((analysis.index(T) - 0.5, ti - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=1.0))
    ax.set_xticks(range(len(analysis)))
    ax.set_xticklabels([f"{round(100*L/n_total)}" for L in analysis], fontsize=6.5)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([f"{round(100*T/n_total)}" for T in targets], fontsize=6.5)
    ax.set_xlabel("Analysis layer (depth %)", fontsize=7.5)
    ax.set_ylabel("Instructed target layer (depth %)", fontsize=7.5)
    ax.tick_params(labelsize=6.5, length=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Standardized projection $\Delta$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()

    exts = [out, str(Path(out).with_suffix(".svg"))] + ([str(Path(out).with_suffix(".pdf"))] if pdf else [])
    for ext in exts:
        fig.savefig(ext, bbox_inches="tight"); print(f"wrote {ext}")
    plt.close(fig)
    print(f"[layer_targeting] focal diagonal mean = {diag:+.3f}")
    return diag


def write_md(out_md, diag):
    body = (
        "# Layer_targeting -- caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width, square. Projection channel. Focal "
        f"model **{FOCAL}**: standardized concept projection $\\Delta$ vs the no-instruction "
        "baseline as a function of the instructed target layer (rows) and the analysis layer "
        "read (columns), column-demeaned within each analysis layer, clipped to the swept "
        "target range so both axes share the same layers (a square). A model that could target "
        "layers would brighten the boxed diagonal (analysis layer = instructed target); it is "
        f"flat (focal diagonal mean {diag:+.2f}). The across-model layer-targeting null (every "
        "model near 0) is summarized in the controllability profile figure.\n")
    open(out_md, "w").write(body)
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Layer_targeting.png"))
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    tc.use_data_root(args.data_root)
    print(f"[layer_targeting] focal = {FOCAL}")
    diag = render(FOCAL, args.out, pdf=args.pdf)
    write_md(str(Path(args.out).with_suffix(".md")), diag)


if __name__ == "__main__":
    main()
