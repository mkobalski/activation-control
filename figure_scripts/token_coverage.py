#!/usr/bin/env python3
"""Token_Coverage — POS-coverage focal panel + coverage across models.

Figure 6 (breadth of token coverage). Two rows:
  Two stacked half-width panels: engage (red) & suppress (blue) d' vs no-instruction BY
      part-of-speech category, AVERAGED across all models with a usable main run (mean ±
      SEM). The breadth illustration (the shipped exploratory pos_coverage,
      paper-styled: no titles, no top/right spines, WITHOUT the "weakest" annotation).
  (b) Coverage across models (FROZEN, SCORES): the engage d' at each model's WEAKEST POS
      category (higher = the concept is represented even in its weakest category).

UNLIKE the other paper figures, row (a) needs the scoring layer: the per-POS-category d'
requires the POS tagging (score_data.pos_category_readouts + pos_tags.json +
compute_scores.dprime_stats), so this script imports scripts/ for row (a) only. Row (b) is read from the frozen SCORES_<model>.json (the same value
model_comparison plots). Roster/palette come from the sibling temporal_control.

The companion Token_Targeting figure (token_group across models) is a separate, fully
self-contained script (token_targeting.py).

AAAI conventions: TrueType, 300 dpi, capitalized labels/legends, no titles, no top/right
spines. Caption material -> Token_Coverage.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out, skip  # portable, env-overridable paths

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                          # noqa: E402
import matplotlib.transforms as mtransforms                             # noqa: E402

AC = AC_ROOT
sys.path.insert(0, str(AC)); sys.path.insert(0, str(AC / "scripts"))
import score_data as sd                                                   # noqa: E402  (POS readouts)
import compute_scores as cs                                               # noqa: E402  (dprime_stats)
import temporal_control as tc                                         # noqa: E402  (roster/palette/loaders)

POS_PATH = str(AC / "pos_tags.json")
ENGAGE_C, SUPPRESS_C = "#c0392b", "#2471a3"


def pos_panels(run_dir):
    """{cond: (dp, lo, hi)} per-POS-category CONVENTIONAL d' = (instructed - baseline)/sigma
    for engage (POS) and suppress (NEG), proj -- positive = readout ROSE under the instruction.
    (Suppress uses sign +1, i.e. raw, NOT the sign-flipped score, so its mild backfire reads as
    positive and overlays engage instead of dipping below zero.) The weakest engage category
    equals the frozen coverage score by construction."""
    vals, bases = sd.pos_category_readouts(run_dir, [sd.POS, sd.NEG],
                                           pos_path=POS_PATH, vector_cache=str(tc.VC))
    order = sorted(bases["proj"])
    rng = np.random.default_rng(0)
    out = {}
    for cond, sign in ((sd.POS, +1.0), (sd.NEG, +1.0)):
        blocks, S = cs.per_concept_blocks(vals[("proj", cond)], bases["proj"], order)
        st = cs.dprime_stats(blocks, S, len(sd.CATS), rng, n_perm=0)
        dp = sign * st["dp"]
        lo = np.nanpercentile(sign * st["bavg"], 2.5, axis=0)
        hi = np.nanpercentile(sign * st["bavg"], 97.5, axis=0)
        out[cond] = (dp, lo, hi)
    return out


def _main_run(model):
    import glob
    runs = [r for r in sorted(glob.glob(str(tc.RAW / f"*_{model}_activation_control")))
            if not r.endswith("_lt") and os.path.exists(os.path.join(r, "no_instruction_cache.pkl"))]
    return runs[-1] if runs else None


def average_pos_panels(models):
    """(eng, sup, used): per-POS engage/suppress d' point estimates for every model with a
    usable main run, stacked into (n_models x n_POS) arrays."""
    eng, sup, used = [], [], []
    for m, fam, size in models:
        run = _main_run(m)
        if run is None:
            print(f"  [skip {m}] no main run"); continue
        try:
            pan = pos_panels(run)
        except Exception as e:                                            # noqa: BLE001
            print(f"  [skip {m}] {type(e).__name__}: {e}"); continue
        eng.append(pan[sd.POS][0]); sup.append(pan[sd.NEG][0]); used.append(m)
        print(f"  [ok {m}] ({len(used)})", flush=True)
    return np.array(eng), np.array(sup), used


def render(eng, sup, out, pdf=False):
    """Engage & suppress mean d' per POS on ONE conventional-d' axis (both point UP; positive =
    concept readout ROSE under the instruction). The blue suppress bars are overlaid, in front,
    on the red engage bars at each category: suppression backfires only mildly (~0..1.5), so it
    is dwarfed by engagement (~0..11) -- i.e. instructing the model to NOT think about a concept
    barely moves it (and, if anything, nudges it up), rather than pushing it below baseline."""
    n = len(eng)
    me, se = np.nanmean(eng, 0), np.nanstd(eng, 0) / np.sqrt(n)
    ms, ss = np.nanmean(sup, 0), np.nanstd(sup, 0) / np.sqrt(n)
    x = np.arange(len(sd.CATS))

    e_top = float(np.nanmax(me + se))                       # tallest engage incl. SEM
    top = float(max(np.ceil(e_top), e_top * 1.02))          # upper bound (clean)
    lo_all = float(np.nanmin([np.nanmin(me - se), np.nanmin(ms - ss), 0.0]))
    bot = float(min(0.0, lo_all * 1.05))                    # allow a small dip if any bar < 0

    fig, ax = plt.subplots(figsize=(3.34, 1.95))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.985, bottom=0.30)
    wE, wS = 0.72, 0.42                                     # blue narrower, overlaid IN FRONT
    ax.bar(x, me, wE, yerr=se, color=ENGAGE_C, edgecolor="black", linewidth=0.5,
           capsize=1.6, error_kw=dict(elinewidth=0.7), label="Engage", zorder=2)
    ax.bar(x, ms, wS, yerr=ss, color=SUPPRESS_C, edgecolor="black", linewidth=0.5,
           capsize=1.6, error_kw=dict(elinewidth=0.7), label="Suppress", zorder=3)
    ax.axhline(0, color="#888", lw=0.7, zorder=1)
    ax.set_ylim(bot, top)
    ax.set_ylabel(r"$d'$  (readout $-$ baseline)", fontsize=7.5)
    ax.set_yticks(list(range(0, int(top) + 1, 2)))
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(sd.CATS, rotation=45, ha="right", fontsize=6.5)
    tc._nospine(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.2,
              labelspacing=0.2, borderaxespad=0.2)

    exts = [out, str(Path(out).with_suffix(".svg"))] + ([str(Path(out).with_suffix(".pdf"))] if pdf else [])
    for ext in exts:
        fig.savefig(ext, bbox_inches="tight"); print(f"wrote {ext}")
    plt.close(fig)
    return n


def write_md(out_md, n, used):
    body = (
        "# Token_Coverage — caption material\n\n"
        "*Auto-emitted; not on the figure.* Half-width, single panel. Projection "
        "channel. Engage (*think about*, red) and Suppress (*don't think about*, blue) "
        f"conventional $d'=(\\text{{instructed}}-\\text{{baseline}})/\\sigma$ vs no instruction, per "
        f"part-of-speech category, **averaged across {n} models** (mean +/- SEM across models; each "
        "model read at its POS peak layer). Both series point up (positive = readout rose); the "
        "blue suppress bars are overlaid in front of the red engage bars. Engagement concentrates "
        "on low-content register tokens (punctuation, determiners) yet stays positive on every "
        "category; suppression does NOT push the concept below baseline -- it mildly backfires "
        "(small positive), so the blue bars sit just above zero and are dwarfed by engagement. "
        "The Coverage measure that enters $S$ is the "
        "engage $d'$ at each model's weakest category; its cross-model spread appears in the "
        f"controllability profile figure. Models: {', '.join(used)}.\n")
    open(out_md, "w").write(body)
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=AC_DATA)
    ap.add_argument("--out", default=out("Token_Coverage.png"))
    ap.add_argument("--pdf", action="store_true", help="also emit .pdf (promotion step)")
    args = ap.parse_args()
    tc.use_data_root(args.data_root)
    print("[token_coverage] averaging per-POS d' across models ...", flush=True)
    eng, sup, used = average_pos_panels(tc.MODELS)
    # Per-POS readouts are computed from each model's raw run (not frozen to JSON); a
    # clone ships no raw, so with nothing to average this figure has no data.
    if not used:
        skip("token_coverage needs the raw main runs for the roster models (per-POS "
             "readouts are not shipped as JSON); none are present. Regenerate runs with "
             "`scripts/run.sh` (see README), then re-run this figure.")
    n = render(eng, sup, args.out, pdf=args.pdf)
    write_md(str(Path(args.out).with_suffix(".md")), n, used)


if __name__ == "__main__":
    main()
