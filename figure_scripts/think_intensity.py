#!/usr/bin/env python3
"""Think_Intensity — half-width VERTICAL two-panel intensity figure (Figure 4).

Two stacked AAAI single-column panels (gemma3_27b focal, `--model` tunable):
  (a) lexical: per-token proj at the intensely-peak layer -- think about / don't think
      about / think intensely, plus the NO-INSTRUCTION baseline (dashed light gray).
  (b) numeric: per-token proj at the intensity-ramp PEAK layer -- don't + intensity
      1/4..4/4.
Both panels share the sentence-token x-axis (labels on the bottom panel).

Reads PRECOMPUTED battery outputs via the sibling engage_suppress loaders; the
illustrative per-token traces come raw from results.json and the no-instruction
baseline from the activation cache (es.token_traces). Reuses that sibling for styling,
the model list, and --data-root data access.

The dial (rank vs depth) and the engage/suppress depth panels live in the companion
depth_dial.py (Figure 5); the ORANGE constants and _aligned_limits below are kept
here as the canonical dial styling that script imports.

AAAI-compliant (TrueType, 300 dpi, no top/right spines, no gridlines, no titles).
Caption material -> the companion .md file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out, skip  # portable, env-overridable paths

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt                                            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # sibling helper
import engage_suppress as es                                          # noqa: E402
from roster import FOCAL                                               # noqa: E402  (single-source focal)

RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
REDS = plt.get_cmap("Reds")(np.linspace(0.40, 0.95, 4))                    # intensity 1..4
INTENSE = REDS[3]
ORANGE = "#E8820C"        # Dial rank -- imported by depth_dial


# --------------------------- data (results.json only) ----------------------------

def run_axis(run_dir):
    """(layers, n_total) — the depth axis the PROFILES curves are indexed on."""
    rows = json.load(open(run_dir / "results.json"))["results"]
    n_total = int(yaml.safe_load(open(run_dir / "config.yaml"))["n_layers"])
    layers = sorted({int(x) for r in rows if r.get("is_compliant")
                     for x in (r.get("analysis_layers") or [])})
    return layers, n_total


def token_proj(run_dir, concept, sentence, L, conds):
    """{cond: per-token proj at layer L} for one concept/sentence + the tokens."""
    rows = json.load(open(run_dir / "results.json"))["results"]
    out, toks = {}, None
    for r in rows:
        if (not r.get("is_compliant") or r.get("concept") != concept
                or r["sentence"] != sentence or r["condition_id"] not in conds):
            continue
        if toks is None:
            toks = [t.strip() or "␣" for t in r["anchored_token_strs"][1:]]
        n_tok = len(r["anchored_token_strs"]) - 1
        t = es._proj_tokens_cond(r, L, n_tok)
        if t is not None:
            out[r["condition_id"]] = t
    return toks, out


def unit_peak_layer(run_dir, concept, sentence, hi, lo, layers):
    """Layer with the largest mean per-token DIRECTED effect (hi - lo) for this
    concept/sentence — the clearest POSITIVE illustration layer."""
    rows = json.load(open(run_dir / "results.json"))["results"]
    tri = {r["condition_id"]: r for r in rows if r.get("is_compliant")
           and r.get("concept") == concept and r["sentence"] == sentence
           and r["condition_id"] in (hi, lo)}
    if hi not in tri or lo not in tri:
        return max(layers)
    rh, rl = tri[hi], tri[lo]
    nh, nl = len(rh["anchored_token_strs"]) - 1, len(rl["anchored_token_strs"]) - 1
    best_L, best = layers[-1], -np.inf
    for L in layers:
        th = es._proj_tokens_cond(rh, L, nh); tl = es._proj_tokens_cond(rl, L, nl)
        if th is None or tl is None:
            continue
        m = min(len(th), len(tl))
        eff = float(np.nanmean(th[:m] - tl[:m]))
        if eff > best:
            best, best_L = eff, L
    return best_L


# --------------------------- render ----------------------------------------------

def _aligned_limits(pairs, pad=0.08):
    """ylims for a twin-axis pair with 0 at a common vertical fraction (no clipping).
    Canonical dial helper -- imported by depth_dial for Figure 4's dial panel."""
    ext = [(max(b, 1e-9) * (1 + pad), max(a, 1e-9) * (1 + pad)) for b, a in pairs]
    f = max(b / (a + b) for b, a in ext)
    return [(-(f / (1 - f)) * a if (f / (1 - f)) * a >= b else -b, a) for b, a in ext]


def render_stack(focal, out):
    """Half-width VERTICAL 2-panel (Figure 3): (a) lexical think-intensely trace WITH the
    no-instruction baseline, (b) numeric intensity ramp. Per-token projection at each
    panel's peak layer; the two panels share the sentence-token x-axis."""
    frun = es.run_for(focal)
    layers, n_total = run_axis(frun)
    dpct = es.depth_pcts(layers, n_total)
    _, gn_m, _, _ = es.load_profile(focal, "gain_numeric")
    L_peak = layers[int(np.nanargmax(gn_m))]
    deep = [L for L, d in zip(layers, dpct) if d > 50]
    L_lex = unit_peak_layer(frun, es.CONCEPT, es.SENTENCE, "think_intensely",
                            "dont_think_about", deep or layers)
    toks_lex, tr_lex = token_proj(frun, es.CONCEPT, es.SENTENCE, L_lex,
                                  ["think_about", "dont_think_about", "think_intensely"])
    toks_num, tr_num = token_proj(frun, es.CONCEPT, es.SENTENCE, L_peak,
                                  ["dont_think_about", *RAMP])
    # no-instruction baseline at the lexical panel's layer (from the activation cache)
    _, _, _, base_lex = es.token_traces(frun, focal, es.CONCEPT, es.SENTENCE, L_lex)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.34, 3.9), sharex=True)
    plt.subplots_adjust(left=0.17, right=0.965, top=0.985, bottom=0.135, hspace=0.20)

    # (a) lexical think-intensely + no-instruction baseline
    for cond, col, lab in (("think_about", es.MED_GRAY, "Think about"),
                           ("dont_think_about", es.LIGHT_GRAY, "Don't think about"),
                           ("think_intensely", INTENSE, "Think intensely about")):
        if cond in tr_lex:
            axA.plot(range(len(tr_lex[cond])), tr_lex[cond], color=col, lw=1.2,
                     marker="o", ms=2.2, label=lab)
    axA.plot(range(len(base_lex)), base_lex, color=es.LIGHT_GRAY, lw=1.2, ls="--",
             marker="o", ms=2.2, label="No instruction")
    axA.set_ylabel("Concept vector projection", fontsize=7)
    axA.yaxis.set_major_formatter(es._FixedOrder(3)); axA.yaxis.get_offset_text().set_fontsize(6.0)
    hA, lA = axA.get_legend_handles_labels()
    wantA = ["Think intensely about", "Think about", "Don't think about", "No instruction"]
    hlA = dict(zip(lA, hA))
    axA.legend([hlA[w] for w in wantA if w in hlA], [w for w in wantA if w in hlA],
               frameon=False, fontsize=6.5, loc="upper left", handlelength=1.3,
               labelspacing=0.2, borderaxespad=0.3)
    axA.tick_params(axis="y", labelsize=6.5); es._nospine(axA)
    axA.text(-0.15, 1.0, "(a)", transform=axA.transAxes, fontsize=7, fontweight="bold", va="bottom")

    # (b) numeric intensity ramp
    if "dont_think_about" in tr_num:
        axB.plot(range(len(tr_num["dont_think_about"])), tr_num["dont_think_about"],
                 color=es.LIGHT_GRAY, lw=1.2, marker="o", ms=2.2, label="Don't think about")
    for k, cond in enumerate(RAMP):
        if cond in tr_num:
            axB.plot(range(len(tr_num[cond])), tr_num[cond], color=REDS[k], lw=1.2,
                     marker="o", ms=2.2, label=f"Intensity {k+1}/4")
    axB.set_ylabel("Concept vector projection", fontsize=7)
    axB.yaxis.set_major_formatter(es._FixedOrder(3)); axB.yaxis.get_offset_text().set_fontsize(6.0)
    hB, lB = axB.get_legend_handles_labels()                            # reverse: 4/4 -> Don't
    axB.legend(hB[::-1], lB[::-1], frameon=False, fontsize=6.5, loc="upper left",
               bbox_to_anchor=(0.1, 1.0), handlelength=1.3, labelspacing=0.18, borderaxespad=0.0)
    axB.set_xticks(range(len(toks_num)))
    axB.set_xticklabels(toks_num, rotation=45, ha="right", fontsize=6.0, family="monospace")
    axB.tick_params(axis="y", labelsize=6.5); axB.tick_params(axis="x", pad=1.0); es._nospine(axB)
    axB.text(-0.15, 1.0, "(b)", transform=axB.transAxes, fontsize=7, fontweight="bold", va="bottom")

    svg = str(Path(out).with_suffix(".svg"))
    fig.savefig(svg, bbox_inches="tight"); print(f"wrote {svg}")
    es._savefig(fig, out)
    return dict(zip(layers, dpct)).get(L_peak), dict(zip(layers, dpct)).get(L_lex)


def _write_md(path, body):
    open(path, "w").write(body)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Half-width vertical intensity-trace figure (Fig 3).")
    ap.add_argument("--model", default=FOCAL, help="focal model (default: roster.FOCAL / $AC_FOCAL)")
    ap.add_argument("--out", default=out("Think_Intensity.png"))
    ap.add_argument("--data-root", default=AC_DATA,
                    help="data dir holding raw/ + PROFILES_ (default: $AC_DATA)")
    args = ap.parse_args()
    es.use_data_root(args.data_root)
    focal = args.model
    # This figure plots single-trial per-token traces, which are not frozen to JSON
    # (unlike the other paper figures); it needs the model's raw run, absent in a clone.
    if es.run_for(focal) is None:
        skip(f"think_intensity needs the raw run for {focal!r} (per-token traces are not "
             f"shipped as JSON). Regenerate it with `scripts/run.sh --config "
             f"experiments/main/{focal}.yaml`, then re-run this figure.")

    d_peak, d_lex = render_stack(focal, args.out)

    _write_md(str(Path(args.out).with_suffix(".md")),
              f"# {Path(args.out).stem} — caption material\n\n"
              "*Auto-emitted; not on the figure.* Half-width, two stacked panels. Single-trial "
              f"per-token concept projection for **{focal}**, concept **{es.CONCEPT}** on "
              f"“{es.SENTENCE}”. "
              f"**(a)** Lexical: think about / don’t think about / think intensely at the "
              f"intensely-peak layer ({d_lex}% depth), against the no-instruction baseline "
              "(dashed light gray). "
              f"**(b)** Numeric intensity ramp (don’t think about + intensity 1/4–4/4) at the "
              f"numeric-gain peak layer ({d_peak}% depth).\n")


if __name__ == "__main__":
    main()
