#!/usr/bin/env python3
"""Collapse the controllability battery into ONE scalar per model, in [0,1].

Method (agreed 2026-07-14; full rationale in results/paper/SCALAR.md):

  1. Keep seven measures (drop Token group = near-universal failure, and
     onset/offset = coarse ↓-error): engage, suppress, dial_rank,
     dial_resolution, temporal_control, coverage, layer_targeting. Both readout
     channels (cos, relnorm) are kept as SEPARATE components (coverage is
     cos-only). suppress and layer_targeting were folded back in on 2026-07-23
     (mentor sign-off): the point of the score is to show control is IMPERFECT,
     so an axis models mostly fail (suppress below baseline; layer targeting, a
     designed null) belongs in it and drags the composite down. The conjunctive
     geometric mean keeps a lone large suppress outlier from being rescued by one
     strong axis, and suppress's D_REF is set high enough (3) that the outlier
     does not saturate to full credit.

  2. Map each (measure, channel) score to a probability p in [0,1] where
     0.5 = "no effect / chance", via a measure-appropriate, pre-registered link:
       - d' measures (engage, suppress, dial_resolution, coverage):
             p = Phi(d' / sqrt(2))            # AUROC of the effect
       - dial_rank (Spearman rho in [-1, 1]):
             p = (rho + 1) / 2
       - temporal_control (contrast, in baseline-SD units):
             p = Phi(contrast / sqrt(2))
     Negatives are meaningful: a rebound / wrong-way effect gives p < 0.5 and is
     kept (not floored), so it drags the composite down rather than vanishing.

  3. ABSOLUTE, not panel-relative: the links use fixed constants only, so a
     model's scalar never changes when another model is added or removed.

  4. Aggregate CONJUNCTIVELY (rewards broad control, punishes any weak axis) as
     a weighted GEOMETRIC mean of the RAW p's:
             G = exp( sum_i w_i * ln(p_i) )
     over the raw p in (0,1) -- so a merely at-chance component attenuates as 0.5
     rather than annihilating the product to 0.

  5. MEASURE-EQUAL weights: each of the (present) measures carries 1/N_measures;
     within a measure that weight is split evenly across its present channels.
     So a two-channel measure gives 1/(2N) per channel and coverage gives its
     full 1/N to cos. Missing channels/measures are dropped and the remaining
     weights renormalized, preserving the measure-equal intent.

  6. Rescale AT THE END so chance -> 0 and perfect -> 1:
             S = clip( 2*G - 1, 0, 1 )
     (all-chance G = 0.5 -> S = 0; all-perfect G = 1 -> S = 1; a net-below-chance
     model floors at 0).

No run data or model load: reads only the per-measure JSON emitted by
`compute_scores.py --json` (results/SCORES_<model>.json). This deliberately does
NOT recompute or perturb any measure -- it is a pure downstream reduction.

CI note: the point estimate is exact. A joint confidence interval requires a
single two-way (sentence x concept) bootstrap that recomputes ALL six measures
on each shared resample and then re-aggregates -- the per-measure bootstraps in
the JSON are drawn INDEPENDENTLY and must NOT be combined component-wise. That
joint-bootstrap CI is intentionally left to a follow-up; this script reports the
point estimate only.

Usage:
  python scripts/aggregate_scalar.py                       # all results/SCORES_*.json
  python scripts/aggregate_scalar.py --scores a.json b.json
  python scripts/aggregate_scalar.py --json results/SCALAR.json
"""

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PHI = NormalDist().cdf
_SQRT2 = math.sqrt(2.0)
_EPS = 1e-6

# The SEVEN kept measures, in report order. suppress measures
# `dont_think_about − no_instruction`: no_instruction is already the concept's
# resting floor, so success is bounded and most models sit near chance -- which is
# exactly why it belongs in S (it shows models cannot suppress on command) rather
# than being hidden as a diagnostic; the white-bear rebound (wrong-way) reads as
# p < 0.5 and penalizes. layer_targeting is a designed null (~0 for every model),
# folded in as a near-uniform drag. Excluded from S by design and reported as
# diagnostics only: onset_offset_error (coarse ↓-error) and token_group
# (near-universal failure).
KEPT_MEASURES = ["engage", "suppress", "dial_rank", "dial_resolution",
                 "temporal_control", "coverage", "layer_targeting"]
SHORT = {"engage": "enga", "suppress": "supp", "dial_rank": "rank",
         "dial_resolution": "res", "temporal_control": "temp", "coverage": "cove",
         "layer_targeting": "layr"}
# Selectable channel sets. 'projection' (default) collapses the single proj
# readout -> each measure is one component at full 1/N weight; 'legacy' is the
# original two-channel (cos, relnorm) scalar; 'all' keeps all three.
CHANNEL_SETS = {"projection": ("proj",),
                "legacy": ("cos", "relnorm"),
                "all": ("cos", "relnorm", "proj")}

# ---- score -> p link -------------------------------------------------------------
# Two selectable links (--link), both giving p in [0,1] with 0.5 = chance:
#   'linear' (DEFAULT): p = clip(0.5 + score / (2*D_REF[measure]), 0, 1). Linear and
#     fully discriminating up to D_REF (= the score that means "essentially perfect
#     control" for that axis), then a deliberate ceiling. Chosen to fix the AUROC
#     link's early saturation (a strong axis like Engage compressed to ~1 and stopped
#     discriminating; see METRICS §5).
#   'phi'  (legacy): p = Phi(score/sqrt2) = the AUROC of the effect. Saturates by
#     d'~4, so it is retained only for comparison / the appendix.
# Dial Rank uses (rho+1)/2 under BOTH links (already bounded/linear, rho in [-1,1]).
#
# D_REF provenance -- chosen 2026-07-17 on the 8-model panel (Gemma/Qwen/Llama/
# GPT-OSS), projection channel. Rationale (Engage/Temporal = "magnitude" reading of
# control, i.e. a bigger commanded push counts, capped below runaway/sigma-inflated
# values; the rest anchored on d'~3 = "reliably separable" given each axis's headroom):
#   engage 8, suppress 3.0, dial_resolution 3, temporal_control 5, coverage 1.5,
#   layer_targeting 5.0.
# suppress raised 1.0->3.0 on 2026-07-23: at 1.0 any suppress d' >= 1 saturated to
# full credit, so the lone below-baseline outlier (Gemma 4 31B, d'~1.45, wide CI)
# earned MORE credit than the honest ~0.5 suppressors; 3.0 keeps it off the ceiling
# and preserves the informative ordering. layer_targeting shares temporal's
# standardized-contrast scale (5.0); since every model scores ~0 there, its p ~ 0.5
# for all -> a uniform drag that does not reorder models.
# These are the metric's calibration constants ("the whole ballgame") -- REVISIT and
# re-run all models if the panel grows or the notion of "perfect" per axis changes.
D_REF = {"engage": 8.0, "suppress": 3.0, "dial_resolution": 3.0,
         "temporal_control": 5.0, "coverage": 1.5, "layer_targeting": 5.0}
LINKS = ("linear", "phi")


def _to_prob(measure, cell, link="linear"):
    """Map one (measure, channel) score cell to p in [0,1], 0.5 = null.
    Returns None if the cell has no usable score."""
    score = cell.get("score")
    if score is None or not math.isfinite(score):
        return None
    if measure == "dial_rank":                      # Spearman rho in [-1, 1] (both links)
        p = (score + 1.0) / 2.0
    elif link == "linear":                          # linear-clip against D_REF
        p = 0.5 + score / (2.0 * D_REF[measure])
    elif measure == "dial_resolution":              # phi: already an AUROC-derived d'
        p = cell.get("auroc")                       # exact AUROC when carried
        if p is None or not math.isfinite(p):
            p = _PHI(score / _SQRT2)
    else:                                            # phi: d' or SD-unit contrast
        p = _PHI(score / _SQRT2)
    return min(max(float(p), _EPS), 1.0 - _EPS)


def model_scalar(measures, channels=CHANNEL_SETS["projection"], link="linear"):
    """Return (S, G, components) for one model's `measures` dict, collapsing over
    the given channel set. components: list of dicts {measure, channel, p, weight}."""
    # collect present components, grouped by measure
    by_measure = {}
    for m in KEPT_MEASURES:
        chans = (measures.get(m) or {}).get("channels", {})
        present = []
        for ch in channels:
            cell = chans.get(ch)
            if cell is None:
                continue
            p = _to_prob(m, cell, link)
            if p is not None:
                present.append((ch, p))
        if present:
            by_measure[m] = present

    if not by_measure:
        return None, None, []

    w_measure = 1.0 / len(by_measure)               # measure-equal
    comps = []
    for m, present in by_measure.items():
        w_ch = w_measure / len(present)             # split within measure
        for ch, p in present:
            comps.append({"measure": m, "channel": ch, "p": p, "weight": w_ch})

    log_g = sum(c["weight"] * math.log(c["p"]) for c in comps)
    G = math.exp(log_g)
    S = min(max(2.0 * G - 1.0, 0.0), 1.0)           # rescale: chance->0, perfect->1
    return S, G, comps


def main():
    ap = argparse.ArgumentParser(
        description="Collapse the SCORES battery into one [0,1] scalar per model.")
    ap.add_argument("--scores", nargs="*", default=None,
                    help="score JSON paths; default: all results/SCORES_*.json")
    ap.add_argument("--channels", choices=list(CHANNEL_SETS), default="projection",
                    help="channel set to collapse (default: projection = the paper channel)")
    ap.add_argument("--link", choices=LINKS, default="linear",
                    help="score->p link (default: linear-clip vs D_REF; 'phi' = legacy AUROC)")
    ap.add_argument("--json", default=None,
                    help="also write the scalars + component breakdown here")
    args = ap.parse_args()
    channels = CHANNEL_SETS[args.channels]

    paths = args.scores or sorted(str(p) for p in (PROJECT_ROOT / "results").glob("SCORES_*.json"))
    if not paths:
        ap.error("no score JSONs; run compute_scores.py --json results/SCORES_<model>.json first")

    rows = []
    for p in paths:
        d = json.load(open(p))
        S, G, comps = model_scalar(d.get("measures", {}), channels, args.link)
        rows.append({"model": d.get("model", Path(p).stem), "source": str(p),
                     "scalar": S, "geom_mean": G, "components": comps})

    rows.sort(key=lambda r: (r["scalar"] is None, -(r["scalar"] or 0)))

    print(f"\nControllability scalar  S = clip(2*G - 1, 0, 1),  "
          f"G = measure-equal weighted geometric mean of per-component p"
          f"   [channels: {args.channels}; link: {args.link}]")
    print("=" * 96)
    print(f"{'model':<26}{'S (scalar)':<14}{'G (geo-mean)':<16}per-measure p ({' / '.join(channels)})")
    print("-" * 96)
    for r in rows:
        if r["scalar"] is None:
            print(f"{r['model']:<26}{'n/a':<14}")
            continue
        pm = {}
        for c in r["components"]:
            pm.setdefault(c["measure"], {})[c["channel"]] = c["p"]
        brk = "  ".join(f"{SHORT[m]}:" + "/".join(f"{pm[m][ch]:.2f}" for ch in channels if ch in pm[m])
                        for m in KEPT_MEASURES if m in pm)
        print(f"{r['model']:<26}{r['scalar']:<14.4f}{r['geom_mean']:<16.4f}{brk}")
    print("=" * 96)
    print("p in [0,1], 0.5 = chance; S in [0,1], 0 = at-chance/no control, 1 = perfect.")
    print("Kept: " + ", ".join(KEPT_MEASURES) + ".")
    print("Excluded (diagnostics): onset_offset_error (coarse ↓-error), "
          "token_group (near-universal failure).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"method": "measure-equal weighted geometric mean; S=clip(2G-1,0,1)",
                       "link": args.link, "d_ref": (D_REF if args.link == "linear" else None),
                       "channel_set": args.channels, "channels": list(channels),
                       "kept_measures": KEPT_MEASURES,
                       "models": [{"model": r["model"], "scalar": r["scalar"],
                                   "geom_mean": r["geom_mean"], "source": r["source"],
                                   "components": r["components"]} for r in rows]},
                      f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
