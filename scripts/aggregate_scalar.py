#!/usr/bin/env python3
"""Collapse the controllability battery into ONE scalar per model, in [0,1].

Method (agreed 2026-07-14; full rationale in results/paper/SCALAR.md):

  1. Keep six measures (drop Token group = near-universal failure,
     onset/offset = coarse ↓-error, and dial_resolution = a sign test with an
     unpaired conversion, redundant with dial_rank; removed 2026-08-08):
     engage, suppress, dial_rank,
     temporal_control, coverage, layer_targeting. Both readout
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

# The SIX kept measures, in report order. suppress measures
# `dont_think_about − no_instruction`: no_instruction is already the concept's
# resting floor, so success is bounded and most models sit near chance -- which is
# exactly why it belongs in S (it shows models cannot suppress on command) rather
# than being hidden as a diagnostic; the white-bear rebound (wrong-way) reads as
# p < 0.5 and penalizes. layer_targeting is a designed null (~0 for every model),
# folded in as a near-uniform drag.
#
# dial_resolution was REMOVED from S on 2026-08-08. It is a per-unit sign test
# (win = 1 / 0.5 / 0 over the three adjacent level pairs) converted with the
# UNPAIRED two-sample constant sqrt(2)*PhiInv -- a conversion that only applies to
# independent samples, whereas the pairs are within-unit. It therefore never
# measured magnitude separation, and it ranked models at Spearman 0.99 with
# dial_rank, giving the dial 2/7 of a measure-equal composite. It is still
# COMPUTED and reported in SCORES_<model>.json as a diagnostic; it just no longer
# enters S. Excluded from S by design and reported as diagnostics only:
# dial_resolution, onset_offset_error (coarse ↓-error) and token_group
# (near-universal failure).
KEPT_MEASURES = ["engage", "suppress", "dial_rank",
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
# and preserves the informative ordering. layer_targeting is anchored a priori at
# 5.0 on the STANDARDIZED-CONTRAST scale (the Delta/sigma construction it shares with
# temporal control) -- deliberately NOT derived from its roster max, which is ~0.05:
# calibrating a designed null to its own maximum would spread p across [0.18, 0.99]
# and turn noise around zero into an apparent capability. At 5.0 every model sits at
# p ~ 0.5 -> a uniform drag that does not reorder models. (This anchor is independent
# of temporal control's constant; the two no longer share a value.)
# engage 8.0->16.4 and coverage 1.5->4.11 on 2026-07-24: on the 25-model roster both
# axes were saturating (median p = 1.00; coverage clipped to full credit for 57% of
# models, engage 54%), so they inflated S without discriminating. Each D_REF is now set
# just above that axis's ROSTER-MAX d' (engage 16.06, coverage 4.03; x1.02 margin), so
# the strongest model lands at p~0.99 instead of pinned at 1.0 and the axis stays
# discriminating across its full observed range. This de-compresses the mid-pack.
# temporal_control 5.0->6.90 on 2026-08-08: it was the last axis still saturating
# (roster max 6.7696 > 5.0 pinned olmo31_32b and gemma2_9b at p = 1.0). Set by the
# same rule as engage/coverage -- roster-max x1.02 -- so all three capability axes
# now follow one rule. No axis is pinned afterwards. dial_resolution's entry is kept
# for the diagnostic, which is still computed and reported outside S.
# These are the metric's calibration constants ("the whole ballgame") -- REVISIT and
# re-run all models if the panel grows or the notion of "perfect" per axis changes.
D_REF = {"engage": 16.4, "suppress": 3.0, "dial_resolution": 3.0,
         "temporal_control": 6.90, "coverage": 4.11, "layer_targeting": 5.0}

# ---- OPEN ITEMS on this calibration (flagged 2026-08-10, deferred) ---------------
# 1. layer_targeting's ceiling is very high relative to what is observed: D_REF 5.0
#    against a roster max of 0.0535, i.e. ~93x. Its p therefore spans only
#    [0.497, 0.505] across all 25 models -- a range of 0.009, against engage's 0.467.
#    It contributes a near-constant ~0.5^(1/6) factor to G: it deflates every S by the
#    same amount without separating any two models (dropping it moves the ranking by
#    Spearman 0.999). That is defensible AS A DELIBERATE STATEMENT -- S should show
#    that no model achieves layer-addressable control -- but it is a statement, not a
#    measurement, so "six measures" spans five DISCRIMINATING axes. Decide whether to
#    keep it in S or report it alongside as a standalone null result.
# 2. CONSIDERED AND DECLINED 2026-08-10. Each empirical ceiling is set by a SINGLE
#    model (engage 16.4 and coverage 4.11 by GLM 4.6V, temporal_control 6.90 by
#    Olmo 3.1 32B), so dropping that model would move the ceiling. Trimmed statistics
#    were evaluated as a fix and rejected: any rule that puts the ceiling BELOW the
#    roster max necessarily saturates the top model, which is the exact pathology the
#    2026-07-24 and 2026-08-08 recalibrations removed. Measured on this roster --
#    p95 x1.02 pins 1/2/1 models on engage/coverage/temporal; top-2-mean x1.02 pins
#    1/1/1; the current max x1.02 pins 0/0/0. "Robust to one observation" and "nothing
#    pins" cannot both hold for a single constant. If this is ever revisited, the
#    version worth trying is WINSORIZING (clip scores at the 95th percentile, then
#    calibrate to the clipped max x1.02): the ceiling stops depending on the single
#    largest value and nothing pins, at the cost of no longer separating the top one
#    or two models on that axis.
# ---------------------------------------------------------------------------------
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
    # Bound p to [_EPS, 1]. The LOWER bound protects the logarithm in the geometric
    # mean: log(0) diverges to -inf and would wipe out that model's G. There is no
    # matching need at the top -- log(1) = 0 -- so the upper end is left at exactly 1
    # (the inset to 1-_EPS was dropped 2026-08-10). This lets a measure that genuinely
    # reaches its ceiling read as 1: on the current roster that is Llama 3.1 8B, whose
    # dial rank is exactly rho = 1. Effect on its S is +2.6e-07; no other model moves.
    return min(max(float(p), _EPS), 1.0)


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
