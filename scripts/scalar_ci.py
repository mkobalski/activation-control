#!/usr/bin/env python3
"""Joint two-way bootstrap CI for the controllability scalar S.

The per-measure CIs in SCORES_<model>.json are drawn INDEPENDENTLY, so they
cannot be combined component-wise into an interval for S (that would ignore the
covariance between measures). This module draws ONE two-way cluster resample of
the shared (sentence x concept) population per replicate, recomputes ALL six kept
measures on that same resample, aggregates them into S exactly as
scripts/aggregate_scalar.py does, and reports the 2.5/97.5 percentiles of the
resulting S distribution.

How the single resample is shared. Every measure reuses the *frozen* bootstrap
kernels in compute_scores.py (dprime_stats for engage/suppress/coverage, twoway
for dial/temporal), but instead of each drawing its own multinomial weights we
draw ONE canonical sentence-multiplicity matrix Wsent (B x |S_all|) and ONE
canonical concept-multiplicity matrix Mconc (B x |C_all|), then PROJECT them onto
each measure's own sentence/concept axis (a measure simply sees the multiplicities
of the sentences/concepts it has data for). Replicate b is therefore the same
resample in every measure, which is exactly the joint two-way cluster bootstrap.

Point estimate. Computed here from the observed (equal-weight) statistics; it is
validated to match aggregate_scalar.py's JSON-based S (same links, same weights).

Data source: stored run artifacts only, via score_data (no model load). Reads a
run dir (default: newest results/raw/*_activation_control) so it can
recompute the measures on each resample -- the JSON alone is not enough for a CI.

Usage:
  python scripts/scalar_ci.py --main-run results/raw/<RUN>
  python scripts/scalar_ci.py --main-run results/raw/<RUN> --n-boot 2000 --json out.json
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import compute_scores as cs                                                # noqa: E402
import score_data as sd                                                    # noqa: E402
from score_data import POS, NEG, RAMP, signed_spearman, CATS              # noqa: E402
import aggregate_scalar as agg                                            # noqa: E402

_PHI = NormalDist().cdf
_PHI_INV = NormalDist().inv_cdf
_SQRT2 = math.sqrt(2.0)
_EPS = 1e-6

TEMP_CONDS = sd.TARGET_GROUPS["temporal_control"]   # loc_beginning, persist_once, loc_end


# ---- per-replicate p links (0.5 = chance), matching aggregate_scalar ------------
# LINK is set from --link in main(); the d' measures route through _p_score(measure).

LINK = "linear"


def _p_phi(reps):
    return np.clip(np.array([_PHI(x / _SQRT2) if np.isfinite(x) else np.nan for x in reps]),
                   _EPS, 1 - _EPS)


def _p_rank(reps):
    return np.clip((np.asarray(reps, float) + 1.0) / 2.0, _EPS, 1 - _EPS)


def _p_score(measure, reps):
    """Link for a d'/contrast measure's replicate array, per the active LINK:
    'linear' = clip(0.5 + score/(2*D_REF[measure])); 'phi' = Phi(score/sqrt2)."""
    if LINK == "linear":
        return np.clip(0.5 + np.asarray(reps, float) / (2.0 * agg.D_REF[measure]), _EPS, 1 - _EPS)
    return _p_phi(reps)


# ---- input extraction (feature-only; no stats) ----------------------------------

def _blocks_with_concepts(vals_cond, bases_ch, concept_order):
    """cs.per_concept_blocks, but also returning all_sents and the KEPT concept
    order so a shared draw can be projected onto both axes."""
    all_sents = sorted({s for c in bases_ch for s in bases_ch[c]})
    s_idx = {s: i for i, s in enumerate(all_sents)}
    blocks, kept = [], []
    for c in concept_order:
        ss = sorted(set(vals_cond.get(c, {})) & set(bases_ch.get(c, {})))
        if len(ss) < 3:
            continue
        X = np.vstack([vals_cond[c][s] for s in ss])
        B = np.vstack([bases_ch[c][s] for s in ss])
        blocks.append((X, B, np.array([s_idx[s] for s in ss])))
        kept.append(c)
    return blocks, all_sents, kept


def _dial_units(run):
    """UR/UW unit arrays + sid/cid per channel (copied feature-extraction from
    compute_scores.rows_3_4; no bootstrap here)."""
    layers, vals, _ = sd.unit_layer_readouts(run, RAMP)
    n_L = len(layers)
    nl_total = sd.run_n_layers(run)
    deep = [li for li, L in enumerate(layers) if L / nl_total >= 0.5]
    units = {ch: dict(rank=[], win=[]) for ch in ("cos", "relnorm", "proj")}
    sids, cids = [], []
    keys = sorted({(c, s) for cond in RAMP
                   for c in vals[("cos", cond)] for s in vals[("cos", cond)][c]})
    for c, s in keys:
        for ch in ("cos", "relnorm", "proj"):
            V = np.full((4, n_L), np.nan)
            for k, cond in enumerate(RAMP):
                v = vals[(ch, cond)].get(c, {}).get(s)
                if v is not None:
                    V[k] = v
            rk = np.full(n_L, np.nan)
            wn = np.full((3, n_L), np.nan)
            for li in range(n_L):
                col = V[:, li]
                lv = [k for k in range(4) if np.isfinite(col[k])]
                if len(lv) >= 3:
                    rk[li] = signed_spearman(lv, [col[k] for k in lv])
                for k in range(3):
                    a, b = col[k + 1], col[k]
                    if np.isfinite(a) and np.isfinite(b):
                        wn[k, li] = 1.0 if a > b else (0.5 if a == b else 0.0)
            units[ch]["rank"].append(rk)
            units[ch]["win"].append(wn)                  # full (3 pairs x n_L)
        sids.append(s); cids.append(c)
    sid, cid = np.array(sids), np.array(cids)
    return {ch: (np.vstack(units[ch]["rank"]), np.stack(units[ch]["win"]), sid, cid)
            for ch in ("cos", "relnorm", "proj")}


def _temporal_blocks(run):
    """s_block[ch][cond] = (su, sids, cids) for the temporal conds (copied
    standardized-contrast construction from compute_scores.rows_targeting)."""
    store = sd.location_units(run)
    out = {}
    for ch in ("cos", "relnorm", "proj"):
        sig_on, sig_off = {}, {}
        for cond in TEMP_CONDS:
            bi, bo = store[(ch, cond, "base", "G_in")], store[(ch, cond, "base", "G_out")]
            per_c = defaultdict(lambda: ([], []))
            for (s, c), v in bi.items():
                if (s, c) in bo:
                    per_c[c][0].append(v); per_c[c][1].append(bo[(s, c)])
            for c, (vi, vo) in per_c.items():
                if len(vi) >= 3:
                    sig_on[(cond, c)] = np.std(vi, ddof=1)
                    sig_off[(cond, c)] = np.std(vo, ddof=1)
        s_block = {}
        for cond in TEMP_CONDS:
            gi_l, go_l = store[(ch, cond, "loc", "G_in")], store[(ch, cond, "loc", "G_out")]
            gi_t, go_t = store[(ch, cond, "think", "G_in")], store[(ch, cond, "think", "G_out")]
            keys = sorted(set(gi_l) & set(go_l) & set(gi_t) & set(go_t))
            su, sids, cids = [], [], []
            for (s, c) in keys:
                if (cond, c) not in sig_on or sig_on[(cond, c)] <= 0 or sig_off[(cond, c)] <= 0:
                    continue
                so, sf = sig_on[(cond, c)], sig_off[(cond, c)]
                su.append((gi_l[(s, c)] / so - go_l[(s, c)] / sf)
                          - (gi_t[(s, c)] / so - go_t[(s, c)] / sf))
                sids.append(s); cids.append(c)
            s_block[cond] = (np.array(su), np.array(sids), np.array(cids))
        out[ch] = s_block
    return out


# ---- joint bootstrap ------------------------------------------------------------

def _rank_stat(ms):                                       # mean rank at the fixed L*
    return float(np.nanmean(ms[0]))


def _res_stat(ms):
    A = float(np.nanmean(ms[0]))
    A = min(max(A, 1e-6), 1 - 1e-6)
    return float(np.sqrt(2) * _PHI_INV(A))


def _temp_stat(ms):
    return float(np.nanmean([m[0] for m in ms]))


COVERAGE_CH = ("cos", "proj")     # coverage exists only on these channels


def joint_bootstrap(run, *, channels, n_boot=2000, seed=0):
    """Return (S_obs, S_lo, S_hi, detail) for the given channel set. detail
    carries per-component observed scores and marginal CIs for validation against
    the JSON. Weights follow the measure-equal / split-within scheme of
    aggregate_scalar.py, derived from the components actually built."""
    # ---- extract every measure's inputs once ----
    layers_list, valsES, basesES = sd.unit_layer_readouts(run, [POS, NEG])
    order_es = sorted(basesES["cos"])
    layers_es = len(layers_list)
    valsCov, basesCov = sd.pos_category_readouts(run, [POS])
    dial = _dial_units(run)
    temporal = _temporal_blocks(run)

    # ---- canonical (sentence, concept) universes across ALL measures ----
    S_all, C_all = set(), set()

    def note(sents, concs):
        S_all.update(sents); C_all.update(concs)

    # engage always; suppress only if it's a kept measure (excluded from S 2026-07-17).
    es_conds = [(POS, +1.0)] + ([(NEG, -1.0)] if "suppress" in agg.KEPT_MEASURES else [])
    es_inputs = {}      # (cond, ch) -> (blocks, all_sents, kept, sign)
    for ch in channels:
        for cond, sign in es_conds:
            blocks, all_sents, kept = _blocks_with_concepts(valsES[(ch, cond)], basesES[ch], order_es)
            es_inputs[(cond, ch)] = (blocks, all_sents, kept, sign)
            note(all_sents, kept)
    cov_inputs = {}     # ch -> (blocks, all_sents, kept)
    for ch in channels:
        if ch not in COVERAGE_CH:
            continue
        cb, cs_, ck = _blocks_with_concepts(valsCov[(ch, POS)], basesCov[ch], sorted(basesCov[ch]))
        cov_inputs[ch] = (cb, cs_, ck)
        note(cs_, ck)
    for ch in channels:
        UR, UW, sid, cid = dial[ch]
        note(sid, cid)
        for cond in TEMP_CONDS:
            _, sids, cids = temporal[ch][cond]
            note(sids, cids)

    S_all = sorted(S_all); C_all = sorted(C_all)
    si = {s: i for i, s in enumerate(S_all)}
    ci = {c: i for i, c in enumerate(C_all)}

    # ---- ONE canonical two-way resample, shared by projection ----
    rng = np.random.default_rng(seed)
    Wsent = rng.multinomial(len(S_all), np.full(len(S_all), 1 / len(S_all)),
                            size=n_boot).astype(float)
    Mconc = rng.multinomial(len(C_all), np.full(len(C_all), 1 / len(C_all)),
                            size=n_boot).astype(float)

    def projW(sent_list):
        return Wsent[:, [si[s] for s in sent_list]]

    def projM(conc_list):
        return Mconc[:, [ci[c] for c in conc_list]]

    dummy = np.random.default_rng(0)
    comp = {}     # name -> dict(obs=float, reps=array(B), p=array(B), weight=float)

    # engage / suppress (dprime kernel, peak over layers) + coverage (min over POS)
    for (cond, ch), (blocks, all_sents, kept, sign) in es_inputs.items():
        st = cs.dprime_stats(blocks, len(all_sents), layers_es, dummy,
                             W=projW(all_sents), Mc=projM(kept), n_perm=0)
        obs = float(np.nanmax(sign * st["dp"]))
        reps = np.nanmax(sign * st["bavg"], axis=1)
        meas = "engage" if cond == POS else "suppress"
        comp[f"{meas}|{ch}"] = dict(obs=obs, reps=reps, p=_p_score(meas, reps))
    for ch, (cb, cs_, ck) in cov_inputs.items():
        st = cs.dprime_stats(cb, len(cs_), len(CATS), dummy,
                             W=projW(cs_), Mc=projM(ck), n_perm=0)
        cov_reps = np.nanmin(st["bavg"], axis=1)
        comp[f"coverage|{ch}"] = dict(obs=float(np.nanmin(st["dp"])), reps=cov_reps,
                                      p=_p_score("coverage", cov_reps))

    # dial rank / resolution (twoway kernel) -- BOTH-PEAK: both read at the same
    # peak-dial layer L* (argmax mean-unit rank), matching compute_scores.rows_3_4.
    for ch in channels:
        UR, WIN, sid, cid = dial[ch]
        sents_ax, concs_ax = cs.twoway_axes([(UR, sid, cid)])
        Ws, Wc = projW(sents_ax), projM(concs_ax)
        li = int(np.nanargmax(np.nanmean(UR, axis=0)))
        URstar = UR[:, li:li + 1]
        UWstar = WIN[:, :, li]
        obs_r, reps_r = cs.twoway([(URstar, sid, cid)], stat=_rank_stat, Ws=Ws, Wc=Wc, return_reps=True)
        obs_s, reps_s = cs.twoway([(UWstar, sid, cid)], stat=_res_stat, Ws=Ws, Wc=Wc, return_reps=True)
        comp[f"dial_rank|{ch}"] = dict(obs=obs_r, reps=reps_r, p=_p_rank(reps_r))
        comp[f"dial_resolution|{ch}"] = dict(obs=obs_s, reps=reps_s,
                                             p=_p_score("dial_resolution", reps_s))

    # temporal control (twoway kernel over the three temporal conds)
    for ch in channels:
        blocks = [temporal[ch][c] for c in TEMP_CONDS if len(temporal[ch][c][0])]
        sents_ax, concs_ax = cs.twoway_axes(blocks)
        obs_t, reps_t = cs.twoway(blocks, stat=_temp_stat,
                                  Ws=projW(sents_ax), Wc=projM(concs_ax), return_reps=True)
        comp[f"temporal_control|{ch}"] = dict(obs=obs_t, reps=reps_t,
                                              p=_p_score("temporal_control", reps_t))

    # measure-equal weights, split within a measure across its built channels
    # (matches aggregate_scalar.model_scalar: w = (1/N_measures)/n_channels).
    by_measure = {}
    for nm in comp:
        by_measure.setdefault(nm.split("|")[0], []).append(nm)
    Nm = len(by_measure)
    for nms in by_measure.values():
        for nm in nms:
            comp[nm]["weight"] = (1.0 / Nm) / len(nms)

    # ---- aggregate each replicate into S (measure-equal weighted geometric mean) ----
    names = list(comp)
    W = np.array([comp[n]["weight"] for n in names])
    P = np.vstack([comp[n]["p"] for n in names])            # (n_comp x B)
    logG = (W[:, None] * np.log(P)).sum(0)                  # any NaN component -> NaN replicate
    G = np.exp(logG)
    S_reps = np.clip(2.0 * G - 1.0, 0.0, 1.0)
    S_reps = S_reps[np.isfinite(S_reps)]
    dropped = n_boot - len(S_reps)

    # observed S from the equal-weight (obs) statistics
    p_obs = {}
    for n in names:
        m = n.split("|")[0]
        s = comp[n]["obs"]
        if m == "dial_rank":
            p_obs[n] = (s + 1) / 2
        elif LINK == "linear":
            p_obs[n] = 0.5 + s / (2.0 * agg.D_REF[m])
        else:
            p_obs[n] = _PHI(s / _SQRT2)
        p_obs[n] = min(max(p_obs[n], _EPS), 1 - _EPS)
    logG_obs = sum(comp[n]["weight"] * math.log(p_obs[n]) for n in names)
    S_obs = min(max(2.0 * math.exp(logG_obs) - 1.0, 0.0), 1.0)

    detail = {n: dict(obs=comp[n]["obs"], p_obs=p_obs[n],
                      lo=float(np.nanpercentile(comp[n]["reps"], 2.5)),
                      hi=float(np.nanpercentile(comp[n]["reps"], 97.5)))
              for n in names}
    return (S_obs, float(np.percentile(S_reps, 2.5)), float(np.percentile(S_reps, 97.5)),
            dict(components=detail, n_boot=n_boot, dropped=dropped,
                 n_sentences=len(S_all), n_concepts=len(C_all)))


def main():
    ap = argparse.ArgumentParser(description="Joint two-way bootstrap CI for the controllability scalar S.")
    ap.add_argument("--main-run", default=None,
                    help="run dir; default: newest results/raw/*_activation_control")
    ap.add_argument("--channels", choices=list(agg.CHANNEL_SETS), default="projection",
                    help="channel set to collapse (default: projection = the paper channel)")
    ap.add_argument("--link", choices=agg.LINKS, default="linear",
                    help="score->p link (default: linear-clip vs D_REF; 'phi' = legacy AUROC)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write S + CI + component detail here")
    args = ap.parse_args()
    channels = agg.CHANNEL_SETS[args.channels]
    global LINK
    LINK = args.link
    if args.main_run is None:
        args.main_run = cs._latest_run()
        if args.main_run is None:
            ap.error("no results/raw/*_activation_control run found; pass --main-run")
        print(f"[auto] --main-run {args.main_run}")

    print(f"joint two-way bootstrap ({args.n_boot} replicates, seed {args.seed}, "
          f"channels={args.channels}={'/'.join(channels)}) ...", flush=True)
    S_obs, S_lo, S_hi, detail = joint_bootstrap(args.main_run, channels=channels,
                                                n_boot=args.n_boot, seed=args.seed)
    model = sd._resolve_model(args.main_run, None)

    print("\n" + "=" * 78)
    print(f"model: {model}   ({detail['n_sentences']} sentences x {detail['n_concepts']} concepts, "
          f"{detail['dropped']} degenerate replicates dropped)   channels={args.channels}")
    print("-" * 78)
    print(f"  S = {S_obs:.4f}   95% joint CI [{S_lo:.4f}, {S_hi:.4f}]")
    print("-" * 78)
    print("  per-component observed score and marginal 95% CI (native units):")
    for n, d in detail["components"].items():
        print(f"    {n:26s} {d['obs']:+8.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]   p={d['p_obs']:.3f}")
    print("=" * 78)

    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump({"model": model, "main_run": str(args.main_run),
                       "channel_set": args.channels, "channels": list(channels),
                       "link": args.link, "d_ref": (agg.D_REF if args.link == "linear" else None),
                       "scalar": S_obs, "ci_lo": S_lo, "ci_hi": S_hi,
                       "n_boot": args.n_boot, "seed": args.seed,
                       "n_sentences": detail["n_sentences"],
                       "n_concepts": detail["n_concepts"],
                       "dropped_replicates": detail["dropped"],
                       "components": detail["components"]}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
