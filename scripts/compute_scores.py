#!/usr/bin/env python3
"""Compute the SCORES.md battery, TWO-COLUMN (cos | relnorm), with uncertainty.

Reads ONLY the stored run artifacts, via scripts/score_data.py (results.json,
no_instruction_cache.pkl, the concept-vector cache, pos_tags.json). No figure
script is imported anywhere in the scoring path -- editing/retiring a figure
cannot change a score.

Channel conventions: instructions define a direction on BOTH channels --
cosine: engage = toward the concept, suppress = away; relative norm:
engage = norm UP (add activation mass), suppress = norm DOWN. Row 10
(coverage) is cosine-only (mixed signs on the norm channel).

Uncertainty = 95% two-way cluster bootstrap (sentences AND concepts):
  rows 1, 2, 10 : sigma recomputed per replicate; peak/min taken INSIDE each
                  replicate (mildly optimistic for extrema).
  targeting/dial rows : unit-level two-way bootstrap, sigma / win definitions
                  fixed at their observed values.
  (Onset/offset error: computed but excluded from the scalar; see row_onset_offset.)

Usage:
  python scripts/compute_scores.py
  python scripts/compute_scores.py --main-run ... --lt-run ...
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_data as sd                                                    # noqa: E402
from score_data import POS, NEG, RAMP, signed_spearman                    # noqa: E402


def _latest_run(pattern="*_activation_control", *, needs_baseline=True):
    """Newest matching run under results/raw/ (timestamp-led names sort
    chronologically), or None. Used only as a convenience default so the
    battery follows whatever model was last run instead of a fixed run dir.

    `needs_baseline`: only consider runs carrying a no_instruction_cache.pkl (the
    baseline the battery needs), so layer-targeting/one-off runs sharing the same
    dir suffix don't get picked as the main run."""
    runs = sorted((PROJECT_ROOT / "results/raw").glob(pattern))
    if needs_baseline:
        runs = [r for r in runs if (r / "no_instruction_cache.pkl").exists()]
    return str(runs[-1].relative_to(PROJECT_ROOT)) if runs else None


PHI_INV = NormalDist().inv_cdf

# Channels emitted to JSON. proj (projection = ||r||*cos) is the paper's default
# collapse channel; cos/relnorm are retained for the legacy two-channel scalar.
CHANNELS = ("cos", "relnorm", "proj")
COVERAGE_CHANNELS = ("cos", "proj")   # relnorm deliberately excluded (bipolar by POS)


# ---- statistics ------------------------------------------------------------------

def bh_fdr(pmat):
    """Benjamini-Hochberg q-values over the non-nan cells."""
    pmat = np.asarray(pmat, float)
    flat = pmat.ravel()
    idx = np.where(~np.isnan(flat))[0]
    q = np.full_like(flat, np.nan)
    if len(idx) == 0:
        return q.reshape(pmat.shape)
    p = flat[idx]
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty(m)
    qv[order] = np.clip(ranked, 0, 1)
    q[idx] = qv
    return q.reshape(pmat.shape)


def dprime_stats(per_concept, S_global, n_L, rng, n_boot=2000, n_perm=5000,
                 W=None, Mc=None):
    """Concept-averaged d' per column + TWO-WAY cluster-bootstrap replicates.

    per_concept: list of (X, B, idx); X, B (S_c x n_L) condition/baseline
    values over the concept's sentences, idx = global sentence indices.
    (Frozen copy of the promoted Fig 2 machinery -- kept verbatim, including
    the sign-flip draw, so the battery numbers are reproducible bit-for-bit.)

    W (n_boot x S_global) sentence-multiplicity and Mc (n_boot x nC) concept-
    multiplicity draws may be supplied externally so several measures share ONE
    two-way resample per replicate (the joint bootstrap in scalar_ci.py); when
    given, n_boot follows W. Passing n_perm=0 skips the permutation null (the
    joint-CI path needs only bavg)."""
    nC = len(per_concept)
    if W is not None:
        n_boot = W.shape[0]
    else:
        W = rng.multinomial(S_global, np.full(S_global, 1.0 / S_global),
                            size=n_boot).astype(float)
    if Mc is None:
        Mc = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=n_boot).astype(float)
    E = rng.integers(0, 2, size=(n_perm, S_global)) * 2.0 - 1.0 if n_perm > 0 \
        else np.zeros((0, S_global))
    obs = np.full((nC, n_L), np.nan)
    boot = np.full((nC, n_boot, n_L), np.nan)
    null = np.full((nC, n_perm, n_L), np.nan)
    for ci, (X, B, idx) in enumerate(per_concept):
        if X.shape[0] < 3:
            continue
        Wc, Ec = W[:, idx], E[:, idx]
        for li in range(n_L):
            x, b = X[:, li], B[:, li]
            ok = ~(np.isnan(x) | np.isnan(b))
            if ok.sum() < 3:
                continue
            x, b = x[ok], b[ok]
            sd_ = b.std(ddof=1)
            if sd_ <= 0:
                continue
            obs[ci, li] = (x.mean() - b.mean()) / sd_
            w = Wc[:, ok]
            n = w.sum(1)
            good = n > 1
            mB = np.divide(w @ b, n, out=np.full(n_boot, np.nan), where=good)
            mX = np.divide(w @ x, n, out=np.full(n_boot, np.nan), where=good)
            eB2 = np.divide(w @ (b * b), n, out=np.full(n_boot, np.nan), where=good)
            var = (eB2 - mB ** 2) * np.divide(n, n - 1, out=np.ones(n_boot), where=good)
            sd_b = np.sqrt(np.clip(var, 1e-24, None))
            boot[ci, :, li] = (mX - mB) / sd_b
            d = x - b
            null[ci, :, li] = (Ec[:, ok] @ d) / ok.sum() / sd_
    dp = np.nanmean(obs, axis=0)
    wts = Mc.T[:, :, None]
    okb = ~np.isnan(boot)
    num = np.nansum(np.where(okb, boot, 0.0) * wts, axis=0)
    den = (wts * okb).sum(axis=0)
    bavg = np.divide(num, den, out=np.full((n_boot, n_L), np.nan), where=den > 0)
    navg = np.nanmean(null, axis=0)
    p = np.full(n_L, np.nan)
    for li in range(n_L):
        if not np.isnan(dp[li]):
            p[li] = (1 + int((np.abs(navg[:, li]) >= abs(dp[li]) - 1e-15).sum())) \
                / (n_perm + 1)
    return dict(dp=dp, bavg=bavg, q=bh_fdr(p))


def per_concept_blocks(vals_cond, bases_ch, concept_order):
    """(X, B, idx) per concept + global sentence count, from score_data dicts."""
    all_sents = sorted({s for c in bases_ch for s in bases_ch[c]})
    s_idx = {s: i for i, s in enumerate(all_sents)}
    blocks = []
    for c in concept_order:
        ss = sorted(set(vals_cond.get(c, {})) & set(bases_ch.get(c, {})))
        if len(ss) < 3:
            continue
        X = np.vstack([vals_cond[c][s] for s in ss])
        B = np.vstack([bases_ch[c][s] for s in ss])
        blocks.append((X, B, np.array([s_idx[s] for s in ss])))
    return blocks, len(all_sents)


def twoway_axes(values_list):
    """The (sents, concs) sorted resampling axes twoway builds internally -- so a
    caller can project a shared canonical draw onto this measure's own order."""
    sents = sorted({s for _, sid, _ in values_list for s in sid})
    concs = sorted({c for _, _, cid in values_list for c in cid})
    return sents, concs


def twoway(values_list, n_boot=2000, seed=0, stat=None,
           Ws=None, Wc=None, return_reps=False):
    """Joint two-way cluster bootstrap over unit-level values. values_list:
    [(vals (n_u x k), sent_ids, conc_ids)]; stat maps the per-block
    weighted-mean vectors -> scalar.

    Ws (n_boot x len(sents)) / Wc (n_boot x len(concs)) draws may be supplied
    externally (aligned to twoway_axes(values_list)) so several measures share
    ONE resample per replicate. return_reps=True returns (obs, reps) with reps
    the FULL length-n_boot per-replicate array (NaNs kept, indices preserved for
    cross-measure alignment) instead of the (lo, hi) percentiles."""
    rng = np.random.default_rng(seed)
    sents, concs = twoway_axes(values_list)
    s_i = {s: i for i, s in enumerate(sents)}
    c_i = {c: i for i, c in enumerate(concs)}
    if Ws is None:
        Ws = rng.multinomial(len(sents), np.full(len(sents), 1 / len(sents)), size=n_boot).astype(float)
    else:
        n_boot = Ws.shape[0]
    if Wc is None:
        Wc = rng.multinomial(len(concs), np.full(len(concs), 1 / len(concs)), size=n_boot).astype(float)

    obs_means, boot_means = [], []
    for vals, sid, cid in values_list:
        vals = np.asarray(vals, float)
        if vals.ndim == 1:
            vals = vals[:, None]
        fin = np.isfinite(vals)
        v0 = np.where(fin, vals, 0.0)
        obs_means.append(np.divide(v0.sum(0), fin.sum(0),
                                   out=np.full(vals.shape[1], np.nan),
                                   where=fin.sum(0) > 0))
        si = np.array([s_i[s] for s in sid]); ci = np.array([c_i[c] for c in cid])
        w = Ws[:, si] * Wc[:, ci]
        num = w @ v0
        den = w @ fin.astype(float)
        boot_means.append(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
    obs = stat(obs_means)
    reps = np.array([stat([bm[b] for bm in boot_means]) for b in range(n_boot)])
    if return_reps:
        return obs, reps
    reps = reps[np.isfinite(reps)]
    return obs, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# ---- rows ------------------------------------------------------------------------

def rows_1_2(main_run, channels=("cos", "relnorm")):
    """Engage/suppress per channel: peak d' in the INSTRUCTED direction
    (cos: toward/away; relnorm: up/down). SIGNED (no floor) so a negative peak —
    i.e. the model's best layer still moves the wrong way (e.g. a suppression
    rebound) — is reported as a real negative rather than masked as 0. The
    `wrong` companion is the most-wrong-way layer (−min over layers)."""
    layers, vals, bases = sd.unit_layer_readouts(main_run, [POS, NEG])
    concept_order = sorted(bases["cos"])
    rng = np.random.default_rng(0)
    out = {}
    for ch in channels:
        for row, cond, sign in (("engage", POS, +1.0), ("suppress", NEG, -1.0)):
            blocks, S = per_concept_blocks(vals[(ch, cond)], bases[ch], concept_order)
            st = dprime_stats(blocks, S, len(layers), rng)
            dirc = sign * st["dp"]
            li = int(np.nanargmax(dirc))
            pk = np.nanmax(sign * st["bavg"], axis=1)
            out[(row, ch)] = dict(score=float(dirc[li]),
                                  lo=float(np.nanpercentile(pk, 2.5)),
                                  hi=float(np.nanpercentile(pk, 97.5)),
                                  layer=layers[li],
                                  wrong=float(-np.nanmin(dirc)))
    return layers, out


def rows_3_4(main_run, channels=("cos", "relnorm")):
    layers, vals, _ = sd.unit_layer_readouts(main_run, RAMP)
    n_L = len(layers)
    nl_total = sd.run_n_layers(main_run)
    deep = [li for li, L in enumerate(layers) if L / nl_total >= 0.5]

    units = {ch: dict(rank=[], win=[]) for ch in channels}
    sent_ids, conc_ids = [], []
    keys = sorted({(c, s) for cond in RAMP
                   for c in vals[("cos", cond)] for s in vals[("cos", cond)][c]})
    for c, s in keys:
        for ch in channels:
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
        sent_ids.append(s); conc_ids.append(c)

    sid = np.array(sent_ids); cid = np.array(conc_ids)
    out = {}

    def res_stat(ms):
        A = float(np.nanmean(ms[0]))
        A = min(max(A, 1e-6), 1 - 1e-6)
        return float(np.sqrt(2) * PHI_INV(A))

    for ch in channels:
        UR = np.vstack(units[ch]["rank"])                # (n_units, n_L)
        WIN = np.stack(units[ch]["win"])                 # (n_units, 3, n_L)
        # BOTH-PEAK convention (2026-07-15): Rank and Resolution are read at the
        # SAME peak-dial layer L* (argmax mean-unit rank), so they can't silently
        # disagree the way the old Rank@peak / Resolution@deep-pool mix could.
        li = int(np.nanargmax(np.nanmean(UR, axis=0)))
        rank_obs, rank_lo, rank_hi = twoway(
            [(UR[:, li:li + 1], sid, cid)], stat=lambda ms: float(np.nanmean(ms[0])))
        UWstar = WIN[:, :, li]                            # 3 adjacent pairs at L*
        res_obs, res_lo, res_hi = twoway([(UWstar, sid, cid)], stat=res_stat)
        A_star = float(np.nanmean(np.nanmean(UWstar, axis=0)))
        # secondary: deep-band POOLED resolution (the conservative robustness check)
        UWpool = WIN[:, :, deep].reshape(WIN.shape[0], -1)
        resp_obs, resp_lo, resp_hi = twoway([(UWpool, sid, cid)], stat=res_stat)
        A_pool = float(np.nanmean(np.nanmean(UWpool, axis=0)))
        out[("rank", ch)] = (rank_obs, rank_lo, rank_hi, layers[li])
        out[("resolution", ch)] = (res_obs, res_lo, res_hi, A_star)
        out[("resolution_pool", ch)] = (resp_obs, resp_lo, resp_hi, A_pool)
    return out


def rows_targeting(main_run):
    """Temporal control + Token group, each as the in/out CONTRAST (`<group>`):
    did the concept fall in the right position / on the right token group
    (located − generic, standardized)? persist_once feeds 'mid' as a middle-third
    span. Returns out[(measure_key, ch)] = (obs, lo, hi, per_task). Higher =
    better. (The CoM-precision companions were retired 2026-07-14 and are no
    longer computed.)"""
    store = sd.location_units(main_run)
    out = {}
    for ch in CHANNELS:
        sig_on, sig_off = {}, {}
        for cond, _ in sd.TARGET_TASKS:
            bi, bo = store[(ch, cond, "base", "G_in")], store[(ch, cond, "base", "G_out")]
            per_c = defaultdict(lambda: ([], []))
            for (s, c), v in bi.items():
                if (s, c) in bo:
                    per_c[c][0].append(v); per_c[c][1].append(bo[(s, c)])
            for c, (vi, vo) in per_c.items():
                if len(vi) >= 3:
                    sig_on[(cond, c)] = np.std(vi, ddof=1)
                    sig_off[(cond, c)] = np.std(vo, ddof=1)

        s_block, s_task = {}, {}
        for cond, _ in sd.TARGET_TASKS:
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
            s_task[cond] = float(np.nanmean(su)) if su else np.nan

        def grouped(block, task, conds):
            blocks = [block[c] for c in conds if c in block and len(block[c][0])]
            if blocks:
                obs, lo, hi = twoway(blocks, stat=lambda ms: float(np.nanmean([m[0] for m in ms])))
            else:
                obs = lo = hi = np.nan
            return (obs, lo, hi, {c: task.get(c) for c in conds})

        for group, conds in sd.TARGET_GROUPS.items():
            out[(group, ch)] = grouped(s_block, s_task, conds)
    return out


def row_9(lt_run, main_run):
    """Layer targeting per channel. Tolerates a missing LT run (lt_run None or with
    no usable think_at_layer units): returns null cells so a MAIN-only run still
    produces a near-complete SCORES.json -- the later LT run fills this row in."""
    null = {ch: (np.nan, np.nan, np.nan, {}) for ch in CHANNELS}
    if lt_run is None:
        return null
    try:
        data = sd.layer_targeting_units(lt_run, main_run)
    except Exception as e:                                            # noqa: BLE001
        print(f"  [layer_targeting] no LT data ({type(e).__name__}); emitting null")
        return null
    out = {}
    for ch in CHANNELS:
        D, sid, cid, layers_lt = data[ch]
        if D.size == 0 or len(sid) == 0:                             # LT run present but empty
            out[ch] = (np.nan, np.nan, np.nan, {})
            continue
        obs, lo, hi = twoway([(D, sid, cid)], stat=lambda ms: float(np.nanmean(ms[0])))
        per_layer = {int(L): float(round(v, 3))
                     for L, v in zip(layers_lt, np.nanmean(D, axis=0))}
        out[ch] = (obs, lo, hi, per_layer)
    return out


def row_10(main_run, channels=COVERAGE_CHANNELS):
    """Coverage = min-POS engage d' per channel. Returns {ch: (score, lo, hi,
    weakest_cat)}. relnorm excluded by design (norm engage is bipolar across POS);
    proj added for the projection-channel scalar."""
    vals, bases = sd.pos_category_readouts(main_run, [POS, NEG])
    rng = np.random.default_rng(0)
    out = {}
    for ch in channels:
        concept_order = sorted(bases[ch])
        blocks, S = per_concept_blocks(vals[(ch, POS)], bases[ch], concept_order)
        st = dprime_stats(blocks, S, len(sd.CATS), rng)
        gi = int(np.nanargmin(st["dp"]))
        mn = np.nanmin(st["bavg"], axis=1)
        out[ch] = (float(st["dp"][gi]), float(np.nanpercentile(mn, 2.5)),
                   float(np.nanpercentile(mn, 97.5)), sd.CATS[gi])
    return out


# ---- onset/offset error (computed, but EXCLUDED from the scalar as a coarse
#      timing diagnostic) -- ported from the retired battery 2026-07-17, now on all three
#      channels. Told to START thinking about X after the 4th token (an onset gate)
#      or STOP after the first half (an offset gate), how far off is X's actual
#      on/off edge? Lower = better; per-edge CI only (no aggregate CI). ---------------

N_BINS = 10
PERSIST_AFTER, PERSIST_FIRST = "persist_after_fourth", "persist_first_half"
# The two PRECISE token-boundary gates: start-after-4th (onset), first-half (offset).
# (persist_once is NOT here -- it feeds Temporal control as the middle-third span.)
TOKEN_PRECISION_EDGES = [("onset", PERSIST_AFTER), ("offset", PERSIST_FIRST)]
EDGE_LABEL = {PERSIST_AFTER: "After 4th", PERSIST_FIRST: "First half"}
# channel -> depth FRACTION for the timing readout (proj at the targeting depth)
_EDGE_FRAC = {"cos": 1.00, "relnorm": 0.70, "proj": None}     # proj filled from sd.PROJ_F_LOC


def _bins_for(n_tok, n_bins=N_BINS):
    if n_tok <= 1:
        return np.zeros(max(n_tok, 0), dtype=int)
    f = np.arange(n_tok) / (n_tok - 1)
    return np.minimum((f * n_bins).astype(int), n_bins - 1)


def _bin_means(delta_vec, bin_idx, n_bins=N_BINS):
    out = np.full(n_bins, np.nan)
    m = min(len(delta_vec), len(bin_idx))
    for b in range(n_bins):
        sel = [delta_vec[i] for i in range(m) if bin_idx[i] == b and not np.isnan(delta_vec[i])]
        if sel:
            out[b] = float(np.mean(sel))
    return out


def _detect_edges(profile):
    """Half-max rising/falling crossings on a rectified profile -> (onset, offset)
    fractional positions, or (nan, nan) if flat."""
    p = np.clip(profile, 0, None)
    if not np.isfinite(p).all() or p.max() - p.min() <= 1e-12:
        return np.nan, np.nan
    above = np.where(p >= p.min() + 0.5 * (p.max() - p.min()))[0]
    if len(above) == 0:
        return np.nan, np.nan
    return (above[0] + 0.5) / len(p), (above[-1] + 0.5) / len(p)


def _persistence_edges(main_run, ch, *, vector_cache="results/vector_cache",
                       method="baseline", n_boot=2000, seed=0):
    """Edge-timing error (detected − requested, fractional) per gate, on channel
    `ch` in {cos, relnorm, proj}. Returns {(edge, cond): dict(mean, lo, hi, req,
    detected)}. proj = per-token cos * raw ||r|| minus the same for baseline."""
    frac = sd.PROJ_F_LOC if ch == "proj" else _EDGE_FRAC[ch]
    L = sd._layer_for_fraction(main_run, frac)
    need_cos, need_norm = ch in ("cos", "proj"), ch in ("relnorm", "proj")
    rows = sd.load_rows(main_run)
    by_sent = defaultdict(list)
    for r in rows:
        if r.get("is_compliant"):
            by_sent[r["sentence"]].append(r)
    cache = sd.load_baseline(main_run)
    vecs = sd.load_vectors(vector_cache, sd._resolve_model(main_run, None), [L], method)
    conds = [PERSIST_AFTER, PERSIST_FIRST]

    prof = defaultdict(dict)
    f5 = {}
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n = len(toks)
        if n < 4:
            continue
        classes = [sd.classify(t) for t in toks]
        bins = _bins_for(n)
        base_cos_all = concepts_L = raw_norm = base_rn = None
        if need_cos and L in vecs:
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            base_cos_all, concepts_L = A @ vecs[L][2].T, vecs[L][0]
        if need_norm:
            raw_norm = np.asarray(ent["norms"][L], np.float32)[:n]
            base_rn = sd.relnorm(raw_norm, classes)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in conds and c:
                byc[cond][c] = r
                concepts.add(c)

        def dsig(row, base_c):
            tc, tn = sd.trace(row, "cosine_sim", L), sd.trace(row, "norms", L)
            if ch == "cos":
                v = sd._fit_len(np.asarray(tc, np.float32)[:n], n) if tc is not None else None
            elif ch == "relnorm":
                v = sd._fit_len(sd.relnorm(np.asarray(tn, np.float32)[:n], classes), n) \
                    if tn is not None else None
            else:                                        # proj = cos * raw ||r||
                if tc is None or tn is None:
                    v = None
                else:
                    v = (sd._fit_len(np.asarray(tc, np.float32)[:n], n)
                         * sd._fit_len(np.asarray(tn, np.float32)[:n], n))
            return (v - base_c) if (v is not None and base_c is not None) else None

        for c in sorted(concepts):
            if need_cos and (base_cos_all is None or c not in concepts_L):
                continue
            if ch == "cos":
                base_c = sd._fit_len(base_cos_all[:, concepts_L.index(c)], n)
            elif ch == "relnorm":
                base_c = sd._fit_len(base_rn, n)
            else:
                base_c = sd._fit_len(base_cos_all[:, concepts_L.index(c)] * raw_norm, n)
            for cond in conds:
                r_ = byc.get(cond, {}).get(c)
                if r_ is None:
                    continue
                sig = dsig(r_, base_c)
                if sig is not None:
                    prof[cond][(s, c)] = _bin_means(sig, bins)
                    if cond == PERSIST_AFTER:
                        f5[(s, c)] = 4.0 / (n - 1)

    rng = np.random.default_rng(seed)
    req = {PERSIST_AFTER: float(np.mean(list(f5.values()))) if f5 else np.nan,  # onset
           PERSIST_FIRST: 0.5}                                                  # offset
    out = {}
    for edge, cond in TOKEN_PRECISION_EDGES:
        keys = list(prof[cond])
        if len(keys) < 3:
            out[(edge, cond)] = dict(mean=np.nan, lo=np.nan, hi=np.nan, req=np.nan, detected=np.nan)
            continue
        P = np.vstack([prof[cond][k] for k in keys])
        on0, off0 = _detect_edges(np.nanmean(P, 0))
        det0 = on0 if edge == "onset" else off0
        errs = []
        for _ in range(n_boot):
            bi = rng.integers(0, len(keys), size=len(keys))
            on, off = _detect_edges(np.nanmean(P[bi], 0))
            dv = on if edge == "onset" else off
            if not np.isnan(dv):
                errs.append(dv - req[cond])
        errs = np.array(errs)
        out[(edge, cond)] = dict(
            mean=float(det0 - req[cond]) if not np.isnan(det0) else np.nan,
            lo=float(np.percentile(errs, 2.5)) if len(errs) else np.nan,
            hi=float(np.percentile(errs, 97.5)) if len(errs) else np.nan,
            req=req[cond], detected=det0)
    return out


def row_onset_offset(main_run, channels=CHANNELS):
    """Onset/offset error per channel: mean |detected − requested| over the two
    token-boundary gates. Lower = better. Returns {ch: (mean_abs_err, edges)} with
    edges = [(edge, label, mean, lo, hi), ...]."""
    out = {}
    for ch in channels:
        stats = _persistence_edges(main_run, ch)
        errs, edges = [], []
        for edge, cond in TOKEN_PRECISION_EDGES:
            e = stats[(edge, cond)]
            if np.isfinite(e["mean"]):
                errs.append(abs(e["mean"]))
                edges.append((edge, EDGE_LABEL[cond], e["mean"], e["lo"], e["hi"]))
        out[ch] = (float(np.mean(errs)) if errs else np.nan, edges)
    return out


# ---- depth PROFILES (mean + band per depth) for the paper's curve panels --------
# Precomputed so the figures read FROZEN bands consistent with the peaks in SCORES
# (the band's peak equals the SCORES bar by construction -- same dprime_stats call).
# Six curves: engage/suppress (vs baseline), gain lexical/numeric (high vs low
# endpoint), rank lexical/numeric (signed Spearman). Same cheap inputs as SCORES
# (results.json + no_instruction_cache.pkl + vector_cache; never results.pkl).

INTENSELY = "think_intensely"


def _depth_pcts(layers, n_total):
    """Requested-fraction depth axis (multiples of 5%), model-independent."""
    if len(layers) == 20:
        return list(range(5, 101, 5))
    return [int(round(100 * L / n_total / 5) * 5) for L in layers]


def _profile_dprime(x_vals, y_vals, order, n_L, rng, sign=1.0):
    """d' per depth (+ 95% band) of x vs y (y as the 'baseline'/low endpoint)."""
    blocks, S = per_concept_blocks(x_vals, y_vals, order)
    st = dprime_stats(blocks, S, n_L, rng, n_perm=0)
    return (sign * st["dp"],
            np.nanpercentile(sign * st["bavg"], 2.5, axis=0),
            np.nanpercentile(sign * st["bavg"], 97.5, axis=0))


def _curve_twoway(U, sids, cids, n_boot=2000, seed=0):
    """Mean-over-units curve (per depth) + 95% two-way (sentence x concept) cluster
    bootstrap band. U: (n_units x n_L)."""
    rng = np.random.default_rng(seed)
    sents, concs = sorted(set(sids)), sorted(set(cids))
    si = {s: i for i, s in enumerate(sents)}
    cind = {c: i for i, c in enumerate(concs)}
    sidx = np.array([si[s] for s in sids])
    cidx = np.array([cind[c] for c in cids])
    Ws = rng.multinomial(len(sents), np.full(len(sents), 1 / len(sents)), size=n_boot).astype(float)
    Wc = rng.multinomial(len(concs), np.full(len(concs), 1 / len(concs)), size=n_boot).astype(float)
    w = Ws[:, sidx] * Wc[:, cidx]                     # (n_boot, n_u)
    fin = np.isfinite(U)
    num = w @ np.where(fin, U, 0.0)
    den = w @ fin.astype(float)
    reps = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return (np.nanmean(U, axis=0),
            np.nanpercentile(reps, 2.5, axis=0),
            np.nanpercentile(reps, 97.5, axis=0))


def _profile_rank(vals, ch, conds, levels, order, n_L, n_boot, seed=0):
    """Mean signed-Spearman-rank curve (+ band) over units, per depth, for the given
    intensity levels (2-level lexical or 4-level numeric)."""
    keys = None
    for cond in conds:
        pres = {(c, s) for c in vals[(ch, cond)] for s in vals[(ch, cond)][c]}
        keys = pres if keys is None else (keys & pres)
    keys = sorted(keys)
    need = max(2, len(conds) - 1)                     # 2-level needs both; 4-level needs >=3
    UR = np.full((len(keys), n_L), np.nan)
    for u, (c, s) in enumerate(keys):
        M = np.vstack([np.asarray(vals[(ch, cond)][c][s], float) for cond in conds])   # (n_cond, n_L)
        for li in range(n_L):
            col = M[:, li]
            ok = np.isfinite(col)
            if ok.sum() >= need:
                UR[u, li] = sd.signed_spearman([levels[k] for k in range(len(conds)) if ok[k]],
                                               [col[k] for k in range(len(conds)) if ok[k]])
    sids = np.array([s for (c, s) in keys])
    cids = np.array([c for (c, s) in keys])
    return _curve_twoway(UR, sids, cids, n_boot, seed)


def rows_profiles(main_run, channels=CHANNELS, n_boot=2000):
    """Depth-profile curves for the paper's ES-(b) / TI-(c)/(d) panels. Returns
    (depth_pct, curves) with curves[name][ch] = (mean, lo, hi) arrays over depth."""
    conds = [POS, NEG, INTENSELY] + list(RAMP)
    layers, vals, bases = sd.unit_layer_readouts(main_run, conds)
    n_L = len(layers)
    depth = _depth_pcts(layers, sd.run_n_layers(main_run))
    names = ("engage", "suppress", "gain_lexical", "gain_numeric", "rank_lexical", "rank_numeric")
    curves = {n: {} for n in names}
    for ch in channels:
        order = sorted(bases[ch])
        rng = np.random.default_rng(0)
        curves["engage"][ch] = _profile_dprime(vals[(ch, POS)], bases[ch], order, n_L, rng, +1.0)
        curves["suppress"][ch] = _profile_dprime(vals[(ch, NEG)], bases[ch], order, n_L, rng, -1.0)
        curves["gain_lexical"][ch] = _profile_dprime(vals[(ch, INTENSELY)], vals[(ch, POS)], order, n_L, rng)
        curves["gain_numeric"][ch] = _profile_dprime(vals[(ch, RAMP[3])], vals[(ch, RAMP[0])], order, n_L, rng)
        curves["rank_lexical"][ch] = _profile_rank(vals, ch, [POS, INTENSELY], [0, 1], order, n_L, n_boot)
        curves["rank_numeric"][ch] = _profile_rank(vals, ch, list(RAMP), [1, 2, 3, 4], order, n_L, n_boot)
    return depth, curves


# ---- output ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute the SCORES.md battery (two-column) from the stored run data.")
    ap.add_argument("--main-run", default=None,
                    help="main run dir; default: newest results/raw/*_activation_control")
    ap.add_argument("--lt-run", default=None,
                    help="layer-targeting run dir; default: newest results/raw/*_activation_control_lt")
    ap.add_argument("--json", default=None,
                    help="also write the battery as structured JSON here (for cross-model plotting)")
    ap.add_argument("--profiles-json", default=None,
                    help="also write the depth-profile curves here (for the paper's ES-b / TI-c/d panels)")
    args = ap.parse_args()
    if args.main_run is None:
        args.main_run = _latest_run()
        if args.main_run is None:
            ap.error("no results/raw/*_activation_control run found; pass --main-run")
        print(f"[auto] --main-run {args.main_run}")
    if args.lt_run is None:
        # LT runs carry the _lt tag (run_experiment._run_name) and no baseline
        # cache of their own -- they are baselined against the main run. Missing is
        # OK: layer_targeting is emitted null (a main-only run still scores).
        args.lt_run = _latest_run(pattern="*_activation_control_lt", needs_baseline=False)
        if args.lt_run is None:
            print("[auto] no *_activation_control_lt run -> layer_targeting will be null")
        print(f"[auto] --lt-run {args.lt_run}")

    print("rows 1/2 (engage/suppress depth profiles) ...", flush=True)
    layers, r12 = rows_1_2(args.main_run, channels=CHANNELS)
    print("rows 3/4 (intensity ramp) ...", flush=True)
    r34 = rows_3_4(args.main_run, channels=CHANNELS)
    print("targeting (temporal control + token group; in/out contrast) ...", flush=True)
    tgt = rows_targeting(args.main_run)
    print("row 9 (layer targeting) ...", flush=True)
    r9 = row_9(args.lt_run, args.main_run)
    print("row 10 (POS coverage) ...", flush=True)
    r10 = row_10(args.main_run)
    print("onset/offset error (timing edges; excluded from scalar) ...", flush=True)
    r_oo = row_onset_offset(args.main_run)

    def cell(sc, lo, hi):
        return f"{sc:+.2f} [{lo:+.2f},{hi:+.2f}]" if np.isfinite(sc) else "n/a"

    def tcell(ch, key):        # targeting measure (obs, lo, hi, tasks)
        return cell(*tgt[(key, ch)][:3])

    W = 34
    print("\n" + "=" * 110)
    print(f"{'row':<4}{'subscore':<26}{'cosine':<{W}}{'relative norm':<{W}}")
    print("-" * 110)
    for row, label in (("engage", "1  Engage"), ("suppress", "2  Suppress")):
        cells = []
        for ch in ("cos", "relnorm"):
            d = r12[(row, ch)]
            cells.append(f"{cell(d['score'], d['lo'], d['hi'])} @L{d['layer']}")
        print(f"{label:<30}{cells[0]:<{W}}{cells[1]:<{W}}")
        wrongs = [f"wrong-way {r12[(row, ch)]['wrong']:+.2f}" for ch in ("cos", "relnorm")]
        print(f"{'':<30}{wrongs[0]:<{W}}{wrongs[1]:<{W}}")
    cells = [f"{cell(*r34[('rank', ch)][:3])} @L{r34[('rank', ch)][3]}" for ch in ("cos", "relnorm")]
    print(f"{'3  Dial Rank':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    cells = [f"{cell(*r34[('resolution', ch)][:3])} (A={r34[('resolution', ch)][3]:.3f})" for ch in ("cos", "relnorm")]
    print(f"{'4  Dial Resolution':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    cells = [tcell(ch, "temporal_control") for ch in ("cos", "relnorm")]
    print(f"{'5  Temporal control':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    for ch in ("cos", "relnorm"):
        tasks = "  ".join(f"{k.replace('loc_', '').replace('persist_', '')}:{v:+.2f}"
                          for k, v in tgt[("temporal_control", ch)][3].items())
        print(f"{'':<10}{ch:>8} tasks:  {tasks}")
    lab7 = "7  Coverage (min POS d')"
    cov = r10["cos"]
    print(f"{lab7:<30}{cov[0]:.2f} [{cov[1]:+.2f},{cov[2]:+.2f}] weakest {cov[3]}   "
          f"(cos; proj {r10['proj'][0]:.2f} weakest {r10['proj'][3]})")
    cells = [tcell(ch, "token_group") for ch in ("cos", "relnorm")]
    print(f"{'8  Token group':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    for ch in ("cos", "relnorm"):
        tasks = "  ".join(f"{k.replace('loc_', '')}:{v:+.2f}"
                          for k, v in tgt[("token_group", ch)][3].items())
        print(f"{'':<10}{ch:>8} tasks:  {tasks}")
    cells = [f"{r9[ch][0]:+.3f} [{r9[ch][1]:+.3f},{r9[ch][2]:+.3f}]" for ch in ("cos", "relnorm")]
    print(f"{'11 Layer targeting':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    cells = [f"{r_oo[ch][0]:.3f}" if np.isfinite(r_oo[ch][0]) else "n/a" for ch in ("cos", "relnorm")]
    print(f"{'10 Onset/offset error (↓)':<30}{cells[0]:<{W}}{cells[1]:<{W}}  proj {r_oo['proj'][0]:.3f}")
    print("=" * 110)
    print("Data source: stored run artifacts only (results.json, no_instruction_cache.pkl,")
    print("vector_cache, pos_tags.json) via scripts/score_data.py -- no figure code imported.")
    print("Temporal control / Token group = located−generic in/out contrast (↑).")
    print("CIs: 95% two-way cluster bootstrap (sentences × concepts); peak/min-inside-replicate")
    print("for Engage/Suppress/Coverage (mildly optimistic/pessimistic).")
    print("Onset/offset error (↓) is computed but EXCLUDED from the scalar (timing diagnostic);")
    print("Suppress & Layer targeting ARE in S (7 measures) since 2026-07-23.")
    print("(Retired: Temporal control (CoM), Token group (CoM) -- no longer computed.)")

    if args.json:
        _dump_json(args.json, args.main_run, args.lt_run, r12, r34, tgt, r9, r10, r_oo)
        print(f"wrote {args.json}")

    if args.profiles_json:
        print("depth profiles (engage/suppress/rank/gain curves) ...", flush=True)
        depth, curves = rows_profiles(args.main_run)
        _dump_profiles(args.profiles_json, args.main_run, depth, curves)
        print(f"wrote {args.profiles_json}")


def _dump_profiles(path, main_run, depth, curves):
    """Serialize the depth-profile curves: curves[name][ch] = (mean, lo, hi) arrays.
    All three channels emitted; the paper figures read ch='proj'."""
    import json as _json

    def _arr(a):
        return [float(x) if np.isfinite(x) else None for x in a]

    def _cell(t):
        return {"mean": _arr(t[0]), "lo": _arr(t[1]), "hi": _arr(t[2])}

    payload = {"model": sd._resolve_model(main_run, None), "main_run": str(main_run),
               "note": "paper figures read ch='proj'; cos/relnorm emitted for the appendix",
               "depth_pct": list(depth),
               "curves": {name: {ch: _cell(curves[name][ch]) for ch in curves[name]}
                          for name in curves}}
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2)


def _dump_json(path, main_run, lt_run, r12, r34, tgt, r9, r10, r_oo):
    """Serialize the battery to structured JSON for cross-model plotting.

    Schema: measures[key] = {label, better (high|low), channels{cos|relnorm:
    {score, lo, hi, ...extras}}}. Only score/lo/hi drive the bars; extras
    (layer, auroc, weakest, per-task) are carried for tooltips/annotation. Every
    measure here is dimensionless and comparable ACROSS MODELS within a (measure,
    channel) cell — the plot must not compare cos-vs-relnorm magnitudes."""
    import json as _json

    def both(fn):
        return {ch: fn(ch) for ch in CHANNELS}

    def _f(x):
        return float(x) if np.isfinite(x) else None

    def base(t):   # (obs, lo, hi, ...) -> {score, lo, hi}, non-finite -> None
        return {"score": _f(t[0]), "lo": _f(t[1]), "hi": _f(t[2])}

    measures = {
        "engage": {"label": "Engage", "better": "high", "channels": both(
            lambda ch: {"score": r12[("engage", ch)]["score"],
                        "lo": r12[("engage", ch)]["lo"], "hi": r12[("engage", ch)]["hi"],
                        "layer": int(r12[("engage", ch)]["layer"]),
                        "wrong": r12[("engage", ch)]["wrong"]})},
        "suppress": {"label": "Suppress", "better": "high", "channels": both(
            lambda ch: {"score": r12[("suppress", ch)]["score"],
                        "lo": r12[("suppress", ch)]["lo"], "hi": r12[("suppress", ch)]["hi"],
                        "layer": int(r12[("suppress", ch)]["layer"]),
                        "wrong": r12[("suppress", ch)]["wrong"]})},
        "dial_rank": {"label": "Dial Rank", "better": "high", "range": [-1, 1], "channels": both(
            lambda ch: {**base(r34[("rank", ch)]), "layer": int(r34[("rank", ch)][3])})},
        "dial_resolution": {"label": "Dial Resolution", "better": "high", "channels": both(
            lambda ch: {**base(r34[("resolution", ch)]), "auroc": float(r34[("resolution", ch)][3])})},
        "dial_resolution_pool": {"label": "Dial Resolution (deep-pool)", "better": "high",
                                 "channels": both(
            lambda ch: {**base(r34[("resolution_pool", ch)]), "auroc": float(r34[("resolution_pool", ch)][3])})},
        "temporal_control": {"label": "Temporal control", "better": "high", "channels": both(
            lambda ch: base(tgt[("temporal_control", ch)]))},
        "coverage": {"label": "Coverage (POS)", "better": "high", "channels": {
            ch: {"score": _f(r10[ch][0]), "lo": _f(r10[ch][1]), "hi": _f(r10[ch][2]),
                 "weakest": r10[ch][3]} for ch in r10}},
        "token_group": {"label": "Token group", "better": "high", "channels": both(
            lambda ch: base(tgt[("token_group", ch)]))},
        "layer_targeting": {"label": "Layer targeting", "better": "high", "channels": both(
            lambda ch: base(r9[ch]))},
        # Computed but EXCLUDED from the scalar (timing-edge error; lower is better,
        # no aggregate CI -- only per-edge). `edges` carries the per-gate detail.
        "onset_offset_error": {"label": "Onset/offset error", "better": "low", "channels": both(
            lambda ch: {"score": _f(r_oo[ch][0]), "lo": None, "hi": None,
                        "edges": [{"edge": e, "gate": g, "mean": _f(m), "lo": _f(lo), "hi": _f(hi)}
                                  for (e, g, m, lo, hi) in r_oo[ch][1]]})},
    }
    payload = {"model": sd._resolve_model(main_run, None),
               "main_run": str(main_run), "lt_run": str(lt_run), "measures": measures}
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
