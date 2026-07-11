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
  rows 3, 4, 6, 7, 9 : unit-level two-way bootstrap, sigma / win definitions
                  fixed at their observed values.
  row 8         : per-edge profile-bootstrap CIs; composite is a point
                  estimate (window conventions to be pre-committed first).

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
from score_data import POS, NEG, RAMP, N_LAYERS_TOTAL, signed_spearman    # noqa: E402

MAIN = "results/raw/20260704_212244_gemma3_27b_write_introspection_main"
LT = "results/raw/20260708_002002_gemma3_27b_write_introspection_main"
PHI_INV = NormalDist().inv_cdf


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


def dprime_stats(per_concept, S_global, n_L, rng, n_boot=2000, n_perm=5000):
    """Concept-averaged d' per column + TWO-WAY cluster-bootstrap replicates.

    per_concept: list of (X, B, idx); X, B (S_c x n_L) condition/baseline
    values over the concept's sentences, idx = global sentence indices.
    (Frozen copy of the promoted Fig 2 machinery -- kept verbatim, including
    the sign-flip draw, so the battery numbers are reproducible bit-for-bit.)"""
    nC = len(per_concept)
    W = rng.multinomial(S_global, np.full(S_global, 1.0 / S_global),
                        size=n_boot).astype(float)
    Mc = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=n_boot).astype(float)
    E = rng.integers(0, 2, size=(n_perm, S_global)) * 2.0 - 1.0
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


def twoway(values_list, n_boot=2000, seed=0, stat=None):
    """Joint two-way cluster bootstrap over unit-level values. values_list:
    [(vals (n_u x k), sent_ids, conc_ids)]; stat maps the per-block
    weighted-mean vectors -> scalar."""
    rng = np.random.default_rng(seed)
    sents = sorted({s for _, sid, _ in values_list for s in sid})
    concs = sorted({c for _, _, cid in values_list for c in cid})
    s_i = {s: i for i, s in enumerate(sents)}
    c_i = {c: i for i, c in enumerate(concs)}
    Ws = rng.multinomial(len(sents), np.full(len(sents), 1 / len(sents)), size=n_boot).astype(float)
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
    reps = reps[np.isfinite(reps)]
    return obs, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# ---- rows ------------------------------------------------------------------------

def rows_1_2(main_run):
    """Engage/suppress per channel: peak d' in the INSTRUCTED direction
    (cos: toward/away; relnorm: up/down), floored at 0 + wrong-way extreme."""
    layers, vals, bases = sd.unit_layer_readouts(main_run, [POS, NEG])
    concept_order = sorted(bases["cos"])
    rng = np.random.default_rng(0)
    out = {}
    for ch in ("cos", "relnorm"):
        for row, cond, sign in (("engage", POS, +1.0), ("suppress", NEG, -1.0)):
            blocks, S = per_concept_blocks(vals[(ch, cond)], bases[ch], concept_order)
            st = dprime_stats(blocks, S, len(layers), rng)
            dirc = sign * st["dp"]
            li = int(np.nanargmax(dirc))
            pk = np.nanmax(sign * st["bavg"], axis=1)
            unf = float(dirc[li])
            out[(row, ch)] = dict(score=max(0.0, unf), unfloored=unf,
                                  lo=float(np.nanpercentile(pk, 2.5)),
                                  hi=float(np.nanpercentile(pk, 97.5)),
                                  layer=layers[li],
                                  wrong=float(-np.nanmin(dirc)))
    return layers, out


def rows_3_4(main_run):
    layers, vals, _ = sd.unit_layer_readouts(main_run, RAMP)
    n_L = len(layers)
    nl_total = sd.run_meta(main_run).get("n_layers") or N_LAYERS_TOTAL
    deep = [li for li, L in enumerate(layers) if L / nl_total >= 0.5]

    units = {ch: dict(rank=[], win=[]) for ch in ("cos", "relnorm")}
    sent_ids, conc_ids = [], []
    keys = sorted({(c, s) for cond in RAMP
                   for c in vals[("cos", cond)] for s in vals[("cos", cond)][c]})
    for c, s in keys:
        for ch in ("cos", "relnorm"):
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
            units[ch]["win"].append(wn[:, deep].ravel())
        sent_ids.append(s); conc_ids.append(c)

    sid = np.array(sent_ids); cid = np.array(conc_ids)
    out = {}
    for ch in ("cos", "relnorm"):
        UR = np.vstack(units[ch]["rank"]); UW = np.vstack(units[ch]["win"])
        rank_obs, rank_lo, rank_hi = twoway(
            [(UR, sid, cid)], stat=lambda ms: float(np.nanmax(ms[0])))
        li = int(np.nanargmax(np.nanmean(UR, axis=0)))

        def res_stat(ms):
            A = float(np.nanmean(ms[0]))
            A = min(max(A, 1e-6), 1 - 1e-6)
            return float(np.sqrt(2) * PHI_INV(A))
        res_obs, res_lo, res_hi = twoway([(UW, sid, cid)], stat=res_stat)
        A_obs = float(np.nanmean(np.nanmean(UW, axis=0)))
        out[("rank", ch)] = (rank_obs, rank_lo, rank_hi, layers[li])
        out[("resolution", ch)] = (res_obs, res_lo, res_hi, A_obs)
    return out


def rows_6_7(main_run):
    store = sd.location_units(main_run)
    conds = [c for c, _ in sd.LOC_CONDS]
    out = {}
    for ch in ("cos", "relnorm"):
        sig_on, sig_off = {}, {}
        for cond in conds:
            bi, bo = store[(ch, cond, "base", "G_in")], store[(ch, cond, "base", "G_out")]
            per_c = defaultdict(lambda: ([], []))
            for (s, c), v in bi.items():
                if (s, c) in bo:
                    per_c[c][0].append(v); per_c[c][1].append(bo[(s, c)])
            for c, (vi, vo) in per_c.items():
                if len(vi) >= 3:
                    sig_on[(cond, c)] = np.std(vi, ddof=1)
                    sig_off[(cond, c)] = np.std(vo, ddof=1)

        S_blocks, com_blocks = [], []
        S_tasks, com_tasks = {}, {}
        for cond in conds:
            gi_l, go_l = store[(ch, cond, "loc", "G_in")], store[(ch, cond, "loc", "G_out")]
            gi_t, go_t = store[(ch, cond, "think", "G_in")], store[(ch, cond, "think", "G_out")]
            cm_l, cm_t = store[(ch, cond, "loc", "CoM")], store[(ch, cond, "think", "CoM")]
            keys = sorted(set(gi_l) & set(go_l) & set(gi_t) & set(go_t))
            su, gu, sids, cids = [], [], [], []
            for (s, c) in keys:
                if (cond, c) not in sig_on or sig_on[(cond, c)] <= 0 or sig_off[(cond, c)] <= 0:
                    continue
                so, sf = sig_on[(cond, c)], sig_off[(cond, c)]
                s_val = (gi_l[(s, c)] / so - go_l[(s, c)] / sf) \
                    - (gi_t[(s, c)] / so - go_t[(s, c)] / sf)
                su.append(s_val)
                gu.append(cm_t.get((s, c), np.nan) - cm_l.get((s, c), np.nan))
                sids.append(s); cids.append(c)
            S_blocks.append((np.array(su), np.array(sids), np.array(cids)))
            com_blocks.append((np.array(gu), np.array(sids), np.array(cids)))
            S_tasks[cond] = float(np.nanmean(su))
            com_tasks[cond] = float(np.nanmean(gu))

        S_obs, S_lo, S_hi = twoway(S_blocks, stat=lambda ms: float(np.nanmean([m[0] for m in ms])))
        C_obs, C_lo, C_hi = twoway(com_blocks, stat=lambda ms: float(np.nanmean([m[0] for m in ms])))
        out[("S", ch)] = (S_obs, S_lo, S_hi, S_tasks)
        out[("com", ch)] = (C_obs, C_lo, C_hi, com_tasks)
    return out


def row_8(main_run):
    out = {}
    for ch in ("cos", "relnorm"):
        stats = sd.persistence_edges(main_run, metric=ch)
        edges, errs = [], []
        for edge, cond_list in sd.EDGE_TASKS:
            for c in cond_list:
                e = stats[(edge, c)]
                if np.isfinite(e["mean"]):
                    errs.append(abs(e["mean"]))
                    edges.append((edge, sd.PERSIST_LABEL[c], e["mean"], e["lo"], e["hi"]))
        out[ch] = (float(np.mean(errs)) if errs else np.nan, edges)
    return out


def row_9(lt_run, main_run):
    data = sd.layer_targeting_units(lt_run, main_run)
    out = {}
    for ch in ("cos", "relnorm"):
        D, sid, cid, layers_lt = data[ch]
        obs, lo, hi = twoway([(D, sid, cid)], stat=lambda ms: float(np.nanmean(ms[0])))
        per_layer = {int(L): float(round(v, 3))
                     for L, v in zip(layers_lt, np.nanmean(D, axis=0))}
        out[ch] = (obs, lo, hi, per_layer)
    return out


def row_10(main_run):
    vals, bases = sd.pos_category_readouts(main_run, [POS, NEG])
    concept_order = sorted(bases["cos"])
    rng = np.random.default_rng(0)
    blocks, S = per_concept_blocks(vals[("cos", POS)], bases["cos"], concept_order)
    st = dprime_stats(blocks, S, len(sd.CATS), rng)
    gi = int(np.nanargmin(st["dp"]))
    mn = np.nanmin(st["bavg"], axis=1)
    return (float(st["dp"][gi]), float(np.nanpercentile(mn, 2.5)),
            float(np.nanpercentile(mn, 97.5)), sd.CATS[gi])


# ---- output ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute the SCORES.md battery (two-column) from the stored run data.")
    ap.add_argument("--main-run", default=MAIN)
    ap.add_argument("--lt-run", default=LT)
    args = ap.parse_args()

    print("rows 1/2 (engage/suppress depth profiles) ...", flush=True)
    layers, r12 = rows_1_2(args.main_run)
    print("rows 3/4 (intensity ramp) ...", flush=True)
    r34 = rows_3_4(args.main_run)
    print("rows 6/7 (location targeting) ...", flush=True)
    r67 = rows_6_7(args.main_run)
    print("row 8 (persistence edges, both channels) ...", flush=True)
    r8 = row_8(args.main_run)
    print("row 9 (layer targeting) ...", flush=True)
    r9 = row_9(args.lt_run, args.main_run)
    print("row 10 (POS coverage) ...", flush=True)
    r10 = row_10(args.main_run)

    def cell(sc, lo, hi):
        return f"{sc:+.2f} [{lo:+.2f},{hi:+.2f}]"

    W = 34
    print("\n" + "=" * 110)
    print(f"{'row':<4}{'subscore':<26}{'cosine':<{W}}{'relative norm':<{W}}")
    print("-" * 110)
    for row, label in (("engage", "1  Engage (headline C)"), ("suppress", "2  Suppress (compliance)")):
        cells = []
        for ch in ("cos", "relnorm"):
            d = r12[(row, ch)]
            cells.append(f"{cell(d['score'], d['lo'], d['hi'])} @L{d['layer']}")
        print(f"{label:<30}{cells[0]:<{W}}{cells[1]:<{W}}")
        wrongs = [f"wrong-way {r12[(row, ch)]['wrong']:+.2f}" for ch in ("cos", "relnorm")]
        print(f"{'':<30}{wrongs[0]:<{W}}{wrongs[1]:<{W}}")
    cells = [f"{cell(*r34[('rank', ch)][:3])} @L{r34[('rank', ch)][3]}" for ch in ("cos", "relnorm")]
    print(f"{'3  Dial monotonicity (rho)':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    cells = [f"{cell(*r34[('resolution', ch)][:3])} (A={r34[('resolution', ch)][3]:.3f})" for ch in ("cos", "relnorm")]
    print(f"{'4  Dial resolution':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    print(f"{'5  (absorbed into row 1 relnorm cell 2026-07-10)':<30}")
    cells = [cell(*r67[("S", ch)][:3]) for ch in ("cos", "relnorm")]
    print(f"{'6  Addressability S':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    for ch in ("cos", "relnorm"):
        tasks = "  ".join(f"{k.replace('loc_','')}:{v:+.2f}" for k, v in r67[("S", ch)][3].items())
        print(f"{'':<10}{ch:>8} tasks:  {tasks}")
    cells = [f"{r67[('com', ch)][0]:+.3f} [{r67[('com', ch)][1]:+.3f},{r67[('com', ch)][2]:+.3f}]" for ch in ("cos", "relnorm")]
    print(f"{'7  Spatial precision (CoM)':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    cells = [f"{r8[ch][0]:.2f}" if np.isfinite(r8[ch][0]) else "n/a" for ch in ("cos", "relnorm")]
    print(f"{'8  Timing precision':<30}{cells[0] + ' (lower=better)':<{W}}{cells[1] + ' (@L43)':<{W}}")
    for ch in ("cos", "relnorm"):
        for edge, lab, m, lo, hi in r8[ch][1]:
            print(f"{'':<10}{ch:>8} - {edge} {lab}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    cells = [f"{r9[ch][0]:+.3f} [{r9[ch][1]:+.3f},{r9[ch][2]:+.3f}]" for ch in ("cos", "relnorm")]
    print(f"{'9  Layer addressability':<30}{cells[0]:<{W}}{cells[1]:<{W}}")
    for ch in ("cos", "relnorm"):
        pl = "  ".join(f"L{L}:{v:+.2f}" for L, v in r9[ch][3].items())
        print(f"{'':<10}{ch:>8} per layer:  {pl}")
    lab10 = "10 Coverage (min POS d')"
    print(f"{lab10:<30}{r10[0]:.2f} [{r10[1]:+.2f},{r10[2]:+.2f}] weakest {r10[3]}   (cosine only; mixed signs on relnorm)")
    print("=" * 110)
    print("Data source: stored run artifacts only (results.json, no_instruction_cache.pkl,")
    print("vector_cache, pos_tags.json) via scripts/score_data.py -- no figure code imported.")
    print("Channel conventions: cosine instructed direction = toward/away the concept;")
    print("relnorm = norm up (engage) / norm down (suppress). CIs: 95% two-way cluster")
    print("bootstrap (sentences x concepts); peak-inside-replicate for rows 1/2/10 (mildly")
    print("optimistic); sigma/win definitions fixed for rows 3/4/6/7/9; row 8 per-edge only.")


if __name__ == "__main__":
    main()
