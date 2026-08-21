#!/usr/bin/env python3
"""Diagnostic: find the PROJECTION channel's peak DEPTHS for the two fixed-depth
battery views, which is how PROJ_F_POS and PROJ_F_LOC were set, by analogy to the
COS_F_POS/COS_F_LOC and RN_F_POS/RN_F_LOC constants.

  PROJ_F_POS  = depth (fraction) maximizing the ENGAGE proj d' (drives the POS /
                Coverage readout, cf. COS_F_POS = 0.90).
  PROJ_F_LOC  = depth (fraction) maximizing the TEMPORAL-CONTROL proj in/out
                contrast group score (drives location/targeting, cf.
                COS_F_LOC = 1.00).

Reads only stored run artifacts via score_data (no model load). Imports and
reuses score_data helpers verbatim; edits nothing. proj readout per token =
cosine_sim[L] * raw_norm[L], baseline = base_cos[L] * raw_norm_baseline[L] --
the same reconstruction unit_layer_readouts uses for its 'proj' channel.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import score_data as sd            # noqa: E402
import compute_scores as cs        # noqa: E402


def frac_pct(i, n_analysis):
    """Canonical requested fraction for analysis-layer index i (5%..100% when the
    run sweeps 20 layers), per the project's FRACS convention."""
    if n_analysis == 20:
        return (i + 1) * 5
    return round(100 * (i + 1) / n_analysis)


# ---------- PROJ_F_POS: engage proj d' vs depth ----------------------------------

def engage_proj_profile(run):
    layers, vals, bases = sd.unit_layer_readouts(run, [sd.POS, sd.NEG])
    order = sorted(bases["proj"])
    blocks, S = cs.per_concept_blocks(vals[("proj", sd.POS)], bases["proj"], order)
    rng = np.random.default_rng(0)
    st = cs.dprime_stats(blocks, S, len(layers), rng, n_perm=0)
    return layers, st["dp"]


# ---------- PROJ_F_LOC: temporal-control proj contrast vs depth ------------------

def _proj_tokens(row, L, n):
    cs_tr = sd.trace(row, "cosine_sim", L)
    nm_tr = sd.trace(row, "norms", L)
    if cs_tr is None or nm_tr is None:
        return None
    cv = sd._fit_len(np.asarray(cs_tr, np.float32)[:n], n)
    nv = sd._fit_len(np.asarray(nm_tr, np.float32)[:n], n)
    if cv is None or nv is None:
        return None
    return cv * nv


TEMP_CONDS = sd.TARGET_GROUPS["temporal_control"]        # loc_beginning, persist_once, loc_end
TEMP_KIND = {"loc_beginning": "begin", "persist_once": "mid", "loc_end": "end"}


def temporal_proj_store_at(run, L, *, vector_cache="results/vector_cache", method="baseline"):
    """proj-channel location store at a single analysis layer L (mirrors
    score_data.location_units for the temporal conds, proj-only)."""
    rows = sd.load_rows(run)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    pos_words = sd.pos_by_sentence()
    cache = sd.load_baseline(run)
    vecs = sd.load_vectors(vector_cache, sd._resolve_model(run, None), [L], method)
    concepts_L = vecs[L][0] if L in vecs else []
    wanted = set(TEMP_CONDS) | {sd.POS}

    store = defaultdict(dict)
    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if words is None or toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n = len(toks)
        if n < 3:
            continue
        upos = sd.token_upos(s, toks, words)
        f = np.arange(n) / (n - 1)
        masks = {}
        for kind in ("begin", "mid", "end"):
            T = sd._loc_mask(kind, f, upos)
            if T.sum() >= 1 and (~T).sum() >= 1:
                masks[kind] = (T, ~T, sd._dist_to_target(f, T))

        acts = np.asarray(ent["activations"][L], np.float32)[:n]
        an = acts / (np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8)
        base_cos_all = an @ vecs[L][2].T if L in vecs else None
        base_norm = np.asarray(ent["norms"][L], np.float32)[:n]

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        for c in sorted(concepts):
            if base_cos_all is None or c not in concepts_L:
                continue
            base_proj_c = sd._fit_len(base_cos_all[:, concepts_L.index(c)] * base_norm, n)

            def sig(row):
                v = _proj_tokens(row, L, n)
                return (v - base_proj_c) if (v is not None and base_proj_c is not None) else None

            think_sig = sig(byc.get(sd.POS, {}).get(c)) if c in byc.get(sd.POS, {}) else None
            for cond in TEMP_CONDS:
                kind = TEMP_KIND[cond]
                if kind not in masks:
                    continue
                T, O, d = masks[kind]
                loc_sig = sig(byc.get(cond, {}).get(c)) if c in byc.get(cond, {}) else None
                if loc_sig is not None:
                    m = sd._loc_metrics(loc_sig, T, O, d)
                    if m:
                        for mk, v in m.items():
                            store[(cond, "loc", mk)][(s, c)] = v
                if think_sig is not None:
                    m = sd._loc_metrics(think_sig, T, O, d)
                    if m:
                        for mk, v in m.items():
                            store[(cond, "think", mk)][(s, c)] = v
                if base_proj_c is not None:
                    m = sd._loc_metrics(np.asarray(base_proj_c, float), T, O, d)
                    if m:
                        for mk in ("G_in", "G_out"):
                            store[(cond, "base", mk)][(s, c)] = m[mk]
    return store


def temporal_proj_score(store):
    """The temporal_control group point estimate from a proj store (mirrors
    compute_scores.rows_targeting: standardized located-minus-generic in/out
    contrast, mean over the three temporal tasks)."""
    sig_on, sig_off = {}, {}
    for cond in TEMP_CONDS:
        bi, bo = store.get((cond, "base", "G_in"), {}), store.get((cond, "base", "G_out"), {})
        per_c = defaultdict(lambda: ([], []))
        for (s, c), v in bi.items():
            if (s, c) in bo:
                per_c[c][0].append(v); per_c[c][1].append(bo[(s, c)])
        for c, (vi, vo) in per_c.items():
            if len(vi) >= 3:
                sig_on[(cond, c)] = np.std(vi, ddof=1)
                sig_off[(cond, c)] = np.std(vo, ddof=1)
    task_means = []
    for cond in TEMP_CONDS:
        gi_l, go_l = store.get((cond, "loc", "G_in"), {}), store.get((cond, "loc", "G_out"), {})
        gi_t, go_t = store.get((cond, "think", "G_in"), {}), store.get((cond, "think", "G_out"), {})
        keys = sorted(set(gi_l) & set(go_l) & set(gi_t) & set(go_t))
        su = []
        for (s, c) in keys:
            if (cond, c) not in sig_on or sig_on[(cond, c)] <= 0 or sig_off[(cond, c)] <= 0:
                continue
            so, sf = sig_on[(cond, c)], sig_off[(cond, c)]
            su.append((gi_l[(s, c)] / so - go_l[(s, c)] / sf)
                      - (gi_t[(s, c)] / so - go_t[(s, c)] / sf))
        task_means.append(float(np.nanmean(su)) if su else np.nan)
    return float(np.nanmean(task_means)), task_means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = args.run
    model = sd._resolve_model(run, None)
    n_total = sd.run_n_layers(run)

    layers, dp = engage_proj_profile(run)
    n = len(layers)
    print(f"\nmodel: {model}   n_layers={n_total}   analysis layers={n}\n")
    print("=== PROJ_F_POS candidate: engage proj d' vs depth ===")
    print(f"{'idx':>3} {'L':>4} {'frac%':>6} {'engage d′':>10}")
    for i, (L, v) in enumerate(zip(layers, dp)):
        star = "  <== peak" if i == int(np.nanargmax(dp)) else ""
        print(f"{i:>3} {L:>4} {frac_pct(i, n):>5}% {v:>10.3f}{star}")
    ip = int(np.nanargmax(dp))
    print(f"\n  -> PROJ_F_POS = {frac_pct(ip, n)/100:.2f}  (L={layers[ip]}, d′={dp[ip]:.3f})\n")

    print("=== PROJ_F_LOC candidate: temporal-control proj contrast vs depth ===")
    print(f"{'idx':>3} {'L':>4} {'frac%':>6} {'temporal':>10}  (begin / mid / end)")
    scores = []
    for i, L in enumerate(layers):
        store = temporal_proj_store_at(run, L)
        sc, tasks = temporal_proj_score(store)
        scores.append(sc)
        t = " / ".join(f"{x:+.2f}" if np.isfinite(x) else " nan" for x in tasks)
        print(f"{i:>3} {L:>4} {frac_pct(i, n):>5}% {sc:>10.3f}  ({t})")
    il = int(np.nanargmax(scores))
    print(f"\n  -> PROJ_F_LOC = {frac_pct(il, n)/100:.2f}  (L={layers[il]}, score={scores[il]:.3f})\n")


if __name__ == "__main__":
    main()
