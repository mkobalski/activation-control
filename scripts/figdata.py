#!/usr/bin/env python3
"""Precompute FOCAL-panel figure inputs into FIGDATA_<model>.json.

The paper figures for layer targeting and temporal control/precision illustrate
focal-model behaviour with per-position, per-word and per-(target×analysis)-layer
projection aggregates. Those aggregates used to be recomputed at *figure* time by
reading the per-trial results.json (hundreds of MB) and the concept-vector cache
off the run store -- slow, and out of place in a plotting script. This module
computes them ONCE, in the scoring layer (which already reads raw), and freezes
them to FIGDATA_<model>.json so the figure scripts read a small JSON like every
other paper figure.

Emitted (projection channel = ||r||*cos, at the targeting depth = 90%):
  profiles       : per-position (10-bin) Δ vs the no-instruction baseline, pooled
                   over sentence × concept, for think_about / loc_beginning /
                   persist_once / loc_end / persist_first_half
  word_profiles  : per-WORD Δ for think_about / persist_after_fourth (clipped to
                   the shortest sentence's word count)
  layer_targeting: (target × analysis) column-demeaned standardized Δ grid, only
                   when an LT run is present (else null)

Bands are the 95% two-way (sentence × concept) cluster bootstrap (B=2000), the
same scheme the battery's scalar CI uses. The compute functions were moved here
verbatim from the figure scripts temporal_control / temporal_precision /
layer_targeting; those scripts now read the JSON this writes.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

import score_data as sd

PROJ_F_LOC = 0.90                                     # targeting depth fraction
N_BOOT = 2000
FRAC_CONDS = ["think_about", "loc_beginning", "persist_once", "loc_end", "persist_first_half"]
WORD_CONDS = ["think_about", "persist_after_fourth"]


# --------------------------- low-level readers (moved verbatim from the figures) ---

def _proj_tokens(row, L, n):
    cs = (row.get("cosine_sim") or {}); nm = (row.get("norms") or {})
    tc = cs.get(str(L), cs.get(L)); tn = nm.get(str(L), nm.get(L))
    if not tc or not tn:
        return None
    tc = np.asarray(tc, np.float32); tn = np.asarray(tn, np.float32)
    m = min(len(tc), len(tn), n)
    out = np.full(n, np.nan, np.float32)
    out[:m] = tc[:m] * tn[:m]
    return out


def _cluster_band(U, sids, cids, n_boot=N_BOOT, seed=0):
    """95% band for a mean-over-units curve, joint two-way (sentence × concept)
    cluster bootstrap with multinomial multiplicities; one resample shared across
    all columns (identical scheme to the battery's scalar CI)."""
    U = np.asarray(U, float)
    rng = np.random.default_rng(seed)
    _, inv_s = np.unique(np.asarray(sids), return_inverse=True)
    _, inv_c = np.unique(np.asarray(cids), return_inverse=True)
    n_s, n_c = inv_s.max() + 1, inv_c.max() + 1
    Wsent = rng.multinomial(n_s, np.full(n_s, 1 / n_s), size=n_boot).astype(float)
    Mconc = rng.multinomial(n_c, np.full(n_c, 1 / n_c), size=n_boot).astype(float)
    W = Wsent[:, inv_s] * Mconc[:, inv_c]                 # (n_boot, n_units)
    fin = np.isfinite(U)
    Uz = np.where(fin, U, 0.0)
    num = W @ Uz
    den = W @ fin.astype(float)
    boot = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return lo, hi


def _unit_vectors(cache_dir, model, L):
    """(concepts, unit_vec_matrix) at layer L from the concept-vector cache."""
    concepts, _V, Vn = sd.load_vectors(str(cache_dir), model, [L])[L]
    return concepts, Vn


# ------------------------------- aggregates --------------------------------------

def position_profile(run_dir, model, cache_dir, conds, n_bins=10):
    """{cond: (centers, mean, lo, hi)} per-position-bin projection Δ (cond −
    no_instruction) at the targeting depth, pooled over sentence × concept."""
    L = sd._layer_for_fraction(run_dir, PROJ_F_LOC)
    rows = sd.load_rows(run_dir)
    cache = sd.load_baseline(run_dir)
    concepts_L, Vn = _unit_vectors(cache_dir, model, L)
    idx = defaultdict(dict)
    for r in rows:
        if r["condition_id"] in conds and r.get("concept"):
            idx[r["sentence"]].setdefault(r["condition_id"], {})[r["concept"]] = r
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = {c: [] for c in conds}; acc_s = {c: [] for c in conds}; acc_c = {c: [] for c in conds}
    for sent, ent in cache.items():
        if sent not in idx:
            continue
        toks = ent["anchored_token_strs"][1:]
        n = len(toks)
        if n < 3:
            continue
        A = np.asarray(ent["activations"][L], np.float32)[:n]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        base_cos = An @ Vn.T
        base_norm = np.asarray(ent["norms"][L], np.float32)[:n]
        f = np.arange(n) / (n - 1)
        b = np.clip(np.digitize(f, edges) - 1, 0, n_bins - 1)
        for cond in conds:
            for c, r in idx[sent].get(cond, {}).items():
                if c not in concepts_L:
                    continue
                pt = _proj_tokens(r, L, n)
                if pt is None:
                    continue
                delta = pt - base_cos[:, concepts_L.index(c)] * base_norm
                vec = np.full(n_bins, np.nan)
                for bi in range(n_bins):
                    mk = b == bi
                    if mk.any() and np.isfinite(delta[mk]).any():
                        vec[bi] = float(np.nanmean(delta[mk]))
                acc[cond].append(vec); acc_s[cond].append(sent); acc_c[cond].append(c)
    out = {}
    for cond in conds:
        if not acc[cond]:
            continue
        U = np.vstack(acc[cond])
        lo, hi = _cluster_band(U, np.asarray(acc_s[cond]), np.asarray(acc_c[cond]))
        out[cond] = (centers, np.nanmean(U, axis=0), lo, hi)
    return out, L


def _word_start(i, tok):
    return i == 0 or (tok[:1] == " " and any(ch.isalnum() for ch in tok))


def _word_spans(toks):
    starts = [i for i, t in enumerate(toks) if _word_start(i, t)]
    bounds = starts + [len(toks)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(starts))]


def min_word_count(run_dir):
    cache = sd.load_baseline(run_dir)
    return min(len(_word_spans(ent["anchored_token_strs"][1:])) for ent in cache.values())


def word_profile(run_dir, model, cache_dir, conds, max_words):
    """{cond: (word_idx 1..max_words, mean, lo, hi)} per-WORD projection Δ (cond −
    no_instruction) at the targeting depth, clipped to the shortest sentence's word
    count. Each word = mean of its tokens. Same readout/depth/band as the fraction path."""
    L = sd._layer_for_fraction(run_dir, PROJ_F_LOC)
    rows = sd.load_rows(run_dir)
    cache = sd.load_baseline(run_dir)
    concepts_L, Vn = _unit_vectors(cache_dir, model, L)
    idx = defaultdict(dict)
    for r in rows:
        if r["condition_id"] in conds and r.get("concept"):
            idx[r["sentence"]].setdefault(r["condition_id"], {})[r["concept"]] = r
    acc = {c: [] for c in conds}; acc_s = {c: [] for c in conds}; acc_c = {c: [] for c in conds}
    for sent, ent in cache.items():
        if sent not in idx:
            continue
        toks = ent["anchored_token_strs"][1:]
        n = len(toks)
        if n < 3:
            continue
        words = _word_spans(toks)
        A = np.asarray(ent["activations"][L], np.float32)[:n]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        base_cos = An @ Vn.T
        base_norm = np.asarray(ent["norms"][L], np.float32)[:n]
        for cond in conds:
            for c, r in idx[sent].get(cond, {}).items():
                if c not in concepts_L:
                    continue
                pt = _proj_tokens(r, L, n)
                if pt is None:
                    continue
                delta = pt - base_cos[:, concepts_L.index(c)] * base_norm
                vec = np.full(max_words, np.nan)
                for wi in range(min(max_words, len(words))):
                    a, b = words[wi]
                    seg = delta[a:b]
                    if np.isfinite(seg).any():
                        vec[wi] = float(np.nanmean(seg))
                acc[cond].append(vec); acc_s[cond].append(sent); acc_c[cond].append(c)
    out = {}
    x = np.arange(1, max_words + 1)
    for cond in conds:
        if not acc[cond]:
            continue
        U = np.vstack(acc[cond])
        lo, hi = _cluster_band(U, np.asarray(acc_s[cond]), np.asarray(acc_c[cond]))
        out[cond] = (x, np.nanmean(U, axis=0), lo, hi)
    return out


def layer_grid(lt_run, main_run, cache_dir):
    """(targets, analysis, grid) where grid[t, l] = column-demeaned standardized
    concept projection Δ (think_at_layer target=targets[t]) at analysis layer
    analysis[l]. The mean of the diagonal ≈ the frozen layer_targeting score."""
    meta = yaml.safe_load(open(Path(lt_run) / "config.yaml"))
    targets = [int(x) for x in meta["prompt_layers"]]
    analysis = [int(x) for x in meta["analysis_layers"]]
    model = sd._resolve_model(main_run, None)
    rows = sd.load_rows(lt_run)
    cache = sd.load_baseline(main_run)
    vecs = sd.load_vectors(str(cache_dir), model, analysis)
    concepts = vecs[analysis[0]][0]
    ci = {c: i for i, c in enumerate(concepts)}

    base = {}
    for s, ent in cache.items():
        toks = ent["anchored_token_strs"][1:]; n = len(toks)
        if n < 1:
            continue
        d = {}
        for L in analysis:
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            cos = An @ vecs[L][2].T                                       # (n_tok, n_concepts)
            nrm = np.asarray(ent["norms"][L], np.float32)[:n]
            d[L] = (cos * nrm[:, None]).mean(0)                           # token-mean proj per concept
        base[s] = d
    sig = {(c, L): float(np.std([base[s][L][ci[c]] for s in base], ddof=1))
           for c in concepts for L in analysis}
    base_mean = {(c, L): float(np.mean([base[s][L][ci[c]] for s in base]))
                 for c in concepts for L in analysis}

    idx = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r.get("is_compliant") and r["condition_id"] == "think_at_layer"
                and r.get("concept") in ci and r.get("prompt_layer") is not None):
            idx[int(r["prompt_layer"])][r["concept"]].append(r)

    grid_raw = np.full((len(targets), len(analysis)), np.nan)
    for ti, T in enumerate(targets):
        for li, L in enumerate(analysis):
            per_c = []
            for c in concepts:
                s = sig.get((c, L))
                if not s or s <= 0:
                    continue
                vals = []
                for r in idx.get(T, {}).get(c, []):
                    n = len(r["anchored_token_strs"]) - 1
                    pt = _proj_tokens(r, L, n)
                    if pt is not None:
                        vals.append(float(np.nanmean(pt)))
                if vals:
                    per_c.append((np.mean(vals) - base_mean[(c, L)]) / s)
            if per_c:
                grid_raw[ti, li] = float(np.mean(per_c))
    grid = grid_raw - np.nanmean(grid_raw, axis=0, keepdims=True)         # column-demean
    return targets, analysis, grid


# ------------------------------- serialization -----------------------------------

def _ser(a):
    """List with NaN -> null (the PROFILES_<model>.json convention; the figure
    loaders map null back to NaN)."""
    return [None if (x is None or not np.isfinite(x)) else float(x) for x in np.asarray(a, float)]


def build(model, data_root, main_run, lt_run):
    """Compute the focal aggregates for `model` and return the FIGDATA dict."""
    data_root = Path(data_root)
    cache_dir = data_root / "vector_cache"

    prof, L = position_profile(main_run, model, cache_dir, FRAC_CONDS)
    profiles = {cond: {"centers": _ser(c), "mean": _ser(m), "lo": _ser(lo), "hi": _ser(hi)}
                for cond, (c, m, lo, hi) in prof.items()}

    mw = min_word_count(main_run)
    wprof = word_profile(main_run, model, cache_dir, WORD_CONDS, mw)
    word_conds = {cond: {"x": [int(v) for v in x], "mean": _ser(m), "lo": _ser(lo), "hi": _ser(hi)}
                  for cond, (x, m, lo, hi) in wprof.items()}

    lt = None
    if lt_run is not None:
        targets, analysis, grid = layer_grid(lt_run, main_run, cache_dir)
        lt = {"targets": [int(t) for t in targets], "analysis": [int(a) for a in analysis],
              "grid": [_ser(row) for row in grid]}

    return {
        "model": model,
        "main_run": str(main_run),
        "lt_run": (str(lt_run) if lt_run is not None else None),
        "targeting_depth_frac": PROJ_F_LOC,
        "targeting_layer": int(L),
        "n_boot": N_BOOT,
        "profiles": profiles,
        "word_profiles": {"min_word_count": int(mw), "conds": word_conds},
        "layer_targeting": lt,
    }


def main():
    ap = argparse.ArgumentParser(description="Freeze focal-panel figure inputs to FIGDATA_<model>.json.")
    ap.add_argument("--main-run", required=True, help="the model's MAIN run dir")
    ap.add_argument("--lt-run", help="the model's layer-targeting (_lt) run dir (optional)")
    ap.add_argument("--model", help="model short-name (default: from the run's config.yaml)")
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent.parent / "results"),
                    help="dir holding vector_cache/ and where FIGDATA_ is written (default: results/)")
    ap.add_argument("--json", help="output path (default: <data-root>/FIGDATA_<model>.json)")
    args = ap.parse_args()

    model = sd._resolve_model(args.main_run, args.model)
    out = args.json or str(Path(args.data_root) / f"FIGDATA_{model}.json")
    data = build(model, args.data_root, args.main_run, args.lt_run)
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    lt = "grid" if data["layer_targeting"] else "no LT run (grid omitted)"
    print(f"wrote {out}  (targeting layer {data['targeting_layer']}, {len(data['profiles'])} profiles, {lt})")


if __name__ == "__main__":
    main()
