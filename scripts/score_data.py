#!/usr/bin/env python3
"""Data layer for the SCORES.md battery: stored run artifacts -> per-unit readouts.

This module reads ONLY the stored data files of a run —
    results/raw/<RUN>/results.json          (per-trial traces)
    results/raw/<RUN>/no_instruction_cache.pkl
    results/vector_cache/<model>_layer<L>_<method>.pt
    pos_tags.json
— and produces the canonical per-(sentence, concept) unit readouts the scorer
consumes. It imports NO figure scripts: the benchmark must not change when a
figure is edited, renamed, or retired. (The primitive functions below are
verbatim copies of the analysis-suite versions, frozen here on purpose; the
figure scripts remain downstream views of the same stored data.)

Readouts: cos = cosine(concept vector, residual); relnorm = ||r|| / trial's
content-token mean norm. All values are token-mean per trial unless a function
says otherwise. CPU-only, no model load.
"""

import json
import pickle

import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

POS, NEG, BASE = "think_about", "dont_think_about", "no_instruction"
RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]
N_LAYERS_TOTAL = 62


# ---- primitives (frozen copies; see module docstring) ---------------------------

def rankdata(a):
    """Average-rank (ties shared), 1-based."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    r = np.empty(len(a))
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def signed_spearman(levels, vals):
    """Signed Spearman rho of vals vs. levels; n>=2 (n=2 -> ±1); ties -> NaN."""
    if len(levels) < 2:
        return np.nan
    x = rankdata(np.asarray(levels, float)); y = rankdata(np.asarray(vals, float))
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return np.nan if d == 0 else float((x * y).sum() / d)


def classify(s):
    t = s.strip().lower()
    if t in ("the", "a", "and", ",", ".", "hello"):
        return t
    if s.startswith("<") and s.endswith(">"):
        return "special"
    return "content"


def load_vectors(cache_dir, model, layers, method="baseline"):
    """{L: (concepts, V, unit-normalized V)} from the .pt concept-vector cache."""
    out = {}
    for L in layers:
        p = Path(cache_dir) / f"{model}_layer{L}_{method}.pt"
        if not p.exists():
            print(f"  [warn] missing {p}")
            continue
        d = torch.load(p, weights_only=False)
        concepts = list(d.keys())
        V = np.stack([d[c].float().cpu().numpy().astype(np.float32) for c in concepts])
        out[L] = (concepts, V, V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8))
    return out


def run_meta(run_dir):
    """The run's saved config.yaml: model, n_layers, analysis/prompt layers."""
    p = Path(run_dir) / "config.yaml"
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_rows(run_dir):
    """Parsed results.json rows, memoized on disk (results_rows.cache.pkl)."""
    src = Path(run_dir) / "results.json"
    cache = Path(run_dir) / "results_rows.cache.pkl"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        with open(cache, "rb") as f:
            return pickle.load(f)
    with open(src) as f:
        rows = json.load(f)["results"]
    try:
        tmp = cache.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    except OSError:
        pass
    return rows


def load_baseline(run_dir):
    return pickle.load(open(Path(run_dir) / "no_instruction_cache.pkl", "rb"))


def trace(r, key, L):
    d = r.get(key) or {}
    t = d.get(str(L), d.get(L))
    return np.asarray(t, np.float32) if t else None


def relnorm(norm_vec, classes):
    content = [i for i, c in enumerate(classes) if c == "content" and i < len(norm_vec)]
    return norm_vec / np.mean(norm_vec[content]) if content else None


def pos_by_sentence(pos_path="pos_tags.json"):
    entries = json.load(open(pos_path))["entries"]
    return {e["text"]: e["words"] for e in entries}


def token_upos(sentence, toks, words):
    """UPOS per model token via char-span overlap with the spaCy words."""
    out, cur = [], 0
    for t in toks:
        ts = t.strip()
        if not ts:
            out.append(None); continue
        j = sentence.find(ts, cur)
        if j < 0:
            j = cur
        a, b = j, j + len(ts)
        cur = b
        tag = None
        for w in words:
            if a < w["end"] and b > w["start"]:
                tag = w["upos"]; break
        out.append(tag)
    return out


def bins_for(n_tok, n_bins):
    if n_tok <= 1:
        return np.zeros(max(n_tok, 0), dtype=int)
    f = np.arange(n_tok) / (n_tok - 1)
    return np.minimum((f * n_bins).astype(int), n_bins - 1)


def bin_means(delta_vec, bin_idx, n_bins):
    out = np.full(n_bins, np.nan)
    m = min(len(delta_vec), len(bin_idx))
    for b in range(n_bins):
        sel = [delta_vec[i] for i in range(m) if bin_idx[i] == b and not np.isnan(delta_vec[i])]
        if sel:
            out[b] = float(np.mean(sel))
    return out


# ---- extraction: per-unit condition/baseline readouts (rows 1-4) ----------------

def unit_layer_readouts(run_dir, conds, *, vector_cache="results/vector_cache",
                        method="baseline", model="gemma3_27b"):
    """Token-mean readouts per (channel, condition, concept, sentence, layer).

    Returns (layers, vals, bases):
      vals[(ch, cond)][concept][sentence]  = np vec over layers
      bases[ch][concept][sentence]         = same for the no_instruction baseline
        (relnorm baseline is concept-agnostic, duplicated per concept)
    """
    rows = load_rows(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    layers = sorted({int(x) for r in comp for x in (r.get("analysis_layers") or [])})
    cache = load_baseline(run_dir)
    vecs = load_vectors(vector_cache, model, layers, method)
    concepts_L = vecs[layers[0]][0] if layers[0] in vecs else []

    vals = {(m, c): defaultdict(dict) for m in ("cos", "relnorm") for c in conds}
    bases = {m: defaultdict(dict) for m in ("cos", "relnorm")}

    for s, sub in by_sent.items():
        ent = cache.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if ent is None or toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]

        base_cos, base_rel = {}, {}
        for L in layers:
            A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            base_cos[L] = An @ vecs[L][2].T if L in vecs else None
            rl = relnorm(np.asarray(ent["norms"][L], np.float32)[:n_tok], classes)
            base_rel[L] = float(np.nanmean(rl)) if rl is not None else np.nan

        byc = defaultdict(dict)
        for r in sub:
            if r["condition_id"] in conds and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r

        for c in concepts_L:
            ci = concepts_L.index(c)
            bc = np.array([float(np.nanmean(base_cos[L][:, ci]))
                           if base_cos[L] is not None else np.nan for L in layers])
            br = np.array([base_rel[L] for L in layers])
            bases["cos"][c][s] = bc
            bases["relnorm"][c][s] = br
            for cond in conds:
                r_ = byc.get(cond, {}).get(c)
                if r_ is None:
                    continue
                xc = np.full(len(layers), np.nan); xr = np.full(len(layers), np.nan)
                for li, L in enumerate(layers):
                    tr = trace(r_, "cosine_sim", L)
                    if tr is not None:
                        xc[li] = float(np.nanmean(np.asarray(tr, np.float32)[:n_tok]))
                    nr = trace(r_, "norms", L)
                    if nr is not None:
                        rl = relnorm(np.asarray(nr, np.float32)[:n_tok], classes)
                        if rl is not None:
                            xr[li] = float(np.nanmean(rl))
                vals[("cos", cond)][c][s] = xc
                vals[("relnorm", cond)][c][s] = xr
    return layers, vals, bases


# ---- extraction: per-POS-category readouts (row 10) -----------------------------

CATS = ["NOUN", "VERB", "DET", "PUNCT", "ADP", "PRON", "ADJ", "ADV", "CCONJ"]
COS_L_POS, RN_L_POS = 55, 46          # the channels' peak layers for the POS view


def pos_category_readouts(run_dir, conds, *, pos_path="pos_tags.json",
                          vector_cache="results/vector_cache",
                          method="baseline", model="gemma3_27b"):
    """Per-sentence category-mean readouts at the peak layers (cos@55, rn@46).

    Returns (vals, bases): vals[(ch, cond)][concept][sentence] = vec over CATS;
    bases[ch][concept][sentence] likewise for no_instruction."""
    rows = load_rows(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)

    pos_words = pos_by_sentence(pos_path)
    cache = load_baseline(run_dir)
    layer_of = {"cos": COS_L_POS, "relnorm": RN_L_POS}
    vecs = load_vectors(vector_cache, model, sorted(set(layer_of.values())), method)
    nG = len(CATS)

    def cat_means(v, cat_of):
        out = np.full(nG, np.nan)
        for g in range(nG):
            sel = v[cat_of == g]
            sel = sel[~np.isnan(sel)]
            if len(sel):
                out[g] = float(sel.mean())
        return out

    vals = {(m, cid): defaultdict(dict) for m in ("cos", "relnorm") for cid in conds}
    bases = {m: defaultdict(dict) for m in ("cos", "relnorm")}

    for s, sub in by_sent.items():
        words = pos_words.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if words is None or toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        cat_of = np.array([CATS.index(u) if u in CATS else -1
                           for u in token_upos(s, toks, words)])

        base_tok = {}
        for metric, L in layer_of.items():
            if metric == "cos":
                A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                base_tok[metric] = An @ vecs[L][2].T if L in vecs else None
            else:
                base_tok[metric] = relnorm(np.asarray(ent["norms"][L], np.float32)[:n_tok], classes)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            if r["condition_id"] in conds and r.get("concept"):
                byc[r["condition_id"]][r["concept"]] = r
                concepts.add(r["concept"])

        for metric, L in layer_of.items():
            concepts_L = vecs[L][0] if L in vecs else []
            for c in sorted(concepts):
                if metric == "cos":
                    if base_tok["cos"] is None or c not in concepts_L:
                        continue
                    bvec = cat_means(np.asarray(base_tok["cos"][:, concepts_L.index(c)], float), cat_of)
                else:
                    if base_tok["relnorm"] is None:
                        continue
                    bvec = cat_means(np.asarray(base_tok["relnorm"], float), cat_of)
                bases[metric][c][s] = bvec
                for cid in conds:
                    r_ = byc.get(cid, {}).get(c)
                    if r_ is None:
                        continue
                    if metric == "cos":
                        tr = trace(r_, "cosine_sim", L)
                        v = np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
                    else:
                        tr = trace(r_, "norms", L)
                        v = relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None
                    if v is not None:
                        vals[(metric, cid)][c][s] = cat_means(np.asarray(v, float), cat_of)
    return vals, bases


# ---- extraction: location-targeting units (rows 6-7) -----------------------------

COS_L_LOC, RN_L_LOC = 61, 43          # the channels' peak layers for targeting
LOC_CONDS = [("loc_beginning", "begin"), ("loc_end", "end"),
             ("loc_punctuation", "PUNCT"), ("loc_adjectives", "ADJ")]


def _loc_mask(kind, f, upos):
    if kind == "begin":
        return f <= 1 / 3 + 1e-9
    if kind == "end":
        return f >= 2 / 3 - 1e-9
    return np.array([u == kind for u in upos])


def _dist_to_target(f, T):
    tf = f[T]
    return np.array([float(np.min(np.abs(fi - tf))) for fi in f])


def _loc_metrics(sig, T, O, d):
    """Inside gain, outside leakage, CoM error for a per-token signal."""
    v = ~np.isnan(sig)
    Ti, Oi = T & v, O & v
    if Ti.sum() < 1 or Oi.sum() < 1:
        return None
    g_in = float(sig[Ti].mean()); g_out = float(sig[Oi].mean())
    p = np.clip(np.where(v, sig, 0.0), 0, None)
    com = float((p / p.sum() * d).sum()) if p.sum() > 1e-12 else np.nan
    return dict(G_in=g_in, G_out=g_out, CoM=com)


def location_units(run_dir, *, pos_path="pos_tags.json",
                   vector_cache="results/vector_cache",
                   method="baseline", model="gemma3_27b"):
    """store[(channel, cond, series, metric)] = {(s, c): value}, series in
    {loc, think, base}; loc/think values are Δ vs no_instruction, base values
    are the RAW baseline readout (for standardizing the on/off contrast)."""
    rows = load_rows(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    pos_words = pos_by_sentence(pos_path)
    cache = load_baseline(run_dir)
    vecs = load_vectors(vector_cache, model, [COS_L_LOC], method)
    wanted = {c for c, _ in LOC_CONDS} | {POS}

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
        classes = [classify(t) for t in toks]
        upos = token_upos(s, toks, words)
        f = np.arange(n) / (n - 1)
        masks = {}
        for _, kind in LOC_CONDS:
            T = _loc_mask(kind, f, upos)
            if T.sum() >= 1 and (~T).sum() >= 1:
                masks[kind] = (T, ~T, _dist_to_target(f, T))

        acos = np.asarray(ent["activations"][COS_L_LOC], np.float32)[:n]
        acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
        base_cos_all = acos @ vecs[COS_L_LOC][2].T if COS_L_LOC in vecs else None
        base_rn = relnorm(np.asarray(ent["norms"][RN_L_LOC], np.float32)[:n], classes)
        concepts_cosL = vecs[COS_L_LOC][0] if COS_L_LOC in vecs else []

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        def signal(row, channel, base_cos_c):
            if channel == "cos":
                tr = trace(row, "cosine_sim", COS_L_LOC)
                v = np.asarray(tr, np.float32)[:n] if tr is not None else None
                return (v - base_cos_c) if (v is not None and base_cos_c is not None) else None
            tr = trace(row, "norms", RN_L_LOC)
            v = relnorm(np.asarray(tr, np.float32)[:n], classes) if tr is not None else None
            return (v - base_rn) if (v is not None and base_rn is not None) else None

        for c in sorted(concepts):
            base_cos_c = (base_cos_all[:, concepts_cosL.index(c)]
                          if (base_cos_all is not None and c in concepts_cosL) else None)
            for channel in ("cos", "relnorm"):
                think_sig = signal(byc.get(POS, {}).get(c), channel, base_cos_c) \
                    if POS in byc and c in byc[POS] else None
                for cond, kind in LOC_CONDS:
                    if kind not in masks:
                        continue
                    T, O, d = masks[kind]
                    loc_sig = signal(byc.get(cond, {}).get(c), channel, base_cos_c) \
                        if cond in byc and c in byc[cond] else None
                    if loc_sig is not None:
                        m = _loc_metrics(loc_sig, T, O, d)
                        if m:
                            for mk, v in m.items():
                                store[(channel, cond, "loc", mk)][(s, c)] = v
                    if think_sig is not None:
                        m = _loc_metrics(think_sig, T, O, d)
                        if m:
                            for mk, v in m.items():
                                store[(channel, cond, "think", mk)][(s, c)] = v
                    bsig = base_cos_c if channel == "cos" else base_rn
                    if bsig is not None:
                        m = _loc_metrics(np.asarray(bsig, float), T, O, d)
                        if m:
                            for mk in ("G_in", "G_out"):
                                store[(channel, cond, "base", mk)][(s, c)] = m[mk]
    return store


# ---- extraction: persistence edge timing (row 8) ---------------------------------

PERSIST = dict(THROUGH="persist_throughout", FIRST="persist_first_half",
               ONCE="persist_once", AFTER="persist_after_fourth")
PERSIST_LABEL = {PERSIST["THROUGH"]: "Throughout", PERSIST["FIRST"]: "First half",
                 PERSIST["ONCE"]: "Once", PERSIST["AFTER"]: "After 4th"}
EDGE_TASKS = [("onset", [PERSIST["ONCE"], PERSIST["AFTER"]]),
              ("offset", [PERSIST["FIRST"], PERSIST["ONCE"]])]
N_BINS = 10


def _detect(profile):
    """Half-max rising/falling crossings on a rectified profile."""
    p = np.clip(profile, 0, None)
    if not np.isfinite(p).all() or p.max() - p.min() <= 1e-12:
        return np.nan, np.nan
    above = np.where(p >= p.min() + 0.5 * (p.max() - p.min()))[0]
    if len(above) == 0:
        return np.nan, np.nan
    return (above[0] + 0.5) / len(p), (above[-1] + 0.5) / len(p)


def persistence_edges(run_dir, *, metric="cos", layer=None,
                      vector_cache="results/vector_cache", method="baseline",
                      model="gemma3_27b", n_boot=2000, seed=0):
    """Edge-timing errors (detected − requested, fractional) per gating task.

    metric='cos' (layer 61) or 'relnorm' (layer 43). Returns
    {(edge, cond): dict(mean, lo, hi, req, detected)}."""
    L_SIG = layer if layer is not None else (61 if metric == "cos" else 43)
    rows = load_rows(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    cache = load_baseline(run_dir)
    vecs = load_vectors(vector_cache, model, [L_SIG], method)
    conds = list(PERSIST.values())

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
        classes = [classify(t) for t in toks]
        bins = bins_for(n, N_BINS)
        if metric == "cos":
            acos = np.asarray(ent["activations"][L_SIG], np.float32)[:n]
            acos = acos / (np.linalg.norm(acos, axis=1, keepdims=True) + 1e-8)
            base_cos_all = acos @ vecs[L_SIG][2].T if L_SIG in vecs else None
            base_rn = None
        else:
            base_cos_all = None
            base_rn = relnorm(np.asarray(ent["norms"][L_SIG], np.float32)[:n], classes)
        concepts_L = vecs[L_SIG][0] if L_SIG in vecs else []

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in conds and c:
                byc[cond][c] = r
                concepts.add(c)

        def dsig(row, base_c):
            if metric == "cos":
                tr = trace(row, "cosine_sim", L_SIG)
                v = np.asarray(tr, np.float32)[:n] if tr is not None else None
            else:
                tr = trace(row, "norms", L_SIG)
                v = relnorm(np.asarray(tr, np.float32)[:n], classes) if tr is not None else None
            return (v - base_c) if (v is not None and base_c is not None) else None

        for c in sorted(concepts):
            if metric == "cos":
                if base_cos_all is None or c not in concepts_L:
                    continue
                base_c = base_cos_all[:, concepts_L.index(c)]
            else:
                if base_rn is None:
                    continue
                base_c = base_rn
            for cond in conds:
                r_ = byc.get(cond, {}).get(c)
                if r_ is None:
                    continue
                sig = dsig(r_, base_c)
                if sig is not None:
                    prof[cond][(s, c)] = bin_means(sig, bins, N_BINS)
                    if cond == PERSIST["AFTER"]:
                        f5[(s, c)] = 4.0 / (n - 1)

    rng = np.random.default_rng(seed)
    req_on = {PERSIST["ONCE"]: 0.5,
              PERSIST["AFTER"]: float(np.mean(list(f5.values()))) if f5 else np.nan}
    req_off = {PERSIST["FIRST"]: 0.5, PERSIST["ONCE"]: 0.5}
    out = {}
    for edge, cond_list in EDGE_TASKS:
        for cond in cond_list:
            keys = list(prof[cond])
            if len(keys) < 3:
                out[(edge, cond)] = dict(mean=np.nan, lo=np.nan, hi=np.nan,
                                         req=np.nan, detected=np.nan)
                continue
            P = np.vstack([prof[cond][k] for k in keys])
            on0, off0 = _detect(np.nanmean(P, 0))
            det0 = on0 if edge == "onset" else off0
            req = (req_on if edge == "onset" else req_off).get(cond, np.nan)
            errs = []
            for _ in range(n_boot):
                bi = rng.integers(0, len(keys), size=len(keys))
                on, off = _detect(np.nanmean(P[bi], 0))
                dv = on if edge == "onset" else off
                if not np.isnan(dv):
                    errs.append(dv - req)
            errs = np.array(errs)
            out[(edge, cond)] = dict(
                mean=float(det0 - req) if not np.isnan(det0) else np.nan,
                lo=float(np.percentile(errs, 2.5)) if len(errs) else np.nan,
                hi=float(np.percentile(errs, 97.5)) if len(errs) else np.nan,
                req=req, detected=det0)
    return out


# ---- extraction: layer-targeting units (row 9) ------------------------------------

DEEP_LT = [40, 43, 46, 49, 52, 55, 58, 61]


def layer_targeting_units(lt_run, main_run, *, cond="think_at_layer",
                          vector_cache="results/vector_cache",
                          method="baseline", model="gemma3_27b"):
    """Per-unit standardized column-demeaned diagonals for both channels.

    Returns {channel: (D (n_units x 8), sent_ids, conc_ids, per-layer sigma note)}
    where D[u, l] = (Δ[T=l, L=l] − mean_T Δ[T, L=l]) / σ(l) with σ = the main
    run's across-sentence baseline SD (per concept for cos; global for relnorm)."""
    layers_lt = list(run_meta(lt_run).get("prompt_layers") or DEEP_LT)
    rows = load_rows(lt_run)
    comp = [r for r in rows if r.get("is_compliant") and r["condition_id"] == cond
            and r.get("concept") and r.get("prompt_layer") is not None]
    cache = load_baseline(main_run)
    vecs = load_vectors(vector_cache, model, layers_lt, method)
    concepts = vecs[layers_lt[0]][0]

    base_mean, base_rn, classes_of = {}, {}, {}
    for s, ent in cache.items():
        toks = ent["anchored_token_strs"][1:]
        n = len(toks)
        classes = [classify(tk) for tk in toks]
        classes_of[s] = classes
        bm, br = {}, {}
        for L in layers_lt:
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            bm[L] = (An @ vecs[L][2].T).mean(0)
            rl = relnorm(np.asarray(ent["norms"][L], np.float32)[:n], classes)
            br[L] = float(np.nanmean(rl)) if rl is not None else np.nan
        base_mean[s], base_rn[s] = bm, br
    sig_cos = {(c, L): np.std([base_mean[s][L][ci] for s in base_mean], ddof=1)
               for ci, c in enumerate(concepts) for L in layers_lt}
    sig_rn = {L: np.std([base_rn[s][L] for s in base_rn], ddof=1) for L in layers_lt}

    val = {"cos": defaultdict(dict), "relnorm": defaultdict(dict)}
    for r in comp:
        s, c, T = r["sentence"], r["concept"], int(r["prompt_layer"])
        if s not in base_mean or c not in concepts:
            continue
        ci = concepts.index(c)
        classes = classes_of[s]
        dc, dn = {}, {}
        for L in layers_lt:
            tr = trace(r, "cosine_sim", L)
            if tr is not None:
                dc[L] = float(np.nanmean(np.asarray(tr, np.float32))) - float(base_mean[s][L][ci])
            nr = trace(r, "norms", L)
            if nr is not None:
                rl = relnorm(np.asarray(nr, np.float32), classes)
                if rl is not None and np.isfinite(base_rn[s][L]):
                    dn[L] = float(np.nanmean(rl)) - base_rn[s][L]
        if dc:
            val["cos"][(s, c)][T] = dc
        if dn:
            val["relnorm"][(s, c)][T] = dn

    out = {}
    for ch in ("cos", "relnorm"):
        units = {k: v for k, v in val[ch].items() if len(v) == len(layers_lt)}
        keys = sorted(units)
        U = np.full((len(keys), len(layers_lt), len(layers_lt)), np.nan)
        for ui, k in enumerate(keys):
            for ti, T in enumerate(layers_lt):
                for li, L in enumerate(layers_lt):
                    U[ui, ti, li] = units[k][T].get(L, np.nan)
        D = np.full((len(keys), len(layers_lt)), np.nan)
        for ui, (s, c) in enumerate(keys):
            for li, L in enumerate(layers_lt):
                sd = sig_cos[(c, L)] if ch == "cos" else sig_rn[L]
                D[ui, li] = (U[ui, li, li] - np.nanmean(U[ui, :, li])) / sd
        out[ch] = (D, np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
                   layers_lt)
    return out
