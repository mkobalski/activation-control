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
from functools import lru_cache

import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

POS, NEG, BASE = "think_about", "dont_think_about", "no_instruction"
RAMP = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]


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


def _resolve_model(run_dir, model):
    """Vector-cache model short-name. An explicit arg wins; otherwise take it
    from the run's own saved config so the analysis follows whatever model the
    run used (gemma4_31b, llama33_70b, ...). There is NO hardcoded default: a
    run whose config.yaml lacks `model` is an error, never a silent gemma3
    assumption."""
    if model is not None:
        return model
    m = run_meta(run_dir).get("model")
    if not m:
        raise ValueError(
            f"{run_dir}/config.yaml has no 'model'; cannot pick the concept-vector "
            f"cache. Pass model=... explicitly or re-run the experiment (it records "
            f"the model short-name).")
    return m


def run_n_layers(run_dir):
    """The model's total decoder-layer count, from the run's saved config.

    Every depth FRACTION resolves against this, so the analysis follows the
    run's own geometry instead of a fixed model. Fails loudly when the config
    lacks `n_layers` rather than assuming a specific model's depth."""
    n = run_meta(run_dir).get("n_layers")
    if not n:
        raise ValueError(
            f"{run_dir}/config.yaml has no 'n_layers'; cannot resolve depth "
            f"fractions to concrete layers. Re-run the experiment (it records "
            f"n_layers) or pass explicit layer indices.")
    return int(n)


def _fraction_to_layer(frac, n_layers):
    """Relative depth in [0,1] -> concrete layer index (truncate, clamp)."""
    return max(0, min(int(n_layers * frac), n_layers - 1))


def _layer_for_fraction(run_dir, frac):
    """Resolve a canonical depth FRACTION to a concrete recorded layer for this
    run's model geometry. The peak/signal layers throughout the analysis are
    defined as fractions (not fixed ints) so they follow the model: on a
    60-layer model 0.90/0.75/1.00/0.70 -> 54/45/59/42; on a 62-layer model
    -> 55/46/61/43. The fractional target is snapped to the nearest layer the
    run actually recorded so it always exists in the stored data."""
    meta = run_meta(run_dir)
    target = _fraction_to_layer(frac, run_n_layers(run_dir))
    rec = [int(x) for x in (meta.get("analysis_layers") or [])]
    if rec:
        target = min(rec, key=lambda L: (abs(L - target), L))
    return target


@lru_cache(maxsize=None)
def load_rows(run_dir):
    """Parsed results.json rows, memoized IN MEMORY for the life of the process.

    Called ~4x per model (unit_layer_readouts, pos_category_readouts,
    location_units, layer_targeting_units), so the memo matters: parsing the
    ~320MB results.json takes seconds.

    The returned list is SHARED between callers -- treat it as read-only.
    """
    with open(Path(run_dir) / "results.json") as f:
        rows = json.load(f)["results"]
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


def _fit_len(a, n):
    """Coerce a 1-D per-token signal to exactly length n (truncate / pad tail
    with NaN). Harmony (gpt-oss) tokenization occasionally captures one fewer
    sentence-span token in a generation trace than in the no-instruction
    baseline, so a trace can come back at n-1 while the baseline and the
    length-n position masks are at n. Padding keeps everything aligned and lets
    _loc_metrics drop the missing position via its ~isnan filter, rather than
    crashing the whole scoring battery on a broadcast error."""
    if a is None:
        return None
    a = np.asarray(a, np.float32)
    if a.shape[0] == n:
        return a
    out = np.full(n, np.nan, np.float32)
    m = min(a.shape[0], n)
    out[:m] = a[:m]
    return out


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


# (bins_for / bin_means live with the onset/offset-error timing metric in
# scripts/compute_scores.py::_persistence_edges.)


# ---- extraction: per-unit condition/baseline readouts (rows 1-4) ----------------

def unit_layer_readouts(run_dir, conds, *, vector_cache="results/vector_cache",
                        method="baseline", model=None):
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
    vecs = load_vectors(vector_cache, _resolve_model(run_dir, model), layers, method)
    concepts_L = vecs[layers[0]][0] if layers[0] in vecs else []

    vals = {(m, c): defaultdict(dict) for m in ("cos", "relnorm", "proj") for c in conds}
    bases = {m: defaultdict(dict) for m in ("cos", "relnorm", "proj")}

    for s, sub in by_sent.items():
        ent = cache.get(s)
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        if ent is None or toks_row is None:
            continue
        toks = toks_row[1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]

        # base_cos[L]: per-token cosine (n_tok x n_concepts); base_norm[L]: per-token
        # raw ||r|| (the proj channel = cos * raw-norm, both per token, reconstructed
        # from results.json -- no results.pkl needed).
        base_cos, base_rel, base_norm = {}, {}, {}
        for L in layers:
            A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            base_cos[L] = An @ vecs[L][2].T if L in vecs else None
            nrm = np.asarray(ent["norms"][L], np.float32)[:n_tok]
            base_norm[L] = nrm
            rl = relnorm(nrm, classes)
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
            bp = np.array([float(np.nanmean(base_cos[L][:, ci] * base_norm[L]))
                           if base_cos[L] is not None else np.nan for L in layers])
            bases["cos"][c][s] = bc
            bases["relnorm"][c][s] = br
            bases["proj"][c][s] = bp
            for cond in conds:
                r_ = byc.get(cond, {}).get(c)
                if r_ is None:
                    continue
                xc = np.full(len(layers), np.nan); xr = np.full(len(layers), np.nan)
                xp = np.full(len(layers), np.nan)
                for li, L in enumerate(layers):
                    tr = trace(r_, "cosine_sim", L)
                    cvec = np.asarray(tr, np.float32)[:n_tok] if tr is not None else None
                    if cvec is not None:
                        xc[li] = float(np.nanmean(cvec))
                    nr = trace(r_, "norms", L)
                    nvec = np.asarray(nr, np.float32)[:n_tok] if nr is not None else None
                    if nvec is not None:
                        rl = relnorm(nvec, classes)
                        if rl is not None:
                            xr[li] = float(np.nanmean(rl))
                    if cvec is not None and nvec is not None:
                        m = min(len(cvec), len(nvec))
                        xp[li] = float(np.nanmean(cvec[:m] * nvec[:m]))
                vals[("cos", cond)][c][s] = xc
                vals[("relnorm", cond)][c][s] = xr
                vals[("proj", cond)][c][s] = xp
    return layers, vals, bases


# ---- extraction: per-POS-category readouts (row 10) -----------------------------

CATS = ["NOUN", "VERB", "DET", "PUNCT", "ADP", "PRON", "ADJ", "ADV", "CCONJ"]
COS_F_POS, RN_F_POS = 0.90, 0.75      # the channels' peak DEPTHS for the POS view
PROJ_F_POS = 0.90                     # projection channel peak DEPTH (POS view)


def pos_category_readouts(run_dir, conds, *, pos_path="pos_tags.json",
                          vector_cache="results/vector_cache",
                          method="baseline", model=None):
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
    layer_of = {"cos": _layer_for_fraction(run_dir, COS_F_POS),
                "relnorm": _layer_for_fraction(run_dir, RN_F_POS),
                "proj": _layer_for_fraction(run_dir, PROJ_F_POS)}
    vecs = load_vectors(vector_cache, _resolve_model(run_dir, model), sorted(set(layer_of.values())), method)
    nG = len(CATS)

    def cat_means(v, cat_of):
        out = np.full(nG, np.nan)
        for g in range(nG):
            sel = v[cat_of == g]
            sel = sel[~np.isnan(sel)]
            if len(sel):
                out[g] = float(sel.mean())
        return out

    vals = {(m, cid): defaultdict(dict) for m in ("cos", "relnorm", "proj") for cid in conds}
    bases = {m: defaultdict(dict) for m in ("cos", "relnorm", "proj")}

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
            if metric in ("cos", "proj"):
                A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
                bc = An @ vecs[L][2].T if L in vecs else None
                # proj = per-token cosine * raw ||r|| (per concept); cos = cosine alone
                base_tok[metric] = (bc if metric == "cos" else
                                    (bc * np.asarray(ent["norms"][L], np.float32)[:n_tok, None]
                                     if bc is not None else None))
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
                if metric in ("cos", "proj"):
                    if base_tok[metric] is None or c not in concepts_L:
                        continue
                    bvec = cat_means(np.asarray(base_tok[metric][:, concepts_L.index(c)], float), cat_of)
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
                    elif metric == "proj":
                        tc, tn = trace(r_, "cosine_sim", L), trace(r_, "norms", L)
                        v = (np.asarray(tc, np.float32)[:n_tok] * np.asarray(tn, np.float32)[:n_tok]
                             if (tc is not None and tn is not None) else None)
                    else:
                        tr = trace(r_, "norms", L)
                        v = relnorm(np.asarray(tr, np.float32)[:n_tok], classes) if tr is not None else None
                    if v is not None:
                        vals[(metric, cid)][c][s] = cat_means(np.asarray(v, float), cat_of)
    return vals, bases


# ---- extraction: location-targeting units (rows 6-7) -----------------------------

COS_F_LOC, RN_F_LOC = 1.00, 0.70      # the channels' peak DEPTHS for targeting
PROJ_F_LOC = 0.90                     # projection channel peak DEPTH (targeting)
# Spatial targeting tasks, scored by the in/out contrast (location_units).
# persist_once is treated as a target token-SPAN (the middle third) so
# "mid-sentence" shares the same metric as beginning/end. (The precise on/off
# persistence conditions are NOT here -- they feed the onset/offset-error timing
# metric in scripts/compute_scores.py::_persistence_edges.)
TARGET_TASKS = [("loc_beginning", "begin"), ("persist_once", "mid"), ("loc_end", "end"),
                ("loc_punctuation", "PUNCT"), ("loc_adjectives", "ADJ")]
# in/out-contrast tasks grouped into the reported measures
TARGET_GROUPS = {"temporal_control": ["loc_beginning", "persist_once", "loc_end"],
                 "token_group": ["loc_punctuation", "loc_adjectives"]}


def _loc_mask(kind, f, upos):
    if kind == "begin":
        return f <= 1 / 3 + 1e-9                            # first third
    if kind == "end":
        return f >= 2 / 3 - 1e-9                            # last third
    if kind == "mid":
        return (f >= 1 / 3 - 1e-9) & (f <= 2 / 3 + 1e-9)   # middle third (mid-sentence)
    return np.array([u == kind for u in upos])              # PUNCT / ADJ


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
                   method="baseline", model=None):
    """store[(channel, cond, series, metric)] = {(s, c): value}, series in
    {loc, think, base}; loc/think values are Δ vs no_instruction, base values
    are the RAW baseline readout (for standardizing the on/off contrast)."""
    COS_L_LOC = _layer_for_fraction(run_dir, COS_F_LOC)
    RN_L_LOC = _layer_for_fraction(run_dir, RN_F_LOC)
    PROJ_L_LOC = _layer_for_fraction(run_dir, PROJ_F_LOC)
    rows = load_rows(run_dir)
    comp = [r for r in rows if r.get("is_compliant")]
    by_sent = defaultdict(list)
    for r in comp:
        by_sent[r["sentence"]].append(r)
    pos_words = pos_by_sentence(pos_path)
    cache = load_baseline(run_dir)
    vecs = load_vectors(vector_cache, _resolve_model(run_dir, model),
                        sorted({COS_L_LOC, PROJ_L_LOC}), method)
    wanted = {c for c, _ in TARGET_TASKS} | {POS}

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
        for _, kind in TARGET_TASKS:
            T = _loc_mask(kind, f, upos)
            if T.sum() >= 1 and (~T).sum() >= 1:
                masks[kind] = (T, ~T, _dist_to_target(f, T))

        def _cos_all(L):
            if L not in vecs:
                return None, []
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            return A @ vecs[L][2].T, vecs[L][0]
        base_cos_all, concepts_cosL = _cos_all(COS_L_LOC)
        base_pcos_all, concepts_projL = _cos_all(PROJ_L_LOC)          # cosine at the proj layer
        base_projnorm = np.asarray(ent["norms"][PROJ_L_LOC], np.float32)[:n]
        base_rn = _fit_len(relnorm(np.asarray(ent["norms"][RN_L_LOC], np.float32)[:n], classes), n)

        byc = defaultdict(dict)
        concepts = set()
        for r in sub:
            cond, c = r["condition_id"], r.get("concept")
            if cond in wanted and c:
                byc[cond][c] = r
                concepts.add(c)

        def signal(row, channel, base_cos_c, base_proj_c):
            if channel == "cos":
                tr = trace(row, "cosine_sim", COS_L_LOC)
                v = _fit_len(np.asarray(tr, np.float32)[:n], n) if tr is not None else None
                return (v - base_cos_c) if (v is not None and base_cos_c is not None) else None
            if channel == "proj":                                     # cosine * raw ||r||
                tc, tn = trace(row, "cosine_sim", PROJ_L_LOC), trace(row, "norms", PROJ_L_LOC)
                if tc is None or tn is None:
                    return None
                v = _fit_len(np.asarray(tc, np.float32)[:n] * np.asarray(tn, np.float32)[:n], n)
                return (v - base_proj_c) if (v is not None and base_proj_c is not None) else None
            tr = trace(row, "norms", RN_L_LOC)
            v = _fit_len(relnorm(np.asarray(tr, np.float32)[:n], classes), n) if tr is not None else None
            return (v - base_rn) if (v is not None and base_rn is not None) else None

        for c in sorted(concepts):
            base_cos_c = (_fit_len(base_cos_all[:, concepts_cosL.index(c)], n)
                          if (base_cos_all is not None and c in concepts_cosL) else None)
            base_proj_c = (_fit_len(base_pcos_all[:, concepts_projL.index(c)] * base_projnorm, n)
                           if (base_pcos_all is not None and c in concepts_projL) else None)
            for channel in ("cos", "relnorm", "proj"):
                think_sig = signal(byc.get(POS, {}).get(c), channel, base_cos_c, base_proj_c) \
                    if POS in byc and c in byc[POS] else None
                for cond, kind in TARGET_TASKS:
                    if kind not in masks:
                        continue
                    T, O, d = masks[kind]
                    loc_sig = signal(byc.get(cond, {}).get(c), channel, base_cos_c, base_proj_c) \
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
                    bsig = {"cos": base_cos_c, "proj": base_proj_c, "relnorm": base_rn}[channel]
                    if bsig is not None:
                        m = _loc_metrics(np.asarray(bsig, float), T, O, d)
                        if m:
                            for mk in ("G_in", "G_out"):
                                store[(channel, cond, "base", mk)][(s, c)] = m[mk]
    return store


# ---- persistence edge timing -----------------------------------------------------
# The 'Onset/offset error' timing metric (computed, excluded from the scalar) lives
# in scripts/compute_scores.py::_persistence_edges. The retired CoM-precision
# companions are no longer computed.


# ---- extraction: layer-targeting units (row 9) ------------------------------------

def layer_targeting_units(lt_run, main_run, *, cond="think_at_layer",
                          vector_cache="results/vector_cache",
                          method="baseline", model=None):
    """Per-unit standardized column-demeaned diagonals for both channels.

    Returns {channel: (D (n_units x k), sent_ids, conc_ids, per-layer sigma note)}
    where D[u, l] = (Δ[T=l, L=l] − mean_T Δ[T, L=l]) / σ(l) with σ = the main
    run's across-sentence baseline SD (per concept for cos; global for relnorm).
    The targeted layers come from the LT run's own `prompt_layers` config, so
    this follows whatever depths that run swept (no fixed model assumption)."""
    layers_lt = [int(L) for L in (run_meta(lt_run).get("prompt_layers") or [])]
    if not layers_lt:
        raise ValueError(
            f"{lt_run}/config.yaml has no 'prompt_layers'; layer-targeting "
            f"analysis needs the swept target layers. Is this a layer-targeting run?")
    rows = load_rows(lt_run)
    comp = [r for r in rows if r.get("is_compliant") and r["condition_id"] == cond
            and r.get("concept") and r.get("prompt_layer") is not None]
    cache = load_baseline(main_run)
    vecs = load_vectors(vector_cache, _resolve_model(main_run, model), layers_lt, method)
    concepts = vecs[layers_lt[0]][0]

    base_mean, base_pmean, base_rn, classes_of = {}, {}, {}, {}
    for s, ent in cache.items():
        toks = ent["anchored_token_strs"][1:]
        n = len(toks)
        classes = [classify(tk) for tk in toks]
        classes_of[s] = classes
        bm, bpm, br = {}, {}, {}
        for L in layers_lt:
            A = np.asarray(ent["activations"][L], np.float32)[:n]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            cos_tok = An @ vecs[L][2].T
            rawn = np.asarray(ent["norms"][L], np.float32)[:n]
            bm[L] = cos_tok.mean(0)
            bpm[L] = (cos_tok * rawn[:, None]).mean(0)              # proj = cosine * raw ||r||
            rl = relnorm(rawn, classes)
            br[L] = float(np.nanmean(rl)) if rl is not None else np.nan
        base_mean[s], base_pmean[s], base_rn[s] = bm, bpm, br
    sig_cos = {(c, L): np.std([base_mean[s][L][ci] for s in base_mean], ddof=1)
               for ci, c in enumerate(concepts) for L in layers_lt}
    sig_proj = {(c, L): np.std([base_pmean[s][L][ci] for s in base_pmean], ddof=1)
                for ci, c in enumerate(concepts) for L in layers_lt}
    sig_rn = {L: np.std([base_rn[s][L] for s in base_rn], ddof=1) for L in layers_lt}

    val = {"cos": defaultdict(dict), "relnorm": defaultdict(dict), "proj": defaultdict(dict)}
    for r in comp:
        s, c, T = r["sentence"], r["concept"], int(r["prompt_layer"])
        if s not in base_mean or c not in concepts:
            continue
        ci = concepts.index(c)
        classes = classes_of[s]
        dc, dn, dp = {}, {}, {}
        for L in layers_lt:
            tr = trace(r, "cosine_sim", L)
            nr = trace(r, "norms", L)
            cvec = np.asarray(tr, np.float32) if tr is not None else None
            nvec = np.asarray(nr, np.float32) if nr is not None else None
            if cvec is not None:
                dc[L] = float(np.nanmean(cvec)) - float(base_mean[s][L][ci])
            if cvec is not None and nvec is not None:
                m = min(len(cvec), len(nvec))
                dp[L] = float(np.nanmean(cvec[:m] * nvec[:m])) - float(base_pmean[s][L][ci])
            if nvec is not None:
                rl = relnorm(nvec, classes)
                if rl is not None and np.isfinite(base_rn[s][L]):
                    dn[L] = float(np.nanmean(rl)) - base_rn[s][L]
        if dc:
            val["cos"][(s, c)][T] = dc
        if dn:
            val["relnorm"][(s, c)][T] = dn
        if dp:
            val["proj"][(s, c)][T] = dp

    sigma = {"cos": lambda c, L: sig_cos[(c, L)], "proj": lambda c, L: sig_proj[(c, L)],
             "relnorm": lambda c, L: sig_rn[L]}
    out = {}
    for ch in ("cos", "relnorm", "proj"):
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
                D[ui, li] = (U[ui, li, li] - np.nanmean(U[ui, :, li])) / sigma[ch](c, L)
        out[ch] = (D, np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
                   layers_lt)
    return out
