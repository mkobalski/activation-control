#!/usr/bin/env python3
"""Shared focal-model data layer for the figure scripts.

Reads PRECOMPUTED battery outputs from the data root (--data-root / $AC_DATA), so
its numbers are FROZEN and identical to the battery: the depth bands come from
PROFILES_<model>.json. Only the illustrative per-token traces are read raw from
results.json (single trial, no CI).
Nothing under scripts/ or src/ is imported -- just a path to the scoring artifacts.
Titles / layer numbers / model names go in the companion .md files for caption
writing, NOT on the figure.

Projection channel = <r,c_hat> = ||r|| * cos.

This module emits no figure of its own -- see the note at the foot of the file for
what the other scripts import from it.
"""

import argparse
import json
import pickle
from pathlib import Path
from paths import AC_ROOT, AC_DATA, out  # portable, env-overridable paths

import numpy as np
import torch
import matplotlib
from model_family_colors import family_color, family_shades   # canonical palette
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class _FixedOrder(ScalarFormatter):
    """Force a fixed power-of-ten multiplier (e.g. 10^3 shown at the top of the
    axis) so tick labels read as double-digit mantissas."""
    def __init__(self, order):
        super().__init__(useMathText=True)
        self._fixed_order = order
        self.set_scientific(True)
        self.set_useOffset(False)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self._fixed_order

import os
# Data root = the directory holding raw/ (per-run results.json) and vector_cache/.
# This figure script lives in the PAPER repo and reads the activation-control data
# store wherever it is mounted -- no import of the scoring layer. Resolved at runtime
# from --data-root or $AC_DATA (use_data_root, called in main()).
RAW = None
VC = None
DATA_ROOT = None                                      # holds raw/, vector_cache/, and the *.json


def use_data_root(root):
    """Point the loaders at a results-style data directory (contains raw/,
    vector_cache/, and the precomputed SCORES_/PROFILES_ JSONs). Accepts either
    that directory or its parent."""
    global RAW, VC, DATA_ROOT
    if not root:
        raise SystemExit("set --data-root (or $AC_DATA) to the data directory "
                         "holding raw/, vector_cache/ and the SCORES_/PROFILES_ JSONs")
    root = Path(root)
    if not (root / "raw").is_dir() and (root / "results" / "raw").is_dir():
        root = root / "results"                       # tolerate being handed the repo root
    DATA_ROOT = root
    RAW = root / "raw"
    VC = str(root / "vector_cache")


def _load_summary(kind, model):
    """Parsed SCORES_/PROFILES_/SCALAR_CI_ JSON from the data root."""
    p = Path(DATA_ROOT) / f"{kind}_{model}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run compute_scores.py first (need {kind})")
    return json.load(open(p))


def load_score(model, measure, ch="proj"):
    """(score, lo, hi, None) for one measure from SCORES_<model>.json -- the FROZEN
    value the battery reported (bar heights + whiskers, no recompute)."""
    c = _load_summary("SCORES", model)["measures"][measure]["channels"][ch]
    return (c["score"], c["lo"], c["hi"], None)


def load_profile(model, curve, ch="proj"):
    """(depth_pct, mean, lo, hi) for one depth-profile curve from PROFILES_<model>.json."""
    d = _load_summary("PROFILES", model)
    c = d["curves"][curve][ch]

    def arr(k):
        return np.array([np.nan if v is None else v for v in c[k]], float)
    return d["depth_pct"], arr("mean"), arr("lo"), arr("hi")


def run_layers(run_dir, model):
    """Analysis layers present in both the run and the concept-vector cache (the
    ordering PROFILES depth indices map onto), for locating the trace-panel layer."""
    rows = json.load(open(run_dir / "results.json"))["results"]
    layers = sorted({int(x) for r in rows if r.get("is_compliant")
                     for x in (r.get("analysis_layers") or [])})
    return [L for L in layers if L in load_vectors(model, layers)]

CONCEPT = "Bread"
SENTENCE = "The bus was crowded, but I found a seat near the back."

MED_GRAY, LIGHT_GRAY = (0.45, 0.45, 0.45), (0.78, 0.78, 0.78)
ENGAGE_C, SUPPRESS_C = "#c0392b", "#2471a3"

# Roster / family order / display labels: the single canonical source (roster.py),
# which also feeds superplot's x-axis, so the paper figures and the shipped
# model_comparison superplots cannot disagree about who is plotted.
from roster import MODELS, FAMILY_ORDER, DETAIL   # noqa: E402  (single-source roster)


def family_bar_colors(fam, n):
    """n within-family bar colors, lightest (smallest model) -> full strength
    (largest), from the canonical palette. A family with ONE model gets the full
    -strength color, not family_shades(fam, 1) -- that returns the 45%-toward-white
    tint, which would make single-model families read as washed-out next to the
    largest member of a multi-model family."""
    return family_shades(fam, n) if n > 1 else [family_color(fam)]

# The experiment sweeps depth in these ORIGINALLY REQUESTED fractions (5%..100%).
# Depth axes MUST label with these, NOT 100*L/n_layers — the latter round-trips
# fraction -> model-specific layer -> fraction and drifts (e.g. 90% -> L55 -> 89%),
# making the same depth read differently across models.
FRACS_PCT = list(range(5, 101, 5))                       # 5,10,...,100


def depth_pcts(layers, n_total):
    """Depth-% per sorted analysis layer = the requested fraction (multiple of 5),
    model-independent. Falls back to nearest-5% only if the sweep isn't the 20-step."""
    if len(layers) == len(FRACS_PCT):
        return list(FRACS_PCT)
    return [int(round(100 * L / n_total / 5) * 5) for L in layers]


# ------------------------------- data layer --------------------------------------

def run_for(model):
    for d in sorted(RAW.glob(f"*_{model}_activation_control"), reverse=True):
        if (d / "no_instruction_cache.pkl").exists():
            return d
    return None


def load_vectors(model, layers):
    """{L: (concepts, unit_vec_matrix)} from the concept-vector cache."""
    out = {}
    for L in layers:
        p = Path(VC) / f"{model}_layer{L}_baseline.pt"
        if not p.exists():
            continue
        d = torch.load(p, weights_only=False)
        concepts = list(d.keys())
        V = np.stack([d[c].float().cpu().numpy().astype(np.float32) for c in concepts])
        out[L] = (concepts, V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8))
    return out


def _proj_tokens_cond(trial, L, n_tok):
    cs = (trial.get("cosine_sim") or {}); nm = (trial.get("norms") or {})
    a = cs.get(str(L), cs.get(L)); b = nm.get(str(L), nm.get(L))
    if not a or not b:
        return None
    a = np.asarray(a, np.float32)[:n_tok]; b = np.asarray(b, np.float32)[:n_tok]
    m = min(len(a), len(b))
    out = np.full(n_tok, np.nan, np.float32); out[:m] = a[:m] * b[:m]
    return out


def token_traces(run_dir, model, concept, sentence, L):
    """Per-token projection at layer L: (tokens, think, dont, no_instruction)."""
    rows = json.load(open(run_dir / "results.json"))["results"]
    cache = pickle.load(open(run_dir / "no_instruction_cache.pkl", "rb"))
    conceptsL, Vn = load_vectors(model, [L])[L]
    ci = conceptsL.index(concept)
    ent = cache[sentence]
    toks = [t.strip() or "␣" for t in ent["anchored_token_strs"][1:]]
    n_tok = len(toks)
    A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    base = (An @ Vn[ci]) * np.asarray(ent["norms"][L], np.float32)[:n_tok]

    def cond(cid):
        r = next((r for r in rows if r.get("is_compliant") and r["condition_id"] == cid
                  and r.get("concept") == concept and r["sentence"] == sentence), None)
        return _proj_tokens_cond(r, L, n_tok) if r else None
    return toks, cond("think_about"), cond("dont_think_about"), base


# ---------------- precomputed curves (from PROFILES_<model>.json) ----------------

def focal_profiles(run_dir, model):
    """Focal engage/suppress depth curves read from PROFILES_<model>.json (proj) --
    FROZEN and consistent with the SCORES peaks. `layers` (from the run) lets us map
    the engage-peak depth index to the concrete layer for the trace panel."""
    depth, me, le, he = load_profile(model, "engage")
    _, ms, ls, hs = load_profile(model, "suppress")
    return dict(layers=run_layers(run_dir, model), depth=depth,
                mean_e=me, lo_e=le, hi_e=he, mean_s=ms, lo_s=ls, hi_s=hs)


# ------------------------------- render ------------------------------------------

def _nospine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _savefig(fig, out):
    for ext in (out, str(Path(out).with_suffix(".png" if out.endswith(".pdf") else ".pdf"))):
        fig.savefig(ext, bbox_inches="tight")
        print(f"wrote {ext}")
    plt.close(fig)


# This module emits no figure of its own: it is the loader and styling layer the
# other figure scripts import (run_for, focal_profiles, token_traces,
# load_profile, _savefig, the condition colours).
