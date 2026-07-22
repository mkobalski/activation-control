#!/usr/bin/env python3
"""Word-based onset/offset error -- a SUPERSEDING score, emitted to its own files.

The shipped onset/offset error (scripts/compute_scores.py::_persistence_edges, in
SCORES_<model>.json) treats the `persist_after_fourth` ONSET gate as the 4th TOKEN:
the requested onset is `4/(n-1)`. But the instruction is worded "after the fourth
WORD", and on tokenizers that split words into multiple sub-tokens (e.g. GPT-OSS,
Mistral) the 4th-word boundary sits a token or two later than the 4th token. This
script recomputes the onset gate against the actual 4th-WORD boundary and writes
ONSET_OFFSET_WORD_<model>.json, leaving the original SCORES untouched.

Why it is exact and drift-free: in `_persistence_edges` the requested edge `req`
enters ONLY as a constant subtraction (`error = detected - req`); the detected edges
and the bootstrap are independent of it. So the word-based onset is the token-based
onset shifted by a constant (req_token - req_word); the offset gate ("first half" =
0.5) is unchanged. We therefore REUSE the tested `_persistence_edges` for all
detection/bootstrap and only recompute `req` here, cross-checking our unit
replication by asserting the token-req we recompute matches the original's `req`.

Measure conditions (unchanged from the original): onset gate = persist_after_fourth,
offset gate = persist_first_half. (persist_throughout / persist_once are not used.)

Usage:
  python scripts/onset_offset_word.py                 # all models under results/raw
  python scripts/onset_offset_word.py --models gemma3_27b,gptoss_20b_low
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import score_data as sd                                                   # noqa: E402
import compute_scores as cs                                               # noqa: E402

RESULTS = PROJECT_ROOT / "results"
RAW = RESULTS / "raw"
VC = str(RESULTS / "vector_cache")
_RUN_RE = re.compile(r"^\d{8}_\d{6}_(?P<model>.+)_activation_control$")


def _word_start(i, tok):
    """New WORD start: first token, or a space-led token with alphanumeric content.
    anchored_token_strs are tokenizer.decode()'d, so the marker is a plain leading
    space for every tokenizer family -- no per-model handling needed."""
    return i == 0 or (tok[:1] == " " and any(ch.isalnum() for ch in tok))


def _fifth_word_token_index(toks):
    """0-indexed token position where the 5th word begins (the commanded 'on' start
    for 'after the fourth word'), or None if the sentence has < 5 words."""
    w = 0
    for i, t in enumerate(toks):
        if _word_start(i, t):
            w += 1
            if w == 5:
                return i
    return None


def _requested_onsets(run_dir, ch):
    """Recompute the requested onset over the SAME (sentence, concept) units that
    _persistence_edges scores for `persist_after_fourth`, both token-based
    (4/(n-1), to validate the replication) and word-based (5th-word-start fraction).
    Returns (req_token, req_word, n_units)."""
    frac = sd.PROJ_F_LOC if ch == "proj" else cs._EDGE_FRAC[ch]
    L = sd._layer_for_fraction(run_dir, frac)
    need_cos = ch in ("cos", "proj")
    need_norm = ch in ("relnorm", "proj")
    rows = sd.load_rows(run_dir)
    by_sent = defaultdict(list)
    for r in rows:
        if r.get("is_compliant"):
            by_sent[r["sentence"]].append(r)
    cache = sd.load_baseline(run_dir)
    vecs = sd.load_vectors(VC, sd._resolve_model(run_dir, None), [L])
    ftok, fword = {}, {}
    for s, sub in by_sent.items():
        toks_row = next((r["anchored_token_strs"] for r in sub if r.get("anchored_token_strs")), None)
        ent = cache.get(s)
        if toks_row is None or ent is None:
            continue
        toks = toks_row[1:]
        n = len(toks)
        if n < 4:                                        # same floor as _persistence_edges
            continue
        w5 = _fifth_word_token_index(toks)
        concepts_L = None
        if need_cos:
            concepts_L = vecs[L][0] if L in vecs else None
        byc = {}
        for r in sub:
            if r["condition_id"] == cs.PERSIST_AFTER and r.get("concept"):
                byc[r["concept"]] = r
        for c, r_ in byc.items():
            if need_cos and (concepts_L is None or c not in concepts_L):
                continue
            tc, tn = sd.trace(r_, "cosine_sim", L), sd.trace(r_, "norms", L)
            ok = (tc is not None) if ch == "cos" else \
                 (tn is not None) if ch == "relnorm" else (tc is not None and tn is not None)
            if not ok:
                continue
            ftok[(s, c)] = 4.0 / (n - 1)
            if w5 is not None:
                fword[(s, c)] = w5 / (n - 1)
    req_token = float(np.mean(list(ftok.values()))) if ftok else np.nan
    req_word = float(np.mean(list(fword.values()))) if fword else np.nan
    return req_token, req_word, len(ftok)


def word_onset_offset(run_dir, ch):
    """{score, lo, hi, edges[...]} for one channel: original detection/bootstrap
    (via cs._persistence_edges) with the ONSET edge shifted to the word boundary."""
    orig = cs._persistence_edges(run_dir, ch, vector_cache=VC)
    on = orig[("onset", cs.PERSIST_AFTER)]
    off = orig[("offset", cs.PERSIST_FIRST)]
    req_token, req_word, n_units = _requested_onsets(run_dir, ch)

    # Fidelity guard: our replicated token-req must match the original's `req` (proves
    # we selected the identical unit set). Tolerate tiny float error.
    if np.isfinite(on["req"]) and np.isfinite(req_token) and abs(on["req"] - req_token) > 1e-9:
        raise AssertionError(f"[{ch}] token-req replication {req_token} != original {on['req']} "
                             f"(unit set mismatch)")
    shift = on["req"] - req_word                          # req_token - req_word
    onset = dict(edge="onset", gate="After 4th word",
                 mean=on["mean"] + shift if np.isfinite(on["mean"]) else np.nan,
                 lo=on["lo"] + shift if np.isfinite(on["lo"]) else np.nan,
                 hi=on["hi"] + shift if np.isfinite(on["hi"]) else np.nan,
                 req_word=req_word, req_token=req_token, detected=on["detected"])
    offset = dict(edge="offset", gate="First half", mean=off["mean"], lo=off["lo"],
                  hi=off["hi"], req=0.5, detected=off["detected"])
    errs = [abs(e["mean"]) for e in (onset, offset) if np.isfinite(e["mean"])]
    score = float(np.mean(errs)) if errs else np.nan
    return dict(score=score, lo=None, hi=None, n_units=n_units,
                edges=[_clean(onset), _clean(offset)])


def _clean(d):
    return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()}


def discover(models=None):
    best = {}
    for d in sorted(RAW.glob("*_activation_control"), reverse=True):
        if not d.is_dir() or d.name.endswith("_lt") or "layer_target" in d.name:
            continue
        m = _RUN_RE.match(d.name)
        if m and m.group("model") not in best and (d / "results.json").exists() \
                and (d / "no_instruction_cache.pkl").exists():
            best[m.group("model")] = d
    if models:
        best = {k: v for k, v in best.items() if k in models}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=None, help="comma-separated filter (default: all)")
    ap.add_argument("--channels", default="cos,relnorm,proj")
    args = ap.parse_args()
    models = {s.strip() for s in args.models.split(",")} if args.models else None
    channels = [c.strip() for c in args.channels.split(",")]
    runs = discover(models)
    print(f"[onset_offset_word] {len(runs)} model(s)")
    print(f"{'model':22s} {'ch':7s} {'req_tok':>8s} {'req_word':>8s} {'onset_tok':>10s} {'onset_word':>10s} {'score':>7s}")
    for model in sorted(runs):
        out = {"model": model, "measure": "onset_offset_error_word",
               "supersedes": {"file": f"SCORES_{model}.json", "measure": "onset_offset_error",
                              "reason": "onset gate requested position uses the 4th-WORD boundary "
                                        "(the prompt wording) instead of the 4th TOKEN"},
               "method": "onset req = mean fractional position of the 5th-word-start token; "
                         "offset req = 0.5; error = detected - req (half-max crossings); per-edge "
                         "95% bootstrap CI (B=2000, seed=0); score = mean(|onset|,|offset|). A "
                         "constant shift of the onset edge from scripts/compute_scores.py::"
                         "_persistence_edges (offset unchanged).",
               "n_boot": 2000, "seed": 0, "channels": {}}
        for ch in channels:
            res = word_onset_offset(runs[model], ch)
            out["channels"][ch] = res
            on = res["edges"][0]
            orig = cs._persistence_edges(runs[model], ch, vector_cache=VC)[("onset", cs.PERSIST_AFTER)]
            if ch == "proj":
                print(f"{model:22s} {ch:7s} {on['req_token']:8.3f} {on['req_word']:8.3f} "
                      f"{orig['mean']:10.3f} {on['mean']:10.3f} {res['score']:7.3f}")
        p = RESULTS / f"ONSET_OFFSET_WORD_{model}.json"
        json.dump(out, open(p, "w"), indent=2)
        print(f"  wrote {p.name}")


if __name__ == "__main__":
    main()
