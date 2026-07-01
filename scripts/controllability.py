#!/usr/bin/env python3
"""Token-class-resolved activation-controllability measures (CPU-only).

The readout is computed PER TOKEN and scored WITHIN a token class -- never as
a span mean -- because controllability varies sharply by token type (structural
slots `the`/`a`/`and`/`,`/`.` vs `content`; see Update_5-14-26.md). Reporting a
single span-averaged number would blend a high-control period with near-zero-
control content tokens into a meaningless middle.

Per (analysis_layer, token_class, concept) it computes bounded-[0,1] scores:

  C_dir   -- DIRECTIONAL control. AUROC separating a positive condition
             (default think_intensely) from a negative (default
             dont_think_about), rescaled 2*|AUC-0.5| -> [0,1].
  C_grade -- GRADED control. Spearman corr between commanded intensity
             (think_intensity_{1..4}_of_4) and the readout, max(0, rho)->[0,1].

Each in two flavors:
  raw       -- readout = on-concept score  s_on
  specific  -- readout = s_on - mean_{C'!=C} s_off(C')   (cancels effects shared
               across all concept directions, e.g. global magnitude rescaling).

Per-token readout score:
  --metric cos  : cos(concept_vec, r_t)                 (magnitude-invariant)
  --metric proj : cos(concept_vec, r_t) * ||r_t|| = (c.r_t)/||c||

Granularity of a (class) sample:
  --granularity class : per trial, mean over that trial's tokens of the class
                        -> AUROC/Spearman across TRIALS (trial-independent stats)
  --granularity token : every in-class token is its own sample (more samples,
                        but tokens within a trial are correlated)

Needs only saved data -- NO GPU, NO model load:
  <run-dir>/results.pkl                       (per-token activations + token strs)
  <vector-cache>/<model>_layer{L}_baseline.pt (concept vectors; CPU torch.load)

Example:
  python scripts/controllability.py \
      --run-dir results/raw/gemma3_27b_write_introspection_main_layers_40_45_55_20260505_042445 \
      --model-name gemma3_27b --metric cos --granularity class
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


# ----------------------------------------------------------------------------- token classes

# Same buckets as scripts/run_attn_experiment.py (minus `special`, which never
# appears inside the generated sentence span).
def classify(token_str: str) -> str:
    t = token_str.strip().lower()
    if t in ("the", "a", "and", ",", ".", "hello"):
        return t
    if token_str.startswith("<") and token_str.endswith(">"):
        return "special"
    if token_str in ("\n", " \n", "\n\n"):
        return "special"
    return "content"


CLASS_ORDER = ["the", "a", "and", ",", ".", "content", "hello"]


# ----------------------------------------------------------------------------- stats helpers

def rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (ties shared), 1-based. Mirrors scipy.stats.rankdata.

    HOW: sort once, then sweep the sorted array finding maximal runs of equal
    values [i..j]; every element in a tie-run is assigned the mean of the 1-based
    ranks it spans, (i+j)/2 + 1. `order` maps sorted positions back to the
    original index so ranks land on the right elements. Self-contained so the
    script has no scipy dependency (CPU/numpy-only, see module docstring).
    """
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(pos, neg):
    """P(pos > neg) with ties=0.5, via the Mann-Whitney rank statistic.

    HOW: AUROC equals the normalized Mann-Whitney U. Rank the pooled pos+neg
    samples, sum the ranks falling on positives, subtract the minimum possible
    rank sum n_pos*(n_pos+1)/2, and divide by n_pos*n_neg. WHY this form: it is
    exactly the fraction of (pos, neg) pairs where pos>neg (ties counting 0.5),
    i.e. the discrimination probability, computed without forming all pairs.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def spearman(x, y):
    # Spearman rho = Pearson correlation of the ranks. Rank both vectors,
    # mean-center, and form the cosine of the centered rank vectors. Returns NaN
    # for <3 points (uninformative) or a degenerate side (zero variance/denom).
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return np.nan if denom == 0 else float((rx * ry).sum() / denom)


# ----------------------------------------------------------------------------- data access

def get_acts(trial, L):
    """(n_tok, d) float32 sentence-span activations at L (int/str key safe)."""
    d = trial.get("activations")
    if d is None:
        return None
    arr = d.get(L, d.get(str(L)))
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    return arr if arr.ndim == 2 and arr.shape[0] > 0 else None


def get_labels(trial):
    """Token-class per sentence-span position. anchored_token_strs[0] is the
    anchor (prompt-last token); [1:] aligns 1:1 with `activations`."""
    strs = trial.get("anchored_token_strs")
    if not strs or len(strs) < 2:
        return None
    return [classify(s) for s in strs[1:]]


def load_vectors(cache_dir: Path, model: str, layers, method="baseline"):
    out = {}
    for L in layers:
        p = cache_dir / f"{model}_layer{L}_{method}.pt"
        if not p.exists():
            print(f"  [warn] missing vector cache {p}, skipping layer {L}")
            continue
        d = torch.load(p, weights_only=False)
        concepts = list(d.keys())
        V = np.stack([d[c].float().cpu().numpy().astype(np.float32) for c in concepts])
        out[L] = (concepts, V)
    return out


def per_token_scores(acts, V, Vn, ci, metric):
    """Return (s_on_t, s_spec_t): per-token on-concept and specificity-corrected.

    HOW: normalize each token residual r_t, then take cosines against ALL unit
    concept vectors at once (one matmul -> (n_tok, n_concepts)). For metric=proj
    we re-scale by ||r_t|| to recover the signed projection magnitude. s_on is
    the cosine to the trial's own concept; s_off is the mean cosine to every
    OTHER concept. s_spec = s_on - s_off cancels any effect shared across all
    concept directions (e.g. a global magnitude rescaling), isolating the part
    that is specific to this concept. (V is unused but kept for signature symmetry.)
    """
    rn = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8   # (n_tok,1)
    cos = (acts / rn) @ Vn.T                                  # (n_tok, n_concepts)
    if metric == "proj":
        cos = cos * rn
    nc = cos.shape[1]
    s_on = cos[:, ci]                                         # (n_tok,)
    s_off = (cos.sum(axis=1) - s_on) / max(nc - 1, 1)
    return s_on, s_on - s_off


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-name", default="gemma3_27b")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--vector-method", default="baseline")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--metric", choices=["cos", "proj"], default="cos")
    ap.add_argument("--granularity", choices=["class", "token"], default="class")
    ap.add_argument("--pos-cond", default="think_intensely")
    ap.add_argument("--neg-cond", default="dont_think_about")
    ap.add_argument("--intensity-prefix", default="think_intensity_")
    ap.add_argument("--intensity-suffix", default="_of_4")
    ap.add_argument("--min-n", type=int, default=8,
                    help="min samples per side to report a cell (else NaN)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "results.pkl", "rb") as f:
        results = pickle.load(f)["results"]
    print(f"Loaded {len(results)} trials from {run_dir}")

    layers = (sorted(args.layers) if args.layers
              else sorted(int(x) for x in (results[0].get("analysis_layers") or [])))
    cache = load_vectors(Path(args.vector_cache), args.model_name, layers, args.vector_method)
    layers = [L for L in layers if L in cache]
    if not layers:
        sys.exit("No usable vector caches found; check --vector-cache / --model-name.")
    print(f"Layers: {layers}  metric: {args.metric}  granularity: {args.granularity}")

    Vn = {L: cache[L][1] / (np.linalg.norm(cache[L][1], axis=1, keepdims=True) + 1e-8)
          for L in layers}
    concept_idx = {L: {c: i for i, c in enumerate(cache[L][0])} for L in layers}

    # Accumulators keyed by (analysis_layer, token_class, concept). Each maps a
    # bucket name -> list of samples: "pos"/"neg" feed the directional AUROC, and
    # "grade" holds (intensity_level, readout) pairs for the graded Spearman.
    # Two parallel accumulators for the two readout flavors (see module docstring).
    raw = defaultdict(lambda: defaultdict(list))   # readout = s_on
    spec = defaultdict(lambda: defaultdict(list))  # readout = s_on - s_off

    def intensity_level(cid):
        # Parse the commanded intensity from condition ids shaped like
        # think_intensity_<k>_of_4 -> k; anything else returns None.
        if not (cid.startswith(args.intensity_prefix) and cid.endswith(args.intensity_suffix)):
            return None
        mid = cid[len(args.intensity_prefix):-len(args.intensity_suffix)]
        return int(mid) if mid.isdigit() else None

    # Single pass over trials: keep only compliant, concept-bearing trials whose
    # condition is one of the directional poles or an intensity rung, then route
    # each in-class token's readout into the matching (key, bucket).
    for t in results:
        if not t.get("is_compliant") or t.get("concept") is None:
            continue
        cid = t.get("condition_id")
        lvl = intensity_level(cid)
        is_pos, is_neg = cid == args.pos_cond, cid == args.neg_cond
        if not (is_pos or is_neg or lvl is not None):
            continue
        labels = get_labels(t)
        if labels is None:
            continue
        for L in layers:
            acts = get_acts(t, L)
            if acts is None:
                continue
            ci = concept_idx[L].get(t["concept"])
            if ci is None:
                continue
            n = min(len(acts), len(labels))
            if n == 0:
                continue
            s_on, s_spec = per_token_scores(acts[:n], cache[L][1], Vn[L], ci, args.metric)
            lab = labels[:n]

            # group token positions by class
            by_cls = defaultdict(list)
            for ti, cl in enumerate(lab):
                if cl == "special":
                    continue
                by_cls[cl].append(ti)

            # One bucket per class present in this trial. At class granularity a
            # trial contributes a single sample (mean over its in-class tokens),
            # keeping samples trial-independent; at token granularity every
            # in-class token is its own (correlated) sample.
            for cl, idxs in by_cls.items():
                key = (L, cl, t["concept"])
                idxs = np.asarray(idxs)
                if args.granularity == "class":
                    on_samples = [float(s_on[idxs].mean())]
                    sp_samples = [float(s_spec[idxs].mean())]
                else:  # token
                    on_samples = [float(v) for v in s_on[idxs]]
                    sp_samples = [float(v) for v in s_spec[idxs]]
                if is_pos:
                    raw[key]["pos"] += on_samples; spec[key]["pos"] += sp_samples
                if is_neg:
                    raw[key]["neg"] += on_samples; spec[key]["neg"] += sp_samples
                if lvl is not None:
                    raw[key]["grade"] += [(lvl, v) for v in on_samples]
                    spec[key]["grade"] += [(lvl, v) for v in sp_samples]

    # ---- reduce each (layer, class, concept) bucket to bounded [0,1] scores ----
    def cdir(d):
        # Directional control: how well the readout separates pos from neg.
        # Fold AUROC about 0.5 (2*|AUC-0.5|) so 0.5->0 (no separation) and either
        # extreme ->1; require >=min_n per side or the cell is left NaN.
        a = auroc(d.get("pos", []), d.get("neg", []))
        if np.isnan(a) or len(d.get("pos", [])) < args.min_n or len(d.get("neg", [])) < args.min_n:
            return np.nan
        return 2 * abs(a - 0.5)

    def cgrade(d):
        # Graded control: monotonic agreement between commanded intensity and the
        # readout, via Spearman over the (level, value) pairs. Clamp to [0,1]
        # (negative correlations count as no control); NaN if too few pairs.
        g = d.get("grade", [])
        if len(g) < max(args.min_n, 3):
            return np.nan
        rho = spearman([x for x, _ in g], [y for _, y in g])
        return np.nan if np.isnan(rho) else max(0.0, rho)

    rows = []
    keys = sorted(set(list(raw.keys()) + list(spec.keys())),
                  key=lambda k: (k[0], CLASS_ORDER.index(k[1]) if k[1] in CLASS_ORDER else 99, k[2]))
    for key in keys:
        L, cl, concept = key
        r, s = raw.get(key, {}), spec.get(key, {})
        rows.append({
            "layer": L, "class": cl, "concept": concept,
            "n_pos": len(r.get("pos", [])), "n_neg": len(r.get("neg", [])),
            "n_grade": len(r.get("grade", [])),
            "Cdir_raw": cdir(r), "Cdir_specific": cdir(s),
            "Cgrade_raw": cgrade(r), "Cgrade_specific": cgrade(s),
        })

    def col_mean(sub, k):
        v = [x[k] for x in sub if not np.isnan(x[k])]
        return np.mean(v) if v else np.nan

    # ---- report: per (layer, class), averaged across concepts ----
    print(f"\nC_dir = 2*|AUROC({args.pos_cond} vs {args.neg_cond})-0.5|   "
          f"C_grade = max(0, Spearman(intensity, readout))")
    print("raw = on-concept ; specific = on minus mean off-concept "
          "(cancels global magnitude scaling)\n")
    print(f"{'layer':>5} {'class':>8} {'Cdir_raw':>9} {'Cdir_spec':>9} "
          f"{'Cgrd_raw':>9} {'Cgrd_spec':>9} {'concepts':>9}")
    for L in layers:
        for cl in CLASS_ORDER:
            sub = [x for x in rows if x["layer"] == L and x["class"] == cl]
            sub = [x for x in sub if not (np.isnan(x["Cdir_raw"]) and np.isnan(x["Cdir_specific"]))]
            if not sub:
                continue
            print(f"{L:>5} {cl:>8} "
                  f"{col_mean(sub,'Cdir_raw'):>9.2f} {col_mean(sub,'Cdir_specific'):>9.2f} "
                  f"{col_mean(sub,'Cgrade_raw'):>9.2f} {col_mean(sub,'Cgrade_specific'):>9.2f} "
                  f"{len(sub):>9}")
        print()

    # ---- save full per-(layer,class,concept) detail ----
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "controllability_by_class.csv"
    cols = ["layer", "class", "concept", "n_pos", "n_neg", "n_grade",
            "Cdir_raw", "Cdir_specific", "Cgrade_raw", "Cgrade_specific"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(",".join(
                ("" if isinstance(row[c], float) and np.isnan(row[c])
                 else f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]))
                for c in cols) + "\n")
    print(f"Wrote {out_csv}  ({len(rows)} layer/class/concept cells)")


if __name__ == "__main__":
    main()
