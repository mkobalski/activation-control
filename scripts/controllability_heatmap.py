#!/usr/bin/env python3
"""Per-token controllability heatmaps for a single sentence (CPU-only).

For ONE sentence (whose token positions align across all trials), compute the
controllability measures at every (analysis_layer, token_position) and render a
heatmap per measure: x = layers, y = tokens, averaged over the concepts. Reads
only saved data (results.pkl + cached concept vectors) -- NO GPU, NO model load.

This runs automatically at the end of scripts/run_experiment.py (via
generate_heatmaps), and can also be invoked standalone on any run directory.

READOUT (per trial, layer, token), selected by --metric:
  cos      direction channel: cos(v_C, r_t)                  (magnitude-invariant)
  relnorm  magnitude channel: ||r_t|| / (trial's content-token mean norm)
  norm     magnitude channel: raw ||r_t||                    (Cohen's d only)
  proj     cos(v_C, r_t) * ||r_t||
For cos/proj a specificity-corrected readout (on-concept minus mean off-concept)
gives the `_specific` panels. relnorm/norm are concept-agnostic (no specificity
correction; and for raw `norm` the unit-carrying `gain` is dropped -- only the
scale-invariant Cohen's d and the rank measure are cross-model comparable).

MEASURES at (layer, token):
  Rank        mean_C Spearman(intensity 1..4, readout), signed, in [-1,1]
              (raw + specific for cos; raw for the magnitude channel).
  gain        mean_C (readout@int4 - readout@int1)   (effect size; signed).
  Cohen's d   mean_C Delta / sd_C Delta              (standardized, unbounded).
  engage/suppress   two-panel figure: (think - neutral) and (dont - neutral),
              shared symmetric scale.

SIGNIFICANCE: permutation nulls -- design-only for Rank (two-sided, null centered
0); sign-flip per cell for gain / Cohen's d / engage-suppress -- then
Benjamini-Hochberg FDR across the map. Numbers are shown only where significant.

STYLING: signed measures use a diverging blue(neg)/white(0)/red(pos) map with
symmetric limits. Fractional differences are shown as raw Deltacos (cosine) or
x100 (relative-norm channel).
"""

import argparse
import itertools
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAT = "The cat watched the bird through the window by the door near the garden."
RAMP_DEFAULT = [f"think_intensity_{i}_of_4" for i in (1, 2, 3, 4)]


def classify(s):
    t = s.strip().lower()
    if t in ("the", "a", "and", ",", ".", "hello"):
        return t
    if s.startswith("<") and s.endswith(">"):
        return "special"
    return "content"


# ---- stats (rank-based) ------------------------------------------------------

def rankdata(a):
    """Average-rank (ties shared), 1-based. Mirrors scipy.stats.rankdata."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    sa = a[order]
    r = np.empty(len(a))
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rho = Pearson correlation of ranks; NaN if <3 points or degenerate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx, ry = rankdata(x) - rankdata(x).mean(), rankdata(y) - rankdata(y).mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return np.nan if d == 0 else float((rx * ry).sum() / d)


# ---- significance: Rank null (design-only, computed once, cached) ------------
# Under H0 (intensity has no effect), a concept's Spearman over the 4 levels is
# uniform over the 24 orderings of 4 items, so the cell (mean over n concepts) is
# the mean of n i.i.d. draws from that 24-value set -- independent of the actual
# values, so the null is the same for every full cell. The SIGNED mean is used
# (no max(0,.) clip), so the null is centered at 0 and the p-value is two-sided
# (a reliably negative average -- net anti-control -- is just as real).
_NPERM = 200_000
_signed24 = np.array([spearman([0, 1, 2, 3], list(p))
                      for p in itertools.permutations(range(4))])
_null_cache = {}


def rank_null(n, rng):
    key = ("rank", n)
    if key not in _null_cache and n >= 1:
        _null_cache[key] = np.sort(rng.choice(_signed24, size=(_NPERM, n)).mean(axis=1))
    return _null_cache.get(key)


def pval_two_sided(obs, null):
    if null is None or np.isnan(obs):
        return np.nan
    a = abs(obs)
    n_high = len(null) - int(np.searchsorted(null, a - 1e-12, side="left"))
    n_low = int(np.searchsorted(null, -a + 1e-12, side="right"))
    return (1 + n_high + n_low) / (len(null) + 1)


def bh_fdr(pmat):
    """Benjamini-Hochberg q-values over the non-nan cells of a 2-D matrix."""
    flat = pmat.ravel()
    idx = np.where(~np.isnan(flat))[0]
    q = np.full_like(flat, np.nan)
    if len(idx) == 0:
        return q.reshape(pmat.shape)
    p = flat[idx]
    order = np.argsort(p)
    m = len(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity
    qv = np.empty(m)
    qv[order] = np.clip(ranked, 0, 1)
    q[idx] = qv
    return q.reshape(pmat.shape)


# ---- data --------------------------------------------------------------------

def get_acts(trial, L):
    """(n_tok, d) float32 sentence-span activations at L (int/str key safe)."""
    d = trial.get("activations") or {}
    arr = d.get(L, d.get(str(L)))
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    return arr if arr.ndim == 2 and arr.shape[0] > 0 else None


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


def per_token_readout(acts, V, Vn, ci, metric, classes=None):
    """(n_tok,) (s_on, s_spec) for concept index ci.

    metric cos/proj : directional; returns (s_on, s_on - mean_off-concept).
    metric norm     : raw residual L2 norm (concept-agnostic; s_spec = NaN).
    metric relnorm  : norm / trial's content-token mean norm (concept-agnostic).
    """
    rn1 = np.linalg.norm(acts, axis=1)                     # (n_tok,)
    if metric in ("norm", "relnorm"):
        nan = np.full(rn1.shape, np.nan)
        if metric == "norm":
            return rn1, nan
        content = [i for i, c in enumerate(classes or []) if c == "content"]
        denom = rn1[content].mean() if content else np.nan
        return (rn1 / denom if denom == denom else nan), nan  # NaN denom -> all-NaN
    rn = rn1[:, None] + 1e-8
    cos = (acts / rn) @ Vn.T                                # (n_tok, n_concepts)
    if metric == "proj":
        cos = cos * rn
    s_on = cos[:, ci]
    s_off = (cos.sum(1) - s_on) / max(cos.shape[1] - 1, 1)
    return s_on, s_on - s_off


# ---- core: compute + render all heatmaps for one (run, metric, sentence) -----

def generate_heatmaps(run_dir, model_name, *, vector_cache="results/vector_cache",
                      method="baseline", metric="cos", sentence=CAT,
                      pos_cond="think_about", neg_cond="dont_think_about",
                      ramp=None, ramp_name="intensity_1to4",
                      baseline_cond="no_instruction", alpha=0.05,
                      out_dir=None, results=None, verbose=True):
    """Render the controllability heatmaps for one sentence + metric.

    Best-effort: returns [] (with a printed reason) instead of raising if the
    sentence isn't present, the vectors are missing, or the metric needs content
    tokens the sentence lacks -- so it is safe to call automatically after a run.
    `results` may be passed in-memory (the run's trial list) to skip re-loading.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ramp = list(ramp) if ramp else list(RAMP_DEFAULT)

    def _skip(msg):
        if verbose:
            print(f"  [skip heatmaps {metric}] {msg}")
        return []

    if results is None:
        with open(run_dir / "results.pkl", "rb") as f:
            results = pickle.load(f)["results"]

    sub = [r for r in results if r.get("is_compliant") and r["sentence"] == sentence
           and r.get("concept") is not None]
    if not sub:
        return _skip(f"no compliant trials for the sentence in {run_dir.name}")

    layers = sorted(int(x) for x in (sub[0].get("analysis_layers") or []))
    vecs = load_vectors(vector_cache, model_name, layers, method)
    layers = [L for L in layers if L in vecs]
    if not layers:
        return _skip("no usable concept-vector cache found")

    tok_strs = sub[0]["anchored_token_strs"][1:]
    n_tok = len(tok_strs)
    sub = [r for r in sub if len(r["anchored_token_strs"]) - 1 == n_tok]
    classes = [classify(s) for s in tok_strs]
    labels = [f"{s!r}".strip("'") for s in tok_strs]   # y-labels: bare token
    concept_agnostic = metric in ("norm", "relnorm")   # no off-concept correction
    if metric == "relnorm" and "content" not in classes:
        return _skip("metric 'relnorm' needs content tokens; this sentence has none")
    if verbose:
        print(f"heatmaps [{metric}] {run_dir.name}: {n_tok} tokens, layers {layers}, "
              f"{len(sub)} trials")

    # Index trials: data[L][cond][concept] = (s_on[n_tok], s_spec[n_tok]).
    wanted = set([pos_cond, neg_cond, *ramp])
    data = defaultdict(lambda: defaultdict(dict))
    for r in sub:
        cond, concept = r["condition_id"], r["concept"]
        if cond not in wanted:
            continue
        for L in layers:
            acts = get_acts(r, L)
            if acts is None or acts.shape[0] < n_tok:
                continue
            concepts_L, V, Vn = vecs[L]
            if concept not in concepts_L:
                continue
            ci = concepts_L.index(concept)
            data[L][cond][concept] = per_token_readout(acts[:n_tok], V, Vn, ci, metric, classes)

    concepts = vecs[layers[0]][0]
    rng = np.random.default_rng(0)
    shape = (n_tok, len(layers))
    B = 5000
    flavors = ["on"] if concept_agnostic else ["on", "spec"]
    # The ramp-based measures (Rank / gain / Cohen's d) need >=3 intensity levels
    # present; some configs (e.g. layer-targeting) omit the ramp -> skip those but
    # still try engagement/suppression, which needs only pos/neg/baseline.
    present = set().union(*(set(data[L].keys()) for L in data)) if data else set()
    do_ramp = sum(c in present for c in ramp) >= 3

    # ---------- Rank: mean signed Spearman over the intensity ramp ----------
    Rank = {fl: np.full(shape, np.nan) for fl in flavors}
    PRank = {fl: np.full(shape, np.nan) for fl in flavors}
    for li, L in enumerate(layers):
        for ti in range(n_tok):
            for fl in flavors:
                idx = 0 if fl == "on" else 1
                rhos = []
                for c in concepts:
                    levels, vals = [], []
                    for lev, cond in enumerate(ramp):
                        d = data[L].get(cond, {})
                        if c in d:
                            levels.append(lev); vals.append(d[c][idx][ti])
                    if len(levels) >= 3:
                        rho = spearman(levels, vals)
                        if not np.isnan(rho):
                            rhos.append(rho)
                if rhos:
                    obs = float(np.mean(rhos))
                    Rank[fl][ti, li] = obs
                    PRank[fl][ti, li] = pval_two_sided(obs, rank_null(len(rhos), rng))

    # ---------- gain (mean Delta) and Cohen's d (mean Delta / sd Delta) ----------
    Gain = np.full(shape, np.nan); Pgain = np.full(shape, np.nan)
    Cohen = np.full(shape, np.nan); Pcohen = np.full(shape, np.nan)
    for li, L in enumerate(layers):
        for ti in range(n_tok):
            rows = [[data[L][cond][c][0][ti] for cond in ramp]
                    for c in concepts if all(c in data[L].get(cond, {}) for cond in ramp)]
            rows = [r for r in rows if not any(np.isnan(r))]
            if len(rows) < 3:
                continue
            Vd = np.array(rows)                       # (m, R)
            delta = Vd[:, -1] - Vd[:, 0]              # per-concept gain
            gain = float(delta.mean())
            cohen = float(delta.mean() / (delta.std() + 1e-9))
            Gain[ti, li], Cohen[ti, li] = gain, cohen
            signs = rng.choice([-1.0, 1.0], size=(B, len(delta)))   # sign-flip null
            ng = (signs * delta).mean(1)
            nc = (signs * delta).mean(1) / ((signs * delta).std(1) + 1e-9)
            Pgain[ti, li] = (1 + int((np.abs(ng) >= abs(gain) - 1e-15).sum())) / (B + 1)
            Pcohen[ti, li] = (1 + int((np.abs(nc) >= abs(cohen) - 1e-15).sum())) / (B + 1)

    # ---------- rendering ----------
    xt = [str(L) for L in layers]
    paths, csv_rows = [], []
    COL_W = 0.42                               # fixed narrow column width (inches)
    FIG_W = 3.0 + COL_W * len(layers)
    FIG_H = 0.34 * n_tok + 2.2
    # cosine differences are raw (no positive baseline to take a % of); the
    # relative-norm channel is ~1, so x100 reads loosely as percent-of-baseline.
    DSCALE, DDEC, DTAG = (1.0, 3, "") if metric == "cos" else (100.0, 1, "×100")
    tag = lambda c: c.replace("think_intensity_", "int")

    def render(key, M, Pm, title, *, scale=1.0, decimals=2, clabel=""):
        """One diverging blue/red heatmap; numbers only in significant cells."""
        Q = bh_fdr(Pm)
        n_sig = int(np.nansum(Q < alpha)); n_tested = int(np.sum(~np.isnan(Pm)))
        Md = M * scale
        A = np.nanmax(np.abs(Md)) if np.isfinite(np.nanmax(np.abs(Md))) else 1.0
        A = A or 1.0
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        im = ax.imshow(Md, aspect="auto", cmap="RdBu_r", vmin=-A, vmax=A)
        ax.set_xticks(range(len(layers))); ax.set_xticklabels(xt, fontsize=8)
        ax.set_yticks(range(n_tok)); ax.set_yticklabels(labels, fontsize=9, family="monospace")
        ax.set_xlabel("analysis layer"); ax.set_ylabel("token")
        ax.set_title(title + f"\n[{metric}, avg over {len(concepts)} concepts]  ·  "
                     f"{n_sig}/{n_tested} sig (BH q<{alpha})", fontsize=10)
        for ti in range(n_tok):
            for li in range(len(layers)):
                v, q = M[ti, li], Q[ti, li]
                if np.isnan(v):
                    continue
                sig = (not np.isnan(q)) and q < alpha
                csv_rows.append((key, labels[ti], classify(tok_strs[ti]),
                                 layers[li], v, Pm[ti, li], q, int(sig)))
                if not sig:
                    continue
                d = Md[ti, li]
                ax.text(li, ti, f"{d:.{decimals}f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(d) / A > 0.55 else "black")
        fig.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.02)
        ax.text(0.0, -0.06, "numbers shown only where significant (BH-FDR q<%.2g)" % alpha,
                transform=ax.transAxes, fontsize=8, color="#444")
        fig.tight_layout()
        p = out_dir / f"heatmap_{key}_{metric}_sig.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        paths.append(str(p))
        if verbose:
            print(f"  wrote {p.name}  ({n_sig}/{n_tested} sig)")

    if do_ramp:
        # Rank (signed Spearman, [-1,1])
        render("Rank_raw", Rank["on"], PRank["on"],
               f"Rank  (mean signed Spearman over {ramp_name}, on-concept)",
               scale=1.0, decimals=2, clabel="Rank (signed Spearman) [-1,1]")
        if "spec" in flavors:
            render("Rank_specific", Rank["spec"], PRank["spec"],
                   "Rank  (off-concept corrected)",
                   scale=1.0, decimals=2, clabel="Rank (signed Spearman) [-1,1]")
        # gain (only where cross-model comparable: dimensionless readouts)
        if metric in ("cos", "relnorm"):
            render("gain", Gain, Pgain,
                   f"Ramp GAIN: Δ{metric}{DTAG} ({tag(ramp[-1])} − {tag(ramp[0])})",
                   scale=DSCALE, decimals=DDEC, clabel=f"Δ{metric} {DTAG}".strip())
        elif verbose:
            print(f"  [skip gain] raw-unit metric '{metric}' is not cross-model comparable")
        # Cohen's d (standardized ramp gain; scale-invariant)
        render("cohensd", Cohen, Pcohen,
               "Cohen's d  (standardized ramp gain = meanΔ / sdΔ)",
               scale=1.0, decimals=2, clabel="Cohen's d (signed)")
    elif verbose:
        print("  [skip Rank/gain/Cohen's d] fewer than 3 intensity-ramp conditions present")

    # ---------- Engagement vs Suppression (two-panel, condition - neutral) ----------
    base_trials = [r for r in results if r.get("is_compliant")
                   and r["sentence"] == sentence and r["condition_id"] == baseline_cond
                   and len(r.get("anchored_token_strs", [])) - 1 == n_tok]
    if base_trials:
        bt = base_trials[0]
        baseline = {}
        for L in layers:
            acts = get_acts(bt, L)
            if acts is None or acts.shape[0] < n_tok:
                continue
            concepts_L, V, Vn = vecs[L]
            baseline[L] = {c: per_token_readout(acts[:n_tok], V, Vn, ci, metric, classes)[0]
                           for ci, c in enumerate(concepts_L)}

        def _delta_panel(cond):
            # (condition - neutral baseline), averaged over concepts; sign-flip null.
            M = np.full(shape, np.nan); P = np.full(shape, np.nan)
            for li, L in enumerate(layers):
                if L not in baseline:
                    continue
                cd = data[L].get(cond, {})
                for ti in range(n_tok):
                    vals = [cd[c][0][ti] - baseline[L][c][ti]
                            for c in concepts if c in cd and c in baseline[L]]
                    vals = [v for v in vals if not np.isnan(v)]
                    if len(vals) < 3:
                        continue
                    dv = np.array(vals); obs = float(dv.mean()); M[ti, li] = obs
                    signs = rng.choice([-1.0, 1.0], size=(B, len(dv)))
                    null = (signs * dv).mean(1)
                    P[ti, li] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (B + 1)
            return M, bh_fdr(P)

        Eng, Qe = _delta_panel(pos_cond)     # think - neutral   (red = toward concept)
        Sup, Qs = _delta_panel(neg_cond)     # dont  - neutral   (blue = below neutral)
        both = np.concatenate([Eng[~np.isnan(Eng)], Sup[~np.isnan(Sup)]])
        if both.size:
            A = (float(np.nanmax(np.abs(both))) or 0.01) * DSCALE
            fig, axes = plt.subplots(1, 2, figsize=(2 * (COL_W * len(layers)) + 4.0,
                                                    FIG_H), sharey=True)
            pnl = [(axes[0], Eng, Qe, f"ENGAGEMENT\n{pos_cond} − {baseline_cond}"),
                   (axes[1], Sup, Qs, f"SUPPRESSION\n{neg_cond} − {baseline_cond}")]
            im = None
            for ax, M, Q, title in pnl:
                im = ax.imshow(M * DSCALE, aspect="auto", cmap="RdBu_r", vmin=-A, vmax=A)
                ax.set_xticks(range(len(layers))); ax.set_xticklabels(xt, fontsize=8)
                ax.set_xlabel("analysis layer")
                n_sig = int(np.nansum(Q < alpha)); n_tested = int(np.sum(~np.isnan(Q)))
                ax.set_title(f"{title}\n{n_sig}/{n_tested} sig (BH q<{alpha})", fontsize=10)
                for ti in range(n_tok):
                    for li in range(len(layers)):
                        v, q = M[ti, li], Q[ti, li]
                        if np.isnan(v) or np.isnan(q) or q >= alpha:
                            continue
                        d = v * DSCALE
                        ax.text(li, ti, f"{d:.{DDEC}f}", ha="center", va="center",
                                fontsize=7, color="white" if abs(d) / A > 0.55 else "black")
            axes[0].set_yticks(range(n_tok))
            axes[0].set_yticklabels(labels, fontsize=9, family="monospace")
            axes[0].set_ylabel("token")
            fig.suptitle(f"Engagement vs Suppression  [{metric}{DTAG}, avg over "
                         f"{len(concepts)} concepts]  (shared symmetric scale)", fontsize=11)
            fig.colorbar(im, ax=axes, label=f"Δ{metric} {DTAG} from neutral".strip(),
                         fraction=0.046)
            fig.text(0.5, 0.005, "numbers shown only where significant (BH-FDR q<%.2g)" % alpha,
                     ha="center", fontsize=8, color="#444")
            pe = out_dir / f"heatmap_engage_suppress_{metric}.png"
            fig.savefig(pe, dpi=140, bbox_inches="tight"); plt.close(fig)
            paths.append(str(pe))
            if verbose:
                print(f"  wrote {pe.name}")
    elif verbose:
        print(f"  [warn] no '{baseline_cond}' trial for this sentence; "
              f"skipping engagement/suppression")

    # ---------- CSV ----------
    csv_path = out_dir / f"controllability_heatmap_{metric}.csv"
    with open(csv_path, "w") as f:
        f.write("measure,token,class,layer,value,p,q,significant\n")
        for r in csv_rows:
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")
    paths.append(str(csv_path))
    if verbose:
        print(f"  wrote {csv_path.name}")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Controllability heatmaps for one sentence.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-name", default="gemma3_27b")
    ap.add_argument("--vector-cache", default="results/vector_cache")
    ap.add_argument("--vector-method", default="baseline")
    ap.add_argument("--sentence", default=CAT)
    ap.add_argument("--metric", choices=["cos", "proj", "norm", "relnorm"], default="cos")
    ap.add_argument("--pos-cond", default="think_about")
    ap.add_argument("--neg-cond", default="dont_think_about")
    ap.add_argument("--ramp", default=",".join(RAMP_DEFAULT),
                    help="ordered (weakest->strongest) condition ids for Rank/gain")
    ap.add_argument("--ramp-name", default="intensity_1to4")
    ap.add_argument("--baseline-cond", default="no_instruction")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH-FDR q threshold")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    generate_heatmaps(
        args.run_dir, args.model_name, vector_cache=args.vector_cache,
        method=args.vector_method, metric=args.metric, sentence=args.sentence,
        pos_cond=args.pos_cond, neg_cond=args.neg_cond,
        ramp=args.ramp.split(","), ramp_name=args.ramp_name,
        baseline_cond=args.baseline_cond, alpha=args.alpha, out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
