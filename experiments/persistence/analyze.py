"""Custom analysis for the persistence experiment: temporal concept profiles.

Registered kind: ``temporal_profile``.

For each persist_* condition we take every compliant, concept-bearing trial,
read its per-token specificity-corrected concept-cosine at the deep layer,
resample the trace onto a common normalized-position grid ([0,1], `nbins`
points), subtract the neutral-baseline profile for the same (sentence, concept),
and average over trials + concepts. The result is a Δ-cosine profile vs position.

Each condition is scored against the temporal shape it was TOLD to produce
(`_TEMPLATES`): Pearson correlation between the Δ profile and the template.
The templates are heuristic (see each entry) -- a positive score means the
observed profile rises/falls where the instruction said it should.

CPU-only: reads the run's per-trial activations + the cached concept vectors.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.registry import register
from controllability_heatmap import load_vectors, per_token_readout, get_acts


DEFAULT_CONDITIONS = ["persist_first_half", "persist_throughout",
                      "persist_once", "persist_after_fourth"]

# Instructed temporal shape as a function of normalized position x in [0,1].
# Heuristic references for the match score; documented per entry.
_TEMPLATES = {
    # engage in the first half, drop after
    "persist_first_half": lambda x: (x < 0.5).astype(float),
    # flat/high the entire time (constant -> match score is NaN; judged by mean level)
    "persist_throughout": lambda x: np.ones_like(x),
    # a single burst mid-sentence then off (bump centered at x=0.5)
    "persist_once": lambda x: (np.abs(x - 0.5) < 0.15).astype(float),
    # off until ~the fourth word, then on (nominal threshold at x>=0.30)
    "persist_after_fourth": lambda x: (x >= 0.30).astype(float),
}


def _resample(acts, ci, Vn, xs):
    """Spec-corrected cosine trace resampled onto grid `xs`, or None if empty."""
    n = len(acts)
    if n == 0:
        return None
    _, s_spec = per_token_readout(acts, None, Vn, ci, "cos")
    if n == 1:
        return np.full_like(xs, float(s_spec[0]))
    pos = (np.arange(n) + 0.5) / n
    return np.interp(xs, pos, s_spec)


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


@register("temporal_profile")
def temporal_profile(*, run_dir, results, model_name, cfg,
                     conditions=None, baseline_cond="no_instruction",
                     sentence=None, layer=None, nbins=12, **_):
    """Δ concept-cosine vs normalized position per condition + template match."""
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_ids = list(conditions) if conditions else list(DEFAULT_CONDITIONS)
    xs = (np.arange(nbins) + 0.5) / nbins

    all_layers = sorted(int(x) for x in (results[0].get("analysis_layers") or []))
    if not all_layers:
        print("  [temporal_profile] no analysis layers in results; skipping")
        return
    L = int(layer) if layer is not None else all_layers[-1]

    vecs = load_vectors(cfg.concept_vectors.cache_dir, model_name, [L],
                        cfg.concept_vectors.method)
    if L not in vecs:
        print(f"  [temporal_profile] no concept-vector cache for layer {L}; skipping")
        return
    concepts_L, _, Vn = vecs[L]
    ci_of = {c: i for i, c in enumerate(concepts_L)}

    def _keep(r):
        return (r.get("is_compliant") and r.get("concept") is not None
                and (sentence is None or r["sentence"] == sentence))

    # Baseline resampled profile per (sentence, concept).
    base = {}
    for r in results:
        if (not r.get("is_compliant") or r["condition_id"] != baseline_cond
                or (sentence is not None and r["sentence"] != sentence)):
            continue
        s = r["sentence"]
        if s in base:
            continue
        acts = get_acts(r, L)
        if acts is None:
            continue
        base[s] = {c: _resample(acts, ci_of[c], Vn, xs) for c in concepts_L}

    # Accumulate Δ profiles per condition.
    acc = {cid: [] for cid in cond_ids}
    for r in results:
        cid = r.get("condition_id")
        if cid not in acc or not _keep(r):
            continue
        c = r["concept"]
        if c not in ci_of:
            continue
        acts = get_acts(r, L)
        if acts is None:
            continue
        b = base.get(r["sentence"], {}).get(c)
        cond = _resample(acts, ci_of[c], Vn, xs)
        if b is None or cond is None:
            continue
        acc[cid].append(cond - b)

    profiles = {cid: (np.nanmean(np.vstack(v), axis=0) if v else None)
                for cid, v in acc.items()}
    if all(p is None for p in profiles.values()):
        print("  [temporal_profile] no data (check conditions / cache); skipping")
        return

    # ---- plot + match scores ----
    fig, ax = plt.subplots(figsize=(8, 4.6))
    reds = plt.get_cmap("tab10")
    scores = {}
    for k, cid in enumerate(cond_ids):
        prof = profiles[cid]
        if prof is None:
            continue
        tmpl = _TEMPLATES.get(cid, lambda x: np.zeros_like(x))(xs)
        score = _pearson(prof, tmpl)
        scores[cid] = score
        lbl = f"{cid}  (match={score:+.2f}, n={len(acc[cid])})" \
              if score == score else f"{cid}  (n={len(acc[cid])})"
        ax.plot(xs, prof, marker="o", markersize=3, linewidth=1.6,
                color=reds(k % 10), label=lbl)
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xlabel("normalized token position (0 = start, 1 = end)")
    ax.set_ylabel("Δ spec-cosine vs neutral")
    ax.set_title(f"Persistence / temporal control  [layer {L}]\n"
                 f"match = Pearson corr of Δ profile with instructed template",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = out_dir / f"temporal_profile_L{L}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p.name}")

    csv = out_dir / f"temporal_profile_L{L}.csv"
    with open(csv, "w") as f:
        f.write("condition,bin_center,delta_spec_cos\n")
        for cid in cond_ids:
            prof = profiles[cid]
            if prof is None:
                continue
            for xi, v in zip(xs, prof):
                f.write(f"{cid},{xi:.4f},{v:.6f}\n")
        f.write("# match scores (Pearson corr with instructed template):\n")
        for cid, sc in scores.items():
            f.write(f"# {cid},{sc:.4f}\n")
    print(f"  wrote {csv.name}")
