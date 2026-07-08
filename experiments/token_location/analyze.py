"""Custom analysis for the token-location experiment: a targeting diagonal.

Registered kind: ``location_targeting``.

Mirrors the layer-targeting diagonal (scripts/plot_layer_targeting.py), but for
token REGIONS instead of layers. Rows = the instructed region (from the
loc_* condition), cols = the measured region; each cell is the mean
specificity-corrected concept-cosine in that region, minus the same quantity
under the neutral baseline. If location control works, the diagonal
(instructed == measured) should be the brightest cell in each row.

Regions (independent masks over the sentence-span tokens):
  punctuation  token class in {",", "."}
  content      token class == "content"
  beginning    normalized position < begin_frac        (default first third)
  end          normalized position >= 1 - end_frac      (default last third)

CPU-only: reads the run's per-trial activations + the cached concept vectors,
exactly like the other analysis scripts.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.registry import register
# These live in scripts/, which run_experiment.py has put on sys.path before
# importing this module (see run_experiment._import_experiment_analyze).
from controllability_heatmap import load_vectors, per_token_readout, classify, get_acts


# Instructed region for each loc_* condition -> the diagonal cell of its row.
# loc_adjectives targets adjectives, a subset of content words; classify() has no
# adjective class, so its measured region is "content" (the diagonal cell).
DEFAULT_COND_REGION = {
    "loc_punctuation": "punctuation",
    "loc_adjectives": "content",
    "loc_beginning": "beginning",
    "loc_end": "end",
}
REGIONS = ["punctuation", "content", "beginning", "end"]


def _labels(trial, n):
    """Token classes for the sentence-span positions (anchor at [0] dropped)."""
    strs = trial.get("anchored_token_strs") or []
    return [classify(s) for s in strs[1:]][:n]


def _region_mask(region, classes, pos, begin_frac, end_frac):
    """Boolean mask over tokens: which belong to `region`."""
    classes = np.asarray(classes, dtype=object)
    if region == "punctuation":
        return np.array([c in (",", ".") for c in classes], dtype=bool)
    if region == "content":
        return classes == "content"
    if region == "beginning":
        return pos < begin_frac
    if region == "end":
        return pos >= 1.0 - end_frac
    raise ValueError(f"unknown region {region}")


def _region_means(acts, classes, ci, Vn, begin_frac, end_frac):
    """Mean specificity-corrected cosine within each region for one trial+concept.

    Returns {region: mean spec-cosine or NaN if the region has no tokens}.
    """
    n = min(len(acts), len(classes))
    if n == 0:
        return {r: np.nan for r in REGIONS}
    _, s_spec = per_token_readout(acts[:n], None, Vn, ci, "cos")
    pos = (np.arange(n) + 0.5) / n
    out = {}
    for r in REGIONS:
        m = _region_mask(r, classes[:n], pos, begin_frac, end_frac)
        out[r] = float(s_spec[m].mean()) if m.any() else np.nan
    return out


@register("location_targeting")
def location_targeting(*, run_dir, results, model_name, cfg,
                       conditions=None, baseline_cond="no_instruction",
                       sentence=None, layer=None,
                       begin_frac=0.34, end_frac=0.34, **_):
    """Build the instructed-region x measured-region Δ heatmap.

    conditions   optional {condition_id: intended_region}; default DEFAULT_COND_REGION.
    sentence     optional single sentence to restrict to; default = all sentences.
    layer        optional analysis layer; default = deepest recorded layer.
    begin/end_frac  fraction of the (normalized) span counted as beginning/end.
    """
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_region = dict(conditions) if conditions else dict(DEFAULT_COND_REGION)
    cond_ids = list(cond_region.keys())

    all_layers = sorted(int(x) for x in (results[0].get("analysis_layers") or []))
    if not all_layers:
        print("  [location_targeting] no analysis layers in results; skipping")
        return
    L = int(layer) if layer is not None else all_layers[-1]

    vecs = load_vectors(cfg.concept_vectors.cache_dir, model_name, [L],
                        cfg.concept_vectors.method)
    if L not in vecs:
        print(f"  [location_targeting] no concept-vector cache for layer {L}; skipping")
        return
    concepts_L, _, Vn = vecs[L]
    ci_of = {c: i for i, c in enumerate(concepts_L)}

    def _keep(r):
        return (r.get("is_compliant") and r.get("concept") is not None
                and (sentence is None or r["sentence"] == sentence))

    # Baseline (neutral) region means per (sentence, concept): project the
    # concept-less no_instruction residual onto each concept vector, same as the
    # engagement/suppression panels do. Keyed by sentence -> {concept: {region: mean}}.
    base_by_sentence = {}
    for r in results:
        if (not r.get("is_compliant") or r["condition_id"] != baseline_cond
                or (sentence is not None and r["sentence"] != sentence)):
            continue
        s = r["sentence"]
        if s in base_by_sentence:
            continue  # one baseline trial per sentence is enough
        acts = get_acts(r, L)
        if acts is None:
            continue
        classes = _labels(r, len(acts))
        base_by_sentence[s] = {
            c: _region_means(acts, classes, ci_of[c], Vn, begin_frac, end_frac)
            for c in concepts_L
        }

    # Accumulate Δ(condition - baseline) region means over trials/concepts.
    acc = {cid: {r: [] for r in REGIONS} for cid in cond_ids}
    for r in results:
        cid = r.get("condition_id")
        if cid not in cond_region or not _keep(r):
            continue
        c = r["concept"]
        if c not in ci_of:
            continue
        acts = get_acts(r, L)
        if acts is None:
            continue
        base = base_by_sentence.get(r["sentence"], {}).get(c)
        if base is None:
            continue
        cond_means = _region_means(acts, _labels(r, len(acts)), ci_of[c], Vn,
                                   begin_frac, end_frac)
        for reg in REGIONS:
            if not (np.isnan(cond_means[reg]) or np.isnan(base[reg])):
                acc[cid][reg].append(cond_means[reg] - base[reg])

    mat = np.full((len(cond_ids), len(REGIONS)), np.nan, dtype=np.float32)
    for i, cid in enumerate(cond_ids):
        for j, reg in enumerate(REGIONS):
            vals = acc[cid][reg]
            if vals:
                mat[i, j] = float(np.mean(vals))
    if np.all(np.isnan(mat)):
        print("  [location_targeting] no data (check conditions / cache); skipping")
        return

    # ---- render ----
    vmax = float(np.nanmax(np.abs(mat))) or 1.0
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(REGIONS),
                                    1.4 + 0.8 * len(cond_ids)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS)
    ax.set_yticks(range(len(cond_ids))); ax.set_yticklabels(cond_ids)
    ax.set_xlabel("measured region")
    ax.set_ylabel("instructed condition")
    for i, cid in enumerate(cond_ids):
        for j, reg in enumerate(REGIONS):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 * vmax else "black")
            if cond_region.get(cid) == reg:  # diagonal / targeted cell
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=2.0))
    fig.colorbar(im, ax=ax, label="Δ spec-cosine vs neutral", fraction=0.046, pad=0.04)
    ax.set_title(f"Token-location control  [layer {L}, spec-cos]\n"
                 f"boxed = instructed region (diagonal)", fontsize=10)
    fig.tight_layout()
    p = out_dir / f"location_targeting_L{L}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p.name}")

    csv = out_dir / f"location_targeting_L{L}.csv"
    with open(csv, "w") as f:
        f.write("condition,intended_region,measured_region,delta_spec_cos,n\n")
        for cid in cond_ids:
            for reg in REGIONS:
                vals = acc[cid][reg]
                v = f"{np.mean(vals):.6f}" if vals else ""
                f.write(f"{cid},{cond_region.get(cid, '')},{reg},{v},{len(vals)}\n")
    print(f"  wrote {csv.name}")
