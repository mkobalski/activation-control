#!/usr/bin/env python3
"""Runtime EXPLORATORY figures — a quick per-model capability glance.

Rendered on demand into a run's own folder (results/raw/<run>/figures/) by the
direct post-run hook in run_experiment.py. Projection channel throughout, except
`engage_heatmap` which shows all three channels side by side.

Self-contained: builds only on the standard scoring layer (score_data,
compute_scores) -- NO dependency on the retired figure/plotting scripts.

Figures (main runs only; LT runs get none):
  raw_trace_example    per-token projection across conditions (s23 / Bread)
  engage_suppress      engage & suppress d' vs depth
  intensity_gain       endpoint-gain d' vs depth (lexical vs numeric)
  intensity_rank       signed Spearman rho vs depth (lexical vs numeric)
  pos_coverage         engage & suppress d' by POS category
  temporal_control concept concentration at sentence beginning / mid / end
  temporal_precision first-half / after-4th-word / throughout vs generic think
  engage_heatmap       cos / relnorm / projection engagement, tokens x layers

Usage:
  python scripts/explore.py --run-dir results/raw/<RUN>
  python scripts/explore.py --run-dir <RUN> --out <DIR> --only engage_suppress,intensity_rank
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import figstyle as fs                                                     # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402
import score_data as sd                                                   # noqa: E402
import compute_scores as cs                                               # noqa: E402

_PHI_INV = NormalDist().inv_cdf
FOCAL_LABEL, FOCAL_CONCEPT = "s23", "Bread"
# Minimum align_sentence_span similarity for a trial to supply the per-token axis
# labels. 1.0 is an exact transcription; the botched spans seen in practice score
# ~0.77, so this only admits near-verbatim ones. See focal_data.
_AXIS_MIN_ALIGN = 0.98
RAMP = sd.RAMP                                    # think_intensity_{1..4}_of_4
POS, NEG, BASE = sd.POS, sd.NEG, sd.BASE          # think_about / dont_think_about / no_instruction
LEX_HI = "think_intensely"                        # lexical high endpoint
VC = str(PROJECT_ROOT / "results/vector_cache")


# ======================================================================================
# shared data helpers
# ======================================================================================

def resolve_sentence(label, *, files=("sentences.txt", "extra_sentences.txt")):
    """Sentence TEXT for a stable label like 's23', looked up by its `s23:` prefix
    (robust to the file being split/reordered -- never an index into the file)."""
    for fn in files:
        p = PROJECT_ROOT / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip().startswith(f"{label}:"):
                return line.split(":", 1)[1].strip()
    return None


def _analysis_layers(sub):
    r = next((r for r in sub if r.get("cosine_sim")), None)
    return sorted(int(k) for k in r["cosine_sim"]) if r else []


def _proj_tokens(row, L, n):
    """Per-token projection = cosine_sim * raw ||r||, fitted to n tokens."""
    tc, tn = sd.trace(row, "cosine_sim", L), sd.trace(row, "norms", L)
    if tc is None or tn is None:
        return None
    return sd._fit_len(np.asarray(tc, np.float32)[:n] * np.asarray(tn, np.float32)[:n], n)


def focal_data(run_dir, label=FOCAL_LABEL, concept=FOCAL_CONCEPT):
    """Everything the per-token figures (raw_trace_example, engage_heatmap) need for
    one sentence x concept: token labels, analysis layers, per-condition per-token
    {cos,relnorm,proj}[L], and the no_instruction baseline per token per layer."""
    sent = resolve_sentence(label)
    if sent is None:
        return None
    rows = sd.load_rows(run_dir)
    sub = [r for r in rows if r["sentence"] == sent]
    # Token-axis labels: these label per-token measurements taken from the tokens the
    # model ACTUALLY generated, so they must come from a trial whose generation is a
    # faithful transcription -- otherwise the axis shows the model's own interjections
    # (a leaked concept word, a turn-end special token) instead of the sentence.
    #
    # Two traps this avoids, both previously live:
    #  1. `next(r for r in sub ...)` took the FIRST row for this sentence regardless of
    #     concept, so a Lightning trial could label a Bread figure.
    #  2. Nothing filtered is_compliant / alignment_similarity (unlike score_data.py and
    #     compute_scores.py, which both drop non-compliant rows), so a botched
    #     transcription could supply the labels -- e.g. a span that slid off and read
    #     [' crowded', ..., ' **', 'mol', 'ten', '**', ' back'] after the model injected
    #     "**molten**" into the sentence while thinking about Volcanoes.
    # Prefer the plotted concept; require compliance and a near-exact alignment.
    def _label_row(rs):
        ok = [r for r in rs if r.get("anchored_token_strs") and r.get("is_compliant")
              and r.get("alignment_similarity", 0.0) >= _AXIS_MIN_ALIGN]
        ok.sort(key=lambda r: -r.get("alignment_similarity", 0.0))
        return ok[0]["anchored_token_strs"] if ok else None

    toks_row = _label_row([r for r in sub if r.get("concept") == concept]) or _label_row(sub)
    if toks_row is None:
        # Better to emit nothing than a plausible-looking, silently mislabeled figure.
        print(f"[focal_data] no compliant, well-aligned trial for {label}/{concept} "
              f"(align >= {_AXIS_MIN_ALIGN}); skipping per-token figures")
    ent = sd.load_baseline(run_dir).get(sent)
    if not sub or toks_row is None or ent is None:
        return None
    toks = toks_row[1:]
    n = len(toks)
    classes = [sd.classify(t) for t in toks]
    layers = _analysis_layers(sub)
    vecs = sd.load_vectors(VC, sd._resolve_model(run_dir, None), layers)

    base = {}   # L -> dict(cos,norm,relnorm,proj) per token, for `concept`
    for L in layers:
        A = np.asarray(ent["activations"][L], np.float32)[:n]
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        concepts_L, _, Vn = vecs[L]
        if concept not in concepts_L:
            return None
        bc = sd._fit_len((An @ Vn.T)[:, concepts_L.index(concept)], n)
        bn = sd._fit_len(np.asarray(ent["norms"][L], np.float32)[:n], n)
        brl = sd._fit_len(sd.relnorm(bn, classes), n)
        base[L] = dict(cos=bc, norm=bn, relnorm=brl, proj=bc * bn)

    byc = {r["condition_id"]: r for r in sub if r.get("concept") == concept}

    def cond_tokens(cond):
        r = byc.get(cond)
        if r is None:
            return None
        out = {}
        for L in layers:
            tc, tn = sd.trace(r, "cosine_sim", L), sd.trace(r, "norms", L)
            if tc is None or tn is None:
                continue
            cvec = sd._fit_len(np.asarray(tc, np.float32)[:n], n)
            nvec = sd._fit_len(np.asarray(tn, np.float32)[:n], n)
            out[L] = dict(cos=cvec, norm=nvec, relnorm=sd._fit_len(sd.relnorm(nvec, classes), n),
                          proj=cvec * nvec)
        return out

    return dict(sent=sent, concept=concept, toks=toks, n=n, layers=layers,
                base=base, cond=cond_tokens, present=set(byc))


def _unit_table(run_dir, conds):
    """{cond: (n_units x n_L) proj array}, plus (layers, sids, cids), over the units
    (concept, sentence) present in ALL `conds`. Token-mean readouts from score_data."""
    layers, vals, _ = sd.unit_layer_readouts(run_dir, conds)
    keys = None
    for cond in conds:
        present = {(c, s) for c in vals[("proj", cond)] for s in vals[("proj", cond)][c]}
        keys = present if keys is None else (keys & present)
    keys = sorted(keys)
    tab = {cond: np.vstack([vals[("proj", cond)][c][s] for (c, s) in keys]) for cond in conds}
    sids = np.array([s for (c, s) in keys])
    cids = np.array([c for (c, s) in keys])
    return tab, layers, sids, cids


def _cluster_band(U, sids, cids, n_boot=1000, seed=0):
    """95% band for a mean-over-units curve, JOINT TWO-WAY cluster bootstrap over
    both sentences and concepts -- the same resampling scheme scalar_ci.py uses for
    the headline scalar, so a figure's band and the number it illustrates answer the
    same question.

    Why two-way. A "unit" is one (sentence, concept) pair, so with 50 sentences x 10
    concepts there are 500 rows -- but they are not 500 independent draws. The 10
    concepts within a sentence share its baseline and token positions, and the 50
    sentences within a concept share that concept's vector. Treating the 500 as
    exchangeable (a plain SEM, or a bootstrap over rows) understates the spread: on
    gemma3-27b, measured against this function, by ~1.4-2.4x if you ignore only the
    sentence grouping and up to ~4.4x if you ignore both. Concept is the LARGER
    cluster here and its share grows toward the end of the sentence, which is exactly
    where the persistence/positional effects are read off.

    Both axes are resampled with multinomial multiplicities (as in scalar_ci's Wsent /
    Mconc) rather than by index concatenation, and a unit is weighted by the PRODUCT
    of its sentence and concept multiplicity. One resample is shared across all
    columns of U, so the band stays coherent along the curve instead of jittering
    independently per point.
    """
    U = np.asarray(U, float)
    rng = np.random.default_rng(seed)
    _, inv_s = np.unique(np.asarray(sids), return_inverse=True)
    _, inv_c = np.unique(np.asarray(cids), return_inverse=True)
    n_s, n_c = inv_s.max() + 1, inv_c.max() + 1
    # Multiplicity of each sentence / concept in each replicate; expected count 1.
    Wsent = rng.multinomial(n_s, np.full(n_s, 1 / n_s), size=n_boot).astype(float)
    Mconc = rng.multinomial(n_c, np.full(n_c, 1 / n_c), size=n_boot).astype(float)
    W = Wsent[:, inv_s] * Mconc[:, inv_c]              # (n_boot, n_units)
    reps = np.full((n_boot, U.shape[1]), np.nan)
    for j in range(U.shape[1]):
        col = U[:, j]
        m = np.isfinite(col)
        if not m.any():
            continue
        Wm = W[:, m]
        den = Wm.sum(axis=1)
        num = Wm @ col[m]
        with np.errstate(invalid="ignore", divide="ignore"):
            reps[:, j] = np.where(den > 0, num / den, np.nan)
    return np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)


def _d_from_auroc(a):
    return float(np.sqrt(2) * _PHI_INV(min(max(a, 1e-6), 1 - 1e-6)))


# ======================================================================================
# 1. raw_trace_example
# ======================================================================================

def raw_trace_example(run_dir, out):
    d = focal_data(run_dir)
    if d is None:
        print("  [raw_trace_example] focal sentence/concept unavailable; skipped")
        return None
    L = sd._layer_for_fraction(run_dir, sd.PROJ_F_POS)
    if L not in d["layers"]:
        L = min(d["layers"], key=lambda x: abs(x - L))
    xs = np.arange(d["n"])
    labels = [t.strip() or "␣" for t in d["toks"]]
    depth = int(round(sd.PROJ_F_POS * 100))         # requested fraction, not back-computed L/n
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)

    def line(ax, cond, color, lab, **kw):
        ct = d["cond"](cond)
        if ct and L in ct:
            ax.plot(xs, ct[L]["proj"], color=color, marker="o", ms=3, lw=1.4, label=lab, **kw)

    axa.plot(xs, d["base"][L]["proj"], color=fs.LIGHT_GRAY, ls="--", lw=1.3, label="no instruction")
    line(axa, POS, fs.MED_GRAY, "think about")
    line(axa, NEG, fs.LIGHT_GRAY, "don't think about")
    line(axa, LEX_HI, "#7b1f1f", "think intensely")
    axa.set_title(f"engagement + lexical intensity  (L{L}, {depth}% depth)")

    axb.plot(xs, d["base"][L]["proj"], color=fs.LIGHT_GRAY, ls="--", lw=1.3, label="no instruction")
    line(axb, NEG, fs.LIGHT_GRAY, "don't think about")
    for i, cond in enumerate(RAMP):
        line(axb, cond, fs.REDS(0.35 + 0.6 * i / 3), f"intensity {i + 1}/4")
    axb.set_title(f"numeric intensity ramp  (L{L}, {depth}% depth)")

    for ax in (axa, axb):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_xlabel("sentence token")
        ax.legend(frameon=False)
    axa.set_ylabel(r"projection  $\langle r,\hat c\rangle$")
    fig.suptitle(f"raw_trace_example — {d['concept']} / {FOCAL_LABEL}", fontweight="bold")
    fig.tight_layout()
    return fs.save(fig, out)


# ======================================================================================
# 2. engage_suppress — d' vs depth
# ======================================================================================

def engage_suppress(run_dir, out):
    layers, vals, bases = sd.unit_layer_readouts(run_dir, [POS, NEG])
    order = sorted(bases["proj"])
    pcts = fs.depth_pcts(layers, sd.run_n_layers(run_dir))
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cond, sign, color, lab in ((POS, +1.0, fs.ENGAGE_C, "engage"),
                                   (NEG, -1.0, fs.SUPPRESS_C, "suppress")):
        blocks, S = cs.per_concept_blocks(vals[("proj", cond)], bases["proj"], order)
        st = cs.dprime_stats(blocks, S, len(layers), rng, n_perm=0)
        dp = sign * st["dp"]
        lo = np.nanpercentile(sign * st["bavg"], 2.5, axis=0)
        hi = np.nanpercentile(sign * st["bavg"], 97.5, axis=0)
        ax.plot(pcts, dp, color=color, marker="o", ms=3, lw=1.6, label=lab)
        ax.fill_between(pcts, lo, hi, color=color, alpha=0.15)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlabel("depth (%)")
    ax.set_ylabel(r"$d'$ vs no instruction")
    ax.set_title("engage_suppress — sensitivity vs depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# 3. intensity_gain — endpoint-gain d' vs depth
# ======================================================================================

def intensity_gain(run_dir, out):
    conds = [POS, LEX_HI] + RAMP
    tab, layers, sids, cids = _unit_table(run_dir, conds)
    pcts = fs.depth_pcts(layers, sd.run_n_layers(run_dir))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for hi_c, lo_c, color, lab in ((LEX_HI, POS, fs.LEX_C, "think → intensely"),
                                   (RAMP[3], RAMP[0], fs.NUM_C, "intensity 1 → 4")):
        win = (tab[hi_c] > tab[lo_c]).astype(float) + 0.5 * (tab[hi_c] == tab[lo_c])
        auroc = np.nanmean(win, axis=0)
        dp = np.array([_d_from_auroc(a) for a in auroc])
        blo, bhi = _cluster_band(win, sids, cids)
        ax.plot(pcts, dp, color=color, marker="o", ms=3, lw=1.6, label=lab)
        ax.fill_between(pcts, [_d_from_auroc(a) for a in blo],
                        [_d_from_auroc(a) for a in bhi], color=color, alpha=0.15)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlabel("depth (%)")
    ax.set_ylabel(r"endpoint-gain $d'$")
    ax.set_title("intensity_gain — adjacent-endpoint resolution vs depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# 4. intensity_rank — signed Spearman rho vs depth  (NEW proj content)
# ======================================================================================

def intensity_rank(run_dir, out):
    conds = [POS, LEX_HI] + RAMP
    tab, layers, sids, cids = _unit_table(run_dir, conds)
    n_u, n_L = tab[POS].shape
    pcts = fs.depth_pcts(layers, sd.run_n_layers(run_dir))
    fig, ax = plt.subplots(figsize=(7, 4.4))

    def rho_curve(levels, cols):
        U = np.full((n_u, n_L), np.nan)
        for u in range(n_u):
            for li in range(n_L):
                vals_u = [tab[c][u, li] for c in cols]
                if np.all(np.isfinite(vals_u)):
                    U[u, li] = sd.signed_spearman(levels, vals_u)
        return U

    for levels, cols, color, lab in (([0, 1], [POS, LEX_HI], fs.LEX_C, "think → intensely"),
                                     ([1, 2, 3, 4], RAMP, fs.NUM_C, "intensity 1 → 4")):
        U = rho_curve(levels, cols)
        mean = np.nanmean(U, axis=0)
        lo, hi = _cluster_band(U, sids, cids)
        ax.plot(pcts, mean, color=color, marker="o", ms=3, lw=1.6, label=lab)
        ax.fill_between(pcts, lo, hi, color=color, alpha=0.15)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("depth (%)")
    ax.set_ylabel(r"mean signed Spearman $\rho$")
    ax.set_title("intensity_rank — order tracking vs depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# 5. pos_coverage — engage & suppress d' by POS category
# ======================================================================================

def pos_coverage(run_dir, out):
    vals, bases = sd.pos_category_readouts(run_dir, [POS, NEG])
    order = sorted(bases["proj"])
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (cond, sign, color, lab) in zip(axes, ((POS, +1.0, fs.ENGAGE_C, "engage"),
                                                   (NEG, -1.0, fs.SUPPRESS_C, "suppress"))):
        blocks, S = cs.per_concept_blocks(vals[("proj", cond)], bases["proj"], order)
        st = cs.dprime_stats(blocks, S, len(sd.CATS), rng, n_perm=0)
        dp = sign * st["dp"]
        lo = np.nanpercentile(sign * st["bavg"], 2.5, axis=0)
        hi = np.nanpercentile(sign * st["bavg"], 97.5, axis=0)
        x = np.arange(len(sd.CATS))
        ax.bar(x, dp, color=color, edgecolor="black", linewidth=0.4)
        ax.errorbar(x, dp, yerr=[dp - lo, hi - dp], fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
        gi = int(np.nanargmin(dp))
        ax.annotate("weakest", (x[gi], dp[gi]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#555")
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(sd.CATS, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{lab} by POS")
    axes[0].set_ylabel(r"$d'$ vs no instruction")
    fig.suptitle("pos_coverage — breadth across parts of speech", fontweight="bold")
    fig.tight_layout()
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# position-binned projection (shared by temporal_control & temporal_precision)
# ======================================================================================

def _position_profile(run_dir, conds, n_bins=10):
    """{cond: (centers, mean, lo, hi)} of per-position-bin projection Δ (cond −
    no_instruction baseline) at the targeting depth, pooled over sentence×concept."""
    L = sd._layer_for_fraction(run_dir, sd.PROJ_F_LOC)
    rows = sd.load_rows(run_dir)
    cache = sd.load_baseline(run_dir)
    vecs = sd.load_vectors(VC, sd._resolve_model(run_dir, None), [L])
    concepts_L, _, Vn = vecs[L]
    idx = defaultdict(dict)                       # (sentence) -> cond -> {concept: row}
    for r in rows:
        if r["condition_id"] in conds and r.get("concept"):
            idx[r["sentence"]].setdefault(r["condition_id"], {})[r["concept"]] = r
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    # cond -> list of per-unit (n_bins,) rows, plus that unit's sentence / concept.
    acc = {c: [] for c in conds}
    acc_s = {c: [] for c in conds}
    acc_c = {c: [] for c in conds}
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
                # One ROW per (sentence, concept) unit: its mean delta in each bin,
                # NaN where the unit has no token in that bin. Keeping the unit intact
                # (rather than pooling loose numbers per bin) is what lets the band be
                # a two-way cluster bootstrap with ONE resample shared across bins.
                vec = np.full(n_bins, np.nan)
                for bi in range(n_bins):
                    m = b == bi
                    if m.any() and np.isfinite(delta[m]).any():
                        vec[bi] = float(np.nanmean(delta[m]))
                acc[cond].append(vec)
                acc_s[cond].append(sent)
                acc_c[cond].append(c)
    out = {}
    for cond in conds:
        if not acc[cond]:
            continue
        U = np.vstack(acc[cond])
        mean = np.nanmean(U, axis=0)
        # Two-way (sentence x concept) cluster bootstrap -- same scheme as scalar_ci.
        # NOT 1.96*SEM over the pooled units: that treated 50x10 correlated units as
        # 500 independent draws and drew bands up to ~4.4x too narrow on gemma3-27b.
        lo, hi = _cluster_band(U, np.asarray(acc_s[cond]), np.asarray(acc_c[cond]))
        out[cond] = (centers, mean, lo, hi)
    return out


# ======================================================================================
# 6. temporal_control — begin / mid / end
# ======================================================================================

def temporal_control(run_dir, out):
    # Three targeted conditions vs generic think, each with its target span shaded:
    # loc_beginning (first third), persist_once (middle third -- "think ... only once
    # mid-sentence, then stop"), loc_end (last third). persist_once is the same
    # condition Temporal control scores as the 'mid' span (score_data.TARGET_GROUPS).
    prof = _position_profile(run_dir, [POS, "loc_beginning", "persist_once", "loc_end"])
    panels = (("loc_beginning", "think at beginning", (0, 1 / 3)),
              ("persist_once", "think once mid-sentence", (1 / 3, 2 / 3)),
              ("loc_end", "think at end", (2 / 3, 1)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
    for ax, (cond, lab, span) in zip(axes, panels):
        if cond not in prof:
            ax.set_title(f"{lab}\n(no data)")
            ax.set_xlabel("position in sentence")
            continue
        for cc, color, cl in ((POS, fs.MED_GRAY, "generic think"), (cond, fs.ENGAGE_C, lab)):
            cx, m, lo, hi = prof[cc]
            ax.plot(cx, m, color=color, marker="o", ms=3, lw=1.5, label=cl)
            ax.fill_between(cx, lo, hi, color=color, alpha=0.12)
        ax.axvspan(span[0], span[1], color="#f0d000", alpha=0.10)
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_xlabel("position in sentence")
        ax.set_title(lab)
        ax.legend(frameon=False)
    axes[0].set_ylabel(r"projection Δ vs baseline")
    fig.suptitle("temporal_control — does the concept land where commanded?", fontweight="bold")
    fig.tight_layout()
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# 7. temporal_precision — where the concept is "on" (first half / after-4th / throughout)
# ======================================================================================

def _word_start(i, tok):
    """True if generated token `tok` (index i, anchor already stripped) begins a new
    WORD: the first token, or a space-led token with alphanumeric content. Because
    anchored_token_strs are tokenizer.decode()'d, the word marker is a plain leading
    space for EVERY tokenizer family (SentencePiece / BPE / tiktoken), so this needs
    no per-model handling -- only the resulting token index shifts (a tokenizer that
    splits words into more subwords lands the boundary a token or two later)."""
    return i == 0 or (tok[:1] == " " and any(ch.isalnum() for ch in tok))


def _word_spans(toks):
    """Token-index (start, end) range for each WORD in a generated-token list."""
    starts = [i for i, t in enumerate(toks) if _word_start(i, t)]
    bounds = starts + [len(toks)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(starts))]


def _min_word_count(cache):
    """Word count of the shortest sentence; the word axis is clipped here so every
    unit contributes at every word index."""
    return min(len(_word_spans(ent["anchored_token_strs"][1:])) for ent in cache.values())


def _word_profile(run_dir, conds, max_words):
    """Like `_position_profile` but on WORD index (1..max_words), clipped to the
    shortest sentence's word count -- for the `after the fourth word` instruction,
    which is word-based. Each word's value is the mean of its constituent tokens'
    projection Δ (cond − no_instruction). Word index is tokenizer-invariant (word
    count doesn't depend on how a tokenizer splits), so the 4th-word boundary is one
    clean line (word 4→5) for every model. Same readout / depth / two-way band."""
    L = sd._layer_for_fraction(run_dir, sd.PROJ_F_LOC)
    rows = sd.load_rows(run_dir)
    cache = sd.load_baseline(run_dir)
    concepts_L, _, Vn = sd.load_vectors(VC, sd._resolve_model(run_dir, None), [L])[L]
    idx = defaultdict(dict)
    for r in rows:
        if r["condition_id"] in conds and r.get("concept"):
            idx[r["sentence"]].setdefault(r["condition_id"], {})[r["concept"]] = r
    acc = {c: [] for c in conds}
    acc_s = {c: [] for c in conds}
    acc_c = {c: [] for c in conds}
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
                delta = pt - base_cos[:, concepts_L.index(c)] * base_norm    # per token
                vec = np.full(max_words, np.nan)
                for wi in range(min(max_words, len(words))):
                    a, b = words[wi]
                    seg = delta[a:b]                                          # tokens in word wi
                    if np.isfinite(seg).any():
                        vec[wi] = float(np.nanmean(seg))
                acc[cond].append(vec)
                acc_s[cond].append(sent)
                acc_c[cond].append(c)
    out = {}
    x = np.arange(1, max_words + 1)
    for cond in conds:
        if not acc[cond]:
            continue
        U = np.vstack(acc[cond])
        lo, hi = _cluster_band(U, np.asarray(acc_s[cond]), np.asarray(acc_c[cond]))
        out[cond] = (x, np.nanmean(U, axis=0), lo, hi)
    return out


def _persist_panel(ax, prof, cond, lab, span, xlabel):
    for cc, color, cl in ((POS, fs.MED_GRAY, "generic think"), (cond, fs.ENGAGE_C, lab)):
        if cc not in prof:
            continue
        cx, m, lo, hi = prof[cc]
        ax.plot(cx, m, color=color, marker="o", ms=3, lw=1.4, label=cl)
        ax.fill_between(cx, lo, hi, color=color, alpha=0.13)
    ax.axvspan(span[0], span[1], color="#f0d000", alpha=0.12)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(lab)
    ax.legend(frameon=False)


def temporal_precision(run_dir, out):
    """Three panels, each a persistence instruction vs generic think, with the
    commanded 'on' region shaded. first-half / throughout are fractional instructions
    (fraction-of-sentence x); after-the-4th-word is a WORD-based instruction, so it
    uses a WORD-index x clipped at the shortest sentence's word count, with its 'on'
    region shaded from word 5 on (persist_once is not shown here -- it is the mid panel
    of temporal_control)."""
    cache = sd.load_baseline(run_dir)
    min_words = _min_word_count(cache)
    prof_f = _position_profile(run_dir, [POS, "persist_first_half", "persist_throughout"])
    prof_w = _word_profile(run_dir, [POS, "persist_after_fourth"], min_words)

    fig = plt.figure(figsize=(13.5, 4.4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, sharey=ax0)
    ax2 = fig.add_subplot(1, 3, 3, sharey=ax0)
    _persist_panel(ax0, prof_f, "persist_first_half", "first half", (0.0, 0.5),
                   "position in sentence")
    _persist_panel(ax1, prof_w, "persist_after_fourth", "after 4th word",
                   (4.5, min_words + 0.5), "word number from sentence start")
    _persist_panel(ax2, prof_f, "persist_throughout", "throughout", (0.0, 1.0),
                   "position in sentence")
    ax1.set_xlim(0.5, min_words + 0.5)
    ax1.set_xticks(range(1, min_words + 1))
    ax0.set_ylabel(r"projection Δ vs baseline")
    fig.suptitle("temporal_precision — concept held to the commanded span "
                 f"(after-4th by WORD; commanded on from word 5, clipped at shortest "
                 f"sentence = {min_words} words)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fs.note(fig)
    return fs.save(fig, out)


# ======================================================================================
# 8. engage_heatmap — cos / relnorm / proj, tokens x layers
# ======================================================================================

def engage_heatmap(run_dir, out):
    d = focal_data(run_dir)
    if d is None:
        print("  [engage_heatmap] focal sentence/concept unavailable; skipped")
        return None
    think = d["cond"](POS)
    if not think:
        print("  [engage_heatmap] think_about trial missing; skipped")
        return None
    layers = [L for L in d["layers"] if L in think]
    pcts = fs.depth_pcts(layers, sd.run_n_layers(run_dir))
    labels = [t.strip() or "␣" for t in d["toks"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, ch, title in zip(axes, ("cos", "relnorm", "proj"),
                             ("direction (cos)", "magnitude (rel-norm)", "projection ‖r‖·cos")):
        M = np.column_stack([think[L][ch] - d["base"][L][ch] for L in layers])  # (n_tok x n_L)
        lim = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, aspect="auto", cmap=fs.DIVERGE, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(pcts, fontsize=6, rotation=90)
        ax.set_xlabel("depth (%)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_yticks(range(d["n"]))
    axes[0].set_yticklabels(labels, fontsize=6)
    fig.suptitle(f"engage_heatmap — think_about − no_instruction  ({d['concept']} / {FOCAL_LABEL})",
                 fontweight="bold")
    fig.tight_layout()
    return fs.save(fig, out)


# ======================================================================================
# orchestration
# ======================================================================================

RENDERERS = {
    "raw_trace_example": raw_trace_example,
    "engage_suppress": engage_suppress,
    "intensity_gain": intensity_gain,
    "intensity_rank": intensity_rank,
    "pos_coverage": pos_coverage,
    "temporal_control": temporal_control,
    "temporal_precision": temporal_precision,
    "engage_heatmap": engage_heatmap,
}


def is_lt_run(run_dir):
    sets = set((sd.run_meta(run_dir).get("active_sets")) or [])
    return sets == {"layer_location"}


def render_all(run_dir, out_dir, only=None):
    """Render the exploratory suite for a MAIN run into out_dir. LT runs are skipped
    (none of these measures apply). Each figure failure is caught so one bad figure
    never aborts a run's post-processing."""
    run_dir = str(run_dir)
    if is_lt_run(run_dir):
        print("[explore] layer-targeting run -- no exploratory figures")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = only or list(RENDERERS)
    written = []
    for name in names:
        try:
            p = RENDERERS[name](run_dir, out_dir / f"{name}.png")
            if p:
                written.append(p)
        except Exception as e:                                        # noqa: BLE001
            print(f"  [explore:{name}] failed: {type(e).__name__}: {e}")
    print(f"[explore] {len(written)} figure(s) -> {out_dir}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Runtime exploratory figures for one run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/figures)")
    ap.add_argument("--only", default=None, help="comma-separated subset of figure names")
    args = ap.parse_args()
    out = args.out or str(Path(args.run_dir) / "figures")
    only = args.only.split(",") if args.only else None
    render_all(args.run_dir, out, only=only)


if __name__ == "__main__":
    main()
