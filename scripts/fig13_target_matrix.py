#!/usr/bin/env python3
"""Fig 13: simple 8x8 layer-targeting matrices.

y = TARGETED layer (named in the prompt, 40..61); x = ANALYSIS layer (same deep
eight); cell = readout averaged over all tokens x 50 sentences x 10 concepts.
Conditions think_at_layer / think_intensely_at_layer; readouts Δcos and Δrelnorm
(vs the main run's no_instruction baseline). All 64 values printed.

Two figures:
  raw      : the Δ readouts as-is. If the model targeted layers, the diagonal
             would dominate; if not, columns look uniform.
  demeaned : each column minus its mean across the 8 targets — the
             target-SPECIFIC residual on its own color scale. A diagonal band
             here = reallocation toward the named layer. Cells where the
             residual is significant across (sentence x concept) units
             (sign-flip, B=5000, BH-FDR per panel) are marked with a dot.

CPU-only, no model load.
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from controllability_heatmap import classify, bh_fdr, load_vectors        # noqa: E402
from fig2_engage_suppress import _trace, _relnorm                         # noqa: E402

B = 5000
CONDS = ["think_at_layer", "think_intensely_at_layer"]
DEEP = [40, 43, 46, 49, 52, 55, 58, 61]


def build(run_dir, baseline_run, *, vector_cache="results/vector_cache",
          method="baseline", model="gemma3_27b", seed=0):
    rows = json.load(open(Path(run_dir) / "results.json"))["results"]
    comp = [r for r in rows if r.get("is_compliant") and r["condition_id"] in CONDS
            and r.get("concept") and r.get("prompt_layer") is not None]
    cache = pickle.load(open(Path(baseline_run) / "no_instruction_cache.pkl", "rb"))
    vecs = load_vectors(vector_cache, model, DEEP, method)

    # per-sentence baseline cos (per concept) and relnorm at the deep layers
    base_cos, base_rel, sent_meta = {}, {}, {}
    for s, ent in cache.items():
        toks = ent["anchored_token_strs"][1:]
        n_tok = len(toks)
        classes = [classify(t) for t in toks]
        sent_meta[s] = (n_tok, classes)
        bc, brl = {}, {}
        for L in DEEP:
            A = np.asarray(ent["activations"][L], np.float32)[:n_tok]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            bc[L] = An @ vecs[L][2].T
            brl[L] = _relnorm(np.asarray(ent["norms"][L], np.float32)[:n_tok], classes)
        base_cos[s], base_rel[s] = bc, brl

    # unit values: val[cond][metric][(s,c)][T][L] = token-mean Δ readout
    val = {c: {"cos": defaultdict(dict), "relnorm": defaultdict(dict)} for c in CONDS}
    for r in comp:
        s, c, T = r["sentence"], r["concept"], int(r["prompt_layer"])
        if s not in sent_meta:
            continue
        n_tok, classes = sent_meta[s]
        concepts_L = vecs[DEEP[0]][0]
        if c not in concepts_L:
            continue
        ci = concepts_L.index(c)
        dcos, drel = {}, {}
        for L in DEEP:
            tr = _trace(r, "cosine_sim", L)
            if tr is not None:
                v = np.asarray(tr, np.float32)[:n_tok]
                m = min(len(v), n_tok)
                dcos[L] = float(np.nanmean(v[:m] - base_cos[s][L][:m, ci]))
            nr = _trace(r, "norms", L)
            if nr is not None:
                rl = _relnorm(np.asarray(nr, np.float32)[:n_tok], classes)
                if rl is not None and base_rel[s][L] is not None:
                    m = min(len(rl), len(base_rel[s][L]))
                    drel[L] = float(np.nanmean(rl[:m] - base_rel[s][L][:m]))
        if dcos:
            val[r["condition_id"]]["cos"][(s, c)][T] = dcos
        if drel:
            val[r["condition_id"]]["relnorm"][(s, c)][T] = drel

    rng = np.random.default_rng(seed)
    out = {}
    for cond in CONDS:
        for metric in ("cos", "relnorm"):
            units = {k: v for k, v in val[cond][metric].items() if len(v) == len(DEEP)}
            # unit tensor U[u, ti, li]
            keys = sorted(units)
            U = np.full((len(keys), len(DEEP), len(DEEP)), np.nan)
            for ui, k in enumerate(keys):
                for ti, T in enumerate(DEEP):
                    for li, L in enumerate(DEEP):
                        U[ui, ti, li] = units[k][T].get(L, np.nan)
            M = np.nanmean(U, axis=0)                                  # raw 8x8
            Ud = U - np.nanmean(U, axis=1, keepdims=True)              # demean per column
            Md = np.nanmean(Ud, axis=0)

            def _qmat(T3):
                P = np.full((len(DEEP), len(DEEP)), np.nan)
                for ti in range(len(DEEP)):
                    for li in range(len(DEEP)):
                        dv = T3[:, ti, li]; dv = dv[~np.isnan(dv)]
                        if len(dv) >= 3:
                            obs = float(dv.mean())
                            signs = rng.choice([-1.0, 1.0], size=(B, len(dv)))
                            null = (signs * dv).mean(1)
                            P[ti, li] = (1 + int((np.abs(null) >= abs(obs) - 1e-15).sum())) / (B + 1)
                return bh_fdr(P)

            out[(cond, metric)] = dict(raw=M, dem=Md, q=_qmat(Ud), q_raw=_qmat(U),
                                       n=len(keys))
    return out


def _panel(ax, M, title, cbar_label, fig, dec=3, dots=None, num_q=None, alpha=0.05):
    """num_q: q-matrix gating the printed numbers (print only where q<alpha);
    None prints all. dots: q-matrix for corner significance dots (demeaned fig)."""
    A = np.nanmax(np.abs(M)); A = float(A) if np.isfinite(A) and A > 0 else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-A, vmax=A)
    ax.set_xticks(range(len(DEEP))); ax.set_xticklabels(DEEP, fontsize=12)
    ax.set_yticks(range(len(DEEP))); ax.set_yticklabels(DEEP, fontsize=12)
    ax.set_xlabel("Analysis layer (measured at)", fontsize=13)
    ax.set_ylabel("Targeted layer (named in prompt)", fontsize=13)
    ax.set_title(title, fontsize=14)
    for ti in range(len(DEEP)):
        for li in range(len(DEEP)):
            v = M[ti, li]
            if np.isnan(v):
                continue
            show = num_q is None or (not np.isnan(num_q[ti, li]) and num_q[ti, li] < alpha)
            if show:
                ax.text(li, ti, f"{v:.{dec}f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v) / A > 0.55 else "black")
            if dots is not None and not np.isnan(dots[ti, li]) and dots[ti, li] < alpha:
                ax.plot(li + 0.38, ti - 0.38, "o", color="black", ms=4)
    # highlight the diagonal cells (target == analysis layer) with bold borders
    for k in range(len(DEEP)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="#111", linewidth=2.6, zorder=5))
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=9); cb.set_label(cbar_label, fontsize=11)


def render(stats, out_raw, out_dem):
    for which, out in (("raw", out_raw), ("dem", out_dem)):
        # sized to match Fig 2's rendered dimensions (~19.9 x 14.3 in @160dpi)
        fig, axes = plt.subplots(2, 2, figsize=(19.4, 14.1), layout="constrained")
        for ri, cond in enumerate(CONDS):
            for ci, metric in enumerate(["cos", "relnorm"]):
                st = stats[(cond, metric)]
                M = st[which]
                lab = "Δcos" if metric == "cos" else "Δrelnorm"
                if which == "dem":
                    lab += " − column mean"
                _panel(axes[ri][ci], M,
                       f"{cond} — {lab}   (n={st['n']} units)", lab, fig,
                       dots=st["q"] if which == "dem" else None,
                       num_q=None)
        if which == "raw":
            fig.suptitle("Layer targeting, 8×8: readout vs (targeted layer × analysis layer)\n"
                         "cell = Δ vs no_instruction, averaged over tokens × 50 sentences × 10 concepts",
                         fontsize=17, fontweight="bold")
            note = ("Bold boxes = the diagonal (measured at the named layer); if targeting worked, those "
                    "cells would stand out from their columns. All numbers are DESCRIPTIVE — see the "
                    "handoff (Fig13.md) for the targeting statistics and why per-cell significance "
                    "marks are deliberately omitted here.")
        else:
            fig.suptitle("Layer targeting, 8×8: target-SPECIFIC residual\n"
                         "cell = same data with each column's mean across targets removed",
                         fontsize=17, fontweight="bold")
            note = ("What remains after removing the shared (target-independent) depth profile. "
                    "A red diagonal band = reallocation toward the named layer. black dot = "
                    "residual significant across (sentence × concept) units "
                    f"(sign-flip B={B}, BH-FDR q<0.05 per panel).")
        fig.text(0.5, 0.005, note, ha="center", fontsize=11, color="#444")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Fig 13: 8x8 layer-targeting matrices.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--out-raw", default="fig13_target_matrix_raw.png")
    ap.add_argument("--out-dem", default="fig13_target_matrix_demeaned.png")
    args = ap.parse_args()
    stats = build(args.run_dir, args.baseline_run)
    render(stats, args.out_raw, args.out_dem)


if __name__ == "__main__":
    main()
