#!/usr/bin/env python3
"""Regenerate the paper figure suite (results/paper/Fig1..Fig7 + appendix) in one command.

Analysis/plotting is DECOUPLED from the experiment code: everything here is
CPU-only and reads only saved run artifacts (results.json, no_instruction_cache,
concept-vector cache). Each figure remains owned by its own scripts/figN_*.py;
this driver just invokes them with the canonical run dirs and output paths, so
"the paper figures" are reproducible with:

    /workspace/.venv/bin/python scripts/make_paper_figures.py            # all
    /workspace/.venv/bin/python scripts/make_paper_figures.py --only 4,7,A1
    /workspace/.venv/bin/python scripts/make_paper_figures.py --list

Figure map (handoffs in results/paper/FigN.md):
    1   plot1_concept.py            single-concept modulation traces (Bread)
    2   fig2_dprime.py              d' depth profiles (engage/suppress vs neutral)
    3   fig3_dprime.py              endpoint-gain d' depth profiles (+ Fig3_aux, unregistered)
    4   fig4_rank_depth.py          intensity-rank depth profiles (lexical vs numeric)
    5   fig5_dprime.py              engagement/suppression d' by POS category
    6   fig6_location_position.py  positional targeting (beginning/end)  [was Fig 8]
    7   fig7_persistence.py         temporal persistence (Fig7a + Fig7b)  [was Fig 10]
    A1  figA1_position_auroc.py     APPENDIX: AUROC by fractional position (completes Fig 2)

    RETIRED to results/paper/"Exploratory analysis" (scripts stay runnable):
      old Figs 5-6 heatmap family (2026-07-08); old Fig 9 type targeting,
      old Figs 11-12 metric figures (superseded by results/paper/SCORES.md),
      old Fig 13 layer-targeting matrices (2026-07-10); the AUROC/max-norm
      versions of Figs 2, 3, 5(a/b) + the _alt raw-delta drafts (2026-07-10,
      replaced by the d' versions).

Inputs (override with --main-run / --lt-run):
    MAIN_RUN : the 07-04 master (20 layers post-append; has no_instruction_cache)
    LT_RUN   : the 07-08 layer-targeting re-run (revised prompts)
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = ROOT / "results/paper"

MAIN_RUN = "results/raw/20260704_212244_gemma3_27b_write_introspection_main"
LT_RUN = "results/raw/20260708_002002_gemma3_27b_write_introspection_main"

S23 = "The bus was crowded, but I found a seat near the back."


def _fig1(main_run, lt_run):
    # plot1_concept writes plot1_<concept>.png; Fig 1 is the Bread panel (see Fig1.md)
    cmd = [PY, "scripts/plot1_concept.py", "--run-dir", main_run,
           "--sentence", S23, "--cos-layer", "55", "--relnorm-layer", "46",
           "--concepts", "Bread", "--out-dir", str(OUT)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    shutil.move(OUT / "plot1_Bread.png", OUT / "Fig1.png")


# each entry: fig number -> (description, list of commands OR callable, outputs)
def registry(main_run, lt_run):
    F = {}
    F[1] = ("concept modulation traces (Bread)", _fig1, ["Fig1.png"])
    F[2] = ("d' depth profiles (engage/suppress vs neutral)",
            [[PY, "scripts/fig2_dprime.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig2.png")]], ["Fig2.png"])
    F[3] = ("endpoint-gain d' depth profiles (lexical vs numeric)",
            [[PY, "scripts/fig3_dprime.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig3.png")]], ["Fig3.png"])
    F[4] = ("intensity-rank depth profiles (lexical vs numeric)",
            [[PY, "scripts/fig4_rank_depth.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig4.png")]], ["Fig4.png"])
    F[5] = ("engagement/suppression d' by POS category",
            [[PY, "scripts/fig5_dprime.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig5.png")]], ["Fig5.png"])
    F[6] = ("positional targeting (beginning/end)  [was Fig 8]",
            [[PY, "scripts/fig6_location_position.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig6.png")]], ["Fig6.png"])
    F[7] = ("temporal persistence (7a: fractional, 7b: after-4th)  [was Fig 10]",
            [[PY, "scripts/fig7_persistence.py", "--run-dir", main_run,
              "--out-a", str(OUT / "Fig7a.png"), "--out-b", str(OUT / "Fig7b.png")]],
            ["Fig7a.png", "Fig7b.png"])
    # ------------------------------------------------------------------
    # RETIRED (see docstring): old Fig 9 (type targeting), old Figs 11-12
    # (metric figures -> SCORES.md), old Fig 13 (layer-targeting matrices;
    # was the only consumer of LT_RUN). Renderers remain runnable, e.g.:
    #   fig13_target_matrix.py --run-dir <LT_RUN> --baseline-run <MAIN_RUN> ...
    # ------------------------------------------------------------------
    # ---- appendix series (FigA#: sequential, independent of main numbers) ----
    F["A1"] = ("APPENDIX: AUROC by fractional position (completes Fig 2)",
               [[PY, "scripts/figA1_position_auroc.py", "--run-dir", main_run,
                 "--out", str(OUT / "FigA1.png")]], ["FigA1.png"])
    return F


def main():
    ap = argparse.ArgumentParser(description="Regenerate the paper figures (results/paper).")
    ap.add_argument("--only", default=None,
                    help="comma-separated figure numbers (e.g. 4,7,13); default: all")
    ap.add_argument("--list", action="store_true", help="list figures and exit")
    ap.add_argument("--main-run", default=MAIN_RUN)
    ap.add_argument("--lt-run", default=LT_RUN)
    args = ap.parse_args()

    figs = registry(args.main_run, args.lt_run)
    order = sorted([k for k in figs if isinstance(k, int)]) + \
        sorted([k for k in figs if isinstance(k, str)])
    if args.list:
        for n in order:
            print(f"Fig {n:>2}: {figs[n][0]}  ->  {', '.join(figs[n][2])}")
        print("Retired (Exploratory analysis): AUROC/max-norm Figs 2,3,5 + old Figs 9, 11-13 + pre-rename 5-6 family")
        return

    if not args.only:
        wanted = order
    else:
        toks = [t.strip() for t in args.only.split(",")]
        keys = [int(t) if t.isdigit() else t.upper() for t in toks]
        wanted = [k for k in order if k in keys]
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for n in wanted:
        desc, action, outs = figs[n]
        t0 = time.time()
        print(f"\n=== Fig {n}: {desc} ===", flush=True)
        try:
            if callable(action):
                action(args.main_run, args.lt_run)
            else:
                for cmd in action:
                    subprocess.run(cmd, cwd=ROOT, check=True)
            missing = [o for o in outs if not (OUT / o).exists()]
            status = "OK" if not missing else f"MISSING {missing}"
        except subprocess.CalledProcessError as e:
            status = f"FAILED (exit {e.returncode})"
        results.append((n, status, time.time() - t0))

    print("\n" + "=" * 56)
    for n, status, dt in results:
        print(f"Fig {n:>2}: {status:<28} {dt:6.1f}s")
    print("Retired predecessors in Exploratory analysis/; scored measures in SCORES.md")
    bad = [n for n, s, _ in results if s != "OK"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
