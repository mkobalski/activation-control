#!/usr/bin/env python3
"""Regenerate the paper figure suite (results/paper/Fig1..Fig13) in one command.

Analysis/plotting is DECOUPLED from the experiment code: everything here is
CPU-only and reads only saved run artifacts (results.json, no_instruction_cache,
concept-vector cache). Each figure remains owned by its own scripts/figN_*.py;
this driver just invokes them with the canonical run dirs and output paths, so
"the paper figures" are reproducible with:

    /workspace/.venv/bin/python scripts/make_paper_figures.py            # all
    /workspace/.venv/bin/python scripts/make_paper_figures.py --only 4,7,13
    /workspace/.venv/bin/python scripts/make_paper_figures.py --list

Figure map (handoffs in results/paper/FigN.md):
    1   plot1_concept.py            single-concept modulation traces (Bread)
    2   fig2_engage_suppress.py     engage/suppress heatmap, one sentence
    3   fig3_position_...py         ... by absolute token position (clipped)
    4   fig4_fraction_...py         ... by fractional position
    5   fig5_rank_intensity.py      rank vs intensity, by fractional position
    6   fig6_gain_intensity.py      gain vs intensity, by fractional position
    7   fig7_pos_categories.py      four metrics by POS category (bars + CI)
    8   fig8_location_position.py   positional targeting (beginning/end)
    9   fig9_type_targeting.py      type targeting (punct/adjectives)
    10  fig10_persistence.py        temporal persistence (Fig10a + Fig10b)
    11  fig11_localization_metrics  localization metrics
    12  (PENDING - fig12_persistence_metrics.py drafted, not yet promoted)
    13  fig13_target_matrix.py      layer-targeting 8x8 (+ supplementary)

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
    F[2] = ("engage/suppress heatmap (s23)",
            [[PY, "scripts/fig2_engage_suppress.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig2.png")]], ["Fig2.png"])
    F[3] = ("engage/suppress by token position",
            [[PY, "scripts/fig3_position_engage_suppress.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig3.png")]], ["Fig3.png"])
    F[4] = ("engage/suppress by fractional position",
            [[PY, "scripts/fig4_fraction_engage_suppress.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig4.png")]], ["Fig4.png"])
    F[5] = ("rank vs intensity",
            [[PY, "scripts/fig5_rank_intensity.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig5.png")]], ["Fig5.png"])
    F[6] = ("gain vs intensity",
            [[PY, "scripts/fig6_gain_intensity.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig6.png")]], ["Fig6.png"])
    F[7] = ("metrics by POS category",
            [[PY, "scripts/fig7_pos_categories.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig7.png")]], ["Fig7.png"])
    F[8] = ("positional targeting (beginning/end)",
            [[PY, "scripts/fig8_location_position.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig8.png")]], ["Fig8.png"])
    F[9] = ("type targeting (punctuation/adjectives)",
            [[PY, "scripts/fig9_type_targeting.py", "--run-dir", main_run,
              "--out", str(OUT / "Fig9.png")]], ["Fig9.png"])
    F[10] = ("temporal persistence (a: fractional, b: after-4th)",
             [[PY, "scripts/fig10_persistence.py", "--run-dir", main_run,
               "--out-a", str(OUT / "Fig10a.png"), "--out-b", str(OUT / "Fig10b.png")]],
             ["Fig10a.png", "Fig10b.png"])
    F[11] = ("localization metrics",
             [[PY, "scripts/fig11_localization_metrics.py", "--run-dir", main_run,
               "--out", str(OUT / "Fig11.png")]], ["Fig11.png"])
    # ------------------------------------------------------------------
    # Fig 12 — PENDING. Persistence timing metrics (onset/offset error,
    # persistence score, rebound, leakage). Drafted in
    # scripts/fig12_persistence_metrics.py but not yet promoted (the `once`
    # edge detection and the after_fourth fractional-onset convention are
    # still under discussion). Re-enable when settled:
    # F[12] = ("persistence timing metrics",
    #          [[PY, "scripts/fig12_persistence_metrics.py", "--run-dir", main_run,
    #            "--out", str(OUT / "Fig12.png")]], ["Fig12.png"])
    # ------------------------------------------------------------------
    F[13] = ("layer-targeting 8x8 (+ supplementary demeaned)",
             [[PY, "scripts/fig13_target_matrix.py", "--run-dir", lt_run,
               "--baseline-run", main_run,
               "--out-raw", str(OUT / "Fig13.png"),
               "--out-dem", str(OUT / "Fig13_supp_demeaned.png")]],
             ["Fig13.png", "Fig13_supp_demeaned.png"])
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
    if args.list:
        for n in sorted(figs):
            print(f"Fig {n:>2}: {figs[n][0]}  ->  {', '.join(figs[n][2])}")
        print("Fig 12: PENDING (see comment in this script)")
        return

    wanted = sorted(figs) if not args.only else \
        [int(x) for x in args.only.split(",") if int(x) in figs]
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
    print("Fig 12: PENDING (skipped by design)")
    bad = [n for n, s, _ in results if s != "OK"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
