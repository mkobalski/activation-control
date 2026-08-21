#!/usr/bin/env python3
"""Render every paper figure, reporting rendered / skipped / failed.

This is the one-command entry point for regenerating the paper's figures from a
clone. Most figures read only the committed derived JSONs (SCORES_/PROFILES_/
SCALAR_CI_/ONSET_OFFSET_WORD_/FIGDATA_) and render anywhere. Two -- think_intensity
and token_coverage -- still read per-trial raw runs (not shipped in a clone); when
the raw is absent they exit paths.SKIP_EXIT and are reported as SKIPPED, not failed.
A nonzero exit from this driver therefore means a real failure, never a clone-expected
skip.

  python figure_scripts/render_all.py --data-root results --out-dir /tmp/figs

--data-root / --out-dir default to $AC_DATA / $AC_FIG_OUT (the same env vars the
individual scripts read), so `AC_DATA=results AC_FIG_OUT=/tmp/figs python
figure_scripts/render_all.py` works too.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from paths import SKIP_EXIT

HERE = Path(__file__).resolve().parent
# Helper/loader modules that emit no figure of their own.
NON_FIGURES = {"paths.py", "engage_suppress.py", "roster.py", "render_all.py"}


def main():
    ap = argparse.ArgumentParser(description="Render all paper figures with a rendered/skipped/failed summary.")
    ap.add_argument("--data-root", default=os.environ.get("AC_DATA"),
                    help="results dir with the derived JSONs (default: $AC_DATA)")
    ap.add_argument("--out-dir", default=os.environ.get("AC_FIG_OUT"),
                    help="figure output dir (default: $AC_FIG_OUT)")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.data_root:
        env["AC_DATA"] = args.data_root
    if args.out_dir:
        env["AC_FIG_OUT"] = args.out_dir

    scripts = sorted(p for p in HERE.glob("*.py") if p.name not in NON_FIGURES)
    rendered, skipped, failed = [], [], []
    for s in scripts:
        r = subprocess.run([sys.executable, str(s)], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode == 0:
            rendered.append(s.name); print(f"[rendered] {s.name}")
        elif r.returncode == SKIP_EXIT:
            skipped.append(s.name); print(f"[skipped ] {s.name}")
            for line in r.stdout.splitlines():
                if line.startswith("[skip]"):
                    print(f"           {line}")
        else:
            failed.append(s.name); print(f"[FAILED  ] {s.name} (exit {r.returncode})")
            for line in r.stdout.splitlines()[-10:]:
                print(f"           {line}")

    print(f"\nrendered {len(rendered)}   skipped {len(skipped)}   failed {len(failed)}")
    if skipped:
        print("skipped (need a regenerated raw run; see README): " + ", ".join(skipped))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
