"""Portable paths for the figure scripts.

Resolves the activation-control code/data location and the figure output directory
RELATIVE to this file, so the scripts work wherever the repo is checked out. These
scripts live in the repo at <root>/figure_scripts and read the frozen scoring
artifacts in <root>/results. Each path is overridable by an environment variable:

  AC_ROOT      activation-control repo root   (default: the repo containing this file)
  AC_DATA      frozen SCORES/PROFILES dir     (default: $AC_ROOT/results)
  AC_FIG_OUT  figure output directory        (default: <repo>/Figures)
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # the activation-control repo root
AC_ROOT = Path(os.environ.get("AC_ROOT") or _ROOT)
AC_DATA = os.environ.get("AC_DATA") or str(AC_ROOT / "results")
FIG_OUT = Path(os.environ.get("AC_FIG_OUT") or _ROOT / "Figures")


# Make the repo's own modules importable from any figure script: `scripts/` holds
# the scoring layer (score_data, compute_scores) and the single canonical copy of
# model_family_colors.py, which used to be duplicated here and drift.
if str(AC_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(AC_ROOT / "scripts"))


def out(name):
    """Absolute path of a figure output file `name` under FIG_OUT.

    Creates FIG_OUT on first use: on a fresh clone the directory does not exist
    and matplotlib's savefig raises FileNotFoundError instead of creating it."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    return str(FIG_OUT / name)


# A figure that needs inputs a public clone does not ship (per-trial raw runs /
# vector_cache) exits with this code instead of crashing, so the render driver can
# tell "inputs unavailable" apart from success (0) and a real error (anything else).
SKIP_EXIT = 3


def skip(msg):
    """Print a standardized one-line skip notice and exit with SKIP_EXIT.

    Use for the figures that still read raw (think_intensity, token_coverage) when
    the raw is absent -- the paper's derived JSONs reproduce the other figures from a
    clone, but these two need regenerated runs. Keep `msg` actionable (say how to
    regenerate)."""
    print(f"[skip] {msg}")
    raise SystemExit(SKIP_EXIT)
