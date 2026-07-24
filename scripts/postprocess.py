#!/usr/bin/env python3
"""Post-run orchestrator: score a finished run and refresh the cross-model figures.

Runs the standalone steps in sequence -- each remains its own editable script:
    compute_scores.py     -> SCORES_<model>.json + PROFILES_<model>.json
    scalar_ci.py          -> SCALAR_CI_<model>.json   (projection)
    onset_offset_word.py  -> ONSET_OFFSET_WORD_<model>.json (word-based onset gate)
    explore.py            -> per-model exploratory figures (MAIN run only)
    superplot.py          -> cross-model comparison, refreshed from all JSONs

Resolves the model's MAIN + LT runs from results/raw/ and tolerates a missing LT
(a main-only run still scores; layer_targeting fills in when the LT run lands).
Called by scripts/run.sh after run_experiment (the completion trigger); also
runnable by hand: `python scripts/postprocess.py --run-dir <dir>` or `--model <m>`.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import score_data as sd                                                    # noqa: E402

PY = sys.executable
SCRIPTS = ROOT / "scripts"


def find_runs(raw_dir, model):
    """(main_run, lt_run) for a model: newest main carrying a baseline cache, newest
    _lt run (or None). Names follow run_experiment._run_name."""
    main = next((d for d in sorted(raw_dir.glob(f"*_{model}_activation_control"), reverse=True)
                 if (d / "no_instruction_cache.pkl").exists()), None)
    lts = sorted(raw_dir.glob(f"*_{model}_activation_control_lt"), reverse=True)
    return main, (lts[0] if lts else None)


def _step(cmd, label):
    print(f"\n[postprocess] {label}", flush=True)
    rc = subprocess.run([PY] + [str(c) for c in cmd]).returncode
    if rc:
        print(f"  [warn] {label} exited {rc}")
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description="Score a finished run + refresh cross-model figures.")
    ap.add_argument("--run-dir", help="the just-finished run (main or LT); model inferred from it")
    ap.add_argument("--model", help="model short name (alternative to --run-dir)")
    ap.add_argument("--data-root", default=str(ROOT / "results"),
                    help="dir for the SCORES_/PROFILES_/SCALAR_CI_ JSONs (default: results/)")
    args = ap.parse_args()

    if args.model:
        model = args.model
    elif args.run_dir:
        model = sd._resolve_model(args.run_dir, None)
    else:
        ap.error("pass --run-dir or --model")

    dr = Path(args.data_root)
    main_run, lt_run = find_runs(dr / "raw", model)
    if main_run is None:
        print(f"[postprocess] no main run for {model!r} under {dr/'raw'}; nothing to score")
        return
    print(f"[postprocess] model={model}  main={main_run.name}  lt={lt_run.name if lt_run else 'none (layer_targeting null)'}")

    scores = dr / f"SCORES_{model}.json"
    profiles = dr / f"PROFILES_{model}.json"
    ci = dr / f"SCALAR_CI_{model}.json"

    cs = [SCRIPTS / "compute_scores.py", "--main-run", main_run,
          "--json", scores, "--profiles-json", profiles]
    if lt_run is not None:
        cs += ["--lt-run", lt_run]
    _step(cs, "compute_scores -> SCORES + PROFILES")
    sci = [SCRIPTS / "scalar_ci.py", "--main-run", main_run,
           "--channels", "projection", "--json", ci]
    if lt_run is not None:
        sci += ["--lt-run", lt_run]
    _step(sci, "scalar_ci -> SCALAR_CI")
    _step([SCRIPTS / "onset_offset_word.py", "--models", model],
          "onset_offset_word -> ONSET_OFFSET_WORD")
    _step([SCRIPTS / "explore.py", "--run-dir", main_run], "explore -> per-model figures")
    _step([SCRIPTS / "superplot.py", "--data-root", dr], "superplot -> comparison")
    print("\n[postprocess] done")


if __name__ == "__main__":
    main()
