"""The `_lt` run-dir tag and the standalone-set rule must describe the same set.

`run_experiment.LT_ONLY_SETS` decides whether a run is tagged `_lt`;
`config.load_config` drops the always-on controls for `standalone_sets`. If the
two disagree, a run generates the wrong trial count under the right name -- the
failure the trial-count check in the README exists to catch, one step too late.
"""
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _lt_only_sets():
    ns = {}
    for line in (ROOT / "scripts" / "run_experiment.py").read_text().splitlines():
        if line.startswith("LT_ONLY_SETS"):
            exec(line, ns)                                    # noqa: S102 - one literal
            return ns["LT_ONLY_SETS"]
    raise AssertionError("LT_ONLY_SETS not found in scripts/run_experiment.py")


def test_lt_only_sets_matches_standalone_sets():
    cfg = yaml.safe_load((ROOT / "experiments" / "main" / "config.yaml").read_text())
    standalone = set(cfg["experiment"]["standalone_sets"])
    assert _lt_only_sets() == standalone, (
        f"run_experiment.LT_ONLY_SETS={_lt_only_sets()} but "
        f"config.yaml standalone_sets={standalone}"
    )
