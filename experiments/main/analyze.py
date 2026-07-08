"""Shim: register the sibling experiments' custom analysis kinds for the main experiment.

scripts/run_experiment.py auto-imports only the analyze.py sitting NEXT TO the
config being run. The main experiment's token_location / persistence analysis steps reuse
the ``location_targeting`` and ``temporal_profile`` kinds that live in
experiments/token_location/analyze.py and experiments/persistence/analyze.py,
so importing this module execs those two files -- their @register decorators
fire at import time, making the kinds available to the dispatcher. No other
logic belongs here.
"""

import importlib.util
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent

for _name in ("token_location", "persistence"):
    _path = _EXPERIMENTS_DIR / _name / "analyze.py"
    _spec = importlib.util.spec_from_file_location(f"{_name}_analyze", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
