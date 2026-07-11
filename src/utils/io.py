"""JSON + pickle save utilities.

Each run is saved twice: a human-readable JSON (metadata, scalar metrics) and a
pickle (the full objects, including the large numpy activation arrays). The JSON
exists so results can be inspected/diffed without unpickling, but the multi-MB
activation arrays would bloat it and are not JSON-native, so they are replaced
with a placeholder string in the JSON and only stored verbatim in the pickle.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np


def get_run_dir(base_dir: str, experiment_name: str, model_name: str) -> Path:
    """Create and return a unique output directory for one run.

    The directory name is ``<YYYYMMDD_HHMMSS>_<model>_<experiment>`` -- the
    timestamp leads so a plain ``ls``/lexical sort of ``results/raw/`` is
    chronological. Model + experiment follow so runs stay self-describing and
    never collide. The model name is sanitized because HF ids contain "/" (a path
    separator) and spaces, which would otherwise create unintended nested dirs.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    run_dir = Path(base_dir) / "raw" / f"{ts}_{safe_model}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_default(obj):
    """Fallback serializer passed to json.dump for non-JSON-native values.

    json.dump calls this for any object it can't encode. We: replace ndarrays
    with a placeholder (they belong in the pickle, not the JSON); downcast numpy
    scalar types to native float/int so they serialize cleanly; fall back to an
    object's __dict__ for simple dataclass-like objects; and stringify anything
    else as a last resort so a stray value can never crash the whole dump.
    """
    if isinstance(obj, np.ndarray):
        return "__ndarray_stored_in_pickle__"
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def save_results_json(results: List[Dict], path: Path, metrics: Optional[Dict] = None):
    """Write the human-readable JSON copy of a run (arrays elided).

    See the module docstring: the per-sample activation arrays are megabytes
    each and not JSON-native, so we swap them for a placeholder string here and
    rely on the pickle (save_results_pickle) for the real data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip large arrays before dumping to JSON; they live in the pickle.
    clean = []
    for r in results:
        rc = {}
        for k, v in r.items():
            # These two keys hold the big numpy activation tensors; replace the
            # value with a marker so the JSON stays small and readable.
            if k in ("activations", "activations_anchored",
                     "activations_tail", "activations_prompt_special"):
                # keep an honest placeholder: "__dropped__" if the run freed
                # the arrays (--no-pickle), else the stored-in-pickle marker
                rc[k] = v if isinstance(v, str) else "__stored_in_pickle__"
            else:
                rc[k] = v
        clean.append(rc)
    data = {"results": clean, "metrics": metrics or {}, "n_samples": len(results)}
    with open(path, "w") as f:
        # _json_default handles any remaining numpy scalars / nested arrays.
        json.dump(data, f, indent=2, default=_json_default)


def save_results_pickle(results: List[Dict], path: Path, metrics: Optional[Dict] = None):
    """Write the complete run (including activation arrays) as a pickle.

    This is the source of truth for the heavy data; the JSON is only a readable
    summary. Mirrors the JSON layout (results / metrics / n_samples) so either
    file can be loaded the same way.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"results": results, "metrics": metrics or {}, "n_samples": len(results)}, f)


def load_results(path: Path):
    """Load a saved run, preferring the pickle (full data) over the JSON.

    `path` may carry any suffix; we try the .pkl sibling first because it
    contains the activation arrays, and fall back to the .json (arrays elided)
    only if no pickle exists.
    """
    path = Path(path)
    pkl = path.with_suffix(".pkl")
    js = path.with_suffix(".json")
    if pkl.exists():
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        return d["results"], d.get("metrics", {})
    if js.exists():
        with open(js) as f:
            d = json.load(f)
        return d["results"], d.get("metrics", {})
    raise FileNotFoundError(f"No results at {path}")


def load_run_config(run_dir) -> Dict:
    """Load a run's saved ``config.yaml`` summary as a dict ({} if absent/unreadable).

    run_experiment.py writes this per run; the plot scripts read it to recover the
    experiment's declared ``sentences`` / ``concepts`` order (see order_by_config)
    so plot labels are stable across runs instead of following trial order.
    """
    p = Path(run_dir) / "config.yaml"
    if not p.exists():
        return {}
    import yaml  # local import: keeps io usable in yaml-less minimal contexts
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def order_by_config(present, canonical) -> List:
    """Order the items actually present by the config's canonical order.

    `present` is the list of items seen in the results (any order); `canonical`
    is the config's declared order (e.g. cfg.sentences). Items are returned sorted
    by their index in `canonical`; anything not in `canonical` (or when `canonical`
    is missing) keeps its original first-seen order and is appended at the end.
    This is what makes s6 == the cat sentence, and the concept panels config-ordered,
    regardless of the per-run trial shuffle.
    """
    present = list(present)
    if not canonical:
        return present
    rank = {v: i for i, v in enumerate(canonical)}
    appear = {v: i for i, v in enumerate(present)}
    return sorted(present, key=lambda v: (rank.get(v, len(canonical)), appear[v]))
