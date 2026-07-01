"""Thin W&B wrapper so the experiment keeps running if W&B is unavailable.

Design principle: FAIL OPEN. Logging to Weights & Biases is a nice-to-have, not
a requirement for the experiment to produce results. Every function therefore
swallows exceptions (missing package, no network, bad credentials) and turns
logging into a no-op rather than crashing the run. The active run is held in a
module-level `_run`; when it is None all logging calls short-circuit.
"""

from typing import Dict, Optional, List, Any


# Module-global handle to the active W&B run (None == logging disabled). Kept at
# module scope so every helper shares one run without threading it through args.
_run = None


def init_wandb(project: str, name: str, config: Dict,
               entity: Optional[str] = None,
               tags: Optional[List[str]] = None) -> bool:
    """Start a W&B run; return True on success, False if W&B is unavailable.

    On any failure we print a notice, leave `_run` as None, and return False so
    the caller can proceed without logging. This is the entry point that decides
    whether the rest of the module's calls do anything.
    """
    global _run
    try:
        import wandb
        _run = wandb.init(project=project, name=name, config=config,
                          entity=entity, tags=tags or [], reinit=True)
        return True
    except Exception as e:
        # Fail open: report once, then run silently without W&B.
        print(f"W&B init failed (continuing without): {e}")
        _run = None
        return False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log a dict of scalar metrics (no-op if W&B isn't running)."""
    if _run is None:
        return
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        # Never let a logging hiccup interrupt the experiment.
        pass


def log_summary(summary: Dict[str, Any]):
    """Write final summary values onto the run (no-op if W&B isn't running)."""
    if _run is None:
        return
    try:
        import wandb
        for k, v in summary.items():
            wandb.run.summary[k] = v
    except Exception:
        pass


def log_artifact(path: str, name: str, artifact_type: str = "results"):
    """Upload a file (e.g. a results pickle) as a W&B artifact, if possible."""
    if _run is None:
        return
    try:
        import wandb
        art = wandb.Artifact(name=name, type=artifact_type)
        art.add_file(path)
        wandb.log_artifact(art)
    except Exception as e:
        # Unlike the metric loggers, surface this one: a failed artifact upload
        # means the saved file did not make it to W&B, worth seeing in logs.
        print(f"W&B artifact log failed ({path}): {e}")


def finish_wandb():
    """Cleanly close the active run and reset state (safe to call anytime)."""
    global _run
    if _run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
        _run = None
