"""Loading and structural validation for control-under-load run tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RunTables:
    """The three normalized tables emitted by an activation-control run."""

    items: pd.DataFrame
    trials: pd.DataFrame
    readouts: pd.DataFrame


@dataclass(frozen=True)
class RunValidation:
    """Result of checking that run tables can be analyzed safely."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    n_items: int
    n_trials: int
    n_readouts: int
    n_concepts: int
    n_wrong_concepts: int

    def raise_for_errors(self) -> None:
        """Raise a single useful exception when validation failed."""
        if self.errors:
            raise ValueError("invalid run tables: " + "; ".join(self.errors))


_REQUIRED = {
    "items": {"item_id", "axis", "difficulty", "difficulty_bin"},
    "trials": {
        "trial_id",
        "item_id",
        "condition",
        "instructed_concept",
        "is_correct",
    },
    "readouts": {
        "trial_id",
        "item_id",
        "condition",
        "instructed_concept",
        "readout_concept",
        "projections",
    },
}
_CONDITIONS = {"no_instruction", "think_about", "ctrl_think_intensely"}


def load_run_tables(data_dir: str | Path) -> RunTables:
    """Read ``items.parquet``, ``trials.parquet``, and ``readouts.parquet``.

    No experiment code is imported and no schema coercion is performed: malformed
    or lossy inputs are reported by :func:`validate_run_tables`.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"run data directory does not exist: {directory}")

    def read(name: str) -> pd.DataFrame:
        path = directory / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"required run table does not exist: {path}")
        return pd.read_parquet(path, engine="pyarrow")

    return RunTables(items=read("items"), trials=read("trials"), readouts=read("readouts"))


def _duplicates(frame: pd.DataFrame, columns: list[str]) -> int:
    if not set(columns).issubset(frame.columns):
        return 0
    return int(frame.duplicated(columns, keep=False).sum())


def validate_run_tables(tables: RunTables) -> RunValidation:
    """Validate keys, schedules, cross-table identity, and projection shapes."""
    errors: list[str] = []
    warnings: list[str] = []
    frames = {
        "items": tables.items,
        "trials": tables.trials,
        "readouts": tables.readouts,
    }
    for name, frame in frames.items():
        missing = sorted(_REQUIRED[name] - set(frame.columns))
        if missing:
            errors.append(f"{name} is missing columns: {', '.join(missing)}")

    concepts: list[str] = []
    if "readout_concept" in tables.readouts:
        concepts = sorted(
            str(value) for value in tables.readouts["readout_concept"].dropna().unique()
        )
    n_concepts = len(concepts)
    n_wrong = max(n_concepts - 1, 0)
    if n_concepts < 2:
        errors.append("readouts must contain at least two readout concepts")
    elif n_wrong not in {9, 29}:
        warnings.append(
            f"inferred {n_wrong} wrong concepts (production v3/v3.1 use 9/29)"
        )

    if _duplicates(tables.items, ["item_id"]):
        errors.append("items.item_id must be unique")
    if _duplicates(tables.trials, ["trial_id"]):
        errors.append("trials.trial_id must be unique")
    if _duplicates(tables.readouts, ["trial_id", "readout_concept"]):
        errors.append("readouts must be unique by (trial_id, readout_concept)")

    if "condition" in tables.trials:
        unknown = sorted(set(tables.trials["condition"].dropna()) - _CONDITIONS)
        if unknown:
            errors.append(f"trials contains unknown conditions: {unknown}")
    if "condition" in tables.readouts:
        unknown = sorted(set(tables.readouts["condition"].dropna()) - _CONDITIONS)
        if unknown:
            errors.append(f"readouts contains unknown conditions: {unknown}")

    if "item_id" in tables.items and "item_id" in tables.trials:
        item_ids = set(tables.items["item_id"])
        unknown = set(tables.trials["item_id"]) - item_ids
        if unknown:
            errors.append(f"trials references {len(unknown)} unknown item_id value(s)")
    if {"trial_id", "item_id"}.issubset(tables.trials.columns) and {
        "trial_id",
        "item_id",
    }.issubset(tables.readouts.columns):
        trial_pairs = tables.trials[["trial_id", "item_id"]].drop_duplicates()
        readout_trial_ids = set(tables.readouts["trial_id"])
        unknown = readout_trial_ids - set(trial_pairs["trial_id"])
        if unknown:
            errors.append(f"readouts references {len(unknown)} unknown trial_id value(s)")
        joined = tables.readouts[["trial_id", "item_id"]].merge(
            trial_pairs,
            on="trial_id",
            how="inner",
            suffixes=("_readout", "_trial"),
        )
        if (joined["item_id_readout"] != joined["item_id_trial"]).any():
            errors.append("readouts item_id disagrees with its trial")

    if n_concepts and {"item_id", "condition", "instructed_concept"}.issubset(
        tables.trials.columns
    ):
        for item_id, group in tables.trials.groupby("item_id", sort=False):
            baseline = group[group["condition"] == "no_instruction"]
            control = group[group["condition"] == "ctrl_think_intensely"]
            think = group[group["condition"] == "think_about"]
            if len(baseline) != 1 or baseline["instructed_concept"].notna().any():
                errors.append(f"item {item_id!r} must have one uninstructed no_instruction trial")
            if len(control) != 1 or control["instructed_concept"].notna().any():
                errors.append(
                    f"item {item_id!r} must have one uninstructed ctrl_think_intensely trial"
                )
            instructed = set(str(value) for value in think["instructed_concept"].dropna())
            if len(think) != n_concepts or instructed != set(concepts):
                errors.append(
                    f"item {item_id!r} must have one think_about trial per concept"
                )

    if n_concepts and {"trial_id", "readout_concept", "projections"}.issubset(
        tables.readouts.columns
    ):
        lengths: set[int] = set()
        malformed = 0
        for value in tables.readouts["projections"]:
            try:
                length = len(value)
            except (TypeError, ValueError):
                malformed += 1
            else:
                lengths.add(length)
        if malformed or not lengths or 0 in lengths or len(lengths) != 1:
            errors.append("readouts.projections must be non-empty, equal-length sequences")
        expected_rows = len(tables.trials) * n_concepts
        if len(tables.readouts) != expected_rows:
            errors.append(
                f"readouts has {len(tables.readouts)} rows; expected {expected_rows}"
            )
        counts = tables.readouts.groupby("trial_id")["readout_concept"].nunique()
        if len(counts) != len(tables.trials) or (counts != n_concepts).any():
            errors.append("each trial must have exactly one readout for every concept")

    return RunValidation(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_items=len(tables.items),
        n_trials=len(tables.trials),
        n_readouts=len(tables.readouts),
        n_concepts=n_concepts,
        n_wrong_concepts=n_wrong,
    )
