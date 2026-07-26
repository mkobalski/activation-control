from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.control_under_load import (
    RunTables,
    analyze_run,
    load_run_tables,
    validate_run_tables,
)


CONCEPTS = ["alpha", "beta", "gamma"]
LAYERS = [4, 8]


def toy_tables() -> RunTables:
    items = []
    trials = []
    readouts = []
    coefficients = np.asarray([1.0, 2.0, 4.0])
    contrast_coefficients = coefficients - np.asarray([3.0, 2.5, 1.5])
    for item_i in range(6):
        item_id = f"item-{item_i}"
        difficulty = float(item_i)
        items.append(
            {
                "item_id": item_id,
                "axis": "poly",
                "difficulty": difficulty,
                "difficulty_bin": item_i // 3,
            }
        )
        specs = [("no_instruction", None), ("ctrl_think_intensely", None)]
        specs.extend(("think_about", concept) for concept in CONCEPTS)
        baseline = coefficients * (item_i + 1.0)
        for condition, instructed in specs:
            suffix = instructed or "none"
            trial_id = f"{item_id}-{condition}-{suffix}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "item_id": item_id,
                    "condition": condition,
                    "instructed_concept": instructed,
                    "is_correct": (item_i + (0 if instructed is None else 1)) % 2 == 0,
                }
            )
            for read_i, readout_concept in enumerate(CONCEPTS):
                values = baseline.copy()
                projections = np.asarray([values[read_i], values[read_i]])
                if condition == "ctrl_think_intensely":
                    projections += 0.25
                elif condition == "think_about":
                    instructed_i = CONCEPTS.index(instructed)
                    if read_i == instructed_i:
                        scale = abs(contrast_coefficients[instructed_i])
                        projections += [scale, scale * (2.0 + 0.5 * difficulty)]
                readouts.append(
                    {
                        "trial_id": trial_id,
                        "item_id": item_id,
                        "condition": condition,
                        "instructed_concept": instructed,
                        "readout_concept": readout_concept,
                        "projections": projections.tolist(),
                    }
                )
    return RunTables(pd.DataFrame(items), pd.DataFrame(trials), pd.DataFrame(readouts))


def test_load_and_validate_tables(tmp_path):
    tables = toy_tables()
    tables.items.to_parquet(tmp_path / "items.parquet", engine="pyarrow")
    tables.trials.to_parquet(tmp_path / "trials.parquet", engine="pyarrow")
    tables.readouts.to_parquet(tmp_path / "readouts.parquet", engine="pyarrow")

    loaded = load_run_tables(tmp_path)
    validation = validate_run_tables(loaded)

    assert validation.valid
    assert validation.n_items == 6
    assert validation.n_concepts == 3
    assert validation.n_wrong_concepts == 2
    assert validation.warnings


def test_validation_rejects_incomplete_schedule():
    tables = toy_tables()
    broken = RunTables(tables.items, tables.trials.iloc[:-1], tables.readouts)
    validation = validate_run_tables(broken)
    assert not validation.valid
    assert any("think_about" in error for error in validation.errors)


def test_analysis_uses_matched_gap_equal_concept_pool_and_peak_reselection():
    result = analyze_run(toy_tables(), LAYERS, CONCEPTS, n_bootstrap=80, seed=7)

    global_sd = np.std(np.arange(1.0, 7.0), ddof=1)
    assert result["layer_profile"]["values"] == pytest.approx(
        [1.0 / global_sd, 3.25 / global_sd]
    )
    assert result["peak"]["index"] == 1
    assert result["peak"]["layer"] == 8
    assert result["slope"]["value"] == pytest.approx(0.5 / global_sd)
    assert result["gap_definition"]["mismatch_count"] == 2
    assert sum(result["peak"]["bootstrap_selected_indices"].values()) <= 80
    assert set(result["accuracy"]) == {"baseline", "think", "control"}
    assert all(
        value["bootstrap"] == "item-only at observed pooled peak"
        for value in result["per_concept"].values()
    )


def test_curves_use_within_bin_sample_sd_and_bootstrap_is_deterministic():
    first = analyze_run(toy_tables(), LAYERS, CONCEPTS, n_bootstrap=30, seed=123)
    second = analyze_run(toy_tables(), LAYERS, CONCEPTS, n_bootstrap=30, seed=123)

    bin_sd = np.std(np.arange(1.0, 4.0), ddof=1)
    assert first["curve"]["values"][0] == pytest.approx(2.5 / bin_sd)
    assert first["curve"]["values"][1] == pytest.approx(4.0 / bin_sd)
    assert first["peak"] == second["peak"]
    assert first["curve"] == second["curve"]
