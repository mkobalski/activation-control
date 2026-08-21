from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.control_under_load.plotting import (
    FIGURE_NAMES,
    PLOTLY_CONFIG,
    build_figures,
    load_analysis,
    recipe_labels,
    write_plots,
)


def _record(values: list[float], width: float = 0.05) -> dict[str, list[float]]:
    return {
        "values": values,
        "ci_low": [value - width for value in values],
        "ci_high": [value + width for value in values],
    }


def _analysis(*, final: bool) -> dict:
    baseline = [0.95, 0.8, 0.6, 0.4, 0.1]
    gap = [0.30, 0.24, 0.18, 0.14, 0.10]
    analysis = {
        "curves": {"gap": _record(gap, 0.07)},
        "accuracy": {
            "baseline": _record(baseline, 0.03),
            "think": _record([value - 0.01 for value in baseline], 0.04),
            "control": _record([value + 0.01 for value in baseline], 0.04),
        },
    }
    if final:
        concepts = {}
        for index in range(30):
            value = (index - 15) / 100
            if index < 10:
                low, high = value - 0.03, min(value + 0.03, -0.001)
            elif index < 20:
                low, high = min(value - 0.03, -0.001), max(value + 0.03, 0.001)
            else:
                low, high = max(value - 0.03, 0.001), value + 0.03
            concepts[f"Concept {index:02d}"] = {
                "slope": {"value": value, "ci_low": low, "ci_high": high}
            }
        analysis["per_concept"] = concepts
    return analysis


def _config() -> dict:
    return {
        "poly": {
            "bins": [
                {"m": 3, "k": 5, "root_range": 10},
                {"m": 4, "k": 4, "root_range": 12},
                {"m": 5, "k": 3, "root_range": 10},
                {"m": 5, "k": 4, "root_range": 12},
                {"m": 5, "k": 5, "root_range": 16},
            ]
        }
    }


def test_builds_exactly_four_figures_and_source_tables() -> None:
    figures, tables = build_figures(_analysis(final=True), _analysis(final=False), _config())

    assert tuple(figures) == FIGURE_NAMES
    assert tuple(tables) == FIGURE_NAMES
    assert [len(tables[name]) for name in FIGURE_NAMES] == [10, 10, 15, 30]
    assert all(figure.layout.height for figure in figures.values())
    assert all(figure.layout.title.text is None for figure in figures.values())
    assert all(trace.type == "scatter" for figure in figures.values() for trace in figure.data)
    assert PLOTLY_CONFIG["toImageButtonOptions"]["scale"] == 2

    failure = tables["gap_vs_failure_rate"][0]
    assert failure["failure_rate"] == pytest.approx(0.05)
    assert failure["failure_ci_low"] == pytest.approx(0.02)
    assert failure["failure_ci_high"] == pytest.approx(0.08)
    assert {row["run"] for row in tables["gap_by_recipe"]} == {
        "final v3.1",
        "earlier v3",
    }
    assert {row["condition"] for row in tables["answer_accuracy_by_instruction"]} == {
        "No instruction",
        "Think about X",
        "Think intensely",
    }
    assert {row["ci_category"] for row in tables["per_concept_slopes"]} == {
        "below zero",
        "above zero",
        "crosses zero",
    }
    assert [row["slope"] for row in tables["per_concept_slopes"]] == sorted(
        row["slope"] for row in tables["per_concept_slopes"]
    )


def test_recipe_labels_are_categorical_and_ordered() -> None:
    labels = recipe_labels(_config())
    assert labels == [
        "m=3, k=5, R=10",
        "m=4, k=4, R=12",
        "m=5, k=3, R=10",
        "m=5, k=4, R=12",
        "m=5, k=5, R=16",
    ]
    with pytest.raises(ValueError, match="exactly five"):
        recipe_labels({"poly": {"bins": []}})


def test_temporary_json_inputs_write_small_sources_without_static_export(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "final.json"
    comparison_path = tmp_path / "comparison.json"
    config_path = tmp_path / "config.json"
    final_path.write_text(json.dumps(_analysis(final=True)), encoding="utf-8")
    comparison_path.write_text(json.dumps(_analysis(final=False)), encoding="utf-8")
    config_path.write_text(json.dumps(_config()), encoding="utf-8")

    assert load_analysis(final_path)["accuracy"]["baseline"]["values"][0] == 0.95
    outputs = write_plots(
        final_path,
        comparison_path,
        config_path,
        tmp_path / "plots",
        export_images=False,
    )

    assert tuple(outputs) == FIGURE_NAMES
    for files in outputs.values():
        assert set(files) == {"csv", "json"}
        assert Path(files["csv"]).is_file()
        assert Path(files["json"]).is_file()
