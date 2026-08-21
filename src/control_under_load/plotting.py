"""Plot construction and plot-ready source tables for control-under-load results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

EDITORIAL_8 = [
    "#C4650D",
    "#4E728A",
    "#2E6E4E",
    "#988453",
    "#B9605B",
    "#7495AB",
    "#84713A",
    "#31362E",
]
PLOTLY_CONFIG = {
    "responsive": True,
    "displayModeBar": "hover",
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
RUN_COLORS = {"final v3.1": EDITORIAL_8[0], "earlier v3": EDITORIAL_8[1]}
CONDITION_COLORS = {
    "No instruction": EDITORIAL_8[7],
    "Think about X": EDITORIAL_8[0],
    "Think intensely": EDITORIAL_8[1],
}
CATEGORY_COLORS = {
    "below zero": EDITORIAL_8[4],
    "above zero": EDITORIAL_8[2],
    "crosses zero": EDITORIAL_8[3],
}
FIGURE_NAMES = (
    "gap_vs_failure_rate",
    "gap_by_recipe",
    "answer_accuracy_by_instruction",
    "per_concept_slopes",
)


def apply_theme(fig: go.Figure, *, height: int) -> go.Figure:
    """Apply the Editorial 8 report theme with an explicit pixel height."""
    fig.update_layout(
        height=height,
        margin={"t": 40, "r": 24, "b": 24, "l": 24, "autoexpand": True},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#1D272A",
            "family": "'Suisse Intl', -apple-system, BlinkMacSystemFont, Arial, sans-serif",
            "size": 13,
        },
        title=None,
        colorway=EDITORIAL_8,
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#B4B4B4",
            "font": {"family": "ui-monospace, Menlo, monospace", "size": 12},
        },
        legend={
            "orientation": "h",
            "xref": "container",
            "x": 0,
            "xanchor": "left",
            "yref": "container",
            "y": 0,
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
        },
        modebar={
            "orientation": "h",
            "bgcolor": "rgba(0,0,0,0)",
            "color": "#1D272A",
            "activecolor": EDITORIAL_8[0],
            "remove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )
    fig.update_xaxes(automargin=True, showgrid=False, zeroline=False)
    fig.update_yaxes(
        automargin=True,
        showgrid=True,
        gridcolor="rgba(29,39,42,0.12)",
        zeroline=False,
    )
    return fig


def load_analysis(path: str | Path) -> dict[str, Any]:
    """Load one frozen analysis JSON document."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"analysis must be a JSON object: {path}")
    return value


def recipe_labels(config: dict[str, Any]) -> list[str]:
    """Return the configured five recipes in their recorded order."""
    bins = config.get("poly", {}).get("bins", [])
    if len(bins) != 5:
        raise ValueError(f"expected exactly five poly recipes, found {len(bins)}")
    labels = []
    for recipe in bins:
        try:
            labels.append(
                f"m={recipe['m']}, k={recipe['k']}, R={recipe['root_range']}"
            )
        except KeyError as exc:
            raise ValueError(f"recipe is missing {exc.args[0]!r}") from exc
    return labels


def _series(analysis: dict[str, Any], *keys: str) -> dict[str, list[float]]:
    value: Any = analysis
    for key in keys:
        value = value[key]
    required = ("values", "ci_low", "ci_high")
    if not isinstance(value, dict) or any(key not in value for key in required):
        raise ValueError(f"missing recorded series at {'.'.join(keys)}")
    lengths = {len(value[key]) for key in required}
    if lengths != {5}:
        raise ValueError(f"expected five recorded values at {'.'.join(keys)}")
    return value


def gap_failure_rows(
    final: dict[str, Any], comparison: dict[str, Any], labels: list[str]
) -> list[dict[str, Any]]:
    """Build source rows for gap d-prime against baseline failure rate."""
    rows: list[dict[str, Any]] = []
    for run, analysis in (("final v3.1", final), ("earlier v3", comparison)):
        gap = _series(analysis, "curves", "gap")
        accuracy = _series(analysis, "accuracy", "baseline")
        for index, recipe in enumerate(labels):
            rows.append(
                {
                    "run": run,
                    "recipe": recipe,
                    "failure_rate": 1.0 - accuracy["values"][index],
                    # Transforming accuracy reverses the CI endpoints.
                    "failure_ci_low": 1.0 - accuracy["ci_high"][index],
                    "failure_ci_high": 1.0 - accuracy["ci_low"][index],
                    "gap_dprime": gap["values"][index],
                    "gap_ci_low": gap["ci_low"][index],
                    "gap_ci_high": gap["ci_high"][index],
                }
            )
    return rows


def recipe_gap_rows(
    final: dict[str, Any], comparison: dict[str, Any], labels: list[str]
) -> list[dict[str, Any]]:
    """Build source rows for categorical recipe gap and baseline accuracy."""
    rows: list[dict[str, Any]] = []
    for run, analysis in (("final v3.1", final), ("earlier v3", comparison)):
        gap = _series(analysis, "curves", "gap")
        accuracy = _series(analysis, "accuracy", "baseline")
        for index, recipe in enumerate(labels):
            rows.append(
                {
                    "run": run,
                    "recipe": recipe,
                    "gap_dprime": gap["values"][index],
                    "gap_ci_low": gap["ci_low"][index],
                    "gap_ci_high": gap["ci_high"][index],
                    "baseline_accuracy": accuracy["values"][index],
                    "accuracy_ci_low": accuracy["ci_low"][index],
                    "accuracy_ci_high": accuracy["ci_high"][index],
                }
            )
    return rows


def instruction_accuracy_rows(
    final: dict[str, Any], labels: list[str]
) -> list[dict[str, Any]]:
    """Build final-run answer-accuracy rows for the three instructions."""
    rows: list[dict[str, Any]] = []
    conditions = (
        ("No instruction", "baseline"),
        ("Think about X", "think"),
        ("Think intensely", "control"),
    )
    for condition, key in conditions:
        accuracy = _series(final, "accuracy", key)
        for index, recipe in enumerate(labels):
            rows.append(
                {
                    "condition": condition,
                    "recipe": recipe,
                    "accuracy": accuracy["values"][index],
                    "ci_low": accuracy["ci_low"][index],
                    "ci_high": accuracy["ci_high"][index],
                }
            )
    return rows


def concept_slope_rows(final: dict[str, Any]) -> list[dict[str, Any]]:
    """Build sorted final per-concept slope rows and classify each CI."""
    concepts = final.get("per_concept")
    if not isinstance(concepts, dict) or len(concepts) != 30:
        count = len(concepts) if isinstance(concepts, dict) else 0
        raise ValueError(f"expected 30 per-concept results, found {count}")
    rows = []
    for concept, result in concepts.items():
        slope = result["slope"]
        low, high = slope["ci_low"], slope["ci_high"]
        category = (
            "below zero" if high < 0 else "above zero" if low > 0 else "crosses zero"
        )
        rows.append(
            {
                "concept": concept,
                "slope": slope["value"],
                "ci_low": low,
                "ci_high": high,
                "ci_category": category,
            }
        )
    return sorted(rows, key=lambda row: (row["slope"], row["concept"]))


def _asymmetric_error(rows: list[dict[str, Any]], value: str, low: str, high: str):
    return {
        "type": "data",
        "symmetric": False,
        "array": [row[high] - row[value] for row in rows],
        "arrayminus": [row[value] - row[low] for row in rows],
    }


def gap_vs_failure_figure(rows: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for run in RUN_COLORS:
        selected = [row for row in rows if row["run"] == run]
        fig.add_trace(
            go.Scatter(
                x=[row["failure_rate"] for row in selected],
                y=[row["gap_dprime"] for row in selected],
                mode="lines+markers",
                name=run,
                marker={"size": 9, "color": RUN_COLORS[run]},
                line={"color": RUN_COLORS[run]},
                error_x=_asymmetric_error(
                    selected, "failure_rate", "failure_ci_low", "failure_ci_high"
                ),
                error_y=_asymmetric_error(
                    selected, "gap_dprime", "gap_ci_low", "gap_ci_high"
                ),
                customdata=[[row["recipe"]] for row in selected],
                hovertemplate=(
                    "%{customdata[0]}<br>Failure rate: %{x:.3f}"
                    "<br>Gap d′: %{y:.3f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    apply_theme(fig, height=470)
    fig.update_xaxes(title_text="Baseline answer failure rate", tickformat=".0%")
    fig.update_yaxes(title_text="Gap d′")
    return fig


def gap_by_recipe_figure(rows: list[dict[str, Any]], labels: list[str]) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
    )
    for run in RUN_COLORS:
        selected = [row for row in rows if row["run"] == run]
        common = {
            "x": [row["recipe"] for row in selected],
            "mode": "lines+markers",
            "name": run,
            "legendgroup": run,
            "marker": {"size": 8, "color": RUN_COLORS[run]},
            "line": {"color": RUN_COLORS[run]},
        }
        fig.add_trace(
            go.Scatter(
                **common,
                y=[row["gap_dprime"] for row in selected],
                error_y=_asymmetric_error(
                    selected, "gap_dprime", "gap_ci_low", "gap_ci_high"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                **common,
                y=[row["baseline_accuracy"] for row in selected],
                error_y=_asymmetric_error(
                    selected,
                    "baseline_accuracy",
                    "accuracy_ci_low",
                    "accuracy_ci_high",
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    apply_theme(fig, height=620)
    fig.update_xaxes(title_text="Polynomial recipe", categoryorder="array", categoryarray=labels, row=2, col=1)
    fig.update_yaxes(title_text="Gap d′", row=1, col=1)
    fig.update_yaxes(
        title_text="Baseline answer accuracy", tickformat=".0%", range=[0, 1.03], row=2, col=1
    )
    return fig


def instruction_accuracy_figure(
    rows: list[dict[str, Any]], labels: list[str]
) -> go.Figure:
    fig = go.Figure()
    for condition in CONDITION_COLORS:
        selected = [row for row in rows if row["condition"] == condition]
        fig.add_trace(
            go.Scatter(
                x=[row["recipe"] for row in selected],
                y=[row["accuracy"] for row in selected],
                mode="lines+markers",
                name=condition,
                marker={"size": 8, "color": CONDITION_COLORS[condition]},
                line={"color": CONDITION_COLORS[condition]},
                error_y=_asymmetric_error(selected, "accuracy", "ci_low", "ci_high"),
            )
        )
    apply_theme(fig, height=470)
    fig.update_xaxes(title_text="Polynomial recipe", categoryorder="array", categoryarray=labels)
    fig.update_yaxes(title_text="Final answer accuracy", tickformat=".0%", range=[0, 1.03])
    return fig


def concept_slopes_figure(rows: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    for category in CATEGORY_COLORS:
        selected = [row for row in rows if row["ci_category"] == category]
        fig.add_trace(
            go.Scatter(
                x=[row["slope"] for row in selected],
                y=[row["concept"] for row in selected],
                mode="markers",
                name=category,
                marker={"size": 8, "color": CATEGORY_COLORS[category]},
                error_x=_asymmetric_error(selected, "slope", "ci_low", "ci_high"),
            )
        )
    fig.add_vline(x=0, line={"color": "#31362E", "width": 1, "dash": "dot"})
    apply_theme(fig, height=850)
    fig.update_xaxes(title_text="Gap d′ slope per difficulty unit")
    fig.update_yaxes(
        title_text="Concept",
        categoryorder="array",
        categoryarray=[row["concept"] for row in rows],
    )
    return fig


def build_figures(
    final: dict[str, Any], comparison: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, go.Figure], dict[str, list[dict[str, Any]]]]:
    """Build exactly four figures and their small plot-ready source tables."""
    labels = recipe_labels(config)
    tables = {
        "gap_vs_failure_rate": gap_failure_rows(final, comparison, labels),
        "gap_by_recipe": recipe_gap_rows(final, comparison, labels),
        "answer_accuracy_by_instruction": instruction_accuracy_rows(final, labels),
        "per_concept_slopes": concept_slope_rows(final),
    }
    figures = {
        "gap_vs_failure_rate": gap_vs_failure_figure(tables["gap_vs_failure_rate"]),
        "gap_by_recipe": gap_by_recipe_figure(tables["gap_by_recipe"], labels),
        "answer_accuracy_by_instruction": instruction_accuracy_figure(
            tables["answer_accuracy_by_instruction"], labels
        ),
        "per_concept_slopes": concept_slopes_figure(tables["per_concept_slopes"]),
    }
    return figures, tables


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty source table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    final_path: str | Path,
    comparison_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    export_images: bool = True,
) -> dict[str, dict[str, str]]:
    """Write source CSV/JSON and SVG plus 2x PNG renders for all four figures."""
    final = load_analysis(final_path)
    comparison = load_analysis(comparison_path)
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    figures, tables = build_figures(final, comparison, config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    for name in FIGURE_NAMES:
        bundle = destination / name
        bundle.mkdir(parents=True, exist_ok=True)
        csv_path = bundle / "data.csv"
        json_path = bundle / "data.json"
        _write_csv(csv_path, tables[name])
        json_path.write_text(json.dumps(tables[name], indent=2) + "\n", encoding="utf-8")
        files = {"csv": str(csv_path), "json": str(json_path)}
        if export_images:
            svg_path = bundle / f"{name}.svg"
            png_path = bundle / f"{name}.png"
            figures[name].write_image(svg_path, format="svg")
            figures[name].write_image(png_path, format="png", scale=2)
            files.update({"svg": str(svg_path), "png": str(png_path)})
        outputs[name] = files
    return outputs
