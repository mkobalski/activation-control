#!/usr/bin/env python3
"""Standalone CLI for control-under-load analysis, verification, and plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "control-under-load" / "data"
DEFAULT_PLOTS = REPO_ROOT / "control-under-load" / "plots"
DEFAULT_ANALYSIS_OUTPUT = REPO_ROOT / "results" / "control-under-load" / "analysis_poly.json"
EXPECTED_HEADLINE = {
    "slope": -0.025452910826112937,
    "ci_low": -0.04275638720116414,
    "ci_high": 0.010929420665996361,
    "baseline_first": 0.9625,
    "baseline_last": 0.10625,
}
EXPECTED_COUNTS = {"items": 800, "trials": 25600, "readouts": 768000}
EXPECTED_CI_CATEGORIES = {"below zero": 13, "above zero": 2, "crosses zero": 15}
EXPECTED_MODEL_REVISION = "005ad3404e59d6023443cb575daa05336842228a"


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entries(path: Path) -> dict[str, str]:
    """Read common JSON or sha256sum manifest forms."""
    if path.suffix == ".json":
        payload = _json(path)
        entries = payload.get("files", payload)
        if isinstance(entries, list):
            return {
                str(item.get("path", item.get("name"))): str(item["sha256"])
                for item in entries
                if isinstance(item, dict)
                and item.get("path", item.get("name"))
                and item.get("sha256")
            }
        if isinstance(entries, dict):
            parsed = {}
            for name, value in entries.items():
                digest = value.get("sha256") if isinstance(value, dict) else value
                if isinstance(digest, str) and len(digest) == 64:
                    parsed[str(name)] = digest
            return parsed
        return {}
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            parsed[parts[1].lstrip("*")] = parts[0]
    return parsed


def _verify_manifest(directory: Path) -> dict[str, Any]:
    candidates = (
        directory / "MANIFEST.json",
        directory / "sha256_manifest.json",
        directory / "manifest.json",
        directory / "SHA256SUMS",
        directory / "sha256sums.txt",
    )
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        return {"present": False, "checked": 0}
    entries = _manifest_entries(manifest)
    failures = []
    for relative, expected in entries.items():
        target = directory / relative
        if not target.is_file():
            failures.append(f"missing manifest file: {relative}")
        elif _sha256(target) != expected.lower():
            failures.append(f"SHA256 mismatch: {relative}")
    if failures:
        raise ValueError("; ".join(failures))
    return {"present": True, "path": str(manifest), "checked": len(entries)}


def _parquet_metadata(path: Path) -> tuple[int, set[str]]:
    """Read only Parquet footer metadata, never the table payload."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("verify requires pyarrow for Parquet footer metadata") from exc
    parquet = pq.ParquetFile(path)
    return parquet.metadata.num_rows, set(parquet.schema_arrow.names)


def _require_recorded_metrics(analysis: dict[str, Any]) -> None:
    required_paths = (
        ("curves", "gap"),
        ("accuracy", "baseline"),
        ("accuracy", "think"),
        ("accuracy", "control"),
        ("slopes", "gap"),
    )
    for keys in required_paths:
        value: Any = analysis
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"missing recorded metric: {'.'.join(keys)}")
            value = value[key]
    for keys in required_paths[:4]:
        value = analysis[keys[0]][keys[1]]
        if any(name not in value for name in ("values", "ci_low", "ci_high")):
            raise ValueError(f"incomplete recorded metric: {'.'.join(keys)}")


def _assert_close(name: str, actual: Any, expected: float) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{name} mismatch: expected {expected!r}, found {actual!r}")


def _ci_category_counts(analysis: dict[str, Any]) -> dict[str, int]:
    concepts = analysis.get("per_concept")
    if not isinstance(concepts, dict) or len(concepts) != 30:
        count = len(concepts) if isinstance(concepts, dict) else 0
        raise ValueError(f"expected 30 recorded per-concept slopes, found {count}")
    counts: Counter[str] = Counter()
    for concept, result in concepts.items():
        try:
            low = result["slope"]["ci_low"]
            high = result["slope"]["ci_high"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing slope CI for concept {concept}") from exc
        category = "below zero" if high < 0 else "above zero" if low > 0 else "crosses zero"
        counts[category] += 1
    return {
        category: counts[category]
        for category in ("below zero", "above zero", "crosses zero")
    }


def verify(final_dir: Path, comparison_dir: Path) -> dict[str, Any]:
    """Validate files, hashes, footer row counts, and recorded aggregates."""
    final_required = (
        "items.parquet",
        "trials.parquet",
        "readouts.parquet",
        "analysis_poly.json",
        "data_review.json",
        "decision_rule.json",
        "qualitative_traces.json",
        "remote_input_provenance.json",
        "run_summary.json",
        "vector_consistency.json",
    )
    comparison_required = (
        "analysis_poly.json",
        "data_review.json",
        "effective_config.json",
        "provenance.json",
        "run_summary.json",
    )
    missing = [str(final_dir / name) for name in final_required if not (final_dir / name).is_file()]
    missing.extend(
        str(comparison_dir / name)
        for name in comparison_required
        if not (comparison_dir / name).is_file()
    )
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))

    analysis = _json(final_dir / "analysis_poly.json")
    comparison = _json(comparison_dir / "analysis_poly.json")
    summary = _json(final_dir / "run_summary.json")
    config = _json(comparison_dir / "effective_config.json")
    _require_recorded_metrics(analysis)
    _require_recorded_metrics(comparison)
    if len(config.get("poly", {}).get("bins", [])) != 5:
        raise ValueError("comparison effective_config.json must record five poly recipes")

    required_columns = {
        "items.parquet": {"item_id", "axis", "task_text", "expected_answer", "difficulty", "difficulty_bin", "source_split", "group_id", "content_hash", "metadata_json"},
        "trials.parquet": {"trial_id", "item_id", "condition", "instructed_concept", "generated_text", "parsed_answer", "parse_valid", "is_correct", "hit_token_limit", "n_generated_tokens"},
        "readouts.parquet": {"trial_id", "item_id", "condition", "instructed_concept", "readout_concept", "projections"},
    }
    metadata = {
        name: _parquet_metadata(final_dir / name) for name in required_columns
    }
    row_counts = {name: value[0] for name, value in metadata.items()}
    for name, (_, columns) in metadata.items():
        missing_columns = required_columns[name] - columns
        if missing_columns:
            raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    expected = {
        "items.parquet": EXPECTED_COUNTS["items"],
        "trials.parquet": EXPECTED_COUNTS["trials"],
        "readouts.parquet": EXPECTED_COUNTS["readouts"],
    }
    mismatches = {
        name: {"expected": expected[name], "actual": actual}
        for name, actual in row_counts.items()
        if expected[name] is None or expected[name] != actual
    }
    if mismatches:
        raise ValueError(f"Parquet row-count mismatch: {mismatches}")

    slope = analysis["slopes"]["gap"]
    _assert_close("headline slope", slope["value"], EXPECTED_HEADLINE["slope"])
    _assert_close("headline CI low", slope["ci_low"], EXPECTED_HEADLINE["ci_low"])
    _assert_close("headline CI high", slope["ci_high"], EXPECTED_HEADLINE["ci_high"])
    baseline = analysis["accuracy"]["baseline"]["values"]
    _assert_close("easiest baseline accuracy", baseline[0], EXPECTED_HEADLINE["baseline_first"])
    _assert_close("hardest baseline accuracy", baseline[-1], EXPECTED_HEADLINE["baseline_last"])
    if analysis.get("gap_definition", {}).get("mismatch_count") != 29:
        raise ValueError("final analysis must use 29 wrong-concept directions")
    if summary.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("model revision does not match the frozen Gemma-3-27B run")
    review = _json(final_dir / "data_review.json")
    for key in ("n_items", "unique_item_ids", "unique_group_ids", "unique_content_hashes"):
        if review.get(key) != EXPECTED_COUNTS["items"]:
            raise ValueError(f"data review {key} mismatch: {review.get(key)!r}")
    categories = _ci_category_counts(analysis)
    if categories != EXPECTED_CI_CATEGORIES:
        raise ValueError(
            f"per-concept CI categories mismatch: expected {EXPECTED_CI_CATEGORIES}, found {categories}"
        )
    manifest = _verify_manifest(final_dir.parent)
    if not manifest["present"] or manifest["checked"] < 15:
        raise ValueError("data/MANIFEST.json must verify every packaged source file")

    return {
        "ok": True,
        "required_files": len(final_required) + len(comparison_required),
        "manifest": manifest,
        "row_counts": row_counts,
        "headline": {**EXPECTED_HEADLINE},
        "ci_category_counts": categories,
    }


def compare_recomputed(
    recomputed: dict[str, Any], reference: dict[str, Any], *, tolerance: float = 1e-12
) -> dict[str, Any]:
    """Compare every saved plot statistic against the deterministic rerun."""
    comparisons: list[tuple[str, Any, Any]] = []
    for field in ("value", "ci_low", "ci_high"):
        comparisons.append(
            (f"slopes.gap.{field}", recomputed["slopes"]["gap"][field], reference["slopes"]["gap"][field])
        )
        comparisons.append(
            (f"peak.{field}", recomputed["peak"][field], reference["peak"][field])
        )
    for metric in ("curves.gap", "accuracy.baseline", "accuracy.think", "accuracy.control"):
        first, second = metric.split(".")
        for field in ("values", "ci_low", "ci_high"):
            actual = recomputed[first][second][field]
            expected = reference[first][second][field]
            if len(actual) != len(expected):
                raise ValueError(f"{metric}.{field} length mismatch")
            comparisons.extend(
                (f"{metric}.{field}[{index}]", left, right)
                for index, (left, right) in enumerate(zip(actual, expected, strict=True))
            )
    for concept in sorted(reference["per_concept"]):
        for field in ("value", "ci_low", "ci_high"):
            comparisons.append(
                (
                    f"per_concept.{concept}.slope.{field}",
                    recomputed["per_concept"][concept]["slope"][field],
                    reference["per_concept"][concept]["slope"][field],
                )
            )
    failures = []
    for name, actual, expected in comparisons:
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
        ):
            failures.append({"name": name, "expected": expected, "actual": actual})
    if failures:
        preview = failures[:5]
        raise ValueError(f"recomputed analysis mismatch ({len(failures)} cells): {preview}")
    return {"within": True, "cells_checked": len(comparisons), "tolerance": tolerance}


def analyze(
    input_dir: Path,
    output_path: Path,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Path:
    """Run the package analysis API on full tables and write JSON."""
    from src.control_under_load.analysis import analyze_run
    from src.control_under_load.io import load_run_tables, validate_run_tables

    tables = load_run_tables(input_dir)
    validation = validate_run_tables(tables)
    validation.raise_for_errors()
    summary = _json(input_dir / "run_summary.json")
    layers = summary.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("run_summary.json must record the analyzed layers")
    concepts = [
        str(value)
        for value in tables.readouts["readout_concept"].dropna().drop_duplicates()
    ]
    result = analyze_run(
        tables,
        layers=layers,
        concepts=concepts,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_path


def plots(final_dir: Path, comparison_dir: Path, output_dir: Path) -> dict[str, Any]:
    from src.control_under_load.plotting import write_plots

    return write_plots(
        final_dir / "analysis_poly.json",
        comparison_dir / "analysis_poly.json",
        comparison_dir / "effective_config.json",
        output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plots", "analyze", "verify", "all"), help="operation"
    )
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_DATA / "final")
    parser.add_argument(
        "--comparison-dir", type=Path, default=DEFAULT_DATA / "comparison-v3"
    )
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument(
        "--analysis-output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports: dict[str, Any] = {}
    if args.command in {"verify", "all"}:
        reports["verify"] = verify(args.final_dir, args.comparison_dir)
    if args.command in {"analyze", "all"}:
        output = analyze(
            args.final_dir,
            args.analysis_output,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        analysis_report: dict[str, Any] = {"path": str(output)}
        if args.n_bootstrap == 2000 and args.seed == 42:
            analysis_report["reproduction"] = compare_recomputed(
                _json(output), _json(args.final_dir / "analysis_poly.json")
            )
        else:
            analysis_report["reproduction"] = {
                "within": None,
                "detail": "comparison requires the frozen 2,000-draw, seed-42 recipe",
            }
        reports["analyze"] = analysis_report
    if args.command in {"plots", "all"}:
        reports["plots"] = plots(args.final_dir, args.comparison_dir, args.plots_dir)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
