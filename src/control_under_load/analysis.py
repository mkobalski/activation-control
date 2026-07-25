"""Matched control-under-load statistics with clustered bootstrap uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .io import RunTables, validate_run_tables


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("nan")
    return float(np.sum(values[keep] * weights[keep]) / np.sum(weights[keep]))


def _weighted_sd(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    total = float(np.sum(weights[keep]))
    if total <= 1 or np.count_nonzero(weights[keep]) < 3:
        return float("nan")
    mean = np.sum(values[keep] * weights[keep]) / total
    variance = np.sum(weights[keep] * (values[keep] - mean) ** 2) / (total - 1)
    return float(np.sqrt(max(float(variance), 0.0)))


def _weighted_slope(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2:
        return float("nan")
    y, x, weights = y[keep], x[keep], weights[keep]
    x_mean = np.sum(weights * x) / np.sum(weights)
    y_mean = np.sum(weights * y) / np.sum(weights)
    denominator = np.sum(weights * (x - x_mean) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * (x - x_mean) * (y - y_mean)) / denominator)


def _percentile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _metric(value: float, replicas: np.ndarray, n: int) -> dict[str, float | int]:
    return {
        "value": float(value),
        "ci_low": _percentile(replicas, 2.5),
        "ci_high": _percentile(replicas, 97.5),
        "n": int(n),
    }


def _curve(
    values: np.ndarray, replicas: np.ndarray, counts: np.ndarray
) -> dict[str, list[float] | list[int]]:
    return {
        "values": np.asarray(values, dtype=float).tolist(),
        "ci_low": [
            _percentile(replicas[:, index], 2.5) for index in range(len(values))
        ],
        "ci_high": [
            _percentile(replicas[:, index], 97.5) for index in range(len(values))
        ],
        "n": np.asarray(counts, dtype=int).tolist(),
    }


def _normal_concept(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def analyze_run(
    tables: RunTables,
    layers: list[int | float] | tuple[int | float, ...],
    concepts: list[str] | tuple[str, ...],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze a complete run using matched shifts and clustered bootstraps.

    Layer profiles pool concepts with equal weight. The selected layer maximizes
    that pooled profile. Its uncertainty reselects the peak in every independent
    item-by-concept multinomial bootstrap replicate. Per-concept uncertainty is
    item-clustered at the observed pooled peak.
    """
    validation = validate_run_tables(tables)
    validation.raise_for_errors()
    layer_values = list(layers)
    concept_values = [str(value) for value in concepts]
    if not layer_values:
        raise ValueError("layers must not be empty")
    if len(set(layer_values)) != len(layer_values):
        raise ValueError("layers must be unique")
    if len(concept_values) < 2 or len(set(concept_values)) != len(concept_values):
        raise ValueError("concepts must contain at least two unique values")
    observed_concepts = set(str(value) for value in tables.readouts["readout_concept"])
    if set(concept_values) != observed_concepts:
        raise ValueError("concepts must exactly match readouts.readout_concept")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative")

    items = tables.items.reset_index(drop=True)
    axes = sorted(str(value) for value in items["axis"].unique())
    if len(axes) != 1:
        raise ValueError("analyze_run requires items from exactly one axis")
    item_ids = [str(value) for value in items["item_id"]]
    item_index = {value: index for index, value in enumerate(item_ids)}
    difficulties = items["difficulty"].to_numpy(dtype=float)
    bins = sorted(items["difficulty_bin"].unique().tolist())
    bin_index = np.asarray(
        [bins.index(value) for value in items["difficulty_bin"]], dtype=int
    )
    n_items, n_concepts, n_layers = len(items), len(concept_values), len(layer_values)
    concept_index = {value: index for index, value in enumerate(concept_values)}

    trial_rows: dict[str, Any] = {}
    trial_lookup: dict[tuple[int, str, str | None], Any] = {}
    for row in tables.trials.itertuples(index=False):
        item_i = item_index[str(row.item_id)]
        instructed = _normal_concept(row.instructed_concept)
        trial_rows[str(row.trial_id)] = row
        trial_lookup[(item_i, str(row.condition), instructed)] = row

    projections: dict[tuple[int, str, str | None, int], np.ndarray] = {}
    for row in tables.readouts.itertuples(index=False):
        values = np.asarray(row.projections, dtype=float)
        if values.shape != (n_layers,):
            raise ValueError(
                f"trial {row.trial_id!r} has {values.size} projections; "
                f"expected {n_layers}"
            )
        trial = trial_rows[str(row.trial_id)]
        instructed = _normal_concept(row.instructed_concept)
        if instructed != _normal_concept(trial.instructed_concept):
            raise ValueError(f"readout {row.trial_id!r} instructed_concept disagrees with trial")
        if str(row.condition) != str(trial.condition):
            raise ValueError(f"readout {row.trial_id!r} condition disagrees with trial")
        key = (
            item_index[str(row.item_id)],
            str(row.condition),
            instructed,
            concept_index[str(row.readout_concept)],
        )
        projections[key] = values

    baseline = np.empty((n_items, n_concepts, n_layers), dtype=float)
    control = np.empty_like(baseline)
    think = np.empty((n_items, n_concepts, n_concepts, n_layers), dtype=float)
    for item_i in range(n_items):
        for read_i in range(n_concepts):
            baseline[item_i, read_i] = projections[
                (item_i, "no_instruction", None, read_i)
            ]
            control[item_i, read_i] = projections[
                (item_i, "ctrl_think_intensely", None, read_i)
            ]
            for instructed_i, instructed in enumerate(concept_values):
                think[item_i, instructed_i, read_i] = projections[
                    (item_i, "think_about", instructed, read_i)
                ]

    # Each unit is one (item, instructed concept). The baseline contrast is also
    # the noise denominator, so a generic movement shared by all directions is
    # removed before standardization.
    gaps = np.empty((n_items, n_concepts, n_layers), dtype=float)
    baseline_contrasts = np.empty_like(gaps)
    for concept_i in range(n_concepts):
        wrong = np.arange(n_concepts) != concept_i
        matched_shift = think[:, concept_i, concept_i] - baseline[:, concept_i]
        wrong_shifts = think[:, concept_i, wrong].mean(axis=1) - baseline[:, wrong].mean(
            axis=1
        )
        gaps[:, concept_i] = matched_shift - wrong_shifts
        baseline_contrasts[:, concept_i] = baseline[:, concept_i] - baseline[:, wrong].mean(
            axis=1
        )

    ones_i = np.ones(n_items)
    ones_c = np.ones(n_concepts)

    def profile(item_weights: np.ndarray, concept_weights: np.ndarray) -> np.ndarray:
        per_concept = np.full((n_concepts, n_layers), np.nan)
        for concept_i in range(n_concepts):
            for layer_i in range(n_layers):
                sd = _weighted_sd(
                    baseline_contrasts[:, concept_i, layer_i], item_weights
                )
                if np.isfinite(sd) and sd > 0:
                    per_concept[concept_i, layer_i] = _weighted_mean(
                        gaps[:, concept_i, layer_i], item_weights
                    ) / sd
        return np.asarray(
            [
                _weighted_mean(per_concept[:, layer_i], concept_weights)
                for layer_i in range(n_layers)
            ]
        )

    def standardized_at(
        layer_i: int, item_weights: np.ndarray
    ) -> np.ndarray:
        values = np.full((n_items, n_concepts), np.nan)
        for concept_i in range(n_concepts):
            sd = _weighted_sd(
                baseline_contrasts[:, concept_i, layer_i], item_weights
            )
            if np.isfinite(sd) and sd > 0:
                values[:, concept_i] = gaps[:, concept_i, layer_i] / sd
        return values

    def pooled_slope(
        layer_i: int, item_weights: np.ndarray, concept_weights: np.ndarray
    ) -> float:
        values = standardized_at(layer_i, item_weights)
        weights = item_weights[:, None] * concept_weights[None, :]
        return _weighted_slope(
            values.ravel(), np.repeat(difficulties, n_concepts), weights.ravel()
        )

    def pooled_curve(
        layer_i: int, item_weights: np.ndarray, concept_weights: np.ndarray
    ) -> np.ndarray:
        result = np.full(len(bins), np.nan)
        for bin_i in range(len(bins)):
            mask = bin_index == bin_i
            concept_means = np.full(n_concepts, np.nan)
            for concept_i in range(n_concepts):
                sd = _weighted_sd(
                    baseline_contrasts[mask, concept_i, layer_i], item_weights[mask]
                )
                if np.isfinite(sd) and sd > 0:
                    concept_means[concept_i] = _weighted_mean(
                        gaps[mask, concept_i, layer_i] / sd, item_weights[mask]
                    )
            result[bin_i] = _weighted_mean(concept_means, concept_weights)
        return result

    observed_profile = profile(ones_i, ones_c)
    if not np.isfinite(observed_profile).any():
        raise ValueError("no finite d-prime layer profile can be computed")
    peak_i = int(np.nanargmax(observed_profile))
    observed_curve = pooled_curve(peak_i, ones_i, ones_c)
    observed_slope = pooled_slope(peak_i, ones_i, ones_c)

    def accuracy_array(condition: str) -> np.ndarray:
        if condition == "think_about":
            result = np.full((n_items, n_concepts), np.nan)
            for item_i in range(n_items):
                for concept_i, concept in enumerate(concept_values):
                    result[item_i, concept_i] = float(
                        trial_lookup[(item_i, condition, concept)].is_correct
                    )
            return result
        result = np.full(n_items, np.nan)
        for item_i in range(n_items):
            result[item_i] = float(trial_lookup[(item_i, condition, None)].is_correct)
        return result

    accuracy_arrays = {
        "baseline": accuracy_array("no_instruction"),
        "think": accuracy_array("think_about"),
        "control": accuracy_array("ctrl_think_intensely"),
    }

    def accuracy_curve(
        values: np.ndarray, item_weights: np.ndarray, concept_weights: np.ndarray
    ) -> np.ndarray:
        result = np.full(len(bins), np.nan)
        for bin_i in range(len(bins)):
            mask = bin_index == bin_i
            if values.ndim == 1:
                result[bin_i] = _weighted_mean(values[mask], item_weights[mask])
            else:
                weights = item_weights[:, None] * concept_weights[None, :]
                result[bin_i] = _weighted_mean(
                    values[mask].ravel(), weights[mask].ravel()
                )
        return result

    observed_accuracy = {
        name: accuracy_curve(values, ones_i, ones_c)
        for name, values in accuracy_arrays.items()
    }

    rng = np.random.default_rng(seed)
    if n_bootstrap:
        item_draws = rng.multinomial(
            n_items, np.full(n_items, 1 / n_items), size=n_bootstrap
        ).astype(float)
        concept_draws = rng.multinomial(
            n_concepts, np.full(n_concepts, 1 / n_concepts), size=n_bootstrap
        ).astype(float)
    else:
        item_draws = np.empty((0, n_items))
        concept_draws = np.empty((0, n_concepts))
    profile_reps = np.full((n_bootstrap, n_layers), np.nan)
    peak_index_reps = np.full(n_bootstrap, np.nan)
    peak_value_reps = np.full(n_bootstrap, np.nan)
    curve_reps = np.full((n_bootstrap, len(bins)), np.nan)
    slope_reps = np.full(n_bootstrap, np.nan)
    accuracy_reps = {
        name: np.full((n_bootstrap, len(bins)), np.nan) for name in accuracy_arrays
    }
    for boot_i, (item_weights, concept_weights) in enumerate(
        zip(item_draws, concept_draws, strict=True)
    ):
        boot_profile = profile(item_weights, concept_weights)
        profile_reps[boot_i] = boot_profile
        if np.isfinite(boot_profile).any():
            selected = int(np.nanargmax(boot_profile))
            peak_index_reps[boot_i] = selected
            peak_value_reps[boot_i] = boot_profile[selected]
            curve_reps[boot_i] = pooled_curve(selected, item_weights, concept_weights)
            slope_reps[boot_i] = pooled_slope(
                selected, item_weights, concept_weights
            )
        for name, values in accuracy_arrays.items():
            accuracy_reps[name][boot_i] = accuracy_curve(
                values, item_weights, concept_weights
            )

    profile_counts = np.sum(
        np.isfinite(gaps) & np.isfinite(baseline_contrasts), axis=(0, 1)
    )
    curve_counts = np.asarray(
        [
            np.sum(
                np.isfinite(gaps[bin_index == bin_i, :, peak_i])
                & np.isfinite(baseline_contrasts[bin_index == bin_i, :, peak_i])
            )
            for bin_i in range(len(bins))
        ]
    )
    accuracy_counts = {
        name: np.asarray(
            [np.isfinite(values[bin_index == bin_i]).sum() for bin_i in range(len(bins))]
        )
        for name, values in accuracy_arrays.items()
    }

    per_concept: dict[str, Any] = {}
    for concept_i, concept in enumerate(concept_values):
        global_sd = _weighted_sd(baseline_contrasts[:, concept_i, peak_i], ones_i)
        global_values = gaps[:, concept_i, peak_i] / global_sd
        concept_slope = _weighted_slope(global_values, difficulties, ones_i)
        concept_curve = np.asarray(
            [
                _weighted_mean(global_values[bin_index == bin_i], ones_i[bin_index == bin_i])
                for bin_i in range(len(bins))
            ]
        )
        concept_slope_reps = np.full(n_bootstrap, np.nan)
        concept_curve_reps = np.full((n_bootstrap, len(bins)), np.nan)
        for boot_i, item_weights in enumerate(item_draws):
            replicate_sd = _weighted_sd(
                baseline_contrasts[:, concept_i, peak_i], item_weights
            )
            if np.isfinite(replicate_sd) and replicate_sd > 0:
                concept_slope_reps[boot_i] = _weighted_slope(
                    gaps[:, concept_i, peak_i] / replicate_sd,
                    difficulties,
                    item_weights,
                )
            if np.isfinite(replicate_sd) and replicate_sd > 0:
                replicate_values = gaps[:, concept_i, peak_i] / replicate_sd
                for bin_i in range(len(bins)):
                    mask = bin_index == bin_i
                    concept_curve_reps[boot_i, bin_i] = _weighted_mean(
                        replicate_values[mask], item_weights[mask]
                    )
        per_concept[concept] = {
            "curve": _curve(
                concept_curve,
                concept_curve_reps,
                np.asarray(
                    [
                        np.isfinite(gaps[bin_index == bin_i, concept_i, peak_i]).sum()
                        for bin_i in range(len(bins))
                    ]
                ),
            ),
            "slope": _metric(
                concept_slope, concept_slope_reps, int(np.isfinite(global_values).sum())
            ),
            "selected_layer": layer_values[peak_i],
            "bootstrap": "item-only at observed pooled peak",
        }

    pooled_curve_output = _curve(observed_curve, curve_reps, curve_counts)
    pooled_slope_output = _metric(
        observed_slope,
        slope_reps,
        int(np.isfinite(gaps[:, :, peak_i]).sum()),
    )
    layer_profile_output = _curve(observed_profile, profile_reps, profile_counts)
    return {
        "axis": axes[0],
        "layers": layer_values,
        "difficulty_bins": bins,
        "difficulty_scale": "native",
        "gap_definition": {
            "unit": "matched think-minus-baseline shift minus mean wrong-concept shifts",
            "mismatch_count": n_concepts - 1,
            "trend_denominator": "global sample SD of baseline matched-minus-mean-wrong contrast, per concept",
            "curve_denominator": "within-bin sample SD of baseline matched-minus-mean-wrong contrast, per concept",
        },
        "layer_profile": layer_profile_output,
        "layer_profiles": {"gap": layer_profile_output},
        "peak": {
            **_metric(
                observed_profile[peak_i],
                peak_value_reps,
                int(profile_counts[peak_i]),
            ),
            "index": peak_i,
            "layer": layer_values[peak_i],
            "selection_variant": "gap",
            "bootstrap_selected_indices": {
                str(index): int(np.sum(peak_index_reps == index))
                for index in range(n_layers)
                if np.any(peak_index_reps == index)
            },
        },
        "curve": pooled_curve_output,
        "curves": {"gap": pooled_curve_output},
        "slope": pooled_slope_output,
        "slopes": {"gap": pooled_slope_output},
        "per_concept": per_concept,
        "accuracy": {
            name: _curve(
                observed_accuracy[name], accuracy_reps[name], accuracy_counts[name]
            )
            for name in accuracy_arrays
        },
        "n": {
            "items": n_items,
            "concepts": n_concepts,
            "wrong_concepts": n_concepts - 1,
            "bootstrap": n_bootstrap,
        },
    }
