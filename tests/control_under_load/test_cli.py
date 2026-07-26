from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.control_under_load import compare_recomputed


def _analysis() -> dict:
    series = {
        "values": [0.1, 0.2],
        "ci_low": [0.0, 0.1],
        "ci_high": [0.2, 0.3],
    }
    metric = {"value": -0.02, "ci_low": -0.04, "ci_high": 0.01}
    return {
        "slopes": {"gap": metric},
        "peak": {"value": 0.2, "ci_low": 0.1, "ci_high": 0.3},
        "curves": {"gap": series},
        "accuracy": {
            "baseline": series,
            "think": series,
            "control": series,
        },
        "per_concept": {
            "Denim": {"slope": metric},
            "Trees": {"slope": metric},
        },
    }


def test_recomputed_comparison_checks_every_plot_cell() -> None:
    reference = _analysis()
    assert compare_recomputed(deepcopy(reference), reference)["within"] is True

    changed = deepcopy(reference)
    changed["per_concept"]["Trees"]["slope"]["ci_high"] = 0.02
    with pytest.raises(ValueError, match="recomputed analysis mismatch"):
        compare_recomputed(changed, reference)
