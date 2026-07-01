#!/usr/bin/env python3
"""Post-hoc analysis: aggregate cosine-sim traces by condition x layer.

Filters to compliant trials, computes mean cos-sim per token position and
overall mean per (condition, layer), and writes a CSV + summary JSON into
the same run directory.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Directory created by run_experiment.py (contains results.pkl)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    results, metrics = load_results(run_dir / "results")
    print(f"Loaded {len(results)} trials (metrics={metrics})")

    # Only compliant trials (the model actually transcribed the sentence) carry
    # interpretable cos-sim traces, so discard the rest up front.
    compliant = [r for r in results if r["is_compliant"]]
    print(f"Compliant: {len(compliant)}/{len(results)}")

    # Collapse each trial's per-layer cos-sim trace (a list over token positions)
    # to one scalar = the token-mean, then group those scalars by
    # (condition_id, prompt_layer, analysis_layer). Each group thus holds one
    # value per contributing trial, ready to be averaged into the summary row.
    # (condition_id, prompt_layer, analysis_layer) -> list of mean cos-sim values
    grouped = defaultdict(list)
    for r in compliant:
        if not r["cosine_sim"]:
            continue
        for li_str, trace in r["cosine_sim"].items():
            li = int(li_str)  # analysis-layer keys are stored as strings in the dict
            if not trace:
                continue
            grouped[(r["condition_id"], r["prompt_layer"], li)].append(float(np.mean(trace)))

    # One CSV row per group: trial count plus mean/std of the per-trial token-means.
    rows = []
    for (cid, pl, ali), vals in grouped.items():
        rows.append({
            "condition_id": cid,
            "prompt_layer": pl,
            "analysis_layer": ali,
            "n_trials": len(vals),
            "mean_cos": float(np.mean(vals)),
            "std_cos": float(np.std(vals)),
        })

    df = pd.DataFrame(rows).sort_values(
        ["condition_id", "prompt_layer", "analysis_layer"]
    )
    out_csv = run_dir / "cosine_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    summary = {
        "n_results": len(results),
        "n_compliant": len(compliant),
        "n_groups": len(grouped),
    }
    with open(run_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
