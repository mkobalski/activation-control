#!/usr/bin/env python3
"""Analyze attention.json from run_attn_experiment.py.

Plots:
  plot_attn_emergence.png — for each key-class, mean fraction of attention mass
    landing on that class vs. layer, averaged over all queries in the sentence
    span and all trials (separated by condition).
  plot_attn_qclass.png  — for each query-class × key-class pair, mean attention
    mass at L55 (and at the layer with strongest sink effect).
  plot_attn_delta.png   — Δ(think_intensely − dont_think_about) fraction on each
    key-class, per layer.

Also writes attn_summary.csv with the underlying aggregated numbers.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

KEY_CLASSES = ["the", "a", "and", ",", ".", "hello", "special", "content"]
COLORS = {
    "the": "#c0392b", "a": "#d35400", "and": "#e67e22",
    ",": "#16a085", ".": "#2980b9",
    "hello": "#8e44ad", "special": "#f1c40f", "content": "#34495e",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    data = json.load(open(run / "attention.json"))
    trials = [t for t in data["trials"] if t.get("is_compliant", True)]
    layers = [int(L) for L in data["layers"]]
    print(f"trials: {len(trials)}, layers: {layers}")

    # Aggregate: mean fraction of attention from sentence-span queries to each key class,
    # per (condition, layer). Each query contributes equally; trials are aggregated by
    # condition.
    #
    # PER-QUERY NORMALIZATION: each query row's class masses are divided by that
    # query's own total_mass BEFORE summing, so every query contributes a
    # distribution that sums to 1. WHY: total_mass can drift below 1 (e.g. mass
    # on positions outside the recorded key set, or numerical effects), and we
    # want each query weighted equally regardless of its raw total -- otherwise
    # high-mass queries would dominate the average. counts[cond][li] tracks how
    # many query rows landed in the bucket so we can divide to a mean later.
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    counts = defaultdict(lambda: defaultdict(int))
    for t in trials:
        cond = t["condition_id"]
        for li_s, qs in t["per_layer"].items():
            li = int(li_s)
            for q in qs:
                total = q["total_mass"]
                if total <= 0:
                    continue
                for cls in KEY_CLASSES:
                    sums[cond][li][cls] += q["mass_by_key_class"][cls] / total
                counts[cond][li] += 1

    conditions = sorted(sums.keys())
    print(f"conditions: {conditions}")

    # CSV
    csv_path = run / "attn_summary.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["condition", "layer", "key_class", "n_queries", "mean_fraction"])
        for cond in conditions:
            for li in layers:
                n = counts[cond][li]
                if n == 0:
                    continue
                for cls in KEY_CLASSES:
                    w.writerow([cond, li, cls, n, sums[cond][li][cls] / n])
    print(f"wrote {csv_path}")

    # ---- Plot 1: emergence curve (fraction-on-class vs layer) per condition. ----
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        for cls in KEY_CLASSES:
            ys = []
            for li in layers:
                n = counts[cond][li]
                ys.append(sums[cond][li][cls] / n if n else float("nan"))
            ax.plot(layers, ys, marker="o", color=COLORS[cls], label=cls, linewidth=1.6)
        ax.set_title(f"condition: {cond}")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", ncols=2)
    axes[0].set_ylabel("Mean fraction of attention mass from sentence queries")
    fig.suptitle("Attention mass on each key-class, by layer", fontsize=11)
    plt.tight_layout()
    plt.savefig(run / "plot_attn_emergence.png", dpi=130, bbox_inches="tight")
    print("wrote plot_attn_emergence.png")

    # ---- Plot 2: think − dont Δ per class per layer (if both present). ----
    if "think_intensely" in sums and "dont_think_about" in sums:
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for cls in KEY_CLASSES:
            ys = []
            for li in layers:
                nt = counts["think_intensely"][li]
                nd = counts["dont_think_about"][li]
                if nt and nd:
                    ys.append(
                        sums["think_intensely"][li][cls] / nt
                        - sums["dont_think_about"][li][cls] / nd
                    )
                else:
                    ys.append(float("nan"))
            ax.plot(layers, ys, marker="o", color=COLORS[cls], label=cls, linewidth=1.6)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("Δ fraction (think_intensely − dont_think_about)")
        ax.set_title("Does the prompt shift attention toward register positions?")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncols=2)
        plt.tight_layout()
        plt.savefig(run / "plot_attn_delta.png", dpi=130, bbox_inches="tight")
        print("wrote plot_attn_delta.png")

    # ---- Plot 3: stratify queries by their own class (where do `the`-queries vs
    # `content`-queries attend?). ----
    # Same per-query normalization as above, but now also keyed by q_class so we
    # can compare the attention profile of, e.g., `the`-queries vs content queries.
    # Aggregate per (condition, layer, q_class, key_class).
    qsums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    qcounts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for t in trials:
        cond = t["condition_id"]
        for li_s, qs in t["per_layer"].items():
            li = int(li_s)
            for q in qs:
                total = q["total_mass"]
                if total <= 0:
                    continue
                qc = q["q_class"]
                for cls in KEY_CLASSES:
                    qsums[cond][li][qc][cls] += q["mass_by_key_class"][cls] / total
                qcounts[cond][li][qc] += 1

    q_classes_present = sorted({
        qc for c in conditions for li in layers for qc in qsums[c][li]
        if qcounts[c][li][qc] >= 5  # require a few queries
    })
    # Plot at the deepest layer.
    L_target = layers[-1]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]
    width = 0.8 / len(KEY_CLASSES)
    for ax, cond in zip(axes, conditions):
        xs = np.arange(len(q_classes_present))
        for j, cls in enumerate(KEY_CLASSES):
            ys = [
                qsums[cond][L_target][qc][cls] / qcounts[cond][L_target][qc]
                if qcounts[cond][L_target][qc] else 0
                for qc in q_classes_present
            ]
            ax.bar(xs + j * width, ys, width, color=COLORS[cls], label=cls)
        ax.set_xticks(xs + width * (len(KEY_CLASSES) - 1) / 2)
        ax.set_xticklabels(q_classes_present, rotation=45)
        ax.set_title(f"L{L_target} — {cond}")
        ax.set_xlabel("query class")
    axes[0].set_ylabel("Mean fraction of attention mass on key class")
    axes[-1].legend(fontsize=8, loc="best", ncols=2)
    plt.tight_layout()
    plt.savefig(run / "plot_attn_qclass.png", dpi=130, bbox_inches="tight")
    print("wrote plot_attn_qclass.png")


if __name__ == "__main__":
    main()
