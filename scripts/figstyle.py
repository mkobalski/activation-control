"""Shared plotting style + tiny helpers for the RUNTIME EXPLORATORY figures.

Exploratory = quick per-model capability glance produced at run time (see
scripts/explore.py). Unlike the strict AAAI paper figures (scripts/*_a3i.py ->
/workspace/Figures_A3I/, no titles/spines), these keep titles, axis labels and
legends ON so a run's folder is readable at a glance.

Deliberately depends on NOTHING under scripts/ except the standard scoring layer
(score_data / compute_scores) -- none of the retired figure/plotting scripts.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,      # embed TrueType, not Type-3
    "savefig.dpi": 150, "figure.dpi": 150,       # exploratory: lighter than the 300 paper dpi
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
})
import matplotlib.pyplot as plt                                          # noqa: E402

# ---- condition palette (shared with the paper figures for visual continuity) ----
ENGAGE_C, SUPPRESS_C = "#c0392b", "#2471a3"
MED_GRAY, LIGHT_GRAY = (0.45, 0.45, 0.45), (0.78, 0.78, 0.78)
LEX_C, NUM_C = "#2e7d32", "#f9a825"             # lexical (green) vs numeric (yellow) contrasts
REDS = plt.get_cmap("Reds")                     # intensity ramp light->dark
DIVERGE = "RdBu_r"                              # heatmaps: red = above baseline, blue = below

# The experiment sweeps depth in these requested fractions (5%..100%). Depth axes
# label with these, NOT 100*L/n_layers, so the same depth reads identically across
# models (see figure-label conventions).
FRACS_PCT = list(range(5, 101, 5))


def depth_pcts(layers, n_total):
    """Depth-% per sorted analysis layer = the requested fraction (multiple of 5)."""
    if len(layers) == len(FRACS_PCT):
        return list(FRACS_PCT)
    return [int(round(100 * L / n_total / 5) * 5) for L in layers]


def save(fig, path):
    """Write a figure (png; +pdf if path ends .pdf) and close it. Returns the path."""
    fig.savefig(str(path), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return str(path)
