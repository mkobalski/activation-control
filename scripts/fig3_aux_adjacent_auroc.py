#!/usr/bin/env python3
"""Fig 3_aux: the numeric dial's RESOLUTION — adjacent-step AUROCs.

Companion to Fig 3 (endpoint-gain d'; feeds the SCORES.md dial-resolution
row): where Fig 3 measures the dial's range, this scores the ADJACENT steps — 1v2, 2v3, 3v4 — asking a different,
harder question: can the readout reliably distinguish NEIGHBORING dial settings?
(the psychophysical just-noticeable-difference view; still consistency-not-
magnitude, but de-saturated and cross-model comparable like all AUROCs).

Identical measures/statistics to Fig 3 (fig3_gain_auroc.py, reused with a
different contrast list): top = paired AUROC on the token-mean cosine (the
higher level vs the SAME unit's lower level; pairing cancels concept offsets);
bottom = pooled AUROC on the concept-agnostic relative norm. Bands = 95%
cluster bootstrap over the 50 sentences (B=2000); ringed = AUROC != 0.5
(paired: per-unit swap; pooled: within-sentence swap; B=5000), BH-FDR across
layers per curve. 0.5 = chance.

CPU-only. Not in the driver — auxiliary to Fig 3.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fig3_gain_auroc import render                                         # noqa: E402

# adjacent intensity steps, light -> dark
ADJACENT = [
    ("s12", "think_intensity_1_of_4", "think_intensity_2_of_4",
     "Intensity 1 → 2", "#e8c468"),
    ("s23", "think_intensity_2_of_4", "think_intensity_3_of_4",
     "Intensity 2 → 3", "#d4a017"),
    ("s34", "think_intensity_3_of_4", "think_intensity_4_of_4",
     "Intensity 3 → 4", "#8a670e"),
]

FOOTNOTE = ("Companion to Fig 3 (endpoint-gain d') and the SCORES.md dial-resolution row: ADJACENT-step AUROCs of the numeric intensity ramp — the dial's resolution "
            "(can neighboring settings be told apart?), not its range.  Measures/statistics as the retired AUROC Fig 3: "
            "paired AUROC on cosine (per-unit pairing), pooled AUROC on relnorm; bands = 95% cluster bootstrap over "
            "sentences (B=2000); ringed = AUROC≠0.5 (swap permutations, BH-FDR across layers, B=5000).")


def main():
    ap = argparse.ArgumentParser(description="Fig 3_aux (draft): adjacent-step AUROC depth profiles (1v2, 2v3, 3v4).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="fig3_aux_adjacent_auroc.png")
    args = ap.parse_args()
    render(args.run_dir, out=args.out, alpha=args.alpha, contrasts=ADJACENT,
           footnote=FOOTNOTE)


if __name__ == "__main__":
    main()
