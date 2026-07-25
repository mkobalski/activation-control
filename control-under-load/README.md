# Activation control under task load

This directory is the analysis-ready release for the Gemma-3-27B polynomial-load experiment. It contains every final problem, generated answer, correctness score, and concept projection; the frozen v3 comparison; deterministic analysis code; and four publication-ready figures.

## Result

The final v3.1 run estimated a small negative control-gap slope of **−0.02545 d′ per ordered difficulty bin**, with a 95% item × concept bootstrap interval of **[−0.04276, +0.01093]**. The interval includes zero. Its half-width was 0.02684, narrowly missing the preregistered 0.025 powered-null precision target, so the terminal verdict is **zero included, precision target missed**—not evidence that control improves or degrades with load.

Baseline answer accuracy fell from **96.25% to 10.63%** across the five polynomial recipes. The final sample contains **800 unique problems, 25,600 generated answers, 768,000 concept readouts, and 30 concepts**. Of the 30 concept-specific slope intervals, 13 lie below zero, 2 above zero, and 15 cross zero.

## What d′ means here

For item *i*, instructed concept *c*, and layer *l*, the matched-minus-wrong-concept gap is:

```text
gap(i,c,l) = [think-about-c projection on c − baseline projection on c]
             − mean over k≠c [think-about-c projection on k − baseline projection on k]
```

The d′ denominator is the across-item sample standard deviation of the baseline matched-minus-mean-wrong contrast, computed separately for each concept. Difficulty-bin curves recompute that denominator within each bin. The headline slope uses one global denominator per concept, regresses all item × concept units on the native ordered bin index, and selects the pooled peak layer before reporting the slope.

Uncertainty is a deterministic 2,000-draw bootstrap with seed 42. Headline intervals independently resample items and concepts and reselect the pooled peak layer inside every draw. Concept-specific intervals resample items only at the observed pooled peak layer. V3 has 10 concepts and therefore 9 wrong directions per instruction; v3.1 has 30 concepts and 29 wrong directions. These are different estimands and are preserved explicitly.

## Data

### Final v3.1 run: `data/final/`

| file | contents |
|---|---|
| `items.parquet` | 800 problem statements, exact expected answers, ordered difficulty bins, source/group/content identities, and polynomial recipe metadata |
| `trials.parquet` | 25,600 prompts, complete generated answers, exact parsed answers, correctness, parser details, truncation flags, and token counts |
| `readouts.parquet` | 768,000 trial × readout-concept rows; each `projections` cell contains the 20 recorded layer values |
| `analysis_poly.json` | frozen 2,000-draw curves, confidence intervals, peak selection, headline slope, accuracy, and all 30 concept slopes |
| `decision_rule.json` | preregistered precision decision and terminal verdict |
| `data_review.json` | per-bin/cohort counts, uniqueness audit, and examples |
| `run_summary.json` | model revision, counts, layers, parse/truncation rates, and shard provenance |
| `remote_input_provenance.json` | source bundle and ten production-shard identities |
| `vector_consistency.json` | cross-shard concept-vector consistency audit |
| `qualitative_traces.json` | empty placeholder emitted by the final merge; no raw residual tensors are included |

### Earlier v3 comparison: `data/comparison-v3/`

The earlier 400-item run is included as its frozen aggregate analysis plus the effective configuration, data review, run summary, and runtime provenance. Those files are sufficient to reproduce every earlier-run value drawn in the shipped plots. The earlier raw tables are deliberately omitted: they would be needed to re-bootstrap that run or derive new per-concept statistics, but neither operation is part of this comparison.

`data/MANIFEST.json` records every source experiment EID, durable artifact reference, byte size, source object identity, local SHA-256 digest, model revision, generator seed, layers, and row count. `data/SHA256SUMS` is the shell-friendly digest list. The 65.8 MB readout Parquet is committed directly because the repository did not use Git LFS and the file is below GitHub's 100 MB per-file limit; this adds roughly 66 MB to repository history.

## Model and protocol provenance

- Model: `google/gemma-3-27b-it`
- Revision: `005ad3404e59d6023443cb575daa05336842228a`
- Generator/analysis seed: 42
- Selected layers: 3, 6, 9, 12, 15, 18, 21, 24, 27, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61
- Observed pooled peak: layer 55
- Final source experiment: `exp_01kycghv6pe1kvf8beg1vpafw3`
- Comparison source experiment: `exp_01ky9r8c7qfb1sr3rh3m9c2wn5`

The exact polynomial grader is in `src/control_under_load/grader.py`. It requires a final `Final answer:` line, checks symbolic equality, and rejects algebraically equivalent answers that are not written as products of linear factors.

## Reproduce

Install the repository dependencies, then run the bounded integrity gate and regenerate all figures:

```bash
pip install -r requirements.txt
pytest -q tests/control_under_load
ruff check src/control_under_load scripts/control_under_load.py tests/control_under_load
python scripts/control_under_load.py verify
python scripts/control_under_load.py plots
```

The full deterministic CPU reanalysis reads all three Parquet tables and runs the frozen 2,000-draw, seed-42 bootstrap. It writes to the gitignored `results/` tree rather than overwriting the frozen source result, then compares every headline, curve, accuracy, and per-concept slope cell used by the plots:

```bash
python scripts/control_under_load.py analyze
# or verify, reanalyze, compare, and plot in one command:
python scripts/control_under_load.py all
```

For a fast development check, use fewer draws; the CLI intentionally marks that output as non-comparable to the frozen interval:

```bash
python scripts/control_under_load.py analyze --n-bootstrap 20 --analysis-output /tmp/control-load-smoke.json
```

## Plots

`python scripts/control_under_load.py plots` writes one bundle per figure under `plots/`, each with `data.csv`, `data.json`, SVG, and 2× PNG:

1. `gap_vs_failure_rate/`: gap d′ against measured baseline failure rate, with horizontal accuracy uncertainty and vertical gap uncertainty, for final v3.1 and earlier v3.
2. `gap_by_recipe/`: gap d′ across the five literal `(m, k, R)` recipes, with baseline answer accuracy underneath. No invented composite complexity score is used.
3. `answer_accuracy_by_instruction/`: no extra instruction, “think about X,” and generic “think intensely” answer accuracy across the same recipes.
4. `per_concept_slopes/`: all 30 final concept slopes and item-bootstrap intervals, classified by whether their interval lies below, above, or across zero.

No model weights, residual-stream tensors, or new concept directions are included. The projection tables are sufficient to recompute the reported d′ results, but not to rescore the old generations against newly constructed directions.
