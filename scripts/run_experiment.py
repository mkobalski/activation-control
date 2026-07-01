#!/usr/bin/env python3
"""Main runner: think-hard + concept-vector cosine similarity experiment.

Pipeline per trial:
  1. Format prompt from condition template (positive / negative / control /
     baseline; with or without a specific prompt layer).
  2. Generate token-by-token; record residual-stream activations at
     `analysis_layers` for the first `n_sentence_tokens + token_buffer`
     generated tokens.
  3. Align the recorded window to the sentence span and slice activations.
  4. Compute per-token cosine similarity with the (per-layer) concept vector.
  5. Save full activations in pickle; metadata + cos-sim traces in JSON.

Non-compliant trials (generated text doesn't reproduce the target sentence)
are saved WITH a flag; analysis scripts filter on `is_compliant`.
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import setup_env
from src.utils.io import get_run_dir, save_results_json, save_results_pickle
from src.utils.compliance import check_compliance
from src.utils.wandb_utils import init_wandb, log_metrics, log_summary, log_artifact, finish_wandb
from src.utils.layers import resolve_layers
from src.config import load_config, parse_cli_args
from src.models.registry import BASE_MODELS
from src.models.wrapper import load_model
from src.activation.recorder import ActivationRecorder
from src.vectors.extraction import extract_concept_vectors
from src.prompts.builder import format_prompt, build_trials
from src.analysis.alignment import tokenize_sentence, align_sentence_span, slice_activations
from src.analysis.cosine import cosine_traces_per_layer


def main():
    setup_env()
    args = parse_cli_args()
    config = load_config(args.config, overrides=args.overrides)

    rng = random.Random(config.seed)
    run_dir = get_run_dir(config.output_base_dir, config.name, config.model.name)
    print(f"Output: {run_dir}")

    if config.model.name in BASE_MODELS:
        config.compliance.method = "prefix_normalized_levenshtein"
        print(f"Base model detected — using compliance method '{config.compliance.method}'")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    wrapper = load_model(config.model.name, config.model.device,
                         config.model.dtype, config.model.quantization)
    print(f"Model ready in {time.time() - t0:.1f}s (n_layers={wrapper.n_layers})")

    # ── Resolve fractional depths + explicit indices → integer layer ids ─
    analysis_layers = resolve_layers(
        config.analysis_layers.fractions, wrapper.n_layers,
        layers=config.analysis_layers.layers,
    )
    prompt_layers = resolve_layers(
        config.prompt_layers.fractions, wrapper.n_layers,
        layers=config.prompt_layers.layers,
    )
    print(f"analysis_layers = {analysis_layers}")
    print(f"prompt_layers   = {prompt_layers}")

    # ── W&B ───────────────────────────────────────────────────────────────
    if not args.no_wandb:
        init_wandb(
            project=config.wandb.project,
            name=f"{config.name}_{time.strftime('%H%M%S')}",
            config={
                "model": config.model.name,
                "dtype": config.model.dtype,
                "quantization": config.model.quantization,
                "n_layers": wrapper.n_layers,
                "analysis_layers": analysis_layers,
                "prompt_layers": prompt_layers,
                "analysis_fractions": config.analysis_layers.fractions,
                "analysis_layer_overrides": config.analysis_layers.layers,
                "prompt_fractions": config.prompt_layers.fractions,
                "prompt_layer_overrides": config.prompt_layers.layers,
                "n_concepts": len(config.concepts),
                "n_sentences": len(config.sentences),
                "n_conditions": len(config.prompt_conditions),
                "num_repetitions": config.num_repetitions,
                "batch_size": config.batch_size,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "token_buffer": config.token_buffer,
                "compliance_method": config.compliance.method,
                "compliance_threshold": config.compliance.threshold,
                "concept_vector_method": config.concept_vectors.method,
                "seed": config.seed,
            },
            entity=config.wandb.entity,
            tags=config.wandb.tags,
        )

    # ── Concept vectors at every analysis layer ──────────────────────────
    print(f"\nExtracting concept vectors at {len(analysis_layers)} layer(s)...")
    t0 = time.time()
    cv = config.concept_vectors
    concept_vectors_by_layer = extract_concept_vectors(
        model=wrapper,
        concept_words=config.concepts,
        layers=analysis_layers,
        cache_dir=cv.cache_dir,
        extraction_method=cv.method,
        template=cv.template,
        token_idx=cv.token_idx,
        normalize=cv.normalize,
        n_baseline_words=cv.n_baseline_words,
    )
    # Convert concept vectors to numpy once (analysis uses numpy).
    concept_vectors_np = {
        li: {w: v.float().numpy() for w, v in d.items()}
        for li, d in concept_vectors_by_layer.items()
    }
    print(f"Concept vectors ready in {time.time() - t0:.1f}s")

    # ── Register recorder hooks on analysis layers (once) ────────────────
    decoder_layers = wrapper.get_decoder_layers()
    recorder = ActivationRecorder(decoder_layers, analysis_layers)
    recorder.register_hooks()

    # ── Build trial schedule ─────────────────────────────────────────────
    trials = build_trials(
        concepts=config.concepts,
        sentences=config.sentences,
        conditions=config.prompt_conditions,
        prompt_layers=prompt_layers,
        num_repetitions=config.num_repetitions,
    )
    rng.shuffle(trials)
    print(f"\nTotal trials: {len(trials)}")

    # Pre-tokenize sentence lengths so we know the recording window per trial.
    sentence_ntokens = {
        s: len(tokenize_sentence(wrapper.tokenizer, s)) for s in config.sentences
    }

    all_results = []
    t_start = time.time()

    # Group trials into batches of similar prompt length (reduces padding).
    # We sort by sentence length as a cheap proxy; trial order within a batch
    # doesn't matter for the experiment.
    batch_size = max(1, int(config.batch_size))
    trials_sorted = sorted(
        list(enumerate(trials)),
        key=lambda p: sentence_ntokens[p[1]["sentence"]],
    )
    batches = [
        trials_sorted[s:s + batch_size]
        for s in range(0, len(trials_sorted), batch_size)
    ]

    global_step = 0
    for batch in tqdm(batches, desc="Generating (batched)"):
        orig_indices = [oi for oi, _ in batch]
        batch_trials = [t for _, t in batch]

        # Build prompts and per-row recording budgets.
        prompts = [
            format_prompt(
                wrapper, t["template"],
                sentence=t["sentence"], concept=t["concept"],
                layer=t["prompt_layer"],
            )
            for t in batch_trials
        ]
        max_record_per_row = [
            sentence_ntokens[t["sentence"]] + config.token_buffer
            for t in batch_trials
        ]

        t_batch = time.time()
        batch_out = wrapper.generate_batch(
            prompts, recorder,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_record_tokens_per_row=max_record_per_row,
            record_prompt_last_token=True,
        )
        snapshots = recorder.get_snapshots()
        batch_gen_s = time.time() - t_batch

        for row_idx, (orig_i, trial, out, snapshot, prompt) in enumerate(
            zip(orig_indices, batch_trials, batch_out, snapshots, prompts)
        ):
            generated_text = out["text"]
            generated_ids = out["generated_ids"]
            n_prompt_tok = out["n_prompt_tokens"]
            prompt_last_id = out["prompt_last_token_id"]
            n_sent_tok = sentence_ntokens[trial["sentence"]]

            # The recorder captured one extra "anchor" step during prefill (the
            # prompt's last token) before any generated token, so the snapshot
            # has num_tokens = 1 anchor + N generated. Strip the anchor to get the
            # generated ids that line up with recorded positions 1..N.
            n_gen_recorded = max(snapshot.num_tokens - 1, 0)
            recorded_gen_ids = generated_ids[:n_gen_recorded]
            # Find [start, end) within the generated tokens that best matches the
            # target sentence (skips any leading space/quote/preamble the model
            # emits before the sentence proper).
            start, end, align_sim = align_sentence_span(
                wrapper.tokenizer, recorded_gen_ids, trial["sentence"], n_sent_tok,
            )
            # Shift by +1 to convert generated-token indices into snapshot indices
            # (recall snapshot position 0 is the anchor). "anchored" windows keep
            # the anchor at index 0 (start..snap_end); "sentence" windows drop it
            # (snap_start..snap_end) and align 1:1 with the cosine traces.
            snap_start, snap_end = start + 1, end + 1

            anchored_acts = slice_activations(snapshot.activations, start, snap_end)
            anchored_norms = {li: snapshot.norms[li][start:snap_end]
                              for li in snapshot.layer_indices}
            anchored_token_ids = [prompt_last_id] + recorded_gen_ids[start:end]
            anchored_token_strs = [
                wrapper.tokenizer.decode([tid], skip_special_tokens=False)
                for tid in anchored_token_ids
            ]

            sentence_acts = slice_activations(snapshot.activations, snap_start, snap_end)
            sentence_norms = {li: snapshot.norms[li][snap_start:snap_end]
                              for li in snapshot.layer_indices}

            cos_by_layer = {}
            cos_anchored_by_layer = {}
            if trial["concept"] is not None:
                cv_for_concept = {
                    li: concept_vectors_np[li][trial["concept"]]
                    for li in analysis_layers
                }
                cos_by_layer = cosine_traces_per_layer(cv_for_concept, sentence_acts)
                cos_anchored_by_layer = cosine_traces_per_layer(cv_for_concept, anchored_acts)

            is_compliant, compliance_score = check_compliance(
                generated_text, trial["sentence"],
                method=config.compliance.method,
                threshold=config.compliance.threshold,
            )

            result = {
                "trial_idx": orig_i,
                "condition_id": trial["condition_id"],
                "condition_kind": trial["condition_kind"],
                "concept": trial["concept"],
                "sentence": trial["sentence"],
                "prompt_layer": trial["prompt_layer"],
                "rep_idx": trial["rep_idx"],
                "prompt": prompt,
                "generated_text": generated_text,
                "n_prompt_tokens": n_prompt_tok,
                "n_generated_tokens": len(generated_ids),
                "n_recorded_tokens": snapshot.num_tokens,
                "sentence_span": [int(start), int(end)],
                "alignment_similarity": float(align_sim),
                "is_compliant": bool(is_compliant),
                "compliance_score": float(compliance_score),
                "analysis_layers": analysis_layers,
                "prompt_last_token_id": prompt_last_id,
                "anchored_token_ids": anchored_token_ids,
                "anchored_token_strs": anchored_token_strs,
                "activations": {int(k): v for k, v in sentence_acts.items()},
                "activations_anchored": {int(k): v for k, v in anchored_acts.items()},
                "norms": {int(k): v.tolist() for k, v in sentence_norms.items()},
                "norms_anchored": {int(k): v.tolist() for k, v in anchored_norms.items()},
                "cosine_sim": {int(k): v.tolist() for k, v in cos_by_layer.items()},
                "cosine_sim_anchored": {int(k): v.tolist() for k, v in cos_anchored_by_layer.items()},
            }
            all_results.append(result)
            global_step += 1

            # Per-trial W&B metrics — mean cosine per analysis layer.
            per_trial_metrics = {
                "trial/idx": global_step,
                "trial/is_compliant": int(is_compliant),
                "trial/compliance_score": float(compliance_score),
                "trial/alignment_similarity": float(align_sim),
                "trial/n_generated_tokens": len(generated_ids),
                f"trial/condition_kind/{trial['condition_kind']}": 1,
            }
            for li, trace in cos_by_layer.items():
                if len(trace) > 0:
                    per_trial_metrics[f"trial/cos_mean/layer_{li}"] = float(np.mean(trace))
            log_metrics(per_trial_metrics, step=global_step)

        # Per-batch W&B metrics.
        trials_per_s = len(batch) / max(batch_gen_s, 1e-6)
        running_compliance = (
            sum(1 for r in all_results if r["is_compliant"]) / max(len(all_results), 1)
        )
        log_metrics({
            "batch/size": len(batch),
            "batch/gen_time_s": batch_gen_s,
            "batch/trials_per_s": trials_per_s,
            "batch/cumulative_trials": len(all_results),
            "batch/running_compliance_rate": running_compliance,
        }, step=global_step)

    elapsed = time.time() - t_start
    recorder.remove_hooks()

    # ── Summary ───────────────────────────────────────────────────────────
    compliant = [r for r in all_results if r["is_compliant"]]
    compliance_rate = len(compliant) / max(len(all_results), 1)
    print(f"\nDone: {len(all_results)} trials in {elapsed:.1f}s "
          f"({elapsed / max(len(all_results), 1):.2f}s/trial)")
    print(f"Compliance: {len(compliant)}/{len(all_results)} ({compliance_rate:.1%})")

    # ── Save ──────────────────────────────────────────────────────────────
    save_results_json(all_results, run_dir / "results.json",
                      metrics={"compliance_rate": compliance_rate,
                               "elapsed_s": elapsed})
    save_results_pickle(all_results, run_dir / "results.pkl",
                        metrics={"compliance_rate": compliance_rate,
                                 "elapsed_s": elapsed})

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump({
            "name": config.name, "seed": config.seed,
            "model": config.model.name, "n_layers": wrapper.n_layers,
            "analysis_layers": analysis_layers,
            "prompt_layers": prompt_layers,
            "analysis_fractions": config.analysis_layers.fractions,
            "analysis_layer_overrides": config.analysis_layers.layers,
            "prompt_fractions": config.prompt_layers.fractions,
            "prompt_layer_overrides": config.prompt_layers.layers,
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
            "token_buffer": config.token_buffer,
            "num_repetitions": config.num_repetitions,
            "n_concepts": len(config.concepts),
            "n_sentences": len(config.sentences),
            "n_conditions": len(config.prompt_conditions),
            "concept_vector_method": cv.method,
            "compliance_method": config.compliance.method,
            "compliance_threshold": config.compliance.threshold,
        }, f, default_flow_style=False)

    # Per-(kind,layer) mean cosine summary.
    kind_layer_sums: dict = {}
    kind_layer_counts: dict = {}
    for r in compliant:
        kind = r["condition_kind"]
        for li, trace in r["cosine_sim"].items():
            if not trace:
                continue
            key = (kind, int(li))
            kind_layer_sums[key] = kind_layer_sums.get(key, 0.0) + float(np.mean(trace))
            kind_layer_counts[key] = kind_layer_counts.get(key, 0) + 1
    summary = {
        "total_trials": len(all_results),
        "compliant_trials": len(compliant),
        "compliance_rate": compliance_rate,
        "generation_time_s": elapsed,
        "trials_per_s": len(all_results) / max(elapsed, 1e-6),
        "batch_size": config.batch_size,
    }
    for (kind, li), s in kind_layer_sums.items():
        summary[f"mean_cos/{kind}/layer_{li}"] = s / kind_layer_counts[(kind, li)]
    log_summary(summary)
    log_artifact(str(run_dir / "results.json"), name=f"{config.name}_results_json",
                 artifact_type="results")
    log_artifact(str(run_dir / "config.yaml"), name=f"{config.name}_run_config",
                 artifact_type="config")
    finish_wandb()

    # ── Auto-generate controllability heatmaps (CPU-side; best-effort) ─────────
    # Renders the direction (cos) and magnitude (relnorm) heatmaps for the default
    # sentence directly from the in-memory results. Skips gracefully if the run's
    # config lacks the needed sentence/conditions.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from controllability_heatmap import generate_heatmaps
        print("\nGenerating controllability heatmaps...")
        for _metric in ("cos", "relnorm"):
            try:
                generate_heatmaps(run_dir, config.model.name,
                                  vector_cache=cv.cache_dir, method=cv.method,
                                  metric=_metric, results=all_results)
            except Exception as e:
                print(f"  [heatmaps:{_metric}] skipped: {e}")
    except Exception as e:
        print(f"  [heatmaps] unavailable: {e}")

    # ── Auto-generate per-concept trace plots (plot1_cos, plot1_norms) ─────────
    # (Layer-targeting plots 7-12 are NOT auto-run; use scripts/plot_layer_targeting.py.)
    try:
        from plot_results import make_trace_plots
        print("\nGenerating trace plots (plot1_cos, plot1_norms)...")
        make_trace_plots(all_results, run_dir / "plots")
    except Exception as e:
        print(f"  [trace plots] skipped: {e}")

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
