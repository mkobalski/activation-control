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
from src.models.registry import BASE_MODELS, REASONING_MODELS, THINK_TAG_MODELS
from src.utils.harmony import final_channel_span
from src.utils.think_tags import final_answer_span
from src.models.wrapper import load_model
from src.activation.recorder import ActivationRecorder
from src.vectors.extraction import extract_concept_vectors
from src.prompts.builder import format_prompt, build_trials
from src.analysis.alignment import tokenize_sentence, align_sentence_span, slice_activations
from src.analysis.cosine import cosine_traces_per_layer


LT_ONLY_SETS = {"layer_location"}


def _run_name(config):
    """Run-dir label: config.name, plus an `_lt` tag for layer-targeting runs.

    The layer-targeting run and the main battery share ONE config (they differ
    only in active_sets), so without this tag both land in identically-named
    directories -- which is what made `--lt-run` auto-resolution unable to tell
    them apart. Any run whose active_sets is exactly {layer_location} is an LT
    run; everything else keeps config.name unchanged.
    """
    sets = set(getattr(config, "active_sets", None) or [])
    return f"{config.name}_lt" if sets == LT_ONLY_SETS else config.name


def main():
    setup_env()
    args = parse_cli_args()
    config = load_config(args.config, overrides=args.overrides)

    rng = random.Random(config.seed)
    run_dir = get_run_dir(config.output_base_dir, _run_name(config), config.model.name)
    print(f"Output: {run_dir}")

    # Optional sentence subset: run only config.sentence_indices (global indices
    # into the full list). The FULL list is still saved to config.yaml so plot
    # labels stay global (s6 == cat) even in a subset run.
    if config.sentence_indices:
        n = len(config.sentences)
        bad = [i for i in config.sentence_indices if not (0 <= i < n)]
        if bad:
            raise ValueError(f"sentence_indices out of range for {n} sentences: {bad}")
        run_sentences = [config.sentences[i] for i in config.sentence_indices]
        print(f"Sentence subset: running {len(run_sentences)}/{n} "
              f"(indices {config.sentence_indices})")
    else:
        run_sentences = list(config.sentences)

    if config.model.name in BASE_MODELS:
        config.compliance.method = "prefix_normalized_levenshtein"
        print(f"Base model detected — using compliance method '{config.compliance.method}'")

    # Reasoning models (gpt-oss/harmony) emit a chain-of-thought `analysis`
    # channel before the `final` channel that holds the requested sentence. For
    # them we record ALL generated steps (the sentence lands past the CoT, beyond
    # the usual sentence+buffer window) and restrict sentence alignment +
    # compliance to the final channel (see final_channel_span). Non-reasoning
    # models are unchanged.
    is_reasoning = config.model.name in REASONING_MODELS
    # Which final-answer parser to use for reasoning models: think-tag (Qwen3
    # <think>...</think>) vs harmony (gpt-oss channels). Both record all steps and
    # slice the final span; only the span-locating parser differs.
    is_think_tag = config.model.name in THINK_TAG_MODELS
    if is_reasoning:
        style = "think-tag (post-</think> answer)" if is_think_tag else "harmony final channel"
        print(f"Reasoning model detected [{style}] — recording all generated steps "
              f"up to max_new_tokens={config.max_new_tokens}")

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model...")
    t0 = time.time()
    wrapper = load_model(config.model.name, config.model.device,
                         config.model.dtype, config.model.quantization,
                         max_memory=config.model.max_memory)
    # Chat-template reasoning effort (harmony/gpt-oss); read by builder._chat_wrap
    # and extraction.format_extraction_prompt off the wrapper.
    wrapper.reasoning_effort = config.model.reasoning_effort
    # Chat-template thinking toggle (Qwen3 enable_thinking); read by
    # builder._chat_wrap and extraction.format_extraction_prompt off the wrapper.
    wrapper.enable_thinking = config.model.enable_thinking
    print(f"Model ready in {time.time() - t0:.1f}s (n_layers={wrapper.n_layers})")
    if config.model.reasoning_effort:
        print(f"Reasoning effort: {config.model.reasoning_effort}")
    if config.model.enable_thinking is not None:
        print(f"enable_thinking: {config.model.enable_thinking}")

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
                "n_sentences_run": len(run_sentences),
                "n_conditions": len(config.prompt_conditions),
                "active_sets": list(config.active_sets),
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
        sentences=run_sentences,
        conditions=config.prompt_conditions,
        prompt_layers=prompt_layers,
        num_repetitions=config.num_repetitions,
    )
    rng.shuffle(trials)
    print(f"\nTotal trials: {len(trials)}")

    # Pre-tokenize sentence lengths so we know the recording window per trial.
    sentence_ntokens = {
        s: len(tokenize_sentence(wrapper.tokenizer, s)) for s in run_sentences
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
        # Non-reasoning: record just the sentence span + a small buffer (the model
        # writes the sentence immediately). Reasoning: record every generated step
        # up to max_new_tokens, since the sentence sits after a variable-length CoT
        # and we can't know its offset ahead of time. Only the aligned sentence
        # span is ultimately saved either way, so the reasoning path costs transient
        # memory, not pickle size.
        if is_reasoning:
            max_record_per_row = [config.max_new_tokens] * len(batch_trials)
        else:
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
            prompt_special_layers=(analysis_layers
                                   if config.record_special_tokens else None),
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
            # emits before the sentence proper). For reasoning models, restrict the
            # search to the harmony `final` channel span and offset the result back
            # into full-sequence indices, so the recorded activations line up with
            # the sentence as-written (after the CoT), not with the reasoning trace.
            if is_reasoning:
                # Locate the final ANSWER span (excluding the CoT): after the last
                # </think> for think-tag models, or the harmony `final` channel for
                # gpt-oss. Both return (start, end, text) as generated-token indices.
                span_fn = final_answer_span if is_think_tag else final_channel_span
                fin_start, fin_end, final_text = span_fn(
                    wrapper.tokenizer, recorded_gen_ids)
                rel_start, rel_end, align_sim = align_sentence_span(
                    wrapper.tokenizer, recorded_gen_ids[fin_start:fin_end],
                    trial["sentence"], n_sent_tok,
                )
                start, end = fin_start + rel_start, fin_start + rel_end
                compliance_text = final_text
            else:
                start, end, align_sim = align_sentence_span(
                    wrapper.tokenizer, recorded_gen_ids, trial["sentence"], n_sent_tok,
                )
                compliance_text = generated_text
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

            # ---- SPECIAL TOKENS (config.record_special_tokens) ----
            # (a) generated TAIL: everything the recorder captured AFTER the
            #     aligned sentence span — trailing punctuation follow-ons and the
            #     <end_of_turn>/EOS token (recorded before the finished flag,
            #     within the sentence+token_buffer budget). Previously discarded.
            # (b) PROMPT specials: <start_of_turn>/BOS etc., captured at prefill
            #     by generate_batch (prompt_special_layers).
            tail_ids, tail_strs, tail_acts, tail_norms = [], [], {}, {}
            ps_ids, ps_strs, ps_pos, ps_acts, ps_norms = [], [], [], {}, {}
            if config.record_special_tokens:
                tail_ids = recorded_gen_ids[end:n_gen_recorded]
                tail_strs = [wrapper.tokenizer.decode([tid], skip_special_tokens=False)
                             for tid in tail_ids]
                tail_acts = slice_activations(snapshot.activations,
                                              end + 1, n_gen_recorded + 1)
                tail_norms = {li: snapshot.norms[li][end + 1:n_gen_recorded + 1]
                              for li in snapshot.layer_indices}
                psp = out.get("prompt_special")
                if psp is not None:
                    ps_ids = psp["token_ids"]
                    ps_pos = psp["positions"]
                    ps_strs = [wrapper.tokenizer.decode([tid], skip_special_tokens=False)
                               for tid in ps_ids]
                    ps_acts = psp["activations"]
                    ps_norms = {li: np.linalg.norm(a, axis=-1).astype(np.float32)
                                for li, a in ps_acts.items()}

            cos_by_layer = {}
            cos_anchored_by_layer = {}
            cos_tail_by_layer = {}
            cos_ps_by_layer = {}
            if trial["concept"] is not None:
                cv_for_concept = {
                    li: concept_vectors_np[li][trial["concept"]]
                    for li in analysis_layers
                }
                cos_by_layer = cosine_traces_per_layer(cv_for_concept, sentence_acts)
                cos_anchored_by_layer = cosine_traces_per_layer(cv_for_concept, anchored_acts)
                if config.record_special_tokens:
                    cos_tail_by_layer = cosine_traces_per_layer(cv_for_concept, tail_acts)
                    cos_ps_by_layer = cosine_traces_per_layer(cv_for_concept, ps_acts)

            # For reasoning models `compliance_text` is the final-channel text only
            # (CoT excluded); for others it's the full generation. See alignment above.
            is_compliant, compliance_score = check_compliance(
                compliance_text, trial["sentence"],
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
            if is_reasoning:
                # `generated_text` keeps the full generation (CoT + final answer,
                # specials stripped); record the isolated final-answer text + its
                # token span (harmony `final` channel, or post-</think> answer for
                # think-tag models) so downstream can see exactly what was
                # scored/recorded over. Keys stay `final_channel_*` for both styles.
                result["final_channel_text"] = compliance_text
                result["final_channel_span"] = [int(fin_start), int(fin_end)]
            if config.record_special_tokens:
                result.update({
                    "tail_token_ids": tail_ids,
                    "tail_token_strs": tail_strs,
                    "activations_tail": {int(k): v for k, v in tail_acts.items()},
                    "norms_tail": {int(k): np.asarray(v).tolist()
                                   for k, v in tail_norms.items()},
                    "cosine_sim_tail": {int(k): np.asarray(v).tolist()
                                        for k, v in cos_tail_by_layer.items()},
                    "prompt_special_token_ids": ps_ids,
                    "prompt_special_token_strs": ps_strs,
                    "prompt_special_positions": ps_pos,
                    "activations_prompt_special": {int(k): v for k, v in ps_acts.items()},
                    "norms_prompt_special": {int(k): np.asarray(v).tolist()
                                             for k, v in ps_norms.items()},
                    "cosine_sim_prompt_special": {int(k): np.asarray(v).tolist()
                                                  for k, v in cos_ps_by_layer.items()},
                })
            # --no-pickle: the per-trial activation arrays exist only for
            # results.pkl, so drop them immediately to keep host RAM flat
            # (~100-160 GB saved on a full sweep). EXCEPTION: no_instruction
            # rows keep theirs until the end -- the baseline cache is built
            # from them at save time (they are ~50 rows, a few hundred MB).
            if getattr(args, "no_pickle", False) \
                    and trial["condition_id"] != "no_instruction":
                for k in ("activations", "activations_anchored",
                          "activations_tail", "activations_prompt_special"):
                    if k in result:
                        result[k] = "__dropped__"
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
    if getattr(args, "no_pickle", False):
        print("--no-pickle: skipping results.pkl (per-trial activation arrays)")
    else:
        save_results_pickle(all_results, run_dir / "results.pkl",
                            metrics={"compliance_rate": compliance_rate,
                                     "elapsed_s": elapsed},
                            bf16=config.pickle_bf16)
        if config.pickle_bf16:
            print("results.pkl: activations stored in lossless bf16 (uint16 codec)")

    # no_instruction_cache.pkl: the per-sentence BASELINE activations/norms the
    # whole analysis suite keys on (score_data.load_baseline, the figure
    # scripts). Built here directly from the in-memory results, replacing the
    # old extract-from-results.pkl step. Format matches the historical cache:
    # {sentence: {anchored_token_strs, activations {L: (n_tok, d)},
    #             norms {L: (n_tok,)}}}, sentence-window rows (no anchor row).
    #
    # SPECIAL TOKENS (config.record_special_tokens): the baseline's tail
    # (<end_of_turn>/EOS) and prompt-side specials (<bos>/<start_of_turn>/...)
    # are added here too, so special-token analyses (the cos channel needs the
    # baseline projected onto each concept vector) work from this SMALL cache
    # WITHOUT unpickling the multi-GB results.pkl. The no_instruction rows keep
    # their activation arrays in memory even under --no-pickle (they are never
    # dropped), so the raw special-token vectors are available here.
    def _acts(d):   # {L: (n_tok, d)} float32, skipping dropped placeholders
        return {int(L): np.asarray(a, np.float32) for L, a in (d or {}).items()
                if not isinstance(a, str)}

    def _vecs(d):   # {L: (n_tok,)} float32
        return {int(L): np.asarray(v, np.float32) for L, v in (d or {}).items()}

    ni_cache = {}
    for r in all_results:
        if r["condition_id"] != "no_instruction" or not r["is_compliant"]:
            continue
        s = r["sentence"]
        if s in ni_cache:
            continue
        ent = {
            "anchored_token_strs": r["anchored_token_strs"],
            "activations": _acts(r["activations"]),
            "norms": _vecs(r["norms"]),
        }
        if config.record_special_tokens:
            ent.update({
                "tail_token_strs": r.get("tail_token_strs"),
                "activations_tail": _acts(r.get("activations_tail")),
                "norms_tail": _vecs(r.get("norms_tail")),
                "prompt_special_token_strs": r.get("prompt_special_token_strs"),
                "prompt_special_positions": r.get("prompt_special_positions"),
                "activations_prompt_special": _acts(r.get("activations_prompt_special")),
                "norms_prompt_special": _vecs(r.get("norms_prompt_special")),
            })
        ni_cache[s] = ent
    if ni_cache:
        import pickle as _pickle
        with open(run_dir / "no_instruction_cache.pkl", "wb") as f:
            _pickle.dump(ni_cache, f, protocol=_pickle.HIGHEST_PROTOCOL)
        _sp = " (+special-token baseline)" if config.record_special_tokens else ""
        print(f"no_instruction_cache.pkl: {len(ni_cache)} sentences{_sp}")

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
            "n_sentences_run": len(run_sentences),
            "n_conditions": len(config.prompt_conditions),
            "active_sets": list(config.active_sets),
            "condition_sets_defined": sorted(config.condition_sets.keys()),
            # Persist the FULL declared order so the plot scripts label sentences
            # (s0..sN) and order concept panels by config order, not trial order --
            # global even on a subset run. sentence_indices records what was run.
            "sentences": list(config.sentences),
            "sentences_file": config.sentences_file,
            "sentence_indices": list(config.sentence_indices),
            "concepts": list(config.concepts),
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

    # Generation only. All post-processing (scoring + figures + the cross-model
    # superplot) is triggered AFTER completion by the orchestrator, not here --
    # run_experiment.py stays a pure generate-and-save step.
    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    main()
