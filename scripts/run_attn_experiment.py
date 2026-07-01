#!/usr/bin/env python3
"""Per-trial attention recording for register/sink hypothesis testing.

For each trial (concept, sentence, condition) in the config:
  1. Greedy-generate the sentence transcription.
  2. Single forward pass on (prompt + generated_ids) with output_attentions=True.
  3. For each (layer, query position in the sentence span):
       a. Average attention weights across heads.
       b. Classify each key position by token class.
       c. Sum attention mass into per-class buckets.
  4. Save aggregated mass tables (not raw attentions).

Output: results/raw/attn_<TS>/attention.json with per-trial per-layer per-query
fields { q_token, q_class, q_position_in_span, mass_by_key_class }.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import setup_env
from src.utils.io import get_run_dir
from src.utils.layers import resolve_layers
from src.utils.compliance import check_compliance
from src.config import load_config, parse_cli_args
from src.models.wrapper import load_model
from src.prompts.builder import format_prompt, build_trials
from src.analysis.alignment import tokenize_sentence, align_sentence_span


KEY_CLASSES = ["the", "a", "and", ",", ".", "hello", "special", "content"]


def classify(token_str: str) -> str:
    # Map a token string to one bucket. Structural / register tokens (articles,
    # conjunction, punctuation, the "hello" probe) get their own class; chat-
    # template markers, BOS/PAD, and bare newlines are "special"; everything else
    # is "content". This coarse partition lets us ask which *kinds* of positions
    # attention concentrates on (e.g. attention-sink behavior on structural slots).
    t = token_str.strip().lower()
    if t == "the":
        return "the"
    if t == "a":
        return "a"
    if t == "and":
        return "and"
    if t == ",":
        return ","
    if t == ".":
        return "."
    if t == "hello":
        return "hello"
    # Chat-template markers / bos / pad
    if t.startswith("<") and t.endswith(">"):
        return "special"
    if token_str in ("\n", " \n", "\n\n"):
        return "special"
    return "content"


def main() -> None:
    setup_env()
    args = parse_cli_args()
    config = load_config(args.config, overrides=args.overrides)

    run_dir = get_run_dir(config.output_base_dir, config.name, config.model.name)
    print(f"Output: {run_dir}")

    t0 = time.time()
    wrapper = load_model(
        config.model.name, config.model.device,
        config.model.dtype, config.model.quantization,
        attn_implementation=config.model.attn_implementation,
    )
    print(f"Model ready in {time.time() - t0:.1f}s (n_layers={wrapper.n_layers})")

    attn_layers = resolve_layers(
        config.analysis_layers.fractions, wrapper.n_layers,
        layers=config.analysis_layers.layers,
    )
    print(f"attention layers: {attn_layers}")

    trials = build_trials(
        concepts=config.concepts, sentences=config.sentences,
        conditions=config.prompt_conditions, prompt_layers=[],
        num_repetitions=config.num_repetitions,
    )
    print(f"Total trials: {len(trials)}")

    sentence_ntokens = {
        s: len(tokenize_sentence(wrapper.tokenizer, s)) for s in config.sentences
    }

    device = wrapper._input_device
    eos_id = wrapper.tokenizer.eos_token_id
    pad_id = wrapper.tokenizer.pad_token_id

    results: List[Dict] = []
    n_noncompliant = 0
    t_start = time.time()

    for trial in tqdm(trials, desc="trials"):
        prompt = format_prompt(
            wrapper, trial["template"],
            sentence=trial["sentence"], concept=trial["concept"], layer=None,
        )
        n_sent_tok = sentence_ntokens[trial["sentence"]]
        target_gen = n_sent_tok + config.token_buffer

        # Tokenize prompt (chat template already prepended <bos>; don't add another).
        prompt_enc = wrapper.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_enc["input_ids"]
        attn_mask = prompt_enc["attention_mask"]
        n_prompt = int(prompt_ids.shape[1])

        # Greedy generation. Use a simple loop so we can stop on EOS and bound length.
        gen_ids: List[int] = []
        cur_ids = prompt_ids
        cur_mask = attn_mask
        past = None
        with torch.no_grad():
            out = wrapper.model(
                input_ids=cur_ids, attention_mask=cur_mask,
                use_cache=True, output_attentions=False,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            for _ in range(target_gen):
                next_tok = int(torch.argmax(next_logits, dim=-1).item())
                if eos_id is not None and next_tok == eos_id:
                    break
                gen_ids.append(next_tok)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)], dim=1,
                )
                out = wrapper.model(
                    input_ids=torch.tensor([[next_tok]], device=device),
                    attention_mask=cur_mask,
                    past_key_values=past, use_cache=True, output_attentions=False,
                )
                past = out.past_key_values
                next_logits = out.logits[:, -1, :]

        generated_text = wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)
        is_compliant, sim = check_compliance(
            generated_text, trial["sentence"],
            method=config.compliance.method, threshold=config.compliance.threshold,
        )

        # Align sentence span in the generated portion.
        start, end, align_sim = align_sentence_span(
            wrapper.tokenizer, gen_ids, trial["sentence"], n_sent_tok,
        )

        if not is_compliant:
            n_noncompliant += 1

        # Forward pass with output_attentions=True on full (prompt + generated).
        # NOTE: returning per-head attention matrices requires the model to be
        # loaded with the "eager" attention implementation. Fused kernels
        # (flash-attention / SDPA) never materialize the full T*T weight matrix,
        # so they silently return None for `attentions`; config.model
        # .attn_implementation must therefore be "eager" for this script.
        full_ids = torch.cat(
            [prompt_ids, torch.tensor([gen_ids], device=device, dtype=prompt_ids.dtype)],
            dim=1,
        )
        full_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            attn_out = wrapper.model(
                input_ids=full_ids, attention_mask=full_mask,
                use_cache=False, output_attentions=True,
            )
        attentions = attn_out.attentions  # tuple len=n_layers, each (1, H, T, T)

        # Per-token class labels for every position in the full sequence.
        key_token_strs = [
            wrapper.tokenizer.decode([int(t)], skip_special_tokens=False)
            for t in full_ids[0].tolist()
        ]
        key_classes = [classify(s) for s in key_token_strs]

        q_start_full = n_prompt + start
        q_end_full = n_prompt + end

        # For each analysis layer, bucket each sentence-span query's attention
        # mass by the class of the key it attends to. HOW: average the attention
        # weights over heads, restrict rows to the sentence-span queries, then for
        # each query sum its per-key weights into the bucket of that key's class.
        # The result is a small (n_q x n_classes) mass table per layer instead of
        # the full T*T attention tensor -- compact enough to store as JSON.
        per_layer: Dict[int, List[Dict]] = {}
        for li in attn_layers:
            a = attentions[li][0]  # (H, T, T)
            q_slice = a[:, q_start_full:q_end_full, :]  # (H, n_q, T)
            q_avg = q_slice.mean(dim=0).float().cpu().numpy()  # (n_q, T): head-mean
            per_q = []
            for qi in range(q_avg.shape[0]):
                weights = q_avg[qi]  # (T,): this query's attention over all keys
                bins = {c: 0.0 for c in KEY_CLASSES}
                for ki, cls in enumerate(key_classes):
                    bins[cls] += float(weights[ki])  # accumulate into key's class
                q_tok = key_token_strs[q_start_full + qi]
                per_q.append({
                    "q_token": q_tok,
                    "q_class": classify(q_tok),
                    "q_pos_in_span": qi,
                    "total_mass": float(weights.sum()),
                    "mass_by_key_class": bins,
                })
            per_layer[int(li)] = per_q

        results.append({
            "condition_id": trial["condition_id"],
            "condition_kind": trial["condition_kind"],
            "concept": trial["concept"],
            "sentence": trial["sentence"],
            "alignment_similarity": float(align_sim),
            "is_compliant": bool(is_compliant),
            "compliance_score": float(sim),
            "n_prompt_tokens": n_prompt,
            "n_total_tokens": int(full_ids.shape[1]),
            "sentence_span": [int(start), int(end)],
            "key_classes_summary": {
                c: int(sum(1 for k in key_classes if k == c)) for c in KEY_CLASSES
            },
            "per_layer": per_layer,
        })

        # Free attention tensors before next iter.
        del attentions, attn_out
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nDone: {len(results)} trials in {elapsed:.1f}s ({elapsed / max(len(results),1):.2f}s/trial)")
    print(f"Compliance: {len(results) - n_noncompliant}/{len(results)}")

    out_path = run_dir / "attention.json"
    json.dump({"trials": results, "layers": attn_layers}, open(out_path, "w"))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
