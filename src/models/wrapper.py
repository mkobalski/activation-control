"""Model wrapper: loading, activation extraction, and token-by-token generation.

Adapted from introspection-master's wrapper. Unused features (steering hooks,
batched multi-steering) have been removed; we only need:

  * extract_activations(prompts, layer_idx) -- for concept vector extraction
  * generate_token_by_token(prompt, recorder) -- for per-token activation recording

The tokenizer's chat template is used by the prompt builders (src/prompts).
"""

import gc
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.registry import MODEL_NAME_MAP, GEMMA_MODELS


class ModelWrapper:
    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 quantization_config: Optional[BitsAndBytesConfig] = None,
                 attn_implementation: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.hf_path = MODEL_NAME_MAP.get(model_name, model_name)

        print(f"Loading model: {self.hf_path}"
              + (f" (attn={attn_implementation})" if attn_implementation else ""))

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "pretrained_model_name_or_path": self.hf_path,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        except Exception:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype", dtype)
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        self._apply_patches()
        self.n_layers = self._get_n_layers()
        print(f"Model loaded. Layers: {self.n_layers}")

    @property
    def _input_device(self):
        return next(self.model.parameters()).device

    def _get_n_layers(self) -> int:
        if hasattr(self.model, "model"):
            m = self.model.model
            if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
                return len(m.language_model.layers)
            if hasattr(m, "layers"):
                return len(m.layers)
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            return cfg.text_config.num_hidden_layers
        raise ValueError(f"Cannot determine layer count for {self.model_name}")

    def get_decoder_layers(self):
        """Return the ModuleList of decoder layers (language-model layers for Gemma)."""
        if hasattr(self.model, "model"):
            m = self.model.model
            if hasattr(m, "language_model") and hasattr(m.language_model, "layers"):
                return m.language_model.layers
            if hasattr(m, "layers"):
                return m.layers
        raise ValueError(f"Cannot access decoder layers for {self.model_name}")

    def get_layer_module(self, layer_idx: int):
        return self.get_decoder_layers()[layer_idx]

    def _apply_patches(self):
        # Gemma rotary-emb shape fix (copied from introspection-master). Family
        # detected from the short name; families whose modeling module is absent
        # or already fixed upstream are skipped with a note instead of crashing.
        if self.model_name in GEMMA_MODELS:
            mod_name = next((m for m in ("gemma4", "gemma3", "gemma2")
                             if m in self.model_name), "gemma2")
            # gemma4 rewrote apply_rotary_pos_emb to a per-tensor signature
            # (x, cos, sin, ...) instead of gemma2/3's (q, k, cos, sin, ...);
            # the legacy shape fix below matches only the old signature, and
            # gemma4's upstream rotary is already correct, so skip patching it.
            if mod_name == "gemma4":
                print(f"[patch] skipping Gemma rotary fix for {self.model_name}: "
                      "gemma4 uses upstream per-tensor apply_rotary_pos_emb")
                return
            try:
                gemma_module = __import__(
                    f"transformers.models.{mod_name}.modeling_{mod_name}",
                    fromlist=["apply_rotary_pos_emb"],
                )
                if not hasattr(gemma_module, "apply_rotary_pos_emb") \
                        or not hasattr(gemma_module, "rotate_half"):
                    raise ImportError("no rotary symbols to patch")
            except ImportError as e:
                print(f"[patch] skipping Gemma rotary fix for {self.model_name}: {e}")
                return

            def fixed(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                cos = cos.unsqueeze(unsqueeze_dim)
                sin = sin.unsqueeze(unsqueeze_dim)
                if cos.shape[-1] != q.shape[-1]:
                    cos = cos[..., :q.shape[-1]]
                    sin = sin[..., :q.shape[-1]]
                q_embed = (q * cos) + (gemma_module.rotate_half(q) * sin)
                k_embed = (k * cos) + (gemma_module.rotate_half(k) * sin)
                return q_embed, k_embed

            gemma_module.apply_rotary_pos_emb = fixed
            print(f"Applied Gemma rotary fix for {self.model_name}")

    @contextmanager
    def _hook_ctx(self, layer_idx: int, hook_fn):
        handle = self.get_layer_module(layer_idx).register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    def extract_activations(self, prompts: List[str], layer_idx: int,
                            token_idx: int = -1) -> torch.Tensor:
        """Single forward pass; grab the hidden state at `token_idx` from `layer_idx`."""
        activations = []

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            activations.append(h[:, token_idx, :].detach().cpu())

        with self._hook_ctx(layer_idx, hook):
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                    truncation=True).to(self._input_device)
            with torch.no_grad():
                self.model(**inputs, use_cache=False)
        return torch.cat(activations, dim=0)

    def generate_batch(self, prompts: List[str], recorder,
                       max_new_tokens: int = 64,
                       temperature: float = 0.0,
                       max_record_tokens_per_row: Optional[List[int]] = None,
                       record_prompt_last_token: bool = True,
                       prompt_special_layers: Optional[List[int]] = None):
        """Batched token-by-token generation with per-step activation recording.

        Returns a list of per-row dicts with keys:
            text, generated_ids, n_prompt_tokens, prompt_last_token_id
            [+ prompt_special if `prompt_special_layers` is given]
        Recorder snapshots (one per row) are obtained via `recorder.get_snapshots()`.

        `max_record_tokens_per_row[i]` bounds how many generated-token steps
        are recorded for row i (the prompt-last token is always recorded if
        `record_prompt_last_token`). Recording stops per-row once its budget
        is exhausted, but generation continues until all rows finish or hit
        `max_new_tokens`.

        `prompt_special_layers`: if given, the PREFILL forward additionally
        captures the residual stream at every prompt position holding a special
        token (<start_of_turn>, BOS, ...) at those layers. Each row's dict then
        carries prompt_special = {token_ids, positions (0-based within the
        unpadded prompt), activations {layer: (n_special, d) float32}}.
        """
        self.model.eval()
        B = len(prompts)
        recorder.reset(batch_size=B)

        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        device = self._input_device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # With padding_side="left", the last column is each row's real last token.
        prompt_last_token_ids = [int(input_ids[i, -1].item()) for i in range(B)]
        # Per-row prompt token count (excluding left pad).
        n_prompt_per_row = attention_mask.sum(dim=1).tolist()

        # ---- prompt-side special tokens: locate their (row, col) positions ----
        ps_rows, ps_cols, ps_per_row, ps_captured, ps_handles = [], [], None, {}, []
        if prompt_special_layers:
            special_ids = set(self.tokenizer.all_special_ids or [])
            L_pad = input_ids.shape[1]
            ids_cpu = input_ids.cpu()
            att_cpu = attention_mask.cpu()
            ps_per_row = []
            for i in range(B):
                cols = [j for j in range(L_pad)
                        if int(att_cpu[i, j]) == 1 and int(ids_cpu[i, j]) in special_ids]
                ps_per_row.append(cols)
                ps_rows.extend([i] * len(cols))
                ps_cols.extend(cols)

            def _make_ps_hook(li):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if ps_rows:
                        ps_captured[li] = (h[ps_rows, ps_cols, :]
                                           .detach().float().cpu().numpy())
                return hook
            for li in prompt_special_layers:
                ps_handles.append(
                    self.get_layer_module(li).register_forward_hook(_make_ps_hook(li)))

        # Stop on the model's FULL end-of-turn / eos set, not just the tokenizer's
        # single eos_token_id. Chat models like gemma4 end a turn with a custom
        # token (<turn|>=106) distinct from <eos>(=1); their generation_config
        # lists them all (e.g. [1, 106, 50]). Recognizing only <eos> lets
        # generation run past the turn end, where the model re-opens a thought
        # channel and repeats the sentence -- which wrecks compliance.
        gc_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(gc_eos, int):
            gc_eos = [gc_eos]
        eos_ids = {int(x) for x in (gc_eos or [])}
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(int(self.tokenizer.eos_token_id))
        eos_ids_tensor = (torch.tensor(sorted(eos_ids), device=device)
                          if eos_ids else None)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
            else self.tokenizer.eos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated_ids: List[List[int]] = [[] for _ in range(B)]
        recorded_steps = [0] * B
        if max_record_tokens_per_row is None:
            max_record_tokens_per_row = [max_new_tokens] * B

        with torch.no_grad():
            # Prefill.
            if record_prompt_last_token:
                recorder.set_active_mask(np.ones(B, dtype=bool))
                recorder.start_recording()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=True)
            if record_prompt_last_token:
                recorder.stop_recording()
            for h in ps_handles:                    # prefill-only capture
                h.remove()
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            # Per-step flow: (1) pick next token, (2) force pad on finished rows,
            # (3) compute this step's active mask (live & under budget), (4) record
            # while running the forward pass, (5) update finished flags AFTER
            # recording so the EOS token itself is captured. Generation continues
            # until all rows hit EOS or max_new_tokens, even after recording stops.
            for _ in range(max_new_tokens):
                if temperature == 0.0:
                    next_ids = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B,1)
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_ids = torch.multinomial(probs, num_samples=1)

                # Force pad for finished rows so they don't emit new content.
                next_ids = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_ids, pad_id),
                    next_ids,
                )

                # Record per-row active mask for THIS step:
                #   active iff not previously finished AND still within budget.
                active_mask = np.array([
                    (not bool(finished[i].item()))
                    and (recorded_steps[i] < max_record_tokens_per_row[i])
                    for i in range(B)
                ], dtype=bool)

                # Append generated id (skip for already-finished rows).
                toks = next_ids.squeeze(-1).tolist()
                for i, t in enumerate(toks):
                    if not bool(finished[i].item()):
                        generated_ids[i].append(int(t))

                # Update attention mask.
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((B, 1), device=device, dtype=attention_mask.dtype),
                ], dim=1)

                do_record = bool(active_mask.any())
                if do_record:
                    recorder.set_active_mask(active_mask)
                    recorder.start_recording()
                outputs = self.model(input_ids=next_ids, attention_mask=attention_mask,
                                     past_key_values=past, use_cache=True)
                if do_record:
                    recorder.stop_recording()
                    for i in range(B):
                        if active_mask[i]:
                            recorded_steps[i] += 1

                past = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]

                # Update finished flags (post-record so EOS itself is captured).
                if eos_ids_tensor is not None:
                    just_ended = torch.isin(next_ids.squeeze(-1), eos_ids_tensor) & (~finished)
                    finished = finished | just_ended
                if bool(finished.all().item()):
                    break

        # Decode per-row; strip trailing EOS if present.
        results = []
        for i in range(B):
            ids = generated_ids[i]
            if ids and ids[-1] in eos_ids:
                ids_for_decode = ids[:-1]
            else:
                ids_for_decode = ids
            text = self.tokenizer.decode(ids_for_decode, skip_special_tokens=True)
            if self.model_name in GEMMA_MODELS and text.startswith("model\n"):
                text = text[len("model\n"):]
            row = {
                "text": text.strip(),
                "generated_ids": ids,
                "n_prompt_tokens": int(n_prompt_per_row[i]),
                "prompt_last_token_id": prompt_last_token_ids[i],
            }
            if prompt_special_layers:
                cols = ps_per_row[i]
                off = sum(len(ps_per_row[r]) for r in range(i))
                pad_off = input_ids.shape[1] - int(n_prompt_per_row[i])
                row["prompt_special"] = {
                    "token_ids": [int(input_ids[i, j].item()) for j in cols],
                    "positions": [j - pad_off for j in cols],
                    "activations": {li: ps_captured[li][off:off + len(cols)]
                                    for li in prompt_special_layers
                                    if li in ps_captured},
                }
            results.append(row)
        return results

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(model_name: str, device: str = "cuda", dtype: str = "bfloat16",
               quantization: Optional[str] = None,
               attn_implementation: Optional[str] = None) -> ModelWrapper:
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    qconfig = None
    if quantization == "8bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(dtype, torch.bfloat16),
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    return ModelWrapper(model_name, device, dtype_map.get(dtype, torch.bfloat16), qconfig,
                        attn_implementation=attn_implementation)
