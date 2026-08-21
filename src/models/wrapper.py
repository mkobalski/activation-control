"""Model wrapper: loading, activation extraction, and token-by-token generation.

Unused features (steering hooks, batched multi-steering) have been removed; we
only need:

  * extract_activations(prompts, layer_idx) -- for concept vector extraction
  * generate_token_by_token(prompt, recorder) -- for per-token activation recording

The tokenizer's chat template is used by the prompt builders (src/prompts).

Some load-time shims below name checkpoints that are NOT in the panel roster --
Kimi-Linear, Ministral-3-3B, DeepSeek-V4-Flash. They are retained from
exploratory runs: each guards a real failure mode (remote-code API drift, FP8
tensor-mode weights, kernel-backed FP8) that a future checkpoint can hit, and
each is keyed on what the checkpoint does rather than on its name.
"""

import gc
from contextlib import contextmanager
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.registry import MODEL_NAME_MAP, GEMMA_MODELS

# Compat shim for trust_remote_code checkpoints pinned against transformers 4.x:
# OutputRecorder/check_model_inputs moved from utils.generic to
# utils.output_capturing in v5 (Kimi-Linear's modeling file imports the old path).
try:
    import transformers.utils.generic as _tf_generic
    from transformers.utils.output_capturing import OutputRecorder as _OR
    if not hasattr(_tf_generic, "OutputRecorder"):
        _tf_generic.OutputRecorder = _OR
except Exception:
    pass
try:
    # v5 renamed create_causal_mask's `input_embeds` kwarg to `inputs_embeds`;
    # accept the old spelling. Patched pre-import so remote modules' from-imports
    # bind the wrapped function.
    import functools
    import inspect
    import transformers.masking_utils as _tf_mu
    _orig_ccm = _tf_mu.create_causal_mask
    _ccm_params = set(inspect.signature(_orig_ccm).parameters)
    @functools.wraps(_orig_ccm)
    def _ccm_compat(*a, **kw):
        if "input_embeds" in kw and "inputs_embeds" not in kw:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        # v4 callers also pass e.g. cache_position, which v5 derives internally
        kw = {k: v for k, v in kw.items() if k in _ccm_params}
        return _orig_ccm(*a, **kw)
    _tf_mu.create_causal_mask = _ccm_compat
except Exception:
    pass
try:
    # fla-core 0.5 changed fused_kda_gate from (g, A_log, head_dim, g_bias=...)
    # to (g, A_log, dt_bias=...); Kimi-Linear's remote code uses the old form.
    import functools as _ft
    import fla.ops.kda.gate as _fla_gate
    _orig_fkg = _fla_gate.fused_kda_gate
    @_ft.wraps(_orig_fkg)
    def _fkg_compat(g, A_log, *args, **kw):
        args = [x for x in args if not isinstance(x, int)]  # drop old head_dim arg
        if "g_bias" in kw:
            kw["dt_bias"] = kw.pop("g_bias")
        return _orig_fkg(g, A_log, *args, **kw)
    _fla_gate.fused_kda_gate = _fkg_compat
except Exception:
    pass


class ModelWrapper:
    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 quantization_config: Optional[BitsAndBytesConfig] = None,
                 attn_implementation: Optional[str] = None,
                 max_memory: Optional[str] = None):
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
        self._ensure_chat_template()

        load_kwargs = {
            "pretrained_model_name_or_path": self.hf_path,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }
        # Cap GPU-0 weight placement so several runs can share one card. Give the
        # CPU a large budget so a model that exceeds the cap spills to host RAM
        # rather than OOMing at load. NOTE: this bounds WEIGHT placement only, not
        # runtime KV/activation allocations -- size pods with headroom to spare.
        if max_memory is not None and device == "cuda":
            load_kwargs["max_memory"] = {0: max_memory, "cpu": "1500GiB"}
            print(f"[wrapper] max_memory cap: GPU0={max_memory} (co-location mode)")
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        # FP8 dequantize shim. Some checkpoints (e.g. Ministral-3-3B) ship
        # fine-grained FP8 with STATIC, TENSOR-mode activation scaling
        # (weight_block_size=null): the installed finegrained-fp8 kernel only
        # supports static scales with BLOCK-wise weights and raises
        # NotImplementedError at the first matmul. We can't run that kernel, but
        # the checkpoint carries its weight/activation scales, so we ask
        # transformers to DEQUANTIZE to bf16 at load (scales applied, no FP8
        # kernel used). Only kicks in for that unsupported combo and only when the
        # caller hasn't already forced a quantization_config. Logged, like the
        # multimodal/auto-class shims below.
        if quantization_config is None:
            try:
                _qc = getattr(AutoConfig.from_pretrained(self.hf_path, trust_remote_code=True),
                              "quantization_config", None)
                if isinstance(_qc, dict) and str(_qc.get("quant_method", "")).lower() == "fp8" \
                        and _qc.get("weight_block_size") is None \
                        and _qc.get("activation_scheme") == "static":
                    from transformers import FineGrainedFP8Config
                    load_kwargs["quantization_config"] = FineGrainedFP8Config(
                        activation_scheme="static", weight_block_size=None, dequantize=True,
                        modules_to_not_convert=_qc.get("modules_to_not_convert"))
                    load_kwargs.pop("dtype", None)
                    self._fp8_dequant_requested = True
                    print("[wrapper] FP8 static/tensor-mode checkpoint -> dequantizing to bf16 "
                          "(unsupported by the finegrained-fp8 kernel)")
            except Exception as e:
                print(f"[wrapper] FP8 dequantize probe skipped ({type(e).__name__})")

        # Pick the auto class by architecture. Vision-text checkpoints (Mistral-Small-3.1,
        # Gemma-3 multimodal, Llama-4-Scout, GLM-4.6V ...) are registered ONLY under
        # image-text-to-text, so AutoModelForCausalLM raises for them even though we
        # drive them with text-only prompts. The vision tower loads and simply goes
        # unused; the decoder layers we record are reached via language_model.layers
        # (see _get_n_layers / get_decoder_layers).
        auto_cls = AutoModelForCausalLM
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            _cfg = AutoConfig.from_pretrained(self.hf_path, trust_remote_code=True)
            mt = _cfg.model_type
            # trust_remote_code text models (e.g. Kimi-Linear) have a model_type
            # unknown to transformers but ship an auto_map with AutoModelForCausalLM
            # -- those stay on the causal-LM path.
            remote_causal = "AutoModelForCausalLM" in (getattr(_cfg, "auto_map", None) or {})
            if mt not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and not remote_causal:
                from transformers import AutoModelForImageTextToText
                auto_cls = AutoModelForImageTextToText
                print(f"[wrapper] {mt} is not a causal-LM architecture; "
                      f"loading with AutoModelForImageTextToText")
        except Exception as e:
            print(f"[wrapper] auto-class probe failed ({type(e).__name__}); "
                  f"falling back to AutoModelForCausalLM")

        try:
            self.model = auto_cls.from_pretrained(**load_kwargs)
        except Exception:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype", dtype)
            self.model = auto_cls.from_pretrained(**load_kwargs)

        # Kimi-Linear's remote code force-overrides the attn implementation to
        # flash_attention_2; without the flash-attn package transformers falls
        # back to the kernels-community/flash-attn2 hub kernel, which has no
        # build for this system (B200). sdpa handles its MLA q/v head-dim
        # mismatch fine, so flip back post-load.
        import importlib.util as _ilu
        if (getattr(self.model.config, "_attn_implementation", None) == "flash_attention_2"
                and _ilu.find_spec("flash_attn") is None):
            self.model.config._attn_implementation = "sdpa"
            print("[wrapper] flash_attention_2 requested but flash-attn is not "
                  "installed -> overriding to sdpa")

        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()

        # v5 masking calls Cache.get_mask_sizes(query_length: int, layer_idx);
        # v4-era remote cache classes (e.g. Kimi-Linear's) expect a cache_position
        # TENSOR and read .shape[0]. Wrap remote-module implementations to accept
        # the int form.
        import sys as _sys, inspect as _insp
        _mod = _sys.modules.get(type(self.model).__module__)
        if _mod is not None and "transformers_modules" in type(self.model).__module__:
            def _wrap_gms(_gms):
                def _inner(cself, cache_position, layer_idx):
                    if isinstance(cache_position, int):
                        cache_position = torch.empty(cache_position)  # only .shape[0] is read
                    return _gms(cself, cache_position, layer_idx)
                _inner._v5shim = True
                return _inner
            for _cls in vars(_mod).values():
                if _insp.isclass(_cls) and "get_mask_sizes" in vars(_cls) \
                        and not getattr(_cls.get_mask_sizes, "_v5shim", False):
                    _cls.get_mask_sizes = _wrap_gms(_cls.get_mask_sizes)
                    print(f"[wrapper] patched {_cls.__name__}.get_mask_sizes for v5 int query_length")

        # transformers' dequantize=True path only matches `*.weight`-style keys, so
        # fused MoE expert tensors (e.g. Mistral-Small-4 `experts.gate_up_proj`)
        # keep raw FP8 weights while their `*_scale_inv` keys are dropped as
        # UNEXPECTED -> first grouped_mm raises on the FP8 dtype. Fold the scales
        # back in from the cached checkpoint shards. ONLY when we asked for
        # dequantization: kernel-backed FP8 checkpoints (e.g. DeepSeek-V4-Flash's
        # block-wise FP8) legitimately keep FP8 params and must not be touched.
        if getattr(self, "_fp8_dequant_requested", False):
            self._dequantize_leftover_fp8()

        self._apply_patches()
        self.n_layers = self._get_n_layers()
        print(f"Model loaded. Layers: {self.n_layers}")

    def _ensure_chat_template(self):
        """Attach the repo's chat template when the tokenizer ships without one.

        MULTIMODAL checkpoints keep their template on the *processor*, in a separate
        `chat_template.json`, so AutoTokenizer comes back with chat_template=None even
        though the repo has one (Mistral-Small-3.1-24B is the panel's first such case;
        Gemma-3 multimodal, Llama-4-Scout and GLM-4.6V are the same shape). That is
        silently destructive here: src/prompts/builder.py falls back to a bare
        "User:/Assistant:" scaffold when chat_template is missing -- the BASE-model
        path -- so an instruct model would be prompted in the wrong format for the
        whole run and score against the panel as if that were its real behaviour.

        We read chat_template.json directly rather than loading AutoProcessor, which
        would drag in torchvision for a vision tower we never touch (every prompt here
        is text-only). Failing to find one is not an error: genuine base models (see
        registry.BASE_MODELS) are meant to use the User:/Assistant: fallback.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return
        try:
            from huggingface_hub import hf_hub_download
            import json as _json
            raw = _json.load(open(hf_hub_download(self.hf_path, "chat_template.json")))
            tmpl = raw.get("chat_template") if isinstance(raw, dict) else raw
        except Exception as e:
            print(f"[wrapper] no chat_template.json for {self.hf_path} ({type(e).__name__}); "
                  f"prompts will use the bare User:/Assistant: fallback")
            return
        if tmpl:
            self.tokenizer.chat_template = tmpl
            print(f"[wrapper] chat template loaded from chat_template.json "
                  f"({len(tmpl)} chars) — tokenizer shipped none")

    def _dequantize_leftover_fp8(self):
        """Dequantize any parameter left in FP8 after loading.

        transformers' FP8 ``dequantize=True`` loader only pairs scales with
        ``*.weight``-suffixed keys. Checkpoints that ship fused MoE expert
        tensors under other names (``experts.gate_up_proj`` /
        ``experts.down_proj`` + ``*_scale_inv``, e.g. Mistral-Small-4) come out
        of load with raw FP8 expert weights and their scales dropped as
        UNEXPECTED. Recover the scales from the cached checkpoint shards and
        fold them in (same convention as Fp8Dequantize: ``w * scale_inv``,
        block grid derived from the scale's shape).
        """
        fp8_params = [(n, p) for n, p in self.model.named_parameters()
                      if p.dtype == torch.float8_e4m3fn]
        if not fp8_params:
            return
        import json as _json
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        idx_path = hf_hub_download(self.hf_path, "model.safetensors.index.json",
                                   local_files_only=True)
        weight_map = _json.load(open(idx_path))["weight_map"]

        def _tail(name):
            # param names and checkpoint keys differ in prefix order
            # (model.language_model.* vs language_model.model.*); match on the
            # stable suffix starting at "layers.<i>".
            i = name.find("layers.")
            return name[i:] if i >= 0 else name

        ckpt_by_tail = {_tail(k): k for k in weight_map}
        deq = Fp8Dequantize(None)
        shard_cache = {}
        n_done = 0
        for name, p in fp8_params:
            key = ckpt_by_tail.get(_tail(name))
            scale_key = None
            if key is not None:
                cand = key[:-len(".weight")] + ".weight_scale_inv" \
                    if key.endswith(".weight") else key + "_scale_inv"
                if cand in weight_map:
                    scale_key = cand
            if scale_key is None:
                raise RuntimeError(f"[wrapper] FP8 param {name} has no matching "
                                   f"scale in the checkpoint; cannot dequantize")
            shard = weight_map[scale_key]
            if shard not in shard_cache:
                shard_cache[shard] = hf_hub_download(self.hf_path, shard,
                                                     local_files_only=True)
            with safe_open(shard_cache[shard], framework="pt") as f:
                scale = f.get_tensor(scale_key)
            p.data = deq._dequantize_one(p.data, scale.to(p.device),
                                         output_dtype=self.dtype)
            n_done += 1
        print(f"[wrapper] dequantized {n_done} leftover FP8 tensors "
              f"(fused-expert scales recovered from checkpoint)")

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
        # Gemma rotary-emb shape fix. Family
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
               attn_implementation: Optional[str] = None,
               max_memory: Optional[str] = None) -> ModelWrapper:
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
                        attn_implementation=attn_implementation, max_memory=max_memory)
