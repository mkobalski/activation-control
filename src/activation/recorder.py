"""Targeted activation recorder: hooks a chosen subset of decoder layers.

Captures the residual-stream hidden state of the final (most recent) token
at each forward pass, at every layer in `layer_indices`. Supports batched
generation: each recorded step stores a (B, d_model) slab; per-row snapshots
are produced by slicing along the batch axis. Recording is toggled by
`start_recording()` / `stop_recording()` and gated per-row by
`set_active_mask(mask)` so finished sequences don't contribute further steps.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn


@dataclass
class ActivationSnapshot:
    # {layer_idx: np.ndarray of shape (n_tokens, d_model)}
    activations: Dict[int, np.ndarray]
    # {layer_idx: np.ndarray of shape (n_tokens,)}
    norms: Dict[int, np.ndarray]
    layer_indices: List[int]
    num_tokens: int


class ActivationRecorder:
    def __init__(self, decoder_layers: nn.ModuleList, layer_indices: List[int]):
        self._decoder_layers = decoder_layers
        self._layer_indices = list(layer_indices)
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        # Per-step slabs: {layer_idx: list of np.ndarray of shape (B, d_model)}
        self._slabs: Dict[int, List[np.ndarray]] = {li: [] for li in self._layer_indices}
        # Per-step per-row active mask: list of np.ndarray bool of shape (B,)
        self._active_masks: List[np.ndarray] = []
        self._recording: bool = False
        self._batch_size: int = 1
        self._active_mask: Optional[np.ndarray] = None  # current step's mask

    def register_hooks(self) -> None:
        for li in self._layer_indices:
            layer = self._decoder_layers[li]
            self._hooks.append(layer.register_forward_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx: int):
        # Forward hook fired after each decoder layer. During generation every
        # forward pass processes exactly one new position per row, so the hidden
        # state's last token (`h[:, -1, :]`) IS the just-generated token. We grab
        # it for the whole batch as one (B, d) "slab" and stash it; finished rows
        # are filtered out later in get_snapshots() via the per-step mask, so we
        # cheaply record everyone here and prune afterward.
        def hook(module, inp, out):
            if not self._recording:
                return
            h = out[0] if isinstance(out, tuple) else out
            last = h[:, -1, :].detach().float().cpu().numpy()  # (B, d)
            self._slabs[layer_idx].append(last)
        return hook

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        # Called once per generation step, after all hooked layers have fired.
        # We record one boolean mask per step saying which rows were "live" this
        # step (not finished, still within their recording budget). The guard
        # `len(active_masks) < len(slabs[first_layer])` ensures exactly one mask
        # is appended per step even though every layer pushed a slab — we key the
        # count off the first layer's slab count as the canonical step counter.
        if self._recording:
            mask = (self._active_mask if self._active_mask is not None
                    else np.ones(self._batch_size, dtype=bool))
            first_li = self._layer_indices[0]
            if len(self._active_masks) < len(self._slabs[first_li]):
                self._active_masks.append(mask.copy())
        self._recording = False

    def reset(self, batch_size: int = 1) -> None:
        self._slabs = {li: [] for li in self._layer_indices}
        self._active_masks = []
        self._batch_size = batch_size
        self._active_mask = np.ones(batch_size, dtype=bool)

    def set_active_mask(self, mask: np.ndarray) -> None:
        self._active_mask = mask.astype(bool)

    def get_snapshots(self) -> List[ActivationSnapshot]:
        """Return one ActivationSnapshot per row of the batch."""
        B = self._batch_size
        snapshots: List[ActivationSnapshot] = []
        # Stack slabs per layer: shape (T, B, d)
        stacked: Dict[int, np.ndarray] = {}
        for li in self._layer_indices:
            if self._slabs[li]:
                stacked[li] = np.stack(self._slabs[li], axis=0)
            else:
                stacked[li] = np.zeros((0, B, 0))
        masks = (np.stack(self._active_masks, axis=0)
                 if self._active_masks else np.zeros((0, B), dtype=bool))  # (T, B)

        for b in range(B):
            acts: Dict[int, np.ndarray] = {}
            norms: Dict[int, np.ndarray] = {}
            row_mask = masks[:, b] if masks.size else np.zeros((0,), dtype=bool)
            n_tok = int(row_mask.sum()) if masks.size else 0
            for li in self._layer_indices:
                full = stacked[li]  # (T, B, d)
                if full.shape[0] == 0:
                    acts[li] = np.zeros((0, 0))
                    norms[li] = np.zeros((0,), dtype=np.float32)
                    continue
                row = full[:, b, :][row_mask]  # (n_tok, d)
                acts[li] = row
                norms[li] = np.linalg.norm(row, axis=-1).astype(np.float32) \
                    if row.size else np.zeros((0,), dtype=np.float32)
            snapshots.append(ActivationSnapshot(
                activations=acts, norms=norms,
                layer_indices=list(self._layer_indices),
                num_tokens=n_tok,
            ))
        return snapshots

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
