"""Per-token cosine similarity between a concept vector and a layer trajectory."""

from typing import Dict

import numpy as np


def cosine_trace(concept_vec: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Return cos(concept_vec, activations[t]) for each token t.

    concept_vec: (d_model,)
    activations: (n_tokens, d_model)
    """
    if activations.size == 0:
        # No tokens in this span -> empty trace (keeps callers branch-free).
        return np.zeros((0,), dtype=np.float32)
    cv = concept_vec.astype(np.float32)
    A = activations.astype(np.float32)
    # Cosine similarity == dot product of L2-normalized vectors. We normalize
    # the concept vector once and each per-token residual stream row, then the
    # matrix-vector product gives one cosine per token in a single pass.
    # The +1e-8 guards against divide-by-zero for a zero-norm vector (e.g. an
    # all-zero activation row), so the result is 0 instead of NaN/inf.
    cv_n = cv / (np.linalg.norm(cv) + 1e-8)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)  # per-token (row) norms
    A_n = A / (A_norm + 1e-8)
    return (A_n @ cv_n).astype(np.float32)


def cosine_traces_per_layer(
    concept_vectors_by_layer: Dict[int, np.ndarray],
    activations_by_layer: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """{layer: cos_trace over tokens}; skip layers not present in both dicts.

    A concept vector and the recorded activations must both exist for a layer to
    compute a trace; we intersect the two dicts so a missing layer on either side
    is silently skipped rather than raising a KeyError.
    """
    out = {}
    for li, cv in concept_vectors_by_layer.items():
        if li in activations_by_layer:
            out[li] = cosine_trace(cv, activations_by_layer[li])
    return out
