"""Align recorded activations to the span of the transcribed sentence.

We record `n_sentence_tokens + token_buffer` generated tokens. The model may
prepend a leading token (space, opening quote, preamble word) before the
sentence itself. We decode the recorded generated tokens one-by-one and look
for the span of length `n_sentence_tokens` whose decoded text matches the
target sentence most closely. Match score is difflib.SequenceMatcher's ratio
(a Ratcliff-Obershelp similarity in [0, 1] — despite the field being loosely
called "Levenshtein" elsewhere, it is not edit distance). That gives a
per-token activation slice that corresponds to the sentence.
"""

from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np


def tokenize_sentence(tokenizer, sentence: str) -> List[int]:
    """Tokenize the *content* of the sentence (no special tokens)."""
    return tokenizer(sentence, add_special_tokens=False).input_ids


def _similarity(a: str, b: str) -> float:
    """Normalized (0..1) similarity between two strings, case/whitespace-insensitive.

    Lower-casing and stripping makes the match robust to leading/trailing spaces
    and capitalization differences introduced during generation. We use the
    ratio rather than exact equality because the decoded span rarely matches the
    target byte-for-byte (tokenization can split words differently).
    """
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def align_sentence_span(
    tokenizer,
    generated_token_ids: List[int],
    target_sentence: str,
    n_sentence_tokens: int,
) -> Tuple[int, int, float]:
    """Find [start, end) over the recorded window that best matches the sentence.

    Returns (start, end, similarity). `end - start == n_sentence_tokens` unless
    the recorded window is shorter than that, in which case we clamp.
    """
    n_rec = len(generated_token_ids)
    if n_rec == 0:
        # Nothing was recorded (e.g. the model emitted only special tokens).
        return 0, 0, 0.0

    # The sentence occupies a fixed-length window of tokens, but we do not know
    # WHERE in the recorded stream it starts: the model may prepend a leading
    # space, an opening quote, or a short preamble before writing the sentence.
    # So we slide a window of length `span_len` over every possible start
    # offset and decode that slice back to text. `span_len` is clamped to the
    # recorded length in case generation was cut short.
    span_len = min(n_sentence_tokens, n_rec)
    best = (0, span_len, -1.0)
    for start in range(0, n_rec - span_len + 1):
        piece = tokenizer.decode(generated_token_ids[start:start + span_len],
                                 skip_special_tokens=True)
        # Normalized similarity (not exact match) tolerates the small textual
        # drift between the decoded window and the target; we keep the window
        # whose decoded text is closest to the target sentence.
        sim = _similarity(piece, target_sentence)
        if sim > best[2]:
            best = (start, start + span_len, sim)
    return best


def slice_activations(activations_by_layer, start: int, end: int):
    """Return {layer: (n_sentence_tokens, d_model)} sliced to [start, end)."""
    return {li: arr[start:end] for li, arr in activations_by_layer.items()}
