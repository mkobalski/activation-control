"""Sentence-compliance checking: did the model actually transcribe the sentence?

NAMING NOTE: the method ids below are historically called "*_levenshtein" (kept
for backward-compatibility with existing configs and saved results, where
`method: "normalized_levenshtein"` is recorded), but the underlying similarity is
NOT edit-distance/Levenshtein. It is difflib.SequenceMatcher's ratio, a
Ratcliff-Obershelp (longest-matching-subsequence) score in [0, 1]. It behaves
similarly for our purposes (1.0 = identical, lower = more divergent) but is not a
true normalized Levenshtein distance. Do not rename the ids without migrating
configs + results. Adapted from introspection-master.
"""

import re
from difflib import SequenceMatcher
from typing import Tuple


def check_compliance(generated_text: str, target_sentence: str,
                     method: str = "normalized_levenshtein",
                     threshold: float = 0.85) -> Tuple[bool, float]:
    """Did the model actually transcribe the target sentence?

    Returns (passed, score) where `passed` is `score >= threshold`. The scoring
    `method` is looked up in a dispatch table so callers can pick the metric
    appropriate to the model (see the per-method docstrings/comments below).
    """
    method_map = {
        "normalized_levenshtein": _normalized_levenshtein,
        "prefix_normalized_levenshtein": _prefix_normalized_levenshtein,
        "exact_stripped": _exact_stripped,
        "token_overlap": _token_overlap,
    }
    if method not in method_map:
        raise ValueError(f"Unknown compliance method '{method}'")
    score = method_map[method](generated_text, target_sentence)
    return score >= threshold, score


def _normalized_levenshtein(generated: str, target: str) -> float:
    # Compares the WHOLE generated text to the target. Best for instruct models
    # that stop after the sentence: any trailing text drags the ratio down, so
    # this penalizes a model that keeps writing past the sentence.
    return SequenceMatcher(None, generated.lower().strip(), target.lower().strip()).ratio()


def _prefix_normalized_levenshtein(generated: str, target: str) -> float:
    # Base models without EOS discipline write the sentence then keep babbling;
    # score only the first len(target) chars so a faithful prefix scores ~1.0.
    # This is the key difference from _normalized_levenshtein: by truncating the
    # generation to the target length we ignore the post-sentence babble that
    # would otherwise wrongly mark a correct transcription as non-compliant.
    g = generated.lower().strip()
    t = target.lower().strip()
    return SequenceMatcher(None, g[:len(t)], t).ratio()


def _exact_stripped(generated: str, target: str) -> float:
    # Strict 1.0/0.0: strip punctuation, collapse whitespace, lower-case, then
    # require an exact match. Useful when only verbatim transcription counts.
    clean = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s).lower().strip())
    return 1.0 if clean(generated) == clean(target) else 0.0


def _token_overlap(generated: str, target: str) -> float:
    # Lenient bag-of-words recall: fraction of target words that appear anywhere
    # in the generation, ignoring order/duplicates. Tolerates reordering but can
    # over-credit (a word present for any reason counts). Empty target -> 1.0.
    gt = set(generated.lower().split())
    tt = target.lower().split()
    if not tt:
        return 1.0
    return sum(1 for t in tt if t in gt) / len(tt)
