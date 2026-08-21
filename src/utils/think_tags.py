"""Think-tag reasoning parsing for Qwen3-style reasoning models.

Qwen3 checkpoints (with `enable_thinking=True` in the chat template) emit a
chain-of-thought wrapped in think tags before the answer proper, e.g.

    <think>
    ...chain of thought...
    </think>

    The requested sentence, written verbatim.

For the activation-control experiment we only care about the **final answer**:
the text after the closing `</think>` tag, where the model writes the requested
sentence. The reasoning inside the think block precedes it and must be excluded
from both compliance scoring and the activation span we record over (we want
activations over the target sentence, wherever it lands after the CoT).

This mirrors ``src/utils/harmony.py`` for gpt-oss, but for the think-tag format:
``final_answer_span`` locates the answer as a token span within the generated
ids so callers can (a) restrict sentence alignment to it and (b) score
compliance against its text only. It is self-contained, derives the `</think>`
token id(s) from the live tokenizer (single special token or a multi-token
subsequence, whichever the checkpoint uses), and degrades gracefully to "treat
everything as final" when no close tag is present (e.g. a trial where the model
skipped the think block, or a non-thinking model routed here by mistake).
"""

from typing import List, Optional, Tuple

# Canonical think-tag markers. Resolved against the live tokenizer at runtime.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _token_seq(tokenizer, s: str) -> List[int]:
    """Token id(s) for marker `s`.

    Prefer a single dedicated vocab id (Qwen3 registers `<think>`/`</think>` as
    single tokens); fall back to the ordinary encoding subsequence for
    tokenizers that split the marker into pieces. Returns [] if it can't be
    resolved at all (caller then treats the whole generation as final).
    """
    tid = tokenizer.convert_tokens_to_ids(s)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(tid, int) and tid >= 0 and not (unk is not None and tid == unk):
        return [tid]
    try:
        seq = tokenizer.encode(s, add_special_tokens=False)
    except Exception:
        seq = []
    return list(seq)


def _rfind_subseq(hay: List[int], needle: List[int]) -> int:
    """Index of the LAST occurrence of `needle` within `hay`, or -1.

    Last (not first) so a stray `</think>` echoed inside the answer can't win
    over the real close tag; the answer is what follows the final close.
    """
    if not needle:
        return -1
    n, m = len(hay), len(needle)
    for i in range(n - m, -1, -1):
        if hay[i:i + m] == needle:
            return i
    return -1


def final_answer_span(tokenizer, gen_ids: List[int]) -> Tuple[int, int, str]:
    """Token span + decoded text of the answer AFTER the last `</think>`.

    Returns (start, end, text) as indices into `gen_ids`. Falls back to the
    whole sequence (0, len, full text) when no closing think tag is found, so a
    generation that skipped the think block (or a non-think model routed here)
    still yields a sane result instead of crashing.
    """
    close = _token_seq(tokenizer, _THINK_CLOSE)
    idx = _rfind_subseq(gen_ids, close) if close else -1
    if idx < 0:
        return 0, len(gen_ids), tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    start = idx + len(close)
    end = len(gen_ids)
    return start, end, tokenizer.decode(gen_ids[start:end], skip_special_tokens=True).strip()
