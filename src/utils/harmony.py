"""Harmony channel parsing for reasoning models (gpt-oss).

gpt-oss generates in the *harmony* response format: the assistant turn is a
sequence of channel-tagged messages, e.g.

    <|channel|>analysis<|message|>...chain of thought...<|end|>
    <|start|>assistant<|channel|>final<|message|>...the answer...<|return|>

For the activation-control experiment we only care about the **final** channel:
that is where the model writes the requested sentence verbatim. The chain of
thought in the `analysis` channel precedes it and must be excluded from both
compliance scoring and the activation span we record over (we want activations
over the target sentence, wherever it lands in the final channel).

This module locates the final-channel content as a token span within the
generated ids, so callers can (a) restrict sentence alignment to it and
(b) score compliance against its text only. It is intentionally self-contained
and derives the harmony structural token ids from the tokenizer at runtime, so
it adapts to whatever ids the loaded checkpoint uses (and degrades gracefully to
"treat everything as final" if the markers are absent — e.g. a non-harmony
model routed here by mistake).
"""

from typing import List, Optional, Tuple

# Canonical harmony structural tokens. Looked up in the live tokenizer; any that
# the tokenizer doesn't know resolve to None and are simply skipped.
_CHANNEL = "<|channel|>"
_MESSAGE = "<|message|>"
# Tokens that terminate a channel message's content.
_STOPS = ("<|end|>", "<|return|>", "<|call|>", "<|start|>", "<|channel|>")


def _tid(tokenizer, s: str) -> Optional[int]:
    tid = tokenizer.convert_tokens_to_ids(s)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    return tid


def parse_channels(tokenizer, gen_ids: List[int]) -> List[Tuple[str, int, int]]:
    """All harmony channel messages as (channel_name, start, end) token spans.

    `start`/`end` bound the message CONTENT (between <|message|> and the next
    terminator), as indices into `gen_ids`. Returns [] if the harmony markers
    aren't present in `gen_ids`.
    """
    ch_id = _tid(tokenizer, _CHANNEL)
    msg_id = _tid(tokenizer, _MESSAGE)
    if ch_id is None or msg_id is None:
        return []
    stop_ids = {t for t in (_tid(tokenizer, s) for s in _STOPS) if t is not None}

    spans: List[Tuple[str, int, int]] = []
    n = len(gen_ids)
    i = 0
    while i < n:
        if gen_ids[i] != ch_id:
            i += 1
            continue
        # Channel name tokens run from just after <|channel|> to <|message|>.
        j = i + 1
        name_ids: List[int] = []
        while j < n and gen_ids[j] != msg_id:
            name_ids.append(gen_ids[j])
            j += 1
        if j >= n:  # unterminated channel header
            break
        name = tokenizer.decode(name_ids, skip_special_tokens=True).strip()
        content_start = j + 1
        k = content_start
        while k < n and gen_ids[k] not in stop_ids:
            k += 1
        spans.append((name, content_start, k))
        i = k
    return spans


def final_channel_span(tokenizer, gen_ids: List[int]) -> Tuple[int, int, str]:
    """Token span + decoded text of the LAST `final` channel message.

    Falls back to the whole sequence (0, len, full text) when no final channel
    is found, so a malformed / non-harmony generation still yields a sane
    (if non-compliant) result instead of crashing.
    """
    spans = parse_channels(tokenizer, gen_ids)
    finals = [(s, e) for (name, s, e) in spans if name == "final"]
    if not finals:
        return 0, len(gen_ids), tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    s, e = finals[-1]
    return s, e, tokenizer.decode(gen_ids[s:e], skip_special_tokens=True).strip()
