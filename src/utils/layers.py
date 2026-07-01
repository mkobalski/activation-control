"""Helpers for resolving fractional layer depths and explicit indices to layer ints."""

from typing import List, Optional


def fraction_to_layer(fraction: float, n_layers: int) -> int:
    """Map a relative depth in [0, 1] to a concrete layer index.

    Fractions let configs be model-agnostic (e.g. "0.5" == middle layer) so the
    same experiment runs across models of different depths. We multiply by
    n_layers and truncate to an int, then clamp into [0, n_layers - 1] so edge
    values like 1.0 (which would give n_layers, out of range) stay valid.
    """
    idx = int(n_layers * fraction)
    return max(0, min(idx, n_layers - 1))


def resolve_layers(fractions: List[float], n_layers: int,
                   layers: Optional[List[int]] = None) -> List[int]:
    """Convert fractional depths and/or explicit integer indices to sorted unique layer ints.

    Either or both of `fractions` and `layers` may be supplied. Integer indices are
    range-checked against n_layers; out-of-range values raise ValueError.
    """
    # Resolve fractional depths first (these are clamped, never error), then add
    # any explicit integer indices. We collect into a set so a fraction and an
    # explicit index that land on the same layer are de-duplicated.
    out = {fraction_to_layer(f, n_layers) for f in (fractions or [])}
    for li in (layers or []):
        li_int = int(li)
        # Explicit indices are NOT clamped (unlike fractions): an out-of-range
        # index is almost certainly a config mistake, so fail loudly.
        if li_int < 0 or li_int >= n_layers:
            raise ValueError(
                f"layer index {li_int} out of range for n_layers={n_layers}"
            )
        out.add(li_int)
    return sorted(out)
