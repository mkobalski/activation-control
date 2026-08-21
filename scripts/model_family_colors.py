"""Canonical color scale for model families — Activation Controllability paper.

Palette inspired by artificialanalysis.ai chart colors / lab brand colors,
finalized 2026-07-20. Use these EVERYWHERE instead of per-script
colormaps so figures are consistent.

Usage:
    from model_family_colors import FAMILY_COLORS, family_color, family_shades
    ax.bar(x, y, color=family_color("Qwen 3.5 122B"))     # alias-tolerant
    shades = family_shades("Gemma", 3)                    # within-family variants
"""

import re

# Keys match FAMILY_ORDER naming used across figure_scripts.
FAMILY_COLORS = {
    "Gemma":    "#34A853",  # Google green
    "Mistral":  "#FF7000",  # Mistral orange
    "Olmo":     "#F0529C",  # Ai2 pink
    "GPT-OSS":  "#1F2328",  # charcoal (OpenAI black)
    "Kimi":     "#2081F9",  # azure
    "Qwen":     "#C2410C",  # burnt orange (darker than Mistral)
    "GLM":      "#00B5AD",  # cyan-teal
    "Llama":    "#0B3D91",  # deep navy
    "DeepSeek": "#4D6BFE",  # blue-violet (official logo color)
}

# Lowercase substring -> canonical family. Covers model-id strings like
# "gptoss_120b_low", "qwen36_27b", "glm-5.2", "kimi-k2.6".
_ALIASES = {
    "gemma": "Gemma", "mistral": "Mistral", "olmo": "Olmo",
    "gpt-oss": "GPT-OSS", "gptoss": "GPT-OSS", "gpt_oss": "GPT-OSS",
    "kimi": "Kimi", "qwen": "Qwen", "glm": "GLM", "zai": "GLM",
    "llama": "Llama", "deepseek": "DeepSeek",
}


def family_of(name: str) -> str:
    """Map any model name/id ('Llama 3.3 70B', 'qwen36_27b') to its family."""
    s = name.lower()
    for key, fam in _ALIASES.items():
        if key in s:
            return fam
    raise KeyError(f"No model family recognized in {name!r}")


def family_color(name: str) -> str:
    """Hex color for a family or any model name containing a family alias."""
    return FAMILY_COLORS[name] if name in FAMILY_COLORS else FAMILY_COLORS[family_of(name)]


def family_shades(name: str, n: int):
    """n shades of the family color, light -> full strength, for
    distinguishing model sizes within one family (replaces per-family cmaps)."""
    base = family_color(name).lstrip("#")
    r, g, b = (int(base[i:i + 2], 16) for i in (0, 2, 4))
    out = []
    for i in range(n):
        t = 0.45 * (1 - i / max(n - 1, 1))  # 45% toward white for lightest
        out.append("#{:02X}{:02X}{:02X}".format(
            round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t)))
    return out
