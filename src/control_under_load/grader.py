"""Exact symbolic grading for fully factored polynomial answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


@dataclass(frozen=True)
class GradeResult:
    """A symbolic-equivalence and written-form grading decision."""

    parsed: str | None
    is_valid: bool
    is_correct: bool
    symbolic_equal: bool
    factored_form: bool
    error: str | None = None


_X = sympy.Symbol("x")
_FINAL_RE = re.compile(
    r"^(?:\*\*)?Final\s+answer(?:\*\*)?\s*:\s*(?:\*\*)?(.+?)(?:\*\*)?\s*$",
    flags=re.IGNORECASE,
)
_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_PARSE_GLOBALS: dict[str, Any] = {
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
    "Symbol": sympy.Symbol,
}


def _read_group(text: str, position: int) -> tuple[str | None, int]:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        return None, position
    depth = 0
    for index in range(position, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[position + 1 : index], index + 1
    return None, position


def _strip_delimiters(text: str) -> str:
    stripped = text.strip()
    for opening, closing in (("$$", "$$"), (r"\(", r"\)"), (r"\[", r"\]"), ("$", "$")):
        if stripped.startswith(opening) and stripped.endswith(closing):
            return stripped[len(opening) : -len(closing)].strip()
    return stripped


def _unwrap_box(text: str) -> str:
    stripped = text.strip()
    for command in (r"\boxed", r"\fbox"):
        if stripped.startswith(command):
            body, end = _read_group(stripped, len(command))
            if body is not None and not stripped[end:].strip():
                return body.strip()
    return stripped


def _convert_fractions(text: str) -> str:
    text = re.sub(r"\\[dt]frac", r"\\frac", text)
    while True:
        position = text.find(r"\frac")
        if position < 0:
            return text
        numerator, middle = _read_group(text, position + len(r"\frac"))
        denominator, end = _read_group(text, middle)
        if numerator is None or denominator is None:
            return text
        text = f"{text[:position]}(({numerator})/({denominator})){text[end:]}"


def _convert_exponents(text: str) -> str:
    output: list[str] = []
    position = 0
    while position < len(text):
        if text[position] == "^":
            body, end = _read_group(text, position + 1)
            if body is not None:
                output.append(f"**({body})")
                position = end
                continue
        output.append(text[position])
        position += 1
    return "".join(output)


def _normalize(text: str) -> str:
    result = _strip_delimiters(text)
    result = _unwrap_box(result)
    result = _strip_delimiters(result)
    for token in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!", r"\;"):
        result = result.replace(token, " ")
    result = _convert_fractions(result)
    result = result.replace(r"\cdot", "*").replace(r"\times", "*")
    result = _convert_exponents(result)
    return result.replace("{", "(").replace("}", ")").strip()


def _parse_untrusted(text: str) -> sympy.Expr | None:
    if not text.strip():
        return None
    try:
        expression = parse_expr(
            text,
            local_dict={"x": _X},
            global_dict=dict(_PARSE_GLOBALS),
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception:
        return None
    if not isinstance(expression, sympy.Expr) or expression.free_symbols - {_X}:
        return None
    polynomial = expression.as_poly(_X)
    return expression if polynomial is not None else None


def _parse_answer_expression(text: str) -> sympy.Expr | None:
    normalized = _normalize(text)
    if "=" in normalized:
        right = _parse_untrusted(normalized.rsplit("=", 1)[-1])
        if right is not None:
            return right
    return _parse_untrusted(normalized)


def _linear_or_constant(expression: sympy.Expr) -> bool:
    if expression.is_Number:
        return True
    polynomial = expression.as_poly(_X)
    return polynomial is not None and polynomial.degree() <= 1


def _allowed_factor(expression: sympy.Expr) -> bool:
    if isinstance(expression, sympy.Pow):
        base, exponent = expression.args
        return bool(
            exponent.is_Integer
            and exponent.is_positive
            and _linear_or_constant(base)
        )
    return _linear_or_constant(expression)


def _is_factored(expression: sympy.Expr) -> bool:
    factors = expression.args if isinstance(expression, sympy.Mul) else (expression,)
    return all(_allowed_factor(factor) for factor in factors)


def grade_polynomial_answer(text: str, expected: str | sympy.Expr) -> GradeResult:
    """Grade the last non-empty ``Final answer:`` line exactly.

    Algebraically equivalent products of numeric and linear factors are accepted,
    including powers for repeated factors. A degree-two-or-higher expression
    written as an expanded sum is rejected even when symbolically equivalent.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = _FINAL_RE.fullmatch(lines[-1]) if lines else None
    if match is None:
        return GradeResult(None, False, False, False, False, "no_final_line")
    parsed = _parse_answer_expression(match.group(1))
    if parsed is None:
        return GradeResult(None, False, False, False, False, "parse")
    try:
        expected_expression = sympy.sympify(expected, locals={"x": _X})
    except (TypeError, ValueError, sympy.SympifyError) as error:
        raise ValueError("expected must be a polynomial expression in x") from error
    if not isinstance(expected_expression, sympy.Expr) or (
        expected_expression.free_symbols - {_X}
    ) or expected_expression.as_poly(_X) is None:
        raise ValueError("expected must be a polynomial expression in x")
    symbolic_equal = bool(sympy.expand(parsed - expected_expression) == 0)
    factored = _is_factored(parsed)
    return GradeResult(
        parsed=str(parsed),
        is_valid=True,
        is_correct=symbolic_equal and factored,
        symbolic_equal=symbolic_equal,
        factored_form=factored,
    )
