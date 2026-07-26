import pytest

from src.control_under_load import grade_polynomial_answer


@pytest.mark.parametrize(
    "answer",
    [
        "Final answer: (x - 1)(x + 1)",
        "Work first.\nFinal answer: -(1-x)*(x+1)",
        r"Final answer: $\boxed{(x-1)(x+1)}$",
        "**Final answer**: **(x + 1)*(x - 1)**",
    ],
)
def test_accepts_equivalent_factored_forms(answer):
    result = grade_polynomial_answer(answer, "x**2 - 1")
    assert result.is_valid
    assert result.is_correct
    assert result.symbolic_equal
    assert result.factored_form


def test_accepts_exact_rational_coefficient_and_repeated_factor():
    result = grade_polynomial_answer(
        r"Final answer: \frac{1}{2}(x-2)^{2}(x+3)",
        "(x - 2)**2 * (x + 3) / 2",
    )
    assert result.is_correct


def test_rejects_expanded_only_answer_even_when_equal():
    result = grade_polynomial_answer("Final answer: x^2 - 1", "x**2 - 1")
    assert result.is_valid
    assert result.symbolic_equal
    assert not result.factored_form
    assert not result.is_correct


def test_rejects_partially_factored_nonlinear_factor():
    result = grade_polynomial_answer(
        "Final answer: x*(x^2 - 1)", "x*(x-1)*(x+1)"
    )
    assert result.symbolic_equal
    assert not result.factored_form
    assert not result.is_correct


@pytest.mark.parametrize(
    "answer",
    [
        "(x - 1)*(x + 1)",
        "The final answer is (x - 1)*(x + 1)",
        "Final answer: (x - 1)*(x + 1)\nextra text",
    ],
)
def test_requires_final_answer_line_as_last_nonempty_line(answer):
    result = grade_polynomial_answer(answer, "x**2 - 1")
    assert not result.is_valid
    assert result.error == "no_final_line"


def test_rejects_wrong_or_unparseable_expression():
    wrong = grade_polynomial_answer("Final answer: (x-2)*(x+1)", "x**2 - 1")
    unknown_symbol = grade_polynomial_answer("Final answer: (y-1)*(y+1)", "x**2 - 1")
    assert wrong.is_valid and not wrong.is_correct
    assert not unknown_symbol.is_valid
