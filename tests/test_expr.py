import random

import pytest

from agent_factory.sim.expr import ExprError, evaluate, parse, paths

STATE = {"me": {"savings": 4000, "fixed_costs": 1000, "amp": "used"}, "world": {"weather": "clear"}}


def ev(source: str, seed: int = 1):
    return evaluate(source, STATE, random.Random(seed))


def test_arithmetic_and_lookups():
    assert ev("me.savings + 500 - me.fixed_costs * 2") == 2500
    assert ev("round(me.savings / me.fixed_costs, 1)") == 4.0
    assert ev("clamp(me.savings, 0, 1000)") == 1000
    assert ev("'big' if me.savings > 3000 else 'small'") == "big"


def test_comparisons_and_text():
    assert ev("me.amp == 'used' or me.amp == 'new'") is True
    assert ev("me.amp in ['new', 'used']") is True
    assert ev("not world.weather == 'fog'") is True
    assert ev("0 < me.savings < 5000") is True


def test_random_functions_follow_the_seed():
    assert ev("uniform(300, 1500)", seed=7) == ev("uniform(300, 1500)", seed=7)
    assert 300 <= ev("uniform(300, 1500)") <= 1500
    assert ev("chance(1)") is True and ev("chance(0)") is False


def test_paths_lists_the_variables_read():
    assert paths("me.savings + me.fixed_costs if world.weather == 'fog' else 0") == {
        "me.savings", "me.fixed_costs", "world.weather"}


@pytest.mark.parametrize("source", [
    "__import__('os')",
    "open('x')",
    "me.savings.__class__",
    "().__class__",
    "(lambda: 1)()",
    "[x for x in [1]]",
    "savings + 1",
    "me.amp in [fog]",
    "max(*[1, 2])",
    "me.savings if",
])
def test_rejects_anything_outside_the_language(source):
    with pytest.raises(ExprError):
        parse(source)


def test_evaluation_errors_are_expr_errors():
    with pytest.raises(ExprError, match="unknown variable"):
        ev("me.debt + 1")
    with pytest.raises(ExprError, match="division by zero"):
        ev("me.savings / 0")
    with pytest.raises(ExprError, match="exponent"):
        ev("2 ** 1000")
