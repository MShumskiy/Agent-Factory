"""Safe evaluation of the expression language used in scenario files.

Rules, derived values, event conditions and effects, and end conditions are Python-syntax
expressions over the world state (`me.savings`, `world.weather`) plus a few functions. Anything
else (imports, attribute access on other objects, other calls, comprehensions, lambdas) is
rejected when the scenario is loaded.
"""

from __future__ import annotations

import ast
import operator
import random
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any


class ExprError(ValueError):
    """An expression is invalid or failed to evaluate."""


_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMPOPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}
FUNCTIONS = ("min", "max", "abs", "round", "clamp", "uniform", "normal", "randint", "chance")
_MAX_EXPONENT = 100


def _functions(rng: random.Random) -> dict[str, Callable[..., Any]]:
    return {
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
        "uniform": rng.uniform,
        "normal": rng.gauss,
        "randint": rng.randint,
        "chance": lambda p: rng.random() < p,
    }


@lru_cache(maxsize=1024)
def parse(source: str) -> ast.Expression:
    """Parse and validate an expression; raises ExprError for anything outside the language."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"invalid expression {source!r}: {e.msg}") from e
    # A name is only valid as an entity (`me` in `me.savings`) or as a function being called.
    allowed_names: set[int] = set()
    for node in ast.walk(tree):
        _check_node(node, source)
        if isinstance(node, ast.Attribute):
            allowed_names.add(id(node.value))
        elif isinstance(node, ast.Call):
            allowed_names.add(id(node.func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and id(node) not in allowed_names:
            raise ExprError(
                f"in {source!r}: {node.id!r} is not a variable; write entity.variable "
                "(for example me.savings) or quote text ('fog')"
            )
    return tree


def _check_node(node: ast.AST, source: str) -> None:
    match node:
        case ast.Expression() | ast.Constant() | ast.BoolOp() | ast.IfExp() | ast.Compare():
            pass
        case ast.List() | ast.Tuple() | ast.Load():
            pass
        case ast.BinOp(op=op) if type(op) in _BINOPS:
            pass
        case ast.UnaryOp(op=op) if type(op) in _UNARYOPS:
            pass
        case ast.And() | ast.Or():
            pass
        case _ if type(node) in _BINOPS or type(node) in _CMPOPS or type(node) in _UNARYOPS:
            pass
        case ast.Attribute(value=ast.Name(), attr=attr) if not attr.startswith("_"):
            pass
        case ast.Name(id=name) if not name.startswith("_"):
            pass
        case ast.Call(func=ast.Name(id=name), keywords=[]) if name in FUNCTIONS:
            pass
        case ast.Call():
            raise ExprError(f"in {source!r}: only these functions are allowed: {', '.join(FUNCTIONS)}")
        case _:
            raise ExprError(f"in {source!r}: {type(node).__name__} is not allowed")


def paths(source: str) -> set[str]:
    """The `entity.variable` paths an expression reads."""
    return {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(parse(source))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }


def evaluate(source: str, state: Mapping[str, Mapping[str, Any]], rng: random.Random) -> Any:
    return _Evaluator(state, _functions(rng), source).eval(parse(source).body)


class _Evaluator:
    def __init__(self, state: Mapping[str, Mapping[str, Any]], functions: dict, source: str):
        self.state = state
        self.functions = functions
        self.source = source

    def fail(self, message: str) -> ExprError:
        return ExprError(f"in {self.source!r}: {message}")

    def eval(self, node: ast.AST) -> Any:
        match node:
            case ast.Constant(value=value):
                return value
            case ast.List(elts=items) | ast.Tuple(elts=items):
                return [self.eval(item) for item in items]
            case ast.Attribute(value=ast.Name(id=entity), attr=var):
                try:
                    return self.state[entity][var]
                except KeyError:
                    raise self.fail(f"unknown variable {entity}.{var}") from None
            case ast.BinOp(left=left, op=op, right=right):
                a, b = self.eval(left), self.eval(right)
                if isinstance(op, ast.Pow) and isinstance(b, (int, float)) and abs(b) > _MAX_EXPONENT:
                    raise self.fail("exponent too large")
                try:
                    return _BINOPS[type(op)](a, b)
                except ZeroDivisionError:
                    raise self.fail("division by zero") from None
                except TypeError as e:
                    raise self.fail(str(e)) from None
            case ast.UnaryOp(op=op, operand=operand):
                return _UNARYOPS[type(op)](self.eval(operand))
            case ast.BoolOp(op=ast.And(), values=values):
                result: Any = True
                for value in values:
                    result = self.eval(value)
                    if not result:
                        return result
                return result
            case ast.BoolOp(op=ast.Or(), values=values):
                result = False
                for value in values:
                    result = self.eval(value)
                    if result:
                        return result
                return result
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                a = self.eval(left)
                for op, comparator in zip(ops, comparators):
                    b = self.eval(comparator)
                    try:
                        if not _CMPOPS[type(op)](a, b):
                            return False
                    except TypeError as e:
                        raise self.fail(str(e)) from None
                    a = b
                return True
            case ast.IfExp(test=test, body=body, orelse=orelse):
                return self.eval(body) if self.eval(test) else self.eval(orelse)
            case ast.Call(func=ast.Name(id=name), args=args):
                try:
                    return self.functions[name](*(self.eval(arg) for arg in args))
                except (TypeError, ValueError) as e:
                    raise self.fail(f"{name}(): {e}") from None
        raise self.fail(f"cannot evaluate {type(node).__name__}")
