"""Arithmetic that is actually correct.

Language models are unreliable at multi-digit arithmetic — not because they are
bad at maths, but because they are predicting plausible digits rather than
computing. This tool removes the guesswork.

It parses to an AST and walks it, rather than calling eval(). eval() on model
output is a remote code execution hole; there is no safe way to sanitize your
way around that, so we never call it.
"""

from __future__ import annotations

import ast
import math
import operator

from . import Tool, ToolResult, register

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
    "gcd": math.gcd, "pow": pow,
}

MAX_POWER = 10_000  # 2**10**9 will hang the process; refuse it instead


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if op is operator.pow and isinstance(right, (int, float)) and right > MAX_POWER:
            raise ValueError(f"exponent too large (>{MAX_POWER}); refusing to compute")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = getattr(node.func, "id", "?")
            raise ValueError(f"unknown function: {name}")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Compare):
        left = _eval(node.left)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp)
            fn = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
                  ast.GtE: operator.ge, ast.Eq: operator.eq,
                  ast.NotEq: operator.ne}.get(type(op))
            if fn is None or not fn(left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> ToolResult:
    expression = str(expression).strip().rstrip("=").strip()
    if not expression:
        return ToolResult(False, "Empty expression.")
    if len(expression) > 500:
        return ToolResult(False, "Expression too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return ToolResult(False, f"Could not parse '{expression}': {e.msg}")
    try:
        value = _eval(tree)
    except ZeroDivisionError:
        return ToolResult(False, "Division by zero.")
    except (ValueError, OverflowError, TypeError) as e:
        return ToolResult(False, str(e))

    if isinstance(value, float):
        pretty = f"{value:.10g}"
    else:
        pretty = str(value)
    return ToolResult(True, f"{expression} = {pretty}",
                      {"expression": expression, "result": pretty})


register(Tool(
    name="calculate",
    description="Evaluate a mathematical expression exactly. Use for ANY arithmetic "
                "beyond trivial mental maths — do not compute it yourself",
    args={"expression": "e.g. '(1920*1080)/1e6' or 'sqrt(2)*log(100)'"},
    run=calculate,
    icon="∑",
))
