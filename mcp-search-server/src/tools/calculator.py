"""Advanced calculator tool for MCP server."""

import json
import logging
from typing import Any

from mcp.server import FastMCP

logger = logging.getLogger(__name__)


def _safe_eval(expr: str) -> Any:
    """Evaluate a mathematical expression using sympy.

    Supports:
    - Basic arithmetic: +, -, *, /, **, %
    - Functions: sin, cos, tan, asin, acos, atan, sqrt, log, log10, exp, abs, ceil, floor
    - Constants: pi, e, degrees, radians
    - Symbolic math: solve, simplify, expand, factor, diff, integrate
    - Matrices: Matrix, determinant, inverse, eigenvals
    - Complex numbers: I
    - Combinatorics: factorial, binomial
    - Linear algebra: det, inv, eigenvects

    Args:
        expr: Mathematical expression or sympy command

    Returns:
        The evaluated result as a string
    """
    from sympy import (
        E,
        Float,
        I,
        Matrix,
        Rational,
        S,
        acos,
        asin,
        atan,
        binomial,
        cbrt,
        ceiling,
        cos,
        degree,
        det,
        diff,
        exp,
        expand,
        factor,
        factorial,
        floor,
        gamma,
        integrate,
        lambdify,
        limit,
        log,
        nsimplify,
        oo,
        pi,
        rad,
        series,
        simplify,
        sin,
        solve,
        sqrt,
        symbols,
        tan,
    )
    from sympy import (
        N as N_func,
    )

    # log10 / log2 aren't top-level sympy — wrap them
    def _log10(x):
        return log(x, 10)

    def _log2(x):
        return log(x, 2)

    # Create a safe namespace
    namespace = {
        # Symbols
        "x": symbols("x"),
        "y": symbols("y"),
        "z": symbols("z"),
        "n": symbols("n"),
        "t": symbols("t"),
        "theta": symbols("theta"),
        # Functions
        "sin": sin, "cos": cos, "tan": tan,
        "asin": asin, "acos": acos, "atan": atan,
        "sqrt": sqrt, "cbrt": cbrt,
        "log": log, "log10": _log10, "log2": _log2, "exp": exp,
        "factorial": factorial, "binomial": binomial,
        "abs": abs, "ceiling": ceiling, "floor": floor,
        "gamma": gamma,
        # Constants
        "pi": pi, "e": E, "I": I,
        "oo": oo, "infinity": oo,
        "Rational": Rational, "Float": Float,
        # Operations
        "solve": solve, "simplify": simplify,
        "expand": expand, "factor": factor,
        "diff": diff, "integrate": integrate, "limit": limit,
        "series": series, "nsimplify": nsimplify,
        "N": N_func,
        "Matrix": Matrix,
        "det": det,
        "inv": lambda m: m.inv(),
        "eigenvals": lambda m: m.eigenvals(),
        "eigenvects": lambda m: m.eigenvects(),
        "lambdify": lambdify,
        "symbols": symbols,
        "S": S,
        "degree": degree,
        "radians": rad,
    }

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
        # Convert sympy objects to readable strings
        if hasattr(result, "__iter__") and not isinstance(result, (str, Matrix)):
            # List of results (e.g., from solve)
            return str([str(r) for r in result])
        elif isinstance(result, Matrix):
            return str(result)
        else:
            return str(result)
    except Exception:
        # Try numeric evaluation as fallback
        try:
            from sympy import N as N_numeric
            result = eval(expr, {"__builtins__": {}}, namespace)
            return str(N_numeric(result, 15))
        except Exception:
            raise


def calculator_handler(server: FastMCP) -> None:
    """Register the calculator tool."""

    @server.tool()
    async def calculator(expression: str) -> str:
        """Evaluate math expressions using SymPy (arithmetic, trig, algebra, calculus, matrices).

        Examples: "sqrt(2)", "solve(x**2-4, x)", "diff(sin(x),x)", "integrate(x**2,x)",
        "Matrix([[1,2],[3,4]]).det()", "factor(x**4-1)", "limit(sin(x)/x,x,0)", "1+I"
        """
        try:
            result = _safe_eval(expression)
            return json.dumps({
                "status": "success",
                "expression": expression,
                "result": result,
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Calculator error for '%s': %s", expression, str(e))
            return json.dumps({
                "status": "error",
                "expression": expression,
                "error": str(e),
            }, indent=2)

    logger.info("Registered calculator tool")
