"""Toolbox MCP server: safe arithmetic, unit conversion, and world-clock tools.

Runs standalone over Streamable HTTP (the transport WSO2 AI gateways proxy) or stdio:

    python servers/toolbox_server.py                      # http://127.0.0.1:8001/mcp
    python servers/toolbox_server.py --transport stdio
"""

from __future__ import annotations

import argparse
import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

SERVER_NAME = "toolbox"
DEFAULT_PORT = 8001
MAX_EXPONENT = 1000

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}

# Factors to the base unit of each dimension (metre, kilogram, litre).
_LINEAR_UNITS: dict[str, dict[str, float]] = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "mi": 1609.344},
    "mass": {"g": 0.001, "kg": 1.0, "oz": 0.028349523125, "lb": 0.45359237},
    "volume": {"ml": 0.001, "l": 1.0, "gal": 3.785411784},
}
_TEMPERATURE_UNITS = ("c", "f", "k")


def evaluate_expression(expression: str) -> float:
    """Evaluate an arithmetic expression without using eval().

    Raises:
        ValueError: if the expression is malformed or uses anything other than
            numbers, + - * / // % **, parentheses, and the whitelisted functions.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {expression!r}") from exc
    try:
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc
    except OverflowError as exc:
        raise ValueError("result is too large") from exc


def _eval_node(node: ast.expr) -> float:
    match node:
        case ast.Constant(value=bool()):
            raise ValueError("booleans are not numbers")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.Name(id=name) if name in _CONSTANTS:
            return _CONSTANTS[name]
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            return _UNARY_OPS[type(op)](_eval_node(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPS:
            lhs, rhs = _eval_node(left), _eval_node(right)
            if isinstance(op, ast.Pow) and abs(rhs) > MAX_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_EXPONENT} is not allowed")
            return _BINARY_OPS[type(op)](lhs, rhs)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in _FUNCTIONS:
            return _FUNCTIONS[name](*(_eval_node(arg) for arg in args))
    raise ValueError(f"unsupported syntax: {ast.unparse(node)!r}")


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between two units of the same dimension.

    Raises:
        ValueError: if either unit is unknown or the units measure different things.
    """
    src, dst = from_unit.strip().lower(), to_unit.strip().lower()
    if src in _TEMPERATURE_UNITS and dst in _TEMPERATURE_UNITS:
        return _kelvin_to(_to_kelvin(value, src), dst)
    for factors in _LINEAR_UNITS.values():
        if src in factors and dst in factors:
            return value * factors[src] / factors[dst]
    raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}; supported units: {supported_units()}")


def supported_units() -> str:
    """Return a human-readable list of the units convert() understands."""
    groups = {**_LINEAR_UNITS, "temperature": dict.fromkeys(_TEMPERATURE_UNITS)}
    return "; ".join(f"{dimension}: {', '.join(units)}" for dimension, units in groups.items())


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value + 273.15
    if unit == "f":
        return (value - 32) * 5 / 9 + 273.15
    return value


def _kelvin_to(kelvin: float, unit: str) -> float:
    if unit == "c":
        return kelvin - 273.15
    if unit == "f":
        return (kelvin - 273.15) * 9 / 5 + 32
    return kelvin


def current_time(timezone: str, now: datetime | None = None) -> str:
    """Return the current time in an IANA timezone as an ISO-8601 string plus weekday.

    Raises:
        ValueError: if the timezone name is unknown.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {timezone!r}; use an IANA name like 'Asia/Colombo'") from exc
    moment = (now or datetime.now(tz=zone)).astimezone(zone)
    return f"{moment.isoformat(timespec='seconds')} ({moment:%A}) in {timezone}"


def _format_number(value: float) -> str:
    return f"{value:.10g}"


def create_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    allowed_hosts: list[str] | None = None,
) -> FastMCP:
    """Build the toolbox server with its three tools registered."""
    server = FastMCP(
        SERVER_NAME,
        instructions="Arithmetic, unit conversion, and current-time lookups.",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        log_level="WARNING",
        transport_security=_transport_security(port, allowed_hosts),
    )

    @server.tool()
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '(3 + 4) * 2 ** 3' or 'sqrt(2) * pi'.

        Supports + - * / // % **, parentheses, the constants pi and e, and the
        functions sqrt, abs, round, log, log10, sin, cos, tan.
        """
        return f"{expression} = {_format_number(evaluate_expression(expression))}"

    @server.tool()
    def convert_units(value: float, from_unit: str, to_unit: str) -> str:
        """Convert a value between units of length (mm, cm, m, km, in, ft, mi),
        mass (g, kg, oz, lb), volume (ml, l, gal) or temperature (c, f, k)."""
        result = convert(value, from_unit, to_unit)
        return f"{_format_number(value)} {from_unit} = {_format_number(result)} {to_unit}"

    @server.tool()
    def get_current_time(timezone: str = "UTC") -> str:
        """Get the current date and time in an IANA timezone such as 'UTC',
        'Asia/Colombo' or 'America/New_York'."""
        return current_time(timezone)

    return server


def _transport_security(port: int, allowed_hosts: list[str] | None) -> TransportSecuritySettings | None:
    """Keep DNS-rebinding protection on, but let a gateway reach us under extra Host names."""
    if not allowed_hosts:
        return None  # FastMCP's default: protection on for loopback binds only.
    # Proxies that rewrite Host (e.g. WSO2 gateway `hostRewrite: auto`) may omit the port.
    extra = [host for name in allowed_hosts for host in (name, f"{name}:{port}")]
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *extra]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)


def main() -> None:
    """Parse CLI flags and run the server."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="extra Host header name to accept, e.g. host.docker.internal for a gateway in Docker (repeatable)",
    )
    args = parser.parse_args()
    create_server(args.host, args.port, args.allowed_host).run(transport=args.transport)


if __name__ == "__main__":
    main()
