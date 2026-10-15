from datetime import UTC, datetime

import pytest

from servers.toolbox_server import (
    _transport_security,
    convert,
    create_server,
    current_time,
    evaluate_expression,
    supported_units,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("2 ** 10", 1024),
        ("-7 // 2", -4),
        ("10 % 4", 2),
        ("sqrt(16) + abs(-2)", 6),
        ("round(pi, 2)", 3.14),
    ],
)
def test_evaluate_expression_computes_arithmetic(expression: str, expected: float) -> None:
    assert evaluate_expression(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "open('/etc/passwd')",
        "x + 1",
        "(1).real",
        "True + 1",
        "'a' * 3",
        "2 ** 100000",
        "1 / 0",
        "1 +",
    ],
)
def test_evaluate_expression_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(ValueError):
        evaluate_expression(expression)


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1, "km", "m", 1000),
        (1, "mi", "km", 1.609344),
        (12, "in", "ft", 1),
        (1, "lb", "g", 453.59237),
        (1, "gal", "l", 3.785411784),
        (100, "C", "F", 212),
        (32, "f", "c", 0),
        (0, "c", "k", 273.15),
    ],
)
def test_convert_between_units(value: float, src: str, dst: str, expected: float) -> None:
    assert convert(value, src, dst) == pytest.approx(expected)


@pytest.mark.parametrize(("src", "dst"), [("kg", "m"), ("parsec", "m"), ("c", "kg")])
def test_convert_rejects_incompatible_or_unknown_units(src: str, dst: str) -> None:
    with pytest.raises(ValueError, match="cannot convert"):
        convert(1, src, dst)


def test_supported_units_lists_every_dimension() -> None:
    units = supported_units()
    for dimension in ("length", "mass", "volume", "temperature"):
        assert dimension in units


def test_current_time_formats_in_requested_zone() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert current_time("Asia/Colombo", now) == "2026-01-01T17:30:00+05:30 (Thursday) in Asia/Colombo"


def test_current_time_rejects_unknown_zone() -> None:
    with pytest.raises(ValueError, match="unknown timezone"):
        current_time("Mars/Olympus_Mons")


async def test_server_exposes_three_tools() -> None:
    tools = await create_server().list_tools()
    assert sorted(tool.name for tool in tools) == ["calculate", "convert_units", "get_current_time"]


async def test_server_tools_return_text() -> None:
    server = create_server()
    content, _ = await server.call_tool("calculate", {"expression": "sqrt(2) ** 2"})
    assert content[0].text == "sqrt(2) ** 2 = 2"
    content, _ = await server.call_tool("convert_units", {"value": 5, "from_unit": "km", "to_unit": "mi"})
    assert content[0].text == f"5 km = {5000 / 1609.344:.10g} mi"


def test_transport_security_allows_extra_host_with_and_without_port() -> None:
    assert _transport_security(8001, None) is None
    settings = _transport_security(8001, ["host.k3d.internal"])
    assert settings is not None
    assert settings.enable_dns_rebinding_protection
    assert {"host.k3d.internal", "host.k3d.internal:8001", "localhost:*"} <= set(settings.allowed_hosts)
