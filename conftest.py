"""Repository-wide pytest config, to reach benchmarks/ as well as tests/, and skip large data tests by default."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom GeoUtils test options."""

    parser.addoption(
        "--large-data",
        action="store_true",
        default=False,
        help="Run large data tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip large data tests unless explicitly requested."""

    if config.getoption("--large-data"):
        return

    skip_large_data = pytest.mark.skip(reason="Large data test; use --large-data to run.")
    for item in items:
        if "large_data" in item.keywords:
            item.add_marker(skip_large_data)
