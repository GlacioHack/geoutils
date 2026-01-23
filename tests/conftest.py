"""Test configuration."""

import logging
from collections import defaultdict

import pytest


class LoggingWarningCollector(logging.Handler):
    """Helper class to collect logging warnings."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture(autouse=True)
def fail_on_logging_warnings(request):
    """Fixture used automatically in all tests to fail when a logging exceptions of WARNING or above is raised."""

    # The collector is required to avoid teardown, hookwrapper or plugin issues (we collect and fail later)
    collector = LoggingWarningCollector()
    root = logging.getLogger()
    root.addHandler(collector)

    # Run test
    yield

    root.removeHandler(collector)

    # Allow opt-out
    if request.node.get_closest_marker("allow_logging_warnings"):
        return

    # Categorize bad tests
    # IGNORED = ("rasterio",)   # If we want to add a list of "IGNORED" packages in the future
    bad = [
        r
        for r in collector.records
        if r.levelno >= logging.WARNING
        # and not r.name.startswith(IGNORED)
    ]

    # Fail on those exceptions and report exception level, name and message
    if bad:
        msgs = "\n".join(f"{r.levelname}:{r.name}:{r.getMessage()}" for r in bad)
        pytest.fail(
            "Logging warning/error detected:\n" + msgs,
            pytrace=False,
        )
