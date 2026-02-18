"""Test configuration."""
from __future__ import annotations
import os
import logging
from typing import Any

import pytest

import numpy as np
from geoutils import examples, Raster

class LoggingWarningCollector(logging.Handler):
    """Helper class to collect logging warnings."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record)
        except Exception:
            # Logging handlers must never raise
            pass

    def handleError(self, record: logging.LogRecord) -> None:
        # Override default behavior (which can print to stderr) and never raise
        return

def _safe_record_line(r: logging.LogRecord) -> str:
    # Helper to avoid record.getMessage() raising errors if msg/args formatting is invalid
    try:
        msg = r.getMessage()
    except Exception as e:
        # Fall back to a representation that will always work
        msg = f"<message formatting failed: {type(e).__name__}: {e}; msg={r.msg!r} args={r.args!r}>"

    return f"{r.levelname}:{r.name}:{msg}"

@pytest.fixture(autouse=True)
def fail_on_logging_warnings(request: Any) -> Any:
    """Fixture used automatically in all tests to fail when a logging exceptions of WARNING or above is raised."""

    # The collector is required to avoid teardown, hookwrapper or plugin issues (we collect and fail later)
    collector = LoggingWarningCollector()
    root = logging.getLogger()
    root.addHandler(collector)

    # Run test
    try:
        yield
    finally:
        root.removeHandler(collector)
        collector.close()

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
        msgs = "\n".join(_safe_record_line(r) for r in bad)
        pytest.fail("Logging warning/error detected:\n" + msgs, pytrace=False)

@pytest.fixture(scope="module")
def lazy_test_files(tmp_path_factory: Any) -> list[str]:
    """
    Create temporary converted files on disk for lazy tests.

    Those are used to compare Xarray accessor and Raster class with loading/laziness check (i.e. no data loaded).
    So we need to convert all of our integer test examples to float32 with valid nodata ahead of loading,
    and save them to temporary test files.
    """

    # Create temporary directory at module scope
    tmpdir = tmp_path_factory.mktemp("lazy_data")

    list_name = ["everest_landsat_b4", "everest_landsat_rgb", "exploradores_aster_dem"]
    list_fn_out = []
    for name in list_name:

        # Get filepath
        fn = examples.get_path_test(name)

        # Else open, convert
        rast = Raster(fn)
        rast = rast.astype(dtype=np.float32, convert_nodata=False)
        rast.set_nodata(-9999, update_array=False, update_mask=False)

        # Save to file in temporary directory
        fn_out = os.path.join(tmpdir, os.path.splitext(os.path.basename(fn))[0] + "_float32.tif")
        rast.to_file(fn_out)

        list_fn_out.append(fn_out)

    return list_fn_out


@pytest.fixture(scope="module")
def lazy_test_files_tiny(tmp_path_factory: Any) -> list[str]:
    """
    Same as lazy_test_files, for tests that need really tiny files (like polygonize).
    """

    # Create temporary directory at module scope
    tmpdir = tmp_path_factory.mktemp("lazy_data")

    list_name = ["everest_landsat_b4", "everest_landsat_rgb", "exploradores_aster_dem"]
    list_fn_out = []
    for name in list_name:

        # Get filepath
        fn = examples.get_path_test(name)

        # Else open, convert
        rast = Raster(fn)
        rast = rast.icrop((0, 0, 26, 24))
        rast = rast.astype(dtype=np.float32, convert_nodata=False)
        rast.set_nodata(-9999, update_array=False, update_mask=False)

        # Save to file in temporary directory
        fn_out = os.path.join(tmpdir, os.path.splitext(os.path.basename(fn))[0] + "_tiny_float32.tif")
        rast.to_file(fn_out)

        list_fn_out.append(fn_out)

    return list_fn_out