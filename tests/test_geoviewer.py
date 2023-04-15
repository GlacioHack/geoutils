"""Test for geoviewer executable."""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pytest

import geoutils as gu

# Add geoviewer path temporarily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin/geoviewer.py")))
import geoviewer  # noqa


@pytest.mark.parametrize(
    "filename", [gu.examples.get_path("everest_landsat_b4"), gu.examples.get_path("exploradores_aster_dem")]
)  # type: ignore
@pytest.mark.parametrize(
    "option",
    (
        ("-cmap", "Reds"),
        ("-vmin", "-10", "-vmax", "10"),
        ("-band", "1"),
        ("-nocb",),
        ("-clabel", "Test"),
        ("-figsize", "8,8"),
        ("-max_size", "1000"),
        ("-save", "test.png"),
        ("-dpi", "300"),
        ("-nodata", "99"),
        ("-noresampl",),
    ),
)  # type: ignore
def test_geoviewer_valid(capsys, monkeypatch, filename, option):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # To not get exception when testing generic functions such as --help
    try:
        geoviewer.main([filename, *option])
    except SystemExit:
        pass

    # Capture error output (not stdout, just a plot)
    output = capsys.readouterr().err

    # No error should be raised
    assert output == ""

    # Remove file if it was created
    if option[0] == "-save":
        if os.path.exists("test.png"):
            os.remove("test.png")


@pytest.mark.parametrize(
    "filename", [gu.examples.get_path("everest_landsat_b4"), gu.examples.get_path("exploradores_aster_dem")]
)  # type: ignore
@pytest.mark.parametrize(
    "option",
    (
        ("-cmap", "Lols"),
        ("-vmin", "lol"),
        ("-vmin", "lol2"),
        ("-figsize", "blabla"),
        ("-dpi", "300.5"),
        ("-nodata", "lol"),
    ),
)  # type: ignore
def test_geoviewer_invalid(capsys, monkeypatch, filename, option):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # To not get exception when testing generic functions such as --help
    with pytest.raises(ValueError):
        geoviewer.main([filename, *option])
