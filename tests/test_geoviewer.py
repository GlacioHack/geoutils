"""Test for geoviewer executable."""

from __future__ import annotations

import os
import sys
import warnings

import matplotlib.pyplot as plt
import pytest

import geoutils as gu

# Add geoviewer path temporarily
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin/geoviewer.py")))
import geoviewer  # noqa


@pytest.mark.parametrize(
    "filename", [gu.examples.get_path_test("everest_landsat_b4"), gu.examples.get_path_test("exploradores_aster_dem")]
)  # type: ignore
@pytest.mark.parametrize(
    "option",
    (
        (),
        ("-cmap", "Reds"),
        ("-cmap", "Reds_r"),
        ("-vmin", "-10", "-vmax", "10"),
        ("-vmin", "5%", "-vmax", "95%"),
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
def test_geoviewer_valid_1band(capsys, monkeypatch, filename, option):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # The everest example will raise errors when setting a nodata value that exists
    if "B4" in os.path.basename(filename) and len(option) > 0 and option[0] == "-nodata":
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
        )

    # To not get exception when testing generic functions such as --help
    try:
        geoviewer.main([filename, *option])
        plt.close()
    except SystemExit:
        pass

    # Capture error output (not stdout, just a plot)
    output = capsys.readouterr().err

    # No error should be raised
    assert output == ""

    # Remove file if it was created
    if "-save" in option:
        if os.path.exists("test.png"):
            os.remove("test.png")


@pytest.mark.parametrize(
    "filename", [gu.examples.get_path_test("everest_landsat_b4"), gu.examples.get_path_test("exploradores_aster_dem")]
)  # type: ignore
@pytest.mark.parametrize(
    "args",
    (
        (("-band", "0"), IndexError),
        (("-band", "2"), IndexError),
        (("-cmap", "Lols"), ValueError),
        (("-cmap", "Lols"), ValueError),
        (("-vmin", "lol"), ValueError),
        (("-vmin", "lol2"), ValueError),
        (("-vmax", "105%"), ValueError),
        (("-figsize", "blabla"), ValueError),
        (("-dpi", "300.5"), ValueError),
        (("-nodata", "lol"), ValueError),
        (("-nodata", "1e40"), ValueError),
    ),
)  # type: ignore
def test_geoviewer_invalid_1band(capsys, monkeypatch, filename, args):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # The everest example will raise errors when setting a nodata value that exists
    if "B4" in os.path.basename(filename) and len(args) > 0 and args[0] == "-nodata":
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
        )

    # To not get exception when testing generic functions such as --help
    option, error = args
    with pytest.raises(error):
        geoviewer.main([filename, *option])


@pytest.mark.parametrize("filename", [gu.examples.get_path_test("everest_landsat_rgb")])  # type: ignore
@pytest.mark.parametrize(
    "option",
    (
        (),
        ("-band", "1"),
        ("-band", "2"),
        ("-band", "3"),
        ("-clabel", "Test"),
        ("-figsize", "8,8"),
        ("-max_size", "1000"),
        ("-save", "test.png"),
        ("-dpi", "300"),
        ("-nodata", "99"),
        ("-noresampl",),
    ),
)  # type: ignore
def test_geoviewer_valid_3band(capsys, monkeypatch, filename, option):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # The everest RGB example will raise errors when setting a nodata value that exists
    if "RGB" in os.path.basename(filename) and len(option) > 0 and option[0] == "-nodata":
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="New nodata value cells already exist in the data array.*"
        )

    # To not get exception when testing generic functions such as --help
    try:
        geoviewer.main([filename, *option])
        plt.close()
    except SystemExit:
        pass

    # Capture error output (not stdout, just a plot)
    output = capsys.readouterr().err

    # No error should be raised
    assert output == ""

    # Remove file if it was created
    if "-save" in option:
        if os.path.exists("test.png"):
            os.remove("test.png")


@pytest.mark.parametrize(
    "filename",
    [
        gu.examples.get_path_test("everest_landsat_rgb"),
    ],
)  # type: ignore
@pytest.mark.parametrize(
    "args",
    (
        (("-band", "0"), IndexError),
        (("-band", "4"), IndexError),
        (("-nodata", "1e40"), ValueError),
    ),
)  # type: ignore
def test_geoviewer_invalid_3band(capsys, monkeypatch, filename, args):  # type: ignore
    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # To not get exception when testing generic functions such as --help
    option, error = args
    with pytest.raises(error):
        geoviewer.main([filename, *option])
