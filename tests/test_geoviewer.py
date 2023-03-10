from __future__ import annotations

import os
import pytest

import matplotlib.pyplot as plt
import geoutils.geoviewer as gv
import geoutils as gu

@pytest.mark.parametrize("filename", [gu.examples.get_path("everest_landsat_b4"),
                                      gu.examples.get_path("exploradores_aster_dem")])
@pytest.mark.parametrize("option",
                         (("-cmap", "Reds"), ("-vmin", "-10", "-vmax", "10"),
                          ("-band", "1"), ("-nocb", ), ("-clabel", "Test"),
                          ("-figsize", "8,8"), ("-max_size", "1000"),
                          ("-save", "test.png"), ("-dpi", "300"),
                          ("-nodata", "99"), ("-noresampl", )))
def test_geoviewer(capsys, monkeypatch, filename, option):

    # To avoid having the plots popping up during execution
    monkeypatch.setattr(plt, "show", lambda: None)

    # To not get exception when testing generic functions such as --help
    try:
        gv.main([filename, *option])
    except SystemExit:
        pass

    # Capture error output (not stdout, just a plot)
    output = capsys.readouterr().err

    # No error should be raised
    assert output == ''

    # Remove file if it was created
    if option[0] == "-save":
        if os.path.exists("test.png"):
            os.remove("test.png")
