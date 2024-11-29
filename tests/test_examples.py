"""
Test the example files used for testing and documentation
"""

import hashlib
import warnings

import pytest

import geoutils as gu
from geoutils import examples


@pytest.mark.parametrize(
    "example", ["everest_landsat_b4", "everest_landsat_b4_cropped", "everest_landsat_rgb", "exploradores_aster_dem"]
)  # type: ignore
def test_read_paths_raster(example: str) -> None:
    assert isinstance(gu.Raster(examples.get_path(example)), gu.Raster)


@pytest.mark.parametrize("example", ["everest_rgi_outlines", "exploradores_rgi_outlines"])  # type: ignore
def test_read_paths_vector(example: str) -> None:
    warnings.simplefilter("error")
    assert isinstance(gu.Vector(examples.get_path(example)), gu.Vector)


# Original sha256 obtained with `sha256sum filename`
original_sha256 = {
    "everest_landsat_b4": "271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26",
    "everest_landsat_b4_cropped": "0e63d8e9c4770534a1ec267c91e80cd9266732184a114f0bd1aadb5a613215e6",
    "everest_landsat_rgb": "7d0505a8610fd7784cb71c03e5b242715cd1574e978c2c86553d60fd82372c30",
    "everest_rgi_outlines": "d1a5bcd4bd4731a24c2398c016a6f5a8064160fedd5bab10609adacda9ba41ef",
    "exploradores_aster_dem": "dcb0d708d042553cdd2bb4fd82c55b5674a5e0bd6ea46f1a021b396b7d300033",
    "exploradores_rgi_outlines": "19c2dac089ce57373355213fdf2fd72f601bf97f21b04c4920edb1e4384ae2b2",
}


@pytest.mark.parametrize("example", examples.available)  # type: ignore
def test_data_integrity(example: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(examples.get_path(example), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256[example]
