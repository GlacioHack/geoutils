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
    assert isinstance(gu.Raster(examples.get_path_test(example)), gu.Raster)


@pytest.mark.parametrize("example", ["everest_rgi_outlines", "exploradores_rgi_outlines"])  # type: ignore
def test_read_paths_vector(example: str) -> None:
    warnings.simplefilter("error")
    assert isinstance(gu.Vector(examples.get_path(example)), gu.Vector)
    assert isinstance(gu.Vector(examples.get_path_test(example)), gu.Vector)


# Original sha256 obtained with `sha256sum filename`
original_sha256_examples = {
    "everest_landsat_b4": "271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26",
    "everest_landsat_b4_cropped": "0e63d8e9c4770534a1ec267c91e80cd9266732184a114f0bd1aadb5a613215e6",
    "everest_landsat_rgb": "7d0505a8610fd7784cb71c03e5b242715cd1574e978c2c86553d60fd82372c30",
    "everest_rgi_outlines": "d1a5bcd4bd4731a24c2398c016a6f5a8064160fedd5bab10609adacda9ba41ef",
    "exploradores_aster_dem": "dcb0d708d042553cdd2bb4fd82c55b5674a5e0bd6ea46f1a021b396b7d300033",
    "exploradores_rgi_outlines": "19c2dac089ce57373355213fdf2fd72f601bf97f21b04c4920edb1e4384ae2b2",
    "coromandel_lidar": "2f1fff1bb84860a8438e14d39e14bf974236dc6345e64649a131507d0ed844f3",
}


@pytest.mark.parametrize("example", examples.available)  # type: ignore
def test_data_integrity__examples(example: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(examples.get_path(example), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256_examples[example]


original_sha256_test = {
    "everest_landsat_b4": "5aa1a0a1c17efd211e42218ab5e2f3e0e404b96ba5055ac5eebef756ad5c65bc",
    "everest_landsat_b4_cropped": "4244767c31c51f7c7b5fb8eb48df7d6394aa707deb9fe699d5672ff9d2507aef",
    "everest_landsat_rgb": "b77109f8027418cdd36ccab34cc3996bbbf2756b116ecf0fed8e4163cd7aa2f9",
    "everest_rgi_outlines": "3642e2fa5da1d9cad2378e0941985ae47077dba7839e30fdd413f8e868ab2ade",
    "exploradores_aster_dem": "c98f24cb131810dd8b2f4773a8df0821cf31edec79890967a67b8a6fdb89314d",
    "exploradores_rgi_outlines": "2f0281b00a49ad2f0874fb4ee54df1e0d11ad073f826d9ca713430588c15fa15",
    "coromandel_lidar": "95af5de14205c712e7674723d00119f4fa6239a65fb2aa3f7035254ace3194ae",
}


@pytest.mark.parametrize("example_test", examples.available_test)  # type: ignore
def test_data_integrity__tests(example_test: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(examples.get_path_test(example_test), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256_test[example_test]
