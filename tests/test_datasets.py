"""
Test datasets
"""
import hashlib

import pytest

import geoutils as gu
from geoutils import datasets


@pytest.mark.parametrize("test_dataset", ["landsat_B4", "landsat_B4_crop", "landsat_RGB"])  # type: ignore
def test_read_paths_raster(test_dataset: str) -> None:
    assert isinstance(gu.Raster(datasets.get_path(test_dataset)), gu.Raster)


@pytest.mark.parametrize("test_dataset", ["glacier_outlines"])  # type: ignore
def test_read_paths_vector(test_dataset: str) -> None:
    assert isinstance(gu.Vector(datasets.get_path(test_dataset)), gu.Vector)


# Original sha256 obtained with `sha256sum filename`
original_sha256 = {
    "landsat_B4": "271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26",
    "landsat_B4_crop": "8d4cf05377957bec9fe9a9ec0c24ea93255e17b529368e8338b1e1865c8b0c83",
    "landsat_RGB": "7d0505a8610fd7784cb71c03e5b242715cd1574e978c2c86553d60fd82372c30",
    "glacier_outlines": "d1a5bcd4bd4731a24c2398c016a6f5a8064160fedd5bab10609adacda9ba41ef",
}


@pytest.mark.parametrize(
    "test_dataset", ["landsat_B4", "landsat_B4_crop", "landsat_RGB", "glacier_outlines"]
)  # type: ignore
def test_data_integrity(test_dataset: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(datasets.get_path(test_dataset), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256[test_dataset]
