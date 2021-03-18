"""
Test datasets
"""
import pytest
import hashlib

import geoutils.georaster as gr
from geoutils import datasets


@pytest.mark.parametrize(
    "test_dataset", ["landsat_B4", "landsat_B4_crop", "landsat_RGB"]
)
def test_read_paths(test_dataset):
    assert isinstance(gr.Raster(datasets.get_path(test_dataset)), gr.Raster)


# Original sha256 obtained with `sha256sum filename`
original_sha256 = {
    "landsat_B4":
        "271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26",
    "landsat_B4_crop":
        "21b514a627571296eb26690b041f863b9fce4f98037c58a5f6f61deefd639541",
    "landsat_RGB":
        "7d0505a8610fd7784cb71c03e5b242715cd1574e978c2c86553d60fd82372c30"
}


@pytest.mark.parametrize(
    "test_dataset", ["landsat_B4", "landsat_B4_crop", "landsat_RGB"]
)
def test_data_integrity(test_dataset):
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(datasets.get_path(test_dataset), 'rb').read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256[test_dataset]
