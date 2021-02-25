"""
Test datasets
"""
import pytest

import geoutils.georaster as gr
from geoutils import datasets


@pytest.mark.parametrize(
    "test_dataset", ["landsat_B4", "landsat_B4_crop", "landsat_RGB"]
)
def test_read_paths(test_dataset):
    assert isinstance(gr.Raster(datasets.get_path(test_dataset)), gr.Raster)
