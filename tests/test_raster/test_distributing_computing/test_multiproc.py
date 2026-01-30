"""
Tests for multiprocessing functions
"""

from multiprocessing import cpu_count

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import Raster, examples
from geoutils.multiproc import AbstractCluster, ClusterGenerator, MultiprocConfig


class TestMultiProc:

    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    num_workers = min(2, cpu_count())  # Safer limit for CI
    cluster = ClusterGenerator("test", nb_workers=num_workers)

    @pytest.mark.skip()  # type: ignore
    @pytest.mark.parametrize("example", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("tile_size", [10, 20])  # type: ignore
    @pytest.mark.parametrize("cluster", [None, cluster])  # type: ignore
    def test_multiproc_reproject(self, example: str, tile_size: int, cluster: None | AbstractCluster) -> None:
        """Test for multiproc_reproject"""

        r = gu.Raster(example)
        config = MultiprocConfig(tile_size, cluster=cluster)

        # specific for the landsat test case, default nodata 255 cannot be used (see above), so use 0
        if r.nodata is None:
            r.set_nodata(0)

        # - Test reprojection with bounds and resolution -
        dst_bounds = rio.coords.BoundingBox(
            left=r.bounds.left, bottom=r.bounds.bottom + r.res[0], right=r.bounds.right - 2 * r.res[1], top=r.bounds.top
        )
        res_tuple = (r.res[0] * 0.5, r.res[1] * 3)

        # Multiprocessing reprojection
        r_multi = r.reproject(bounds=dst_bounds, res=res_tuple, multiproc_config=config)

        # Assert that the raster has not been loaded during reprojection
        assert not r.is_loaded

        # Single-process reprojection
        r_single = r.reproject(bounds=dst_bounds, res=res_tuple)

        # Assert the results are the same
        assert r_single.raster_equal(r_multi)

        # - Test reprojection with CRS change -
        for out_crs in [rio.crs.CRS.from_epsg(4326)]:

            # Single-process reprojection
            r_single = r.reproject(crs=out_crs)

            # Multiprocessing reprojection
            r_multi = r.reproject(crs=out_crs, multiproc_config=config)

            # Assert the results are the same
            assert r_single.raster_equal(r_multi)

        # Check that reprojection works for several bands in multiproc as well
        for n in [2, 3, 4]:
            img1 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"),
                transform=rio.transform.from_origin(0, 500, 1, 1),
                crs=4326,
                nodata=0,
            )
            img2 = Raster.from_array(
                np.ones((n, 500, 500), dtype="uint8"),
                transform=rio.transform.from_origin(50, 500, 1, 1),
                crs=4326,
                nodata=0,
            )

            out_img_single = img2.reproject(img1)
            out_img_multi = img2.reproject(ref=img1, multiproc_config=config)

            assert out_img_multi.count == n
            assert out_img_multi.shape == (500, 500)
            assert out_img_single.raster_equal(out_img_multi)

    @pytest.mark.skip()  # type: ignore
    def test_map_overlap_multiproc_save_bigTiff(self) -> None:
        """
        Test the multiprocessing map function with a simple operation returning a raster > 4go (BigTIFF format)
        """
