"""Test configuration file."""

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu


class TestConfig:
    def test_config_defaults(self) -> None:
        """Check defaults compared to file"""

        # Read file
        default_config = gu._config.GeoUtilsConfigDict()
        default_config._set_defaults(gu._config._config_ini_file)

        assert default_config == gu.config

    def test_config_set(self) -> None:
        """Check setting a non-default config argument by user"""

        # Default is True
        assert gu.config["shift_area_or_point"]

        # We set it to False and it should be updated
        gu.config["shift_area_or_point"] = False
        assert not gu.config["shift_area_or_point"]

        # Leave the test with the initial default
        gu.config["shift_area_or_point"] = True
        assert gu.config["shift_area_or_point"]

    def test_config_validator(self) -> None:
        """Check setting a config argument with a wrong input type converts it automatically"""

        # We input an "off" value, that should be converted to False
        gu.config["shift_area_or_point"] = "off"
        assert not gu.config["shift_area_or_point"]

        # Leave the test with initial default
        gu.config["shift_area_or_point"] = 1
        assert gu.config["shift_area_or_point"]

        # sampling_method validator
        gu.config["resampling_method"] = "bilinear"
        with pytest.raises(
            ValueError,
            match="'splinef2d' is not a valid*",
        ):
            gu.config["resampling_method"] = "splinef2d"

        # interpolation_method validator
        gu.config["interpolation_method"] = "linear"
        with pytest.raises(
            ValueError,
            match="'bilinear' is not a valid*",
        ):
            gu.config["interpolation_method"] = "bilinear"

    def test_default_resampling_method(self) -> None:
        landsat_b4_crop_path = gu.examples.get_path_test("everest_landsat_b4_cropped")
        raster = gu.Raster(landsat_b4_crop_path)
        raster.set_nodata(0)

        # test resampling_method for reproject
        out_size = (raster.shape[1] // 2, raster.shape[0] // 2)  # Outsize is (ncol, nrow)
        raster_reproj_force = raster.reproject(grid_size=out_size, resampling=gu.config["resampling_method"])
        raster_reproj = raster.reproject(grid_size=out_size)
        assert raster_reproj_force.raster_equal(raster_reproj)

        # test resampling_method for _reproject
        _, data, transform, crs, nodata = gu.raster.transformation._reproject(raster, None, grid_size=out_size)
        raster__reproject = gu.Raster.from_array(
            data=data, transform=transform, crs=crs, nodata=nodata, area_or_point=raster.area_or_point, tags=raster.tags
        )
        assert raster__reproject.raster_equal(raster_reproj)

        # test resampling_method for merge_rasters
        merged_img_force = gu.raster.merge_rasters(
            [raster, raster_reproj_force], resampling_method=gu.config["resampling_method"]
        )
        merged_img = gu.raster.merge_rasters([raster, raster_reproj_force])
        assert merged_img_force.raster_equal(merged_img)

        stack_force = gu.raster.stack_rasters(
            [raster.copy(), raster_reproj_force.copy()], resampling_method=gu.config["resampling_method"]
        )
        stack = gu.raster.stack_rasters([raster.copy(), raster_reproj_force.copy()])
        assert stack_force.raster_equal(stack)

    def test_default_interpolation_method(self) -> None:

        # Test from test_interpolation.py::TestInterpolate::test_interp_points__synthetic
        arr = np.flipud(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3)))
        transform = rio.transform.from_bounds(0, 0, 3, 3, 3, 3)
        raster = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=-9999)
        raster.set_area_or_point("Point", shift_area_or_point=False)
        index_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        index_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        points_x, points_y = raster.ij2xy(i=index_x, j=index_y)

        raster_points_force = raster.interp_points(
            (points_x, points_y), method=gu.config["interpolation_method"], as_array=True
        )
        raster_points = raster.interp_points((points_x, points_y), as_array=True)

        assert np.array_equal(raster_points, raster_points_force)
