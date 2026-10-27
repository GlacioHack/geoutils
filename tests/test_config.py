"""Test configuration file."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr

import geoutils as gu

ResamplingCase = tuple[str, Callable[..., Any], dict[str, Any], list[float]]


@pytest.fixture(params=["reproject", "interp_at_points", "resample_at_points", "grid"])
def resampling_case(request: pytest.FixtureRequest) -> ResamplingCase:
    """Create a resampling call with known interpolation weights."""

    reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.5, 1.5, 0.5, 0.5), 32631)
    options: dict[str, Any] = {"ref": reference}
    if request.param == "grid":
        source: Any = gu.PointCloud.from_xyz([0.5, 1.5, 0.5], [1.5, 1.5, 0.5], [0, 4, 8], 32631)
        options["radius"] = 10
        kind, weights = "point", [0.5, 0.25, 0.25]
    else:
        source = gu.Raster.from_array(np.array([[0, 4], [8, 12.0]]), rio.transform.from_origin(0, 2, 1, 1), 32631)
        kind, weights = "raster", [9 / 16, 3 / 16, 3 / 16, 1 / 16]
        if request.param != "reproject":
            options = {"points": (0.75, 1.25), "as_array": True}
    return kind, getattr(source, request.param), options, weights


class TestConfig:
    """Test module for the global configuration."""

    def test_config_defaults(self) -> None:
        """Checks that configuration defaults match the file."""

        default_config = gu._config.GeoUtilsConfigDict()
        default_config._set_defaults(gu._config._config_ini_file)
        assert default_config == gu.config
        assert gu.config == {
            "warn_area_or_point": False,
            "raster_resampling_method": "linear",
            "point_resampling_method": "linear",
            "interpolation_nodata_handling": "nearest",
            "densify_points": 5000,
            "raster_return_type": "dataarray",
            "pointcloud_return_type": "dataarray",
        }

    @pytest.mark.parametrize(
        "key,value,expected",
        [
            ("warn_area_or_point", True, True),
            ("warn_area_or_point", False, False),
            ("warn_area_or_point", "off", False),
            ("warn_area_or_point", 1, True),
            ("interpolation_nodata_handling", "half_order_up", "half_order_up"),
            ("interpolation_nodata_handling", 1, 1),
            ("interpolation_nodata_handling", "2", 2),
            ("densify_points", 1, 1),
            ("densify_points", 25, 25),
            ("densify_points", "12", 12),
            ("densify_points", np.int64(7), 7),
        ],
    )
    def test_config_validator(self, key: str, value: Any, expected: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that configuration values are converted to the expected types."""

        monkeypatch.setitem(gu.config, key, value)
        assert gu.config[key] == expected
        assert type(gu.config[key]) is type(expected)

    @pytest.mark.parametrize("key", ["raster_resampling_method", "point_resampling_method"])
    @pytest.mark.parametrize("method", ["nearest", "linear", "BILINEAR", "cubic_convolution", "quintic", "idw", "mean"])
    def test_config__resampling_names(self, key: str, method: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that resampling defaults accept names regardless of case."""

        monkeypatch.setitem(gu.config, key, method)
        assert gu.config[key] == method.lower()

    def test_default_reproject(self) -> None:
        """Checks that reprojection, merging and stacking use the configured method."""

        # Open a raster
        raster = gu.Raster(gu.examples.get_path_test("everest_landsat_b4_cropped"))
        raster.set_nodata(0)
        out_size = (raster.shape[1] // 2, raster.shape[0] // 2)
        method = gu.config["raster_resampling_method"]

        # Compare reprojection defaults
        expected = raster.reproject(grid_size=out_size, resampling=method)
        assert raster.reproject(grid_size=out_size).raster_equal(expected)
        _, data, transform, crs, nodata = gu.raster.transformation._reproject(raster, None, grid_size=out_size)
        actual = gu.Raster.from_array(
            data, transform, crs, nodata, area_or_point=raster.area_or_point, tags=raster.tags
        )
        assert actual.raster_equal(expected)

        # Compare merge and stack defaults
        for operation in (gu.raster.merge_rasters, gu.raster.stack):
            actual = operation([raster.copy(), expected.copy()])
            explicit = operation([raster.copy(), expected.copy()], resampling_method=method)
            assert actual.raster_equal(explicit)

    def test_default_interpolation(self) -> None:
        """Checks that point sampling uses the configured method and nodata handling."""

        # Create a small raster
        values = np.flipud(np.arange(1, 10).reshape(3, 3))
        raster = gu.Raster.from_array(values, rio.transform.from_bounds(0, 0, 3, 3, 3, 3), None, nodata=-9999)
        raster.set_area_or_point("Point", shift_half_pixel=False)
        rows, cols = np.indices(raster.shape)
        points = raster.ij2xy(i=rows.ravel(), j=cols.ravel())

        # Compare sampling defaults
        expected = raster.interp_at_points(
            points,
            method=gu.config["raster_resampling_method"],
            nodata_handling=gu.config["interpolation_nodata_handling"],
            as_array=True,
        )
        np.testing.assert_array_equal(raster.interp_at_points(points, as_array=True), expected)
        actual = gu.interface.resampling._resample_at_points(source_raster=raster, points=points, as_array=True)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("method,expected", [("nearest", 0), ("linear", 3), ("bilinear", 3)])
    def test_config__resampling(
        self, resampling_case: ResamplingCase, method: str, expected: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that resampling uses the source default and accepts overrides."""

        # Configure different source defaults
        kind, operation, options, _ = resampling_case
        other = "point" if kind == "raster" else "raster"
        monkeypatch.setitem(gu.config, f"{kind}_resampling_method", method)
        monkeypatch.setitem(gu.config, f"{other}_resampling_method", "linear" if method == "nearest" else "nearest")

        # Compare defaults and overrides
        for input_options in ({}, {"resampling": "nearest"}):
            actual = operation(**options, **input_options)
            if isinstance(actual, gu.Raster):
                actual = actual.to_nanarray()
            # Linear weights: 4/4 + 8/4
            shape = (1,) if "points" in options else (1, 1)
            np.testing.assert_allclose(actual, np.full(shape, 0 if input_options else expected))

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    @pytest.mark.parametrize("explicit", [False, True])
    def test_config__resampling_uncertainty(
        self, resampling_case: ResamplingCase, method: str, explicit: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that estimates and variances use the default or input resampling method."""

        # Configure the source method
        kind, operation, options, weights = resampling_case
        monkeypatch.setitem(gu.config, f"{kind}_resampling_method", method)
        options["resampling"] = "nearest" if explicit else None

        # Propagate measurement errors
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        summary = gu.uncertainty.propagate(operation, error_structure=errors, operation_kwargs=options)
        nearest = explicit or method == "nearest"

        # Variance sums squared weights
        expected_variance = 1 if nearest else np.sum(np.square(weights))
        for result, expected in ((summary.estimate, 0 if nearest else 3), (summary.variance, expected_variance)):
            if isinstance(result, gu.Raster):
                result = result.to_nanarray()
            np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("output_type,expected_type", [("dataarray", xr.DataArray), ("dataset", xr.Dataset)])
    def test_methods__raster_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_type: str, expected_type: type
    ) -> None:
        """Checks that raster outputs follow the default or override and preserve values and loading."""

        # Create matching spatial data
        values = np.arange(6, dtype=float).reshape(2, 3)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32631)
        filename = tmp_path / "raster.tif"
        raster.to_file(filename)
        points = raster.to_pointcloud()
        point_array = points.to_xarray()
        vector = gu.Vector.from_bounds_projected(raster).gdf.vct
        monkeypatch.setitem(gu.config, "raster_return_type", output_type.upper())
        assert gu.config["raster_return_type"] == output_type

        # Compare defaults and overrides
        result_types = {"dataarray": xr.DataArray, "dataset": xr.Dataset}
        for requested in (None, "dataarray", "dataset"):
            options: dict[str, Any] = {} if requested is None else {"as_object_type": requested}
            result_type = expected_type if requested is None else result_types[requested]
            results: list[tuple[Any, Any, bool]] = [
                (gu.raster.from_array(values, raster.transform, raster.crs, **options), values, True),
                (gu.open_raster(str(filename), **options), values, False),
            ]
            if requested is None:
                results.extend(
                    [
                        (point_array.pc.grid(ref=raster, resampling="nearest"), values, True),
                        (points.gdf.pc.grid(ref=raster, resampling="nearest"), values, True),
                        (vector.rasterize(ref=raster, in_value=1), np.ones(raster.shape), True),
                        (vector.create_mask(ref=raster), np.ones(raster.shape, dtype=bool), True),
                    ]
                )
            for result, expected, loaded in results:
                assert isinstance(result, result_type)
                array: Any = result.raster if isinstance(result, xr.Dataset) else result
                assert array.rst.is_loaded is loaded
                assert array.rst.transform == raster.transform and array.rst.crs == raster.crs
                np.testing.assert_array_equal(array.values, expected)
                assert isinstance(result.rst.icrop((0, 0, 2, 1)), result_type)

        # Check both Dataset variables
        dataset = xr.Dataset({"height": point_array, "double": point_array.copy(data=2 * point_array.data)})
        result = dataset.pc.grid(ref=raster, resampling="nearest")
        assert isinstance(result, xr.Dataset)
        np.testing.assert_array_equal(result.height, values)
        np.testing.assert_array_equal(result.double, 2 * values)

    @pytest.mark.parametrize(
        "output_type,expected_type",
        [("dataarray", xr.DataArray), ("dataset", xr.Dataset), ("geodataframe", gpd.GeoDataFrame)],
    )
    def test_methods__pointcloud_outputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_type: str, expected_type: type
    ) -> None:
        """Checks that point cloud outputs follow the default or override."""

        # Create raster and point file
        raster = gu.raster.from_array(np.ones((2, 2)), rio.transform.from_origin(0, 2, 1, 1), 32631)
        filename = tmp_path / "points.gpkg"
        gu.PointCloud.from_xyz([0], [0], [1], 32631, "height").to_file(filename)
        monkeypatch.setitem(gu.config, "pointcloud_return_type", output_type.upper())
        assert gu.config["pointcloud_return_type"] == output_type

        # Compare defaults and overrides
        for explicit in (False, True):
            options: dict[str, Any] = {"as_object_type": "dataarray"} if explicit else {}
            raster_options = {"pointcloud_return_type": "dataarray"} if explicit else {}
            results = [
                gu.pointcloud.from_xyz([0], [0], [1], 32631, **options),
                gu.open_pointcloud(filename, data_name="height", load=not explicit, **options),
                raster.rst.to_pointcloud(**raster_options),
                raster.rst.subsample(1, **raster_options),
            ]
            if not explicit:
                results.append(raster.rst.interp_at_points(([0.5], [1.5]), method="nearest"))
            for result in results:
                assert isinstance(result, xr.DataArray if explicit else expected_type)

    @pytest.mark.parametrize("source_kind", ["raster", "vector", "pointcloud"])
    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("densify_points", [2, 9])
    def test_config__densify_projected_geometry(
        self, source_kind: str, accessor: bool, densify_points: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that projected bounds and footprints use configured or input point counts."""

        # Create matching bounds
        bounds = rio.coords.BoundingBox(-10, 55, 5, 65)
        out_crs = rio.crs.CRS.from_epsg(3995)
        sources = {
            "raster": gu.Raster.from_array(np.zeros((4, 7)), rio.transform.from_bounds(*bounds, 7, 4), 4326),
            "vector": gu.Vector(gpd.GeoDataFrame(geometry=[gu.projtools.bounds2poly(bounds)], crs=4326)),
            "pointcloud": gu.PointCloud.from_xyz([-10, 5], [55, 65], [1, 2], 4326),
        }
        source: Any = sources[source_kind]
        if accessor:
            if source_kind == "raster":
                source = source.to_xarray().rst
            else:
                accessor_name = "vct" if source_kind == "vector" else "pc"
                source = getattr(source.gdf, accessor_name)
        monkeypatch.setitem(gu.config, "densify_points", densify_points)
        assert len(source.footprint.geometry.iloc[0].exterior.coords) == 4 * densify_points + 1

        # Compare defaults and overrides
        for options in ({}, {"densify_points": 1}):
            count = options.get("densify_points", densify_points)
            results = [
                source.get_footprint_projected(out_crs, **options),
                gu.Vector.from_bounds_projected(source, out_crs=out_crs, **options),
            ]
            for result in results:
                assert len(result.geometry.iloc[0].exterior.coords) == 4 * count + 1
            # Limit raster bounds to pixels
            bound_count = min(7, count) if source_kind == "raster" else count
            expected = rio.warp.transform_bounds(4326, out_crs, *bounds, densify_pts=bound_count)
            assert source.get_bbox_projected(out_crs, **options) == expected


class TestConfigErrors:
    """Test module for invalid configuration values."""

    @pytest.mark.parametrize(
        "key,values,message",
        [
            ("warn_area_or_point", [None, "invalid"], "Cannot convert"),
            ("raster_return_type", [None, "invalid", "geodataframe"], "raster_return_type"),
            ("pointcloud_return_type", [None, "invalid"], "pointcloud_return_type"),
            ("raster_resampling_method", [None, "invalid", True, 1], "not a valid resampling method"),
            ("point_resampling_method", [None, "invalid", True, 1], "not a valid resampling method"),
            (
                "interpolation_nodata_handling",
                ["bilinear", "gdal", None, True, -1, 1.5, "-1"],
                "not a valid nodata_handling",
            ),
            (
                "densify_points",
                [None, True, False, 0, -1, 1.5, "invalid", "1.5"],
                "densify_points must be a positive integer",
            ),
        ],
    )
    def test_config__error_invalid_value(self, key: str, values: list[Any], message: str) -> None:
        """Checks an error is raised for invalid values without changing the configuration."""

        original = gu.config[key]
        for value in values:
            with pytest.raises(ValueError, match=message):
                gu.config[key] = value
            assert gu.config[key] == original
