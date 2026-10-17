"""Test PointCloudBase class, parent of PointCloud class and 'pc' Pandas/Xarray accessor."""

from __future__ import annotations

import inspect
import os.path
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio as rio
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from pyproj.crs import CompoundCRS
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

import geoutils as gu
from geoutils import PointCloud, Raster
from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.base import GeometryPointCloudBase, PointCloudBase
from geoutils.pointcloud.pd_accessor import GeoDataFramePointCloudAccessor
from geoutils.pointcloud.xr_accessor import DataArrayPointCloudAccessor
from tests.accessor_helpers import assert_output_equal
from tests.remote_helpers import RemoteFiles


class NeedsTestError(ValueError):
    """Error to remember to add test when a new PointCloudBase method is added."""


class TestClassVsAccessorConsistency:
    """
    Test module for matching outputs and loading through PointCloud, GeoPandas and Xarray accessors.
    """

    ds = gpd.GeoDataFrame(
        data={"b1": np.array([1.0, 2.0, 3.0, 4.0]), "b2": np.array([5.0, 6.0, 7.0, 8.0])},
        geometry=gpd.points_from_xy(x=np.array([0.0, 1.0, 0.0, 1.0]), y=np.array([0.0, 0.0, 1.0, 1.0])),
        crs=CRS.from_epsg(32610),
    )

    # Get all PointCloudBase public properties and methods, ensures we test everything even with API changes
    public_api = {**PointCloudBase.__dict__, **GeometryPointCloudBase.__dict__}
    property_api = {**public_api, **DataArrayPointCloudAccessor.__dict__}
    properties = [
        k for k, v in property_api.items() if not k.startswith("_") and isinstance(v, property) and k != "data_column"
    ]
    methods = [k for k, v in public_api.items() if not k.startswith("_") and not isinstance(v, property)]
    # Ignore deprecated methods (already tested through their new name)
    methods = [m for m in methods if m not in ["get_stats", "set_data_column"]]

    @pytest.mark.parametrize(
        "prop,as_object_type",
        [
            (prop, as_object_type)
            for prop in properties
            for as_object_type in ("geodataframe", "dataarray")
            if as_object_type == "geodataframe" or hasattr(DataArrayPointCloudAccessor, prop)
        ],
    )
    def test_properties__equality_and_loading(self, prop: str, as_object_type: str) -> None:
        """
        Test that properties are exactly equal between a PointCloud and a GeoDataFrame using the "pc" accessor.
        """

        pc = PointCloud(self.ds, data_name="b1")
        ds = self.ds.copy()
        ds.pc.set_data_name("b1")

        if as_object_type == "dataarray":
            ds = ds.pc.to_xarray()

        output_pc = getattr(pc, "gdf" if prop == "ds" else prop)
        output_ds = getattr(ds.pc, prop)

        if prop == "columns" and as_object_type == "dataarray":
            output_pc = output_pc.drop("geometry")
        assert_output_equal(output_pc, output_ds)
        assert pc.is_loaded
        assert ds.pc.is_loaded

    methods_and_kwargs = [
        ("set_data_name", {"new_data_name": "b2"}),
        ("copy", {}),
        ("clip", {"mask": Polygon([(-0.1, -0.1), (0.5, -0.1), (0.5, 1.1), (-0.1, 1.1)])}),
        ("reproject", {"crs": 4326}),
        ("filter", {"method": "median", "radius": 1.1}),
        ("to_xyz", {}),
        ("to_array", {}),
        ("to_tuples", {}),
        ("pointcloud_equal", {"other": "self"}),
        ("pointcloud_allclose", {"other": "self"}),
        ("georeferenced_coords_equal", {"pc": "self"}),
        ("stats", {}),
        ("stats", {"by": {"group": "b2"}, "bins": {"group": 2}, "statistics": "mean"}),
        ("plot", {"max_points": 2, "add_cbar": False}),
        ("subsample", {"subsample": 2, "random_state": 42}),
        ("cosample", {"other": "self", "subsample": 2, "random_state": 42}),
        (
            "estimate_error_structure",
            {
                "other": PointCloud(ds.assign(b1=[0.0, 2.0, 1.0, 5.0]), data_name="b1"),
                "other_precision": "negligible",
                "components": {"measurement": {"magnitude": "constant", "correlation": None}},
                "spread_estimator": np.std,
            },
        ),
        (
            "pairsample",
            {"n_pairs": 4, "min_distance": 0.5, "max_distance": 2, "strategy": "kdtree", "random_state": 42},
        ),
        (
            "variogram",
            {
                "n_pairs": 4,
                "n_lags": 2,
                "min_lag": 0.5,
                "max_lag": 2,
                "strategy": "kdtree",
                "random_state": 42,
            },
        ),
        ("to_geoutils", {}),
        ("to_xarray", {}),
        (
            "grid",
            {"grid_coords": (np.array([0.0, 1.0]), np.array([0.0, 1.0])), "resampling": "nearest"},
        ),
        (
            "grid",
            {
                "grid_coords": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                "resampling": "nearest",
                "data_name": "b2",
            },
        ),
        (
            "krige",
            {
                "variogram": gu.Variogram.from_model("gaussian", effective_range=1, partial_sill=1),
                "grid_coords": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                "max_overlap": 0.01,
            },
        ),
        (
            "random_field",
            {
                "error_structure": gu.ErrorStructure([gu.ErrorComponent("measurement", 1)]),
                "random_state": 42,
            },
        ),
    ]

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in methods_and_kwargs])
    def test_methods__equality_and_loading(self, as_object_type: str, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that method output and loading are the same between a PointCloud and a GeoDataFrame "pc" accessor.
        """

        pc = PointCloud(self.ds, data_name="b1")
        ds = self.ds.copy()
        ds.pc.set_data_name("b1")
        if as_object_type == "dataarray":
            ds = ds.pc.to_xarray()
            if method == "to_xarray":
                # The array already has the requested representation
                assert pc.to_xarray().identical(ds)
                return
        if method == "variogram":
            pytest.importorskip("skgstat")

        args_pc = kwargs.copy()
        args_ds = kwargs.copy()
        if isinstance(args_pc.get("other"), str) and args_pc["other"] == "self":
            args_pc["other"] = pc
            args_ds["other"] = ds
        elif isinstance(args_pc.get("other"), PointCloud):
            other_ds = args_pc["other"].gdf.copy()
            other_ds.pc.set_data_name("b1")
            args_ds["other"] = other_ds.pc.to_xarray() if as_object_type == "dataarray" else other_ds
        if args_pc.get("pc") == "self":
            args_pc["pc"] = pc
            args_ds["pc"] = ds

        output_pc = getattr(pc, method)(**args_pc)
        output_ds = getattr(ds.pc, method)(**args_ds)

        if method == "set_data_name":
            assert output_pc is None
            if as_object_type == "dataarray":
                assert isinstance(output_ds, xr.DataArray)
                assert_output_equal(pc, output_ds)
                assert ds.pc.data_name == "b1"
            else:
                assert output_ds is None
                assert pc.data_name == ds.pc.data_name
        else:
            assert_output_equal(output_pc, output_ds, use_allclose=method in ("grid", "krige"), strict_masked=False)

        assert pc.is_loaded
        assert ds.pc.is_loaded

    class_methods_and_kwargs = [
        (
            "from_xyz",
            {
                "x": ds.geometry.x.values,
                "y": ds.geometry.y.values,
                "z": ds["b1"].values,
                "crs": CRS.from_epsg(32610),
                "data_name": "b1",
            },
        ),
        (
            "from_array",
            {
                "data": np.vstack((ds.geometry.x.values, ds.geometry.y.values, ds["b1"].values)),
                "crs": CRS.from_epsg(32610),
                "data_name": "b1",
            },
        ),
        (
            "from_tuples",
            {
                "tuples_xyz": list(zip(ds.geometry.x.values, ds.geometry.y.values, ds["b1"].values)),
                "crs": CRS.from_epsg(32610),
                "data_name": "b1",
            },
        ),
    ]

    def test_geo_interface__point_features_and_bbox(self) -> None:
        """Checks that a point cloud exposes point features, values and its bounding box."""

        # Create matching class and accessor representations with an explicit point value column
        pointcloud = PointCloud(self.ds, data_name="b1")
        ds = self.ds.copy()
        ds.pc.set_data_name("b1")
        expected_bbox = rio.coords.BoundingBox(left=0, bottom=0, right=1, top=1)
        expected_interface = ds.__geo_interface__

        # Check the common name and compatibility alias without losing point cloud metadata
        assert pointcloud.bbox == expected_bbox
        assert pointcloud.bounds == expected_bbox
        assert ds.pc.bbox == expected_bbox
        assert ds.pc.bounds == expected_bbox

        # Check that the protocol includes each point and both numeric data columns
        assert pointcloud.__geo_interface__ == expected_interface

    @pytest.mark.parametrize("accessor", [GeoDataFramePointCloudAccessor, DataArrayPointCloudAccessor])
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in class_methods_and_kwargs])
    def test_classmethods__equality(self, accessor: type, method: str, kwargs: dict[str, Any]) -> None:
        """Test class method output exactly the same objects."""

        output_pc = getattr(PointCloud, method)(**kwargs)
        output_ds = getattr(accessor, method)(**kwargs)

        assert_output_equal(output_pc, output_ds)

    def test_methods__test_coverage(self) -> None:
        """Test that checks that all existing PointCloudBase methods are tested above."""

        methods_1 = [m[0] for m in self.methods_and_kwargs]
        methods_2 = [m[0] for m in self.class_methods_and_kwargs]
        # File writes are compared in the GeoParquet roundtrip tests
        methods_1.extend(["to_parquet", "to_file"])
        # Discover array-specific methods alongside the shared and geometry APIs
        array_methods = [
            name
            for name, value in DataArrayPointCloudAccessor.__dict__.items()
            if not name.startswith("_") and not isinstance(value, property)
        ]
        methods_1.extend(["crop", "load", "to_las"])
        list_missing = [method for method in set(self.methods + array_methods) if method not in methods_1 + methods_2]

        if len(list_missing) != 0:
            raise NeedsTestError(f"PointCloudBase methods not covered by tests: {list_missing}")

    def test_equality__cross_type_and_tolerance(self) -> None:
        """Check that equality accepts both APIs while allclose tolerates small numeric differences."""

        pointcloud = PointCloud(self.ds, data_name="b1")
        exact_ds = self.ds.copy()
        exact_ds.pc.set_data_name("b1")
        close_ds = self.ds.copy()
        close_ds.geometry = close_ds.geometry.translate(xoff=1e-9)
        close_ds["b1"] += 1e-9
        close_ds.pc.set_data_name("b1")

        assert pointcloud.pointcloud_equal(exact_ds.pc)
        assert exact_ds.pc.pointcloud_equal(pointcloud)
        assert not pointcloud.pointcloud_equal(close_ds)
        assert pointcloud.pointcloud_allclose(close_ds, atol=1e-8)
        assert close_ds.pc.pointcloud_allclose(pointcloud, atol=1e-8)
        assert not pointcloud.pointcloud_allclose(close_ds, rtol=0, atol=1e-10)

    def test_georeferenced_coords_equal__vertical_and_missing_crs(self) -> None:
        """Checks that coordinate equality ignores vertical CRS differences, warns optionally and accepts no CRS."""

        # 1/ Create matching point clouds with horizontal-only and compound CRS metadata
        horizontal_crs = CRS.from_epsg(32610)
        compound_crs = CompoundCRS("Horizontal and vertical test CRS", [horizontal_crs, CRS.from_epsg(5773)])
        pointcloud = PointCloud(self.ds, data_name="b1")
        compound_ds = self.ds.set_crs(compound_crs, allow_override=True)
        compound_ds.pc.set_data_name("b1")

        # 2/ Check that the vertical difference warns but does not change horizontal coordinate equality
        with pytest.warns(UserWarning, match="same 2D CRS but a different vertical CRS"):
            assert pointcloud.georeferenced_coords_equal(compound_ds.pc)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert pointcloud.georeferenced_coords_equal(compound_ds.pc, warn_3d_crs=False)

        # 3/ Check that missing CRS metadata compares safely
        without_crs = self.ds.set_crs(None, allow_override=True)
        other_without_crs = without_crs.copy()

        assert without_crs.pc.georeferenced_coords_equal(other_without_crs)
        assert not pointcloud.georeferenced_coords_equal(without_crs.pc)

    def test_info__crs_name(self) -> None:
        """Checks that point-cloud info reports the CRS name through both class and accessor APIs."""

        # 1/ Define the CRS cases to apply to the same point cloud data
        horizontal_crs = CRS.from_epsg(32610)
        compound_crs = CompoundCRS("Horizontal and vertical test CRS", [horizontal_crs, CRS.from_epsg(5773)])
        crs_cases = [
            (horizontal_crs, "WGS 84 / UTM zone 10N"),
            (compound_crs, "Horizontal and vertical test CRS"),
            (None, None),
        ]

        # 2/ Check that PointCloud and the Pandas accessor report the same readable name
        for crs, expected_name in crs_cases:
            ds = self.ds.set_crs(crs, allow_override=True)
            ds.pc.set_data_name("b1")
            pointcloud = PointCloud(ds, data_name="b1")
            expected_line = f"Coordinate system:  {[expected_name]}"

            assert expected_line in pointcloud.info(verbose=False).split("\n")
            assert expected_line in ds.pc.info(verbose=False).split("\n")

    def test_copy__preserves_dataframe_and_pointcloud_type(self) -> None:
        """Check that copying retains auxiliary columns, indexes and PointCloud outputs."""

        ds = self.ds.copy()
        ds.index = pd.Index([10, 20, 30, 40], name="point_id")
        pointcloud = PointCloud(ds, data_name="b1")
        replacement = np.array([11.0, 12.0, 13.0, 14.0])

        copied = pointcloud.copy(new_array=replacement)
        copied_ds = ds.pc.copy(new_array=replacement)

        assert isinstance(copied, PointCloud)
        assert copied.columns.equals(ds.columns)
        assert copied.index.equals(ds.index)
        assert np.array_equal(copied["b1"], replacement)
        assert np.array_equal(copied["b2"], ds["b2"])
        assert_geodataframe_equal(copied.gdf, copied_ds)
        assert np.array_equal(pointcloud["b1"], ds["b1"])

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("crop", {"bbox": (-1, -1, 0.5, 2)}),
            ("clip", {"mask": (-1, -1, 0.5, 2)}),
            ("reproject", {"crs": 4326}),
            ("translate", {"xoff": 1, "yoff": 2}),
        ],
    )
    def test_point_preserving_vector_methods__return_pointcloud(self, method: str, kwargs: dict[str, Any]) -> None:
        """Check that inherited point-preserving vector methods retain PointCloud semantics."""

        pointcloud = PointCloud(self.ds, data_name="b1")
        result = getattr(pointcloud, method)(**kwargs)

        assert isinstance(result, PointCloud)
        assert result.data_name == "b1"
        assert "b2" in result.columns
        assert isinstance(pointcloud.to_geoutils(), PointCloud)

    def test_shared_methods_and_arithmetic_ownership(self) -> None:
        """Check that shared operations live in the base while arithmetic remains exclusive to PointCloud."""

        shared_methods = {"stats", "get_stats", "grid"}
        geometry_methods = {"from_xyz", "pointcloud_equal", "pointcloud_allclose"}
        assert shared_methods <= set(PointCloudBase.__dict__)
        assert geometry_methods <= set(GeometryPointCloudBase.__dict__)
        assert shared_methods.isdisjoint(PointCloud.__dict__)
        assert "__add__" not in PointCloudBase.__dict__
        assert "__add__" in PointCloud.__dict__

        with pytest.raises(TypeError):
            self.ds.pc + 1


class TestArrayVsGeometryConsistency:
    """
    Test module for veriyfing the consistency of operations that we explicitly separate in GeometryBase and ArrayBase
    for point clouds, depending on their input type (GeoDataFrame or Xarray).

    These tests have some redundancy with the ones above in TestClassVsAccessorConsistency, but leave the flexibility
    to test more complex cases.
    """

    def test_pointcloud_equal__equivalence(self) -> None:
        """Checks that array and geometry point cloud comparisons are equivalent."""

        # Synthetic data for comparisons
        points = gu.DataArrayPointCloudAccessor.from_xyz([0, 1], [2, 3], [4.0, 5.0], 32633)
        cloud = gu.PointCloud.from_xyz([0, 1], [2, 3], [4.0, 5.0], 32633)

        # Equal output
        assert points.pc.pointcloud_equal(cloud)
        assert cloud.pointcloud_equal(points)

        # Close values
        close = points.pc.copy(new_array=points.data + 1e-6)
        assert not cloud.pointcloud_equal(close)
        assert cloud.pointcloud_allclose(close)
        assert close.pc.pointcloud_allclose(cloud)

        # Different dtypes
        integers = points.pc.copy(new_array=np.array([4, 5], dtype=np.int16))
        assert not points.pc.pointcloud_equal(integers)
        assert points.pc.pointcloud_equal(integers, check_dtype=False)

    @pytest.mark.parametrize("method", ["mean", "median", "min", "max", "std", "count"])
    @pytest.mark.parametrize("include_self", [True, False])
    def test_filter__equivalence(self, method: str, include_self: bool) -> None:
        """Checks that array and geometry point cloud filters are equivalent."""

        # Synthetic data for filtering (duplicates and NaNs)
        cloud = gu.PointCloud.from_xyz([0, 0, 1, 2, np.nan], [0, 0, 0, 0, 0], [1, 3, np.nan, 7, 9], 32633)
        points = cloud.to_xarray()

        # Equal output
        options = {"method": method, "radius": 1.1, "include_self": include_self}
        expected = cloud.filter(**options)
        assert points.pc.filter(**options).pc.pointcloud_equal(expected)

    def test_stats__equivalence(self) -> None:
        """Checks that array and geometry point cloud stats are equivalent."""

        # Synthetic data for stats
        cloud = gu.PointCloud.from_xyz([0, 1, 2, 3], [0, 0, 0, 0], [1.0, 3.0, 5.0, 7.0], 32633)
        cloud.gdf["class"] = [0, 0, 1, 1]
        points = cloud.to_xarray()

        # Equal output
        assert points.pc.stats() == cloud.stats()
        result = points.pc.stats(statistics="mean", by={"class": "class"}, categories={"class": [0, 1]})
        expected = cloud.stats(statistics="mean", by={"class": "class"}, categories={"class": [0, 1]})
        assert result.equals(expected)

    def test_subsample__equivalence(self) -> None:
        """Checks that array and geometry point cloud subsampling are equivalent."""

        # Synthetic data for subsampling (masked rows and uint64 IDs)
        cloud = gu.PointCloud.from_xyz(np.arange(11), np.arange(11), np.arange(11, dtype=float), 32633)
        cloud.gdf["id"] = np.arange(11, dtype=np.uint64) + 2**63
        mask = np.arange(11) % 2 == 0

        # Equal output
        points = cloud.to_xarray()
        options = {"subsample": 3, "random_state": 42, "mask": mask}
        assert points.pc.subsample(**options).pc.pointcloud_equal(cloud.subsample(**options))
        np.testing.assert_array_equal(
            points.pc.subsample(**options, as_array=True), cloud.subsample(**options, as_array=True)
        )
        indices = points.pc.subsample(**options, as_array=True, return_indices=True)
        np.testing.assert_array_equal(indices[0], cloud.subsample(**options, as_array=True, return_indices=True)[0])

    def test_cosample__equivalence(self) -> None:
        """Checks that point cloud cosampling returns the expected shared values."""

        # Synthetic data for cosampling (NaNs in different rows)
        points = gu.DataArrayPointCloudAccessor.from_xyz(np.arange(4), np.zeros(4), [1, np.nan, 3, 4], 32633)
        other = points.pc.copy(new_array=np.array([5, 6, np.nan, 8]))

        # Expected rows 0 and 3
        expected = gu.DataArrayPointCloudAccessor.from_xyz(
            [0, 3], [0, 0], [1.0, 4.0], 32633, data_name="self", auxiliary={"other": [5.0, 8.0]}
        )
        expected = expected.assign_coords(point=[0, 3])

        # Equal output
        result = points.pc.cosample(other)
        assert result.pc.pointcloud_equal(expected)

    @pytest.mark.parametrize("resampling", ["nearest", "linear", "mean", "idw"])
    def test_grid__equivalence(self, resampling: str) -> None:
        """Checks that array and geometry point cloud gridding are equivalent."""

        # Synthetic data for gridding (a plane)
        x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
        cloud = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + 2 * y).ravel(), 32633)
        options = {"grid_coords": (np.arange(4.0), np.arange(4.0)), "resampling": resampling}

        # Equal output
        expected = cloud.grid(**options)
        result = cloud.to_xarray().pc.grid(**options)
        assert result.rst.to_geoutils().raster_equal(expected)

    @pytest.mark.parametrize("mode", ["intersects", "within"])
    def test_crop__equivalence(self, mode: Literal["intersects", "within"]) -> None:
        """Checks that array and geometry point cloud cropping are equivalent."""

        # Synthetic data for cropping (points on bounding box edges)
        cloud = gu.PointCloud.from_xyz([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0], 32633)
        points = cloud.to_xarray()
        bbox = (0.0, -1.0, 2.0, 1.0)

        # Equal output
        expected = cloud.crop(bbox=bbox, mode=mode)
        actual = points.pc.crop(bbox=bbox, mode=mode)
        assert actual.pc.pointcloud_equal(expected)
        assert points.pc.point_count == 3

    def test_reproject_crop_clip__equivalence(self) -> None:
        """Checks that array and geometry point cloud reprojection, cropping and clipping are equivalent."""

        # Synthetic data for reprojection, cropping and clipping
        cloud = gu.PointCloud.from_xyz([500000, 500010, 500020], [0, 10, 20], [1, 2, 3], 32633)
        points = cloud.to_xarray()

        # Equal reprojection output
        result = points.pc.reproject(crs=4326)
        assert result.pc.pointcloud_equal(cloud.reproject(crs=4326))
        assert result.pc.crs == CRS.from_epsg(4326)
        assert points.pc.crs == CRS.from_epsg(32633)

        # Equal cropping and clipping output
        bounds = (500000, 0, 500010, 10)
        assert points.pc.crop(bounds).pc.pointcloud_equal(points.isel(point=[0, 1]))
        assert points.pc.clip(box(*bounds)).pc.pointcloud_equal(points.isel(point=[0, 1]))

    def test_cosample__equivalence_raster(self) -> None:
        """Checks that raster cosampling and regular point conversion return equivalent values."""

        # Synthetic raster data for cosampling (upper-left grid nodes)
        raster = gu.Raster.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), from_origin(0, 3, 1, 1), 32633)
        points = raster.to_pointcloud(pointcloud_return_type="dataarray", force_pixel_offset="ul")
        expected = points.rename("self").assign_coords(other=points.variable)

        # Equal cosampling output
        sampled = points.pc.cosample(raster.to_xarray(), resample_method="nearest")
        assert sampled.pc.pointcloud_equal(expected)

        # Equal regular raster output
        restored = gu.Raster.from_pointcloud_regular(
            points, transform=raster.transform, shape=raster.shape, area_or_point=raster.area_or_point
        )
        np.testing.assert_array_equal(restored.data, raster.data)
        assert restored.georeferenced_grid_equal(raster)

    def test_plot__equivalence(self) -> None:
        """Checks that point cloud plotting returns the expected sample and colorbar."""
        from matplotlib import pyplot as plt

        # Synthetic data for plotting
        points = gu.DataArrayPointCloudAccessor.from_xyz(np.arange(11.0), np.zeros(11), np.arange(11.0), 32633)

        # Equal output
        axes, colorbar_axes = points.pc.plot(max_points=4, ax="new", return_axes=True, cbar_title="Height")
        assert len(axes.collections[0].get_offsets()) == 4
        assert colorbar_axes.get_ylabel() == "Height"
        assert points.pc.point_count == 11
        plt.close(axes.figure)

    @pytest.mark.parametrize("correlated", [False, True])
    def test_estimate_error_structure__equivalence(self, correlated: bool) -> None:
        """Checks that array and geometry point cloud error structures are equivalent."""

        if correlated:
            pytest.importorskip("skgstat")

        # Synthetic data for error structures (variable magnitude)
        rng = np.random.default_rng(11)
        x, y = np.meshgrid(np.arange(20.0), np.arange(20.0))
        quality = np.linspace(0, 1, x.size)
        errors = (0.5 + quality) * rng.normal(size=x.size)
        cloud = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), errors, 32633)
        cloud.gdf["quality"] = quality
        reference = cloud.copy(new_array=np.zeros(x.size))

        # Error model options
        options = {
            "other_precision": "negligible",
            "predictors": {"quality": "quality"},
            "components": {
                "measurement": {"magnitude": "heteroscedastic", "correlation": "spherical" if correlated else None}
            },
            "bins": 4,
            "min_count": 10,
            "spread_estimator": np.std,
            "random_state": 5,
        }
        if correlated:
            options.update(n_pairs=1000, n_lags=8, min_lag=1.0, max_lag=10.0)

        # Equal output
        expected = cloud.estimate_error_structure(reference, **options)
        actual = cloud.to_xarray().pc.estimate_error_structure(reference.to_xarray(), **options)
        probes = {"quality": np.array([0.25, 0.75])}
        np.testing.assert_array_equal(actual.predict_magnitude(probes), expected.predict_magnitude(probes))
        if correlated:
            distances = np.arange(10.0)
            np.testing.assert_array_equal(
                actual.predict_correlation(distances), expected.predict_correlation(distances)
            )


class TestAccessorDask:
    """Test module for Dask loading, laziness and eager comparisons through the point cloud accessors."""

    # Use the same compact fixture as the eager class-versus-accessor tests
    ds = TestClassVsAccessorConsistency.ds

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_open__dask(self, as_object_type: Literal["dataarray", "geodataframe"]) -> None:
        """
        Checks that open_pointcloud() returns lazy point arrays or GeoDataFrame partitions when chunks is supplied.
        """

        # Skip cleanly when the optional lazy dataframe backend is unavailable
        dgpd = pytest.importorskip("dask_geopandas")

        # Write a source that can be reopened into two row partitions
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        # Metadata queries should not force the Dask collection into memory
        ds = gu.open_pointcloud(temp_file, data_name="b1", chunks=2, as_object_type=as_object_type)

        assert isinstance(ds, xr.DataArray if as_object_type == "dataarray" else dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == len(self.ds)

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_bbox__dask(self, as_object_type: Literal["dataarray", "geodataframe"]) -> None:
        """Checks that a lazy point cloud reads its bounding box without replacing its Dask collection."""

        # Write the eager points and reopen them as two lazy row partitions
        pytest.importorskip("dask_geopandas")
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)
        lazy = gu.open_pointcloud(temp_file, data_name="b1", chunks=2, as_object_type=as_object_type)

        # Read the saved bounding box while keeping the accessor backed by Dask
        assert lazy.pc.bbox == rio.coords.BoundingBox(left=0, bottom=0, right=1, top=1)
        assert not lazy.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_chunked_methods__equality_loading_laziness(
        self, as_object_type: Literal["dataarray", "geodataframe"]
    ) -> None:
        """
        Checks that chunked point methods match eager outputs and preserve their expected loading and laziness.
        """

        # Load both lazy dataframe and lazy array types used by this test
        pytest.importorskip("dask_geopandas")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.array as da

        # Prepare matching lazy and eager point-cloud interfaces
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_name="b1", chunks=2, as_object_type=as_object_type)
        pc = PointCloud(self.ds, data_name="b1")

        # Array conversion should create a Dask array with the eager values
        array_ds = ds.pc.to_array()
        array_pc = pc.to_array()

        assert isinstance(array_ds, da.Array)
        assert np.array_equal(array_ds.compute(), array_pc)
        assert not ds.pc.is_loaded

        # Loading returns an eager replacement because a Dask collection cannot be mutated in place
        loaded = ds.pc.load()
        assert_output_equal(pc, loaded)
        assert not ds.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_chunked_equality__compares_values_without_loading(
        self, as_object_type: Literal["dataarray", "geodataframe"]
    ) -> None:
        """Check that lazy equality detects changed coordinates and values without replacing either input."""

        pytest.importorskip("dask_geopandas")

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "source.gpkg")
            changed = os.path.join(temp_dir, "changed.gpkg")
            self.ds.to_file(source)
            changed_ds = self.ds.copy()
            changed_ds.geometry = changed_ds.geometry.translate(xoff=10)
            changed_ds.to_file(changed)

            lazy = gu.open_pointcloud(source, data_name="b1", chunks=3, as_object_type=as_object_type)
            lazy_changed = gu.open_pointcloud(changed, data_name="b1", chunks=2, as_object_type=as_object_type)
            eager = PointCloud(self.ds, data_name="b1")

            assert lazy.pc.pointcloud_equal(eager)
            assert lazy.pc.georeferenced_coords_equal(eager)
            assert not lazy_changed.pc.georeferenced_coords_equal(eager)
            assert not lazy_changed.pc.pointcloud_equal(eager)
            assert not lazy.pc.is_loaded
            assert not lazy_changed.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_chunked_reduction_methods__equality_loading_laziness(
        self, as_object_type: Literal["dataarray", "geodataframe"]
    ) -> None:
        """Test Dask point-cloud reductions/subsampling compute small outputs without loading the source."""

        pytest.importorskip("dask_geopandas")

        # Open the same source lazily and eagerly for reduction comparisons
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_name="b1", chunks=2, as_object_type=as_object_type)
        pc = PointCloud(self.ds, data_name="b1")

        # Statistics compute a small dictionary without loading the accessor source
        assert_output_equal(
            pc.stats(["mean", "max", "valid_count"]),
            ds.pc.stats(["mean", "max", "valid_count"]),
        )
        assert not ds.pc.is_loaded

        # Subsampling keeps the requested point rows lazy until the Dask result is computed
        expected = pc.subsample(subsample=2, random_state=42)
        result = ds.pc.subsample(subsample=2, random_state=42)
        assert is_dask_array(result.data) if as_object_type == "dataarray" else is_dask_dataframe(result)
        assert result.pc.data_name == "b1"
        assert_output_equal(expected, result.compute())
        assert not ds.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_open_las__multiprocessing_and_dask(self, as_object_type: Literal["dataarray", "geodataframe"]) -> None:
        """Check LAS chunked loading through Multiprocessing and Dask."""

        pytest.importorskip("laspy")
        pytest.importorskip("lazrs")
        pytest.importorskip("dask")
        if as_object_type == "geodataframe":
            dgpd = pytest.importorskip("dask_geopandas")

        # Establish one eager reference for both chunked backends
        fn_las = gu.examples.get_path_test("coromandel_lidar")

        pc = PointCloud(fn_las)
        pc.load()

        # Multiprocessing reads independent LAS row chunks into one PointCloud
        pc_mp = PointCloud(fn_las)
        pc_mp.load(mp_config=MultiprocConfig(chunks=100))

        assert pc_mp.is_loaded
        assert pc_mp.pointcloud_equal(pc)

        # Dask represents the same LAS chunks as a lazy GeoDataFrame
        ds = gu.open_pointcloud(fn_las, chunks=100, as_object_type=as_object_type)
        assert isinstance(ds, xr.DataArray if as_object_type == "dataarray" else dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == pc.point_count

        # Compute once for a complete coordinate and value comparison
        ds_comp = ds.compute()
        ds_pc = ds_comp.pc.to_geoutils()
        assert ds_pc.georeferenced_coords_equal(pc)
        assert np.allclose(ds_pc.data, pc.data)
        assert not ds.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_grid_las__multiprocessing_and_dask(self, as_object_type: Literal["dataarray", "geodataframe"]) -> None:
        """Test LAS point-cloud gridding with Dask and Multiprocessing."""

        pytest.importorskip("laspy")
        pytest.importorskip("lazrs")
        da = pytest.importorskip("dask.array")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")

        # Build a small eager grid that both chunked implementations must match
        fn_las = gu.examples.get_path_test("coromandel_lidar")

        pc = PointCloud(fn_las)
        pc.load()
        expected = pc.grid(
            shape=(3, 3),
            bounds=pc.bbox,
            resampling="nearest",
            dist_nodata_pixel=100,
        )

        # Dask selects and grids point partitions separately for every output block
        ds = gu.open_pointcloud(fn_las, chunks=100, as_object_type=as_object_type)
        output_dask = ds.pc.grid(
            shape=(3, 3),
            bounds=pc.bbox,
            resampling="nearest",
            dist_nodata_pixel=100,
            chunksizes=(2, 1),
        )

        # Check graph layout and laziness before computing the grid
        assert isinstance(output_dask.data, da.Array)
        assert output_dask.data.chunks == ((2, 1), (1, 1, 1))
        assert not ds.pc.is_loaded
        assert np.array_equal(expected.data, output_dask.compute().values, equal_nan=True)
        assert not ds.pc.is_loaded
        assert not output_dask._in_memory

        # Multiprocessing keeps the PointCloud unloaded while workers read LAS bounds
        pc_file = PointCloud(fn_las)
        output_mp = pc_file.grid(
            shape=(3, 3),
            bounds=pc.bbox,
            resampling="nearest",
            dist_nodata_pixel=100,
            mp_config=MultiprocConfig(chunks=(2, 1)),
        )

        # Its file-backed raster result should equal the eager and Dask results
        assert not pc_file.is_loaded
        assert not output_mp.is_loaded
        assert np.array_equal(expected.data, output_mp.data, equal_nan=True)
        assert not pc_file.is_loaded
        assert output_mp.is_loaded

    @pytest.mark.parametrize(
        "method, options",
        [
            ("copy", {}),
            ("filter", {"radius": 1.1}),
            ("reproject", {"crs": 4326}),
            ("crop", {"bbox": (0.0, -1.0, 6.0, 1.0), "mode": "within"}),
            ("subsample", {"subsample": 5, "random_state": 42}),
            ("cosample", {}),
        ],
    )
    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_methods__loading_laziness(
        self, as_object_type: Literal["dataarray", "geodataframe"], tmp_path: Path, method: str, options: dict[str, Any]
    ) -> None:
        """Checks that point operations return lazy Dask arrays whether they use DataArrays or GeoDataFrames."""
        pytest.importorskip("dask")
        pytest.importorskip("pyarrow")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")

        # Write then read a chunked point cloud file as Geodataframe/DatArray
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.arange(11.0),
            np.zeros(11),
            np.arange(11.0),
            32633,
            auxiliary={"id": np.arange(11, dtype=np.uint64)},
        )
        filename = tmp_path / "points.parquet"
        points.pc.to_parquet(filename, chunks=4)
        points = gu.open_pointcloud(filename, columns="all", as_object_type=as_object_type)
        lazy = gu.open_pointcloud(filename, columns="all", chunks=4, as_object_type=as_object_type)
        eager_options = {**options, **({"other": points} if method == "cosample" else {})}
        lazy_options = {**options, **({"other": lazy} if method == "cosample" else {})}

        # Run the same operation
        expected = getattr(points.pc, method)(**eager_options)
        result = getattr(lazy.pc, method)(**lazy_options)

        # Check laziness and equality
        assert not lazy.pc.is_loaded
        assert not result.pc.is_loaded
        assert not result.pc.is_loaded
        assert_output_equal(expected.pc.to_geoutils(), result.compute())
        assert not lazy.pc.is_loaded
        assert not result.pc.is_loaded

    @pytest.mark.parametrize("operation", ["predict_magnitude", "random_field"])
    @pytest.mark.parametrize("correlated", [False, True])
    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_uncertainty__loading_laziness(
        self, as_object_type: Literal["dataarray", "geodataframe"], tmp_path: Path, operation: str, correlated: bool
    ) -> None:
        """Checks that uncertainty maps stay lazy and agree exactly across uneven point chunks."""

        pytest.importorskip("dask")
        pytest.importorskip("pyarrow")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")
        if correlated:
            pytest.importorskip("gstools")
        # We define variable magnitudes, and a final shorter chunk to check edge case
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        correlation = gu.Variogram.from_model("gaussian", effective_range=3, partial_sill=1) if correlated else None
        model = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude, correlation)])
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            np.arange(11.0),
            np.zeros(11),
            np.ones(11),
            32633,
            auxiliary={"quality": np.linspace(0, 1, 11)},
        )
        filename = tmp_path / "points.parquet"
        points.pc.to_parquet(filename, chunks=4)
        points = gu.open_pointcloud(filename, columns="all", as_object_type=as_object_type)
        lazy = gu.open_pointcloud(filename, columns="all", chunks=4, as_object_type=as_object_type)
        predictors = {"quality": "quality"}

        # Exact equality and laziness checks
        if operation == "predict_magnitude":
            expected = model.predict_magnitude(predictors, like=points)
            result = model.predict_magnitude(predictors, like=lazy)
        else:
            expected = points.pc.random_field(model, predictors=predictors, random_state=5)
            result = lazy.pc.random_field(model, predictors=predictors, random_state=5)
        assert points.pc.is_loaded
        assert not lazy.pc.is_loaded
        assert not result.pc.is_loaded
        assert_output_equal(expected.pc.to_geoutils(), result.compute())
        assert not lazy.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "geodataframe"])
    def test_grid__loading_laziness(self, as_object_type: Literal["dataarray", "geodataframe"], tmp_path: Path) -> None:
        """Checks that grid keeps laziness on both point cloud types."""

        pytest.importorskip("dask")
        pytest.importorskip("pyarrow")
        if as_object_type == "geodataframe":
            pytest.importorskip("dask_geopandas")

        # Open file lazily
        x, y = np.meshgrid(np.arange(4.0), np.arange(4.0))
        points = gu.DataArrayPointCloudAccessor.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), 32633)
        filename = tmp_path / "points.parquet"
        points.pc.to_parquet(filename, chunks=7)
        points = gu.open_pointcloud(filename, columns="all", as_object_type=as_object_type)
        lazy = gu.open_pointcloud(filename, chunks=7, as_object_type=as_object_type)
        options = {"grid_coords": (np.arange(4.0), np.arange(4.0)), "resampling": "nearest"}

        # Run operation
        expected = points.pc.grid(**options)
        result = lazy.pc.grid(**options, chunksizes=(3, 3))

        # Check exact equality and laziness
        assert not lazy.pc.is_loaded
        assert not result.rst.is_loaded
        assert expected.rst.raster_equal(result.compute())
        assert not lazy.pc.is_loaded
        assert not result.rst.is_loaded


class TestPointCloudNames:
    """
    Test module for deprecation of ``data_column`` main attribute (now renamed ``data_name``) for point clouds,
    and deprecated ``ds`` main attribute for PointCloud class through Vector (now renamed ``gdf``).
    """

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "geodataframe", "dataarray"])
    def test_data_name__deprecated_property(self, as_object_type: str) -> None:
        """Checks that using the old data_column property warns once and returns the same name as data_name."""

        # Create a point cloud in each type
        cloud = PointCloud.from_xyz([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], 32633, data_name="height")
        interfaces = {"pointcloud": cloud, "geodataframe": cloud.gdf.pc, "dataarray": cloud.to_xarray().pc}
        points = interfaces[as_object_type]

        # Check calling the property raises a warning, but works the same
        with pytest.warns(DeprecationWarning, match="data_column.*data_name") as recorded:
            assert points.data_column == points.data_name == "height"
        assert len(recorded) == 1
        description = inspect.getdoc(type(points).data_name)
        assert description is not None
        assert "Name of the point cloud's data." in description

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "geodataframe"])
    def test_data_name__deprecated_setter(self, as_object_type: str) -> None:
        """Checks that setting the old data_column warns and selects the proper point values."""

        # Create a point cloud in each type
        cloud = PointCloud.from_xyz([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], 32633, data_name="height")
        cloud.gdf["quality"] = [4.0, 5.0]
        points = cloud if as_object_type == "pointcloud" else cloud.gdf.pc

        # Check warning and proper behaviour
        with pytest.warns(DeprecationWarning, match="data_column.*data_name"):
            points.data_column = "quality"
        assert points.data_name == "quality"
        np.testing.assert_array_equal(points.data, [4.0, 5.0])

    @pytest.mark.parametrize("as_object_type", ["pointcloud", "geodataframe", "dataarray"])
    def test_set_data_name__deprecated_method(self, as_object_type: str) -> None:
        """Checks that set_data_column() warns and follows set_data_name() for every point representation."""

        # Create a point cloud in each type
        cloud = PointCloud.from_xyz([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], 32633, data_name="height")
        cloud.gdf["quality"] = [4.0, 5.0]
        interfaces = {"pointcloud": cloud, "geodataframe": cloud.gdf.pc, "dataarray": cloud.to_xarray().pc}
        points = interfaces[as_object_type]

        # Check warning and proper behaviour
        with pytest.warns(DeprecationWarning, match="set_data_column.*set_data_name"):
            result = points.set_data_column("quality")
        selected = result.pc if as_object_type == "dataarray" else points
        assert selected.data_name == "quality"
        np.testing.assert_array_equal(selected.data, [4.0, 5.0])

    @pytest.mark.parametrize("constructor", ["from_xyz", "from_array", "from_tuples"])
    @pytest.mark.parametrize("point_class", [PointCloud, DataArrayPointCloudAccessor])
    def test_constructors__deprecated_keyword(self, constructor: str, point_class: type) -> None:
        """Checks that the old data_column keyword warns and constructs the same named point values."""

        # We define inputs for the three constructor functions
        arguments = {
            "from_xyz": {"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "z": [2.0, 3.0, 4.0]},
            "from_array": {"data": np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])},
            "from_tuples": {"tuples_xyz": [(0.0, 0.0, 2.0), (1.0, 1.0, 3.0), (2.0, 2.0, 4.0)]},
        }[constructor]
        create = getattr(point_class, constructor)
        expected = create(**arguments, crs=32633, data_name="height")

        # Check warning and old/new creation yields equal point cloud
        with pytest.warns(DeprecationWarning, match="data_column.*data_name"):
            actual = create(**arguments, crs=32633, data_column="height")
        interface = actual.pc if isinstance(actual, xr.DataArray) else actual
        assert interface.pointcloud_equal(expected)

    def test_pointcloud__deprecated_constructor_keyword(self) -> None:
        """Checks the PointCloud() accepts the old data_column with a warning."""

        # Synthetic data to initialize the point cloud
        frame = gpd.GeoDataFrame({"height": [2.0, 3.0]}, geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 1.0]), crs=32633)

        # Check warning and equality
        with pytest.warns(DeprecationWarning, match="data_column.*data_name"):
            cloud = PointCloud(frame, data_column="height")
        assert cloud.data_name == "height"
        np.testing.assert_array_equal(cloud.data, [2.0, 3.0])

    def test_grid__deprecated_keyword(self) -> None:
        """Checks that grid() accepts data_column with a warning."""

        # Synthetic data to grid
        cloud = PointCloud.from_xyz([0.5, 1.5, 0.5, 1.5], [1.5, 1.5, 0.5, 0.5], [0.0] * 4, 32633, data_name="height")
        cloud.gdf["quality"] = [1.0, 2.0, 3.0, 4.0]
        reference = Raster.from_array(np.zeros((2, 2)), from_origin(0.0, 2.0, 1.0, 1.0), 32633)
        expected = cloud.grid(ref=reference, data_name="quality", resampling="nearest")

        # Check warning and equality
        with pytest.warns(DeprecationWarning, match="data_column.*data_name"):
            raster = cloud.grid(ref=reference, data_column="quality", resampling="nearest")
        assert raster.raster_equal(expected)
        assert cloud.data_name == "height"

    def test_to_pointcloud__deprecated_keyword(self) -> None:
        """Checks that Raster.to_pointcloud() accepts data_column_name and names the point values with data_name."""

        # Synthetic data to convert
        raster = Raster.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), from_origin(0.0, 2.0, 1.0, 1.0), 32633)

        # Check warning and equality
        with pytest.warns(DeprecationWarning, match="data_column_name.*data_name"):
            cloud = raster.to_pointcloud(data_column_name="height")
        assert cloud.data_name == "height"
        np.testing.assert_array_equal(cloud.data, [1.0, 2.0, 3.0, 4.0])

    def test_data_name__error_conflicting_keywords(self) -> None:
        """Checks an error is raised when old and new data name keywords are both supplied."""

        with pytest.raises(TypeError, match="both passed"):
            PointCloud.from_xyz([0.0], [0.0], [1.0], 32633, data_name="height", data_column="quality")

    @pytest.mark.parametrize("use_z", [False, True])
    def test_data_name__legacy_metadata(self, use_z: bool) -> None:
        """Checks that saved data_column metadata selects named values or geometry elevations without warnings."""

        # We write the old metadata name, including None for elevations in 3D geometries
        cloud = PointCloud.from_xyz([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], 32633, data_name="height", use_z=use_z)
        frame = cloud.gdf.copy()
        frame.attrs = {"data_column": None if use_z else "height"}

        # Must not trigger an error
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert frame.pc.data_name == (None if use_z else "height")
            copied = frame.pc.copy()
            np.testing.assert_array_equal(copied.pc.data, [2.0, 3.0])
        assert frame.attrs["data_column"] == copied.attrs["data_name"] == (None if use_z else "height")

    def test_gdf__accessor_storage(self) -> None:
        """Checks that accessor storage stays available as ds and only native objects expose gdf."""

        # Get all object types
        pc = PointCloud.from_xyz([0.0, 1.0], [0.0, 1.0], [2.0, 3.0], 32633, data_name="height")
        df = pc.gdf
        da = pc.to_xarray()

        # Check accessors raise no deprecation error, and have no gdf alias
        # Dataset accessors select several point variables and have separate interface checks
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for accessor in [df.pc, df.vct, da.pc]:
                assert not hasattr(accessor, "gdf")
                assert accessor.ds is not None


@pytest.fixture
def remote_points(remote_files: RemoteFiles) -> str:
    """Write a "mocked" GPKG point cloud in the cloud."""

    coordinates = np.arange(7.0)
    frame = gpd.GeoDataFrame(
        {"height": coordinates + 10, "quality": np.arange(7, dtype="int32")},
        geometry=gpd.points_from_xy(coordinates + 500000, coordinates + 5100000),
        crs=32633,
    )
    filename = remote_files.directory / "points.gpkg"
    frame.to_file(filename)
    return filename.name


@pytest.fixture(params=["las", "laz", "copc.laz", "parquet"])
def remote_native_points(request: pytest.FixtureRequest, remote_files: RemoteFiles) -> str:
    """Write "mocked" LAS/LAZ/COPC/GeoParquet point clouds in the cloud."""

    # Skip unavailable format libraries, then create known point values that LAS can represent exactly
    suffix = request.param
    if suffix == "gpkg":
        return request.getfixturevalue("remote_points")
    if suffix == "copc-levels":
        # Copy a COPC file with two levels
        filename = remote_files.directory / "levels.copc.laz"
        shutil.copyfile(request.getfixturevalue("copc_file"), filename)
        return filename.name

    dependencies = ["pyarrow"] if suffix.startswith("parquet") else ["laspy"]
    if suffix.endswith("laz"):
        dependencies.append("lazrs")
    for dependency in dependencies:
        pytest.importorskip(dependency)
    cloud = gu.PointCloud.from_xyz(np.arange(7.0), np.arange(7.0) + 1, np.arange(7.0) + 10, crs=32633)
    cloud.gdf["intensity"] = np.arange(7, dtype="uint16")
    partitioned = suffix == "parquet-partitioned"
    filename = remote_files.directory / ("partitions" if partitioned else f"points.{suffix}")

    # Use PDAL for COPC (only format not supported directly by LasPy)
    if suffix.startswith("parquet"):
        # Write string and integer attributes
        cloud.gdf["label"] = pd.Series(["ground", "tree", "ground", "tree", "ground", "tree", "ground"], dtype=object)
        cloud.to_parquet(str(filename), chunks=3, partitioned=partitioned)
    elif suffix == "copc.laz":
        pdal = shutil.which("pdal")
        if pdal is not None:
            source = remote_files.directory / "source.las"
            cloud.to_las(str(source))
            subprocess.run(
                [pdal, "translate", str(source), str(filename), "--writer=writers.copc"], check=True, timeout=30
            )
        else:
            pytest.skip("Requires PDAL to write the local COPC reference file.")
    else:
        cloud.to_las(str(filename))
    return filename.name


_REMOTE_POINT_FILES = [
    ("gpkg", "height"),
    ("las", None),
    ("laz", None),
    ("copc.laz", None),
    ("parquet", None),
]
_REMOTE_POINT_OPEN_CASES = [
    pytest.param(suffix, data_name, {}, id=suffix) for suffix, data_name in _REMOTE_POINT_FILES
] + [
    pytest.param("parquet-partitioned", None, {}, id="parquet-partitioned"),
    pytest.param("copc-levels", None, {"level": 0}, id="copc-root"),
    pytest.param("copc-levels", None, {"level": range(2)}, id="copc-levels"),
    pytest.param("copc-levels", None, {"downsample": 4.5}, id="copc-downsample"),
]
_REMOTE_POINT_READERS = [
    pytest.param(gu.PointCloud, {}, id="pointcloud"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "dataarray"}, id="dataarray"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "dataset"}, id="dataset"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "geodataframe"}, id="geodataframe"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "dataarray", "chunks": 3}, id="dataarray-chunked"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "dataset", "chunks": 3}, id="dataset-chunked"),
    pytest.param(gu.open_pointcloud, {"as_object_type": "geodataframe", "chunks": 3}, id="geodataframe-chunked"),
]


class TestPointCloudRemote:
    """Test module for reading remote point clouds (in the cloud)."""

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize("remote_native_points,data_name", _REMOTE_POINT_FILES, indirect=["remote_native_points"])
    def test_init__remote(
        self,
        remote_files: RemoteFiles,
        remote_url: Callable[[str], str],
        remote_native_points: str,
        data_name: str | None,
    ) -> None:
        """Checks that PointCloud instantiation works on a remote file."""

        # Open local/remote
        filename = remote_native_points
        expected = gu.PointCloud(remote_files.directory / filename, data_name=data_name)
        remote = gu.PointCloud(remote_url(filename), data_name=data_name)

        # Check metadata, loading and equality after loading
        assert not expected.is_loaded
        assert not remote.is_loaded
        assert remote.crs == expected.crs
        assert remote.bounds == expected.bounds
        assert remote.columns.equals(expected.columns)
        assert remote.point_count == expected.point_count == 7
        expected.load(columns="all")
        remote.load(columns="all")
        assert remote.is_loaded
        assert_geodataframe_equal(remote.gdf, expected.gdf)

        # Check that remote reads requested part of a file
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize(
        "remote_native_points,data_name,options", _REMOTE_POINT_OPEN_CASES, indirect=["remote_native_points"]
    )
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__remote(
        self,
        remote_files: RemoteFiles,
        remote_url: Callable[[str], str],
        remote_native_points: str,
        data_name: str | None,
        as_object_type: Literal["dataarray", "dataset", "geodataframe"],
        options: dict[str, Any],
    ) -> None:
        """Checks that open_pointcloud() opens a remote point cloud file properly."""

        # Open local and remote
        filename = remote_native_points
        options = {"data_name": data_name, "columns": "all", "as_object_type": as_object_type, **options}
        expected = gu.open_pointcloud(remote_files.directory / filename, **options)
        remote = gu.open_pointcloud(remote_url(filename), **options)

        # Check exact equality
        remote_values = remote[next(iter(remote.data_vars))] if as_object_type == "dataset" else remote
        assert remote_values.pc.is_loaded
        assert_output_equal(remote, expected)

        # Check that remote reads requested part of a file
        assert any(method == "GET" and filename in path for method, path, byte_range in remote_files.requests)
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)

    @pytest.mark.filterwarnings("ignore:.*GPKG application_id.*non conformant file extension:RuntimeWarning")
    @pytest.mark.parametrize("remote_url", ["http", "http-query"], indirect=True)
    @pytest.mark.parametrize(
        "remote_native_points,data_name",
        [("gpkg", "height"), ("las", None), ("parquet", None)],
        indirect=["remote_native_points"],
    )
    def test_open_pointcloud__remote_http_urls(
        self,
        remote_files: RemoteFiles,
        remote_url: Callable[[str], str],
        remote_native_points: str,
        data_name: str | None,
    ) -> None:
        """Checks that open_pointcloud() preserves HTTP URLs with/without query strings."""

        # Open local and remote files through GDAL, LasPy or PyArrow
        filename = remote_native_points
        expected = gu.open_pointcloud(remote_files.directory / filename, data_name=data_name, columns="all")
        remote = gu.open_pointcloud(remote_url(filename), data_name=data_name, columns="all")

        # For query URLs, the server refuses requests that lose the token
        xr.testing.assert_identical(remote, expected)
        assert remote.pc.is_loaded
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    def test_crop__remote_native_format(
        self, remote_files: RemoteFiles, remote_url: Callable[[str], str], remote_native_points: str
    ) -> None:
        """Checks crop on remote files."""

        # Open local/remote without loading points
        expected = gu.PointCloud(remote_files.directory / remote_native_points)
        remote = gu.PointCloud(remote_url(remote_native_points))
        bounds = (1.5, 2.5, 4.5, 5.5)

        # Crop
        expected_crop = expected.crop(bounds)
        remote_crop = remote.crop(bounds)
        assert not expected.is_loaded and not remote.is_loaded
        assert not expected_crop.is_loaded and not remote_crop.is_loaded

        # Load crops and check equality
        expected_crop.load(columns="all")
        remote_crop.load(columns="all")
        assert expected_crop.is_loaded and remote_crop.is_loaded
        assert remote_crop.point_count == 3
        assert_geodataframe_equal(remote_crop.gdf, expected_crop.gdf)
        assert not expected.is_loaded and not remote.is_loaded

        # Check that remote reads requested part of a file
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize("remote_native_points", ["las"], indirect=True)
    def test_init__remote_metadata_range_reads(
        self, remote_files: RemoteFiles, remote_url: Callable[[str], str], remote_native_points: str
    ) -> None:
        """Checks that opening remote LAS metadata uses byte ranges without downloading all point records."""

        # Write enough points to exceed 1 MB buffer
        coordinates = np.arange(100_000.0)
        cloud = gu.PointCloud.from_xyz(coordinates, coordinates + 1, coordinates + 10, crs=32633)
        filename = remote_files.directory / remote_native_points
        cloud.to_las(str(filename))

        # Open metadata and check that points stay unloaded
        remote = gu.PointCloud(remote_url(filename.name))
        assert not remote.is_loaded
        assert remote.point_count == len(coordinates)
        assert remote.crs == cloud.crs

        # Count requested bytes, clipping ranges to the file size
        file_size = filename.stat().st_size
        ranges = [byte_range for method, path, byte_range in remote_files.requests if method == "GET"]
        assert ranges
        requested_bytes = 0
        for byte_range in ranges:
            assert byte_range is not None
            start, end = byte_range.removeprefix("bytes=").split("-")
            requested_bytes += min(int(end) + 1, file_size) - int(start)
        assert requested_bytes < file_size

        # Check that remote reads requested part of a file
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)


class TestPointCloudRemoteChunked:
    """Test module for lazy/chunked reading on remote point clouds."""

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize(
        "remote_native_points,data_name,options", _REMOTE_POINT_OPEN_CASES, indirect=["remote_native_points"]
    )
    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_open_pointcloud__remote_laziness(
        self,
        remote_files: RemoteFiles,
        remote_url: Callable[[str], str],
        remote_native_points: str,
        data_name: str | None,
        as_object_type: Literal["dataarray", "dataset", "geodataframe"],
        options: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that remote point chunks stay lazy and compute exactly the eager local points."""

        pytest.importorskip("dask")
        if as_object_type == "geodataframe" or remote_native_points.endswith(".gpkg"):
            pytest.importorskip("dask_geopandas")

        copc_selection = bool(options)
        if copc_selection:
            # Open remote COPC chunks for level selection (larger to match the fixture)
            chunks = 7001
        else:
            # Open local/remote with chunks (smaller for other files)
            chunks = 3
        filename = remote_native_points
        options = {"data_name": data_name, "columns": "all", "as_object_type": as_object_type, **options}
        expected = gu.open_pointcloud(remote_files.directory / filename, **options)

        # We record when points are decoded
        read_offsets = []
        if copc_selection:
            laspy = pytest.importorskip("laspy")
            original = laspy.CopcReader._fetch_and_decompress_points_of_nodes

            def record_nodes(reader: Any, nodes: list[Any]) -> Any:
                read_offsets.extend(node.offset for node in nodes)
                return original(reader, nodes)

            monkeypatch.setattr(laspy.CopcReader, "_fetch_and_decompress_points_of_nodes", record_nodes)

        # Open local and remote selections
        remote = gu.open_pointcloud(remote_url(filename), chunks=chunks, **options)
        expected_values = expected[next(iter(expected.data_vars))] if as_object_type == "dataset" else expected
        remote_values = remote[next(iter(remote.data_vars))] if as_object_type == "dataset" else remote
        assert expected_values.pc.is_loaded
        assert not remote_values.pc.is_loaded
        expected_count = expected_values.pc.point_count if copc_selection else 7
        assert remote_values.pc.point_count == expected_values.pc.point_count == expected_count
        assert not read_offsets

        # Compute and check equality without loading the original object
        computed = remote.compute(scheduler="synchronous")
        assert_output_equal(computed, expected)
        computed_values = computed[next(iter(computed.data_vars))] if as_object_type == "dataset" else computed
        assert computed_values.pc.is_loaded
        assert not remote_values.pc.is_loaded
        if copc_selection:
            assert read_offsets

        # Check that remote reads requested part of a file
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)

    @pytest.mark.parametrize("remote_url", ["s3", "http-query"], indirect=True)
    @pytest.mark.parametrize("remote_native_points", ["laz", "parquet"], indirect=True)
    def test_open_pointcloud__remote_process_workers(
        self, remote_files: RemoteFiles, remote_url: Callable[[str], str], remote_native_points: str
    ) -> None:
        """Checks that Dask processes reopen remote files and compute the same points as eager local reads."""

        pytest.importorskip("dask")

        # Open local/remote with chunks (Dask)
        expected = gu.open_pointcloud(remote_files.directory / remote_native_points, columns="all")
        remote = gu.open_pointcloud(remote_url(remote_native_points), columns="all", chunks=3)
        assert expected.pc.is_loaded
        assert not remote.pc.is_loaded

        # Compute in new processes and check equality
        computed = remote.compute(scheduler="processes", num_workers=2)
        xr.testing.assert_identical(computed, expected)
        assert computed.pc.is_loaded
        assert not remote.pc.is_loaded

        # Check that remote reads requested part of a file
        assert any(method == "GET" and byte_range for method, path, byte_range in remote_files.requests)


class TestPointCloudRemoteErrors:
    """Test module for errors/warnings on remote point cloud files."""

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize("reader,options", _REMOTE_POINT_READERS)
    @pytest.mark.parametrize(
        "suffix,error",
        [("gpkg", pyogrio.errors.DataSourceError), ("las", FileNotFoundError), ("parquet", ValueError)],
    )
    def test_methods__error_missing_remote_file(
        self,
        remote_url: Callable[[str], str],
        reader: Callable[..., Any],
        options: dict[str, Any],
        suffix: str,
        error: type[Exception],
    ) -> None:
        """Checks an error is raised when a remote point file is unavailable."""

        # Skip optional readers
        dependency = {"las": "laspy", "parquet": "pyarrow"}.get(suffix)
        if dependency is not None:
            pytest.importorskip(dependency)
        if options.get("chunks") is not None and suffix == "gpkg":
            pytest.importorskip("dask_geopandas")
        filename = remote_url(f"missing.{suffix}")
        # Check that opening fails immediately, including with chunks
        with pytest.raises(error):
            reader(filename, data_name="height", **options)

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize("reader,options", _REMOTE_POINT_READERS)
    @pytest.mark.parametrize("remote_native_points", ["gpkg", "las", "parquet"], indirect=True)
    def test_methods__error_missing_data_name(
        self,
        remote_url: Callable[[str], str],
        remote_native_points: str,
        reader: Callable[..., Any],
        options: dict[str, Any],
    ) -> None:
        """Checks an error is raised when a requested point attribute is absent from a remote file."""

        if options.get("chunks") is not None and remote_native_points.endswith(".gpkg"):
            pytest.importorskip("dask_geopandas")

        filename = remote_url(remote_native_points)
        with pytest.raises(ValueError, match="missing"):
            reader(filename, data_name="missing", **options)

    @pytest.mark.parametrize("remote_url", ["s3"], indirect=True)
    @pytest.mark.parametrize("reader,options", _REMOTE_POINT_READERS)
    def test_methods__error_non_point_geometries(
        self,
        remote_files: RemoteFiles,
        remote_url: Callable[[str], str],
        reader: Callable[..., Any],
        options: dict[str, Any],
    ) -> None:
        """Checks an error is raised when a remote vector file contains polygons rather than points."""

        if options.get("chunks") is not None:
            pytest.importorskip("dask_geopandas")

        frame = gpd.GeoDataFrame({"height": [10.0]}, geometry=[box(0, 0, 1, 1)], crs=32633)
        filename = remote_files.directory / "polygons.gpkg"
        frame.to_file(filename)
        with pytest.raises(ValueError, match="non-point"):
            reader(remote_url(filename.name), data_name="height", **options)
