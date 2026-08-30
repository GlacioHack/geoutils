"""Test VectorBase class, parent of Vector class and 'vct' Pandas accessor."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_series_equal
from pyproj import CRS
from shapely import Polygon

from geoutils import Raster, Vector
from geoutils.vector.base import VectorBase
from geoutils.vector.pd_accessor import VectorAccessor


def assert_output_equal(output_vector: Any, output_ds: Any, use_allclose: bool = False) -> None:
    """Return equality of different output types."""

    # For vectors: the class returns a Vector, while the accessor usually returns a GeoDataFrame
    if isinstance(output_vector, Vector):
        if isinstance(output_ds, Vector):
            assert output_vector.vector_equal(output_ds)
        else:
            assert isinstance(output_ds, gpd.GeoDataFrame)
            assert output_vector.vector_equal(Vector(output_ds))

    # For rasters
    elif isinstance(output_vector, Raster):
        if use_allclose:
            assert output_vector.raster_allclose(output_ds, warn_failure_reason=True, strict_masked=False)
        else:
            assert output_vector.raster_equal(output_ds, warn_failure_reason=True, strict_masked=False)

    # For GeoPandas objects
    elif isinstance(output_vector, gpd.GeoDataFrame):
        assert_geodataframe_equal(output_vector, output_ds)
    elif isinstance(output_vector, gpd.GeoSeries):
        assert_geoseries_equal(output_vector, output_ds)

    # For Pandas and NumPy objects
    elif isinstance(output_vector, pd.Series):
        assert_series_equal(output_vector, output_ds)
    elif isinstance(output_vector, pd.Index):
        assert output_vector.equals(output_ds)
    elif isinstance(output_vector, np.ndarray):
        assert np.array_equal(output_vector, output_ds)

    # For any other object type
    else:
        assert output_vector == output_ds


class NeedsTestError(ValueError):
    """Error to remember to add test when a new VectorBase method is added."""


class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs of the Vector class and Pandas accessor.

    All shared attributes should be the same, except for the filename which is not kept by the Pandas accessor.
    """

    poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = Polygon([(5, 0), (7, 0), (7, 2), (5, 2)])
    poly3 = Polygon([(0, 5), (2, 5), (2, 7), (0, 7)])
    ds = gpd.GeoDataFrame({"id": [1, 2, 3], "geometry": [poly1, poly2, poly3]}, crs="EPSG:32610")

    # Get all VectorBase public properties and methods, ensures we test everything even with API changes
    properties = [k for k, v in VectorBase.__dict__.items() if not k.startswith("_") and isinstance(v, property)]
    methods = [k for k, v in VectorBase.__dict__.items() if not k.startswith("_") and not isinstance(v, property)]
    methods = [m for m in methods if m not in ["plot", "save"]]

    # Methods tested separately because their output contains class/accessor specific filenames
    methods_exceptions = ["info"]

    @pytest.mark.parametrize("prop", properties)
    def test_properties__equality(self, prop: str) -> None:
        """
        Test that properties are exactly equal between a Vector and a GeoDataFrame using the "vct" accessor.
        """

        vector = Vector(self.ds)
        ds = self.ds.copy()

        output_vector = getattr(vector, prop)
        output_ds = getattr(ds.vct, prop)

        if prop == "name":
            assert output_vector is output_ds
        else:
            assert_output_equal(output_vector, output_ds)

    methods_and_kwargs = [
        ("copy", {}),
        ("vector_equal", {"other": "self"}),
        ("vector_allclose", {"other": "self"}),
        ("crop", {"bbox": (-1, -1, 3, 3)}),
        ("reproject", {"crs": CRS.from_epsg(4326)}),
        ("translate", {"xoff": 1, "yoff": 2}),
        (
            "create_mask",
            {"res": 1, "bounds": (-1, -1, 8, 8), "crs": CRS.from_epsg(32610), "as_array": True},
        ),
        (
            "rasterize",
            {"res": 1, "bounds": (-1, -1, 8, 8), "crs": CRS.from_epsg(32610)},
        ),
        ("query", {"expression": "id == 1"}),
        (
            "proximity",
            {
                "raster": Raster.from_array(
                    data=np.zeros((5, 5)),
                    transform=rio.transform.from_bounds(-1, -1, 8, 8, 5, 5),
                    crs=CRS.from_epsg(32610),
                )
            },
        ),
        ("buffer_metric", {"buffer_size": 1}),
        ("get_bounds_projected", {"out_crs": CRS.from_epsg(4326)}),
        ("get_footprint_projected", {"out_crs": CRS.from_epsg(4326)}),
        ("get_metric_crs", {"local_crs_type": "universal"}),
        ("buffer_without_overlap", {"buffer_size": 1, "metric": False}),
        ("to_geoutils", {}),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in methods_and_kwargs])
    def test_methods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that the method output is exactly the same between a Vector and a GeoDataFrame using the "vct" accessor.
        """

        vector = Vector(self.ds)
        ds = self.ds.copy()
        args_vector = kwargs.copy()
        args_ds = kwargs.copy()
        if args_vector.get("other") == "self":
            args_vector["other"] = vector
            args_ds["other"] = ds

        output_vector = getattr(vector, method)(**args_vector)
        output_ds = getattr(ds.vct, method)(**args_ds)

        assert_output_equal(output_vector, output_ds, use_allclose=method == "proximity")

    inplace_methods_and_kwargs = [
        ("crop", {"bbox": (9, 9, 11, 11), "inplace": True}),
        ("reproject", {"crs": CRS.from_epsg(4326), "inplace": True}),
        ("translate", {"xoff": 1, "yoff": 2, "inplace": True}),
        ("query", {"expression": "id == 1", "inplace": True}),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in inplace_methods_and_kwargs])
    def test_inplace_methods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """Test that in-place methods update the Vector and Pandas accessor consistently."""

        vector = Vector(self.ds)
        ds = self.ds.copy()

        output_vector = getattr(vector, method)(**kwargs)
        output_ds = getattr(ds.vct, method)(**kwargs)

        assert output_vector is None
        assert output_ds is None
        assert vector.vector_equal(Vector(ds))

    def test_info(self) -> None:
        """Test that info() contains the same main information for the class and accessor."""

        vector = Vector(self.ds)
        ds = self.ds.copy()

        output_vector = vector.info(verbose=False)
        output_ds = ds.vct.info(verbose=False)

        for text in ["Coordinate system", "Extent", "Number of features", "Attributes"]:
            assert text in output_vector
            assert text in output_ds

    def test_equality__cross_type_and_tolerance(self) -> None:
        """Check that equality accepts both APIs while allclose tolerates small coordinate differences."""

        vector = Vector(self.ds)
        close_ds = self.ds.copy()
        close_ds.geometry = close_ds.geometry.translate(xoff=1e-9)

        assert vector.vector_equal(self.ds.vct)
        assert self.ds.vct.vector_equal(vector)
        assert not vector.vector_equal(close_ds)
        assert vector.vector_allclose(close_ds, atol=1e-8)
        assert close_ds.vct.vector_allclose(vector, atol=1e-8)
        assert not vector.vector_allclose(close_ds, rtol=0, atol=1e-10)

    def test_shared_methods_are_owned_by_base(self) -> None:
        """Check that GeoUtils operations shared with the accessor are not redefined on Vector."""

        shared_methods = {"vector_equal", "vector_allclose", "crop", "reproject", "rasterize", "proximity"}
        assert shared_methods <= set(VectorBase.__dict__)
        assert shared_methods.isdisjoint(Vector.__dict__)

    def test_methods__test_coverage(self) -> None:
        """Test that checks that all existing VectorBase methods are tested above."""

        methods_1 = [m[0] for m in self.methods_and_kwargs]
        methods_2 = [m[0] for m in self.class_methods_and_kwargs]
        list_missing = [
            method for method in self.methods if method not in methods_1 + methods_2 + self.methods_exceptions
        ]

        if len(list_missing) != 0:
            raise NeedsTestError(f"VectorBase methods not covered by tests: {list_missing}")

    class_methods_and_kwargs = [
        (
            "from_bounds_projected",
            {"raster_or_vector": Vector(ds), "out_crs": CRS.from_epsg(4326)},
        ),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in class_methods_and_kwargs])
    def test_classmethods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """Test class method output exactly the same objects."""

        output_vector = getattr(Vector, method)(**kwargs)
        output_ds = getattr(VectorAccessor, method)(**kwargs)

        assert_output_equal(output_vector, output_ds)
