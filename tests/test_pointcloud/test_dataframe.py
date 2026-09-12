"""Tests for shared point dataframe tools."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.typing import NDArray
from rasterio.coords import BoundingBox

from geoutils._dispatch import is_dask_dataframe
from geoutils._misc import import_optional
from geoutils.pointcloud.dataframe import (
    _assign_point_values,
    _build_pointcloud_output,
    _get_dataframe_attrs,
    _point_array_partitions,
    _point_partition_lengths,
    _select_point_rows,
    _set_dataframe_attrs,
)
from geoutils.pointcloud.pointcloud import PointCloud


def _make_point_frame(point_count: int = 6) -> gpd.GeoDataFrame:
    """Create deterministic points with duplicate labels for positional row checks."""

    positions = np.arange(point_count)
    frame = gpd.GeoDataFrame(
        {"height": positions + 10},
        geometry=gpd.points_from_xy(positions, positions + 1),
        crs=32632,
    )
    frame.index = np.resize([0, 1, 0], point_count)
    return frame


class TestPointDataframe:
    """Test module for eager metadata, output construction, positional assignment and row selection."""

    def test_dataframe_attrs__eager_mapping(self) -> None:
        """Checks that eager metadata is updated without sharing the caller's mapping."""

        # Add metadata through the common helper, then mutate the caller's dictionary
        frame = _make_point_frame()
        attrs = {"source": "fixture"}
        _set_dataframe_attrs(frame, attrs)
        attrs["source"] = "changed"
        _set_dataframe_attrs(frame, {"quality": "checked"})

        # Check that Pandas owns the stored values and later updates keep existing metadata
        assert _get_dataframe_attrs(frame) == {"source": "fixture", "quality": "checked"}

    @pytest.mark.parametrize("as_dataframe", [False, True])
    def test_build_pointcloud_output__eager_metadata(self, as_dataframe: bool) -> None:
        """Checks that eager outputs receive current metadata and the requested public type."""

        # Supply stale spatial metadata so output construction must replace values tied to point locations
        frame = _make_point_frame()
        source_bounds = BoundingBox(-1, -2, 20, 30)
        attrs = {"source": "fixture", "point_count": 100, "bounds": source_bounds}
        original_attrs = attrs.copy()

        # Build either the dataframe accessor representation or a PointCloud object
        result = _build_pointcloud_output(
            frame,
            data_column="height",
            as_dataframe=as_dataframe,
            attrs=attrs,
        )
        output_frame = result if as_dataframe else result.ds

        # Check the output type and metadata derived from the current eager rows
        assert (result is frame) if as_dataframe else isinstance(result, PointCloud)
        assert _get_dataframe_attrs(output_frame) == {
            "source": "fixture",
            "point_count": len(frame),
            "bounds": None,
            "data_column": "height",
            "geometry_type": "Point",
            "crs": frame.crs,
        }
        assert attrs == original_attrs

    def test_build_pointcloud_output__preserve_eager_locations(self) -> None:
        """Checks that unchanged eager locations keep bounds while their row count is refreshed."""

        # Describe the known point locations with deliberately stale row-count metadata
        frame = _make_point_frame()
        source_bounds = BoundingBox(0, 1, 5, 6)
        attrs = {"point_count": 100, "bounds": source_bounds}

        # Mark the output as having the same ordered X/Y coordinates
        result = _build_pointcloud_output(
            frame,
            data_column="height",
            as_dataframe=True,
            attrs=attrs,
            preserve_locations=True,
        )

        # Eager rows provide an exact count, while their cached bounds remain valid
        output_attrs = _get_dataframe_attrs(result)
        assert output_attrs["point_count"] == len(frame)
        assert output_attrs["bounds"] == source_bounds

    def test_assign_point_values__position_with_duplicate_labels(self) -> None:
        """Checks that eager values follow row positions when dataframe labels repeat."""

        # Keep only point geometry and prepare values whose order is easy to recognize
        frame = _make_point_frame()
        geometry = frame[["geometry"]]
        values = np.arange(len(frame)) * 10

        # Assign two columns at the matching point positions
        result = _assign_point_values(geometry, {"self": values, "other": values + 1})

        # Duplicate labels must not duplicate or reorder values through Pandas index alignment
        assert result is not geometry
        assert np.array_equal(result.index, frame.index)
        assert np.array_equal(result.geometry, frame.geometry)
        assert np.array_equal(result["self"], values)
        assert np.array_equal(result["other"], values + 1)
        assert list(geometry.columns) == ["geometry"]

    def test_assign_point_values__empty_mapping(self) -> None:
        """Checks that assigning no values returns the original dataframe unchanged."""

        # Use object identity to establish that the no-op path does not make an unnecessary copy
        frame = _make_point_frame()
        result = _assign_point_values(frame, {})

        # No columns or metadata need rebuilding when the mapping is empty
        assert result is frame

    @pytest.mark.parametrize(
        ("indices", "expected_positions"),
        [
            (np.array([True, False, True, False, False, True]), [0, 2, 5]),
            (np.array([0, 2, 2, 5]), [0, 2, 2, 5]),
        ],
    )
    def test_select_point_rows__eager_position(self, indices: NDArray[Any], expected_positions: list[int]) -> None:
        """Checks that eager masks and row numbers select positions independently of labels."""

        # Select duplicate-indexed rows using either a boolean mask or ordered integer positions
        frame = _make_point_frame()
        result = _select_point_rows(frame, indices)

        # Compare with an independent iloc selection, including its repeated requested row
        expected = frame.iloc[expected_positions]
        assert_geodataframe_equal(result, expected)


class TestPointDataframeChunked:
    """
    Test module for lazy metadata, output construction, partition alignment, assignment and row selection.

    Every computed result is compared with the same positional operation on an eager GeoDataFrame.
    """

    def test_dataframe_attrs__dask_private_copy(self) -> None:
        """Checks that Dask metadata is private, copied and available without running the graph."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        # Construct a lazy point table without GeoUtils metadata
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        assert _get_dataframe_attrs(lazy) == {}
        attrs = {"source": "fixture"}

        # Store and read metadata while recording any Dask tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            _set_dataframe_attrs(lazy, attrs)
            output_attrs = _get_dataframe_attrs(lazy)
        attrs["source"] = "changed"

        # The metadata must be independent of the caller and must not evaluate point rows
        assert tasks == []
        assert output_attrs == {"source": "fixture"}

    @pytest.mark.parametrize("preserve_locations", [False, True])
    def test_build_pointcloud_output__dask_metadata(self, preserve_locations: bool) -> None:
        """Checks that lazy dataframe outputs update metadata without computing point partitions."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        # Supply location metadata from an earlier six-point dataframe
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        source_bounds = BoundingBox(0, 1, 5, 6)
        attrs = {"source": "fixture", "point_count": len(frame), "bounds": source_bounds}
        original_attrs = attrs.copy()

        # Build the lazy output while checking that metadata access does not execute the graph
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            result = _build_pointcloud_output(
                lazy,
                data_column="height",
                as_dataframe=True,
                attrs=attrs,
                preserve_locations=preserve_locations,
            )
            output_attrs = _get_dataframe_attrs(result)

        # Location-dependent metadata is reused only under the explicit caller guarantee
        assert tasks == []
        assert result is lazy and is_dask_dataframe(result)
        assert output_attrs == {
            "source": "fixture",
            "point_count": len(frame) if preserve_locations else None,
            "bounds": source_bounds if preserve_locations else None,
            "data_column": "height",
            "geometry_type": "Point",
            "crs": frame.crs,
        }
        assert attrs == original_attrs

    def test_build_pointcloud_output__dask_to_pointcloud(self) -> None:
        """Checks that object output computes lazy rows and constructs an equivalent eager PointCloud."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Split a point table so object construction must collect its lazy partitions
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)

        # Request the object interface, which is eager by design
        result = _build_pointcloud_output(lazy, data_column="height", as_dataframe=False)

        # The PointCloud contains the same ordered rows and current eager metadata
        assert isinstance(result, PointCloud)
        assert_geodataframe_equal(result.ds, frame)
        assert result.data_column == "height"
        assert _get_dataframe_attrs(result.ds)["point_count"] == len(frame)

    @pytest.mark.parametrize("partitions", [2, 5])
    @pytest.mark.parametrize("native_series", [False, True])
    def test_point_rows__reuse_partition_layout(self, partitions: int, native_series: bool) -> None:
        """Checks that duplicate labels and known row counts do not start Dask computation."""

        import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Create duplicate labels and uneven partitions so label-based alignment would change selected rows
        positions = np.arange(40)
        frame = gpd.GeoDataFrame(
            {"height": positions}, geometry=gpd.points_from_xy(positions, positions + 1), crs=32632
        )
        frame.index = np.tile(["a", "b", "a", "c"], 10)
        lazy = dgpd.from_geopandas(frame, npartitions=partitions, sort=False)
        geometry = lazy[["geometry"]]
        lengths = None if native_series else _point_partition_lengths(geometry)

        # Matching Series need no counts; arrays with other chunks can reuse one earlier partition-length summary
        values = lazy["height"] if native_series else da.from_array(positions, chunks=7)
        mask = lazy["height"] % 3 == 0 if native_series else da.from_array(positions % 3 == 0, chunks=9)
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            assigned = _assign_point_values(geometry, {"self": values, "other": values * 2}, partition_lengths=lengths)
            result = _select_point_rows(assigned, mask, partition_lengths=lengths)
        assert tasks == []
        assert is_dask_dataframe(result)

        # Compare original positions and geometry after computing only the requested lazy result
        output = result.compute()
        expected = frame.iloc[positions % 3 == 0]
        assert np.array_equal(output.index, expected.index)
        assert np.array_equal(output.geometry, expected.geometry)
        assert np.array_equal(output["self"], expected["height"])
        assert np.array_equal(output["other"], expected["height"] * 2)
        assert is_dask_dataframe(lazy) and is_dask_dataframe(result)

    def test_assign_point_values__dask_values_to_eager_rows(self) -> None:
        """Checks that lazy value arrays are computed before positional assignment to eager rows."""

        import_optional("dask")
        import dask.array as da

        # Use chunks that do not follow the eager dataframe's row layout
        frame = _make_point_frame()
        values = da.from_array(np.arange(len(frame)) * 2, chunks=4)

        # Eager point rows require an eager output even when values originated in Dask
        result = _assign_point_values(frame[["geometry"]], {"value": values})

        # Values still follow point positions and the output remains a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        assert not is_dask_dataframe(result)
        assert np.array_equal(result["value"], np.arange(len(frame)) * 2)

    def test_assign_point_values__empty_dask_dataframe(self) -> None:
        """Checks that an empty lazy point table accepts an empty value array with its declared dtype."""

        import_optional("dask")
        import dask.array as da

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Build the zero-row case that has no array blocks to align with point partitions
        frame = _make_point_frame(0)
        lazy = dgpd.from_geopandas(frame, npartitions=1, sort=False)
        values = da.from_array(np.array([], dtype=np.int16), chunks=1)

        # Assigning the empty column should keep a valid lazy geospatial table
        result = _assign_point_values(lazy, {"value": values})
        output = result.compute()

        # The result retains geometry metadata and the requested numeric dtype
        assert is_dask_dataframe(result)
        assert_geodataframe_equal(output.drop(columns="value"), frame)
        assert output["value"].dtype == np.int16
        assert output.empty

    def test_assign_point_values__unknown_array_length(self) -> None:
        """Checks that lazy arrays with unknown chunk lengths are aligned to point partitions."""

        dask = import_optional("dask")
        import dask.array as da

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Hide the array length behind a delayed task while keeping its expected dtype explicit
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        delayed_values = dask.delayed(np.arange)(len(frame))
        values = da.from_delayed(delayed_values, shape=(np.nan,), dtype=np.int64)

        # Discover chunk sizes and then align the values with the point partitions
        result = _assign_point_values(lazy[["geometry"]], {"value": values})

        # The computed rows must match positional eager assignment exactly
        output = result.compute()
        expected = frame[["geometry"]].copy()
        expected["value"] = np.arange(len(frame))
        assert_geodataframe_equal(output, expected)

    def test_select_point_rows__integer_partition_boundaries(self) -> None:
        """Checks that sorted row numbers select across lazy partition boundaries and keep repeats."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        # Choose rows on both sides of partition boundaries, including one requested twice
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        positions = np.array([0, 2, 2, 3, 5])
        lengths = _point_partition_lengths(lazy)

        # Construct the positional selection from known lengths without evaluating point partitions again
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            result = _select_point_rows(lazy, positions, partition_lengths=lengths)

        # Dask keeps the graph lazy and returns the same ordered rows as eager iloc
        assert tasks == []
        assert is_dask_dataframe(result)
        assert_geodataframe_equal(result.compute(), frame.iloc[positions])


class TestPointDataframeErrors:
    """Test module for validation errors raised by eager and chunked point dataframe helpers."""

    @pytest.mark.parametrize("values", [np.arange(5), np.arange(12).reshape(6, 2)])
    def test_assign_point_values__error_invalid_shape(self, values: NDArray[Any]) -> None:
        """Checks that eager assignment rejects missing rows and multidimensional values."""

        # Assign invalid values to a six-row point table
        frame = _make_point_frame()

        # Every assigned column must provide exactly one scalar for each ordered point
        with pytest.raises(ValueError, match="one value per dataframe row"):
            _assign_point_values(frame, {"value": values})

    @pytest.mark.parametrize(
        ("positions", "error", "message"),
        [
            (np.array([[1, 2]]), ValueError, "sorted one-dimensional integer array"),
            (np.array([2, 1]), ValueError, "sorted one-dimensional integer array"),
            (np.array([-1]), IndexError, "outside the dataframe"),
            (np.array([6]), IndexError, "outside the dataframe"),
        ],
    )
    def test_select_point_rows__error_invalid_integer_positions(
        self, positions: NDArray[Any], error: type[Exception], message: str
    ) -> None:
        """Checks that lazy row selection rejects invalid shapes, order and bounds."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Apply invalid global positions to a known six-row, three-partition table
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        lengths = _point_partition_lengths(lazy)

        # Positional selection requires a sorted one-dimensional array within the dataframe
        with pytest.raises(error, match=message):
            _select_point_rows(lazy, positions, partition_lengths=lengths)

    @pytest.mark.parametrize("lengths", [(6,), (2, -1, 5)])
    def test_point_array_partitions__error_invalid_partition_lengths(self, lengths: tuple[int, ...]) -> None:
        """Checks that lazy array alignment rejects missing and negative partition lengths."""

        import_optional("dask")
        import dask.array as da

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Pair a valid six-value array with an invalid description of three point partitions
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        values = da.from_array(np.arange(6), chunks=2)

        # Each point partition needs one nonnegative row count
        with pytest.raises(ValueError, match="one nonnegative row count"):
            _point_array_partitions(lazy, [values], partition_lengths=lengths)

    @pytest.mark.parametrize("values", [np.arange(5), np.arange(12).reshape(6, 2)])
    def test_point_array_partitions__error_invalid_value_shape(self, values: NDArray[Any]) -> None:
        """Checks that lazy array alignment rejects missing rows and multidimensional values."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Align invalid values against a six-row point table with known partition lengths
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)

        # A value column must contain one scalar for every point row
        with pytest.raises(ValueError, match="one value per dataframe row"):
            _point_array_partitions(lazy, [values], partition_lengths=(2, 2, 2))

    @pytest.mark.parametrize("kind", ["dataframe", "partitions"])
    def test_assign_point_values__error_invalid_dask_series(self, kind: str) -> None:
        """Checks that lazy Series assignment requires one matching partition per point partition."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Derive either a two-dimensional dataframe or a Series with the wrong partition count
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        values: Any
        if kind == "dataframe":
            values = lazy[["height"]]
        else:
            values = lazy.repartition(npartitions=2)["height"]

        # Native lazy values must have one dimension and follow the point partition layout
        with pytest.raises(ValueError, match="must follow the dataframe's partition layout"):
            _assign_point_values(lazy[["geometry"]], {"value": values})

    def test_select_point_rows__error_invalid_dask_mask(self) -> None:
        """Checks that a lazy boolean Series must match the point dataframe partition layout."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # Repartition a native mask so its partitions no longer correspond to point partitions
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        mask = (lazy.repartition(npartitions=2)["height"] > 11).astype(bool)

        # Reject the mask before adding an invalid partitionwise selection to the graph
        with pytest.raises(ValueError, match="must follow the dataframe's partition layout"):
            _select_point_rows(lazy, mask)
