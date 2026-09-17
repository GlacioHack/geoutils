"""Tests for shared point dataframe tools."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from numpy.typing import NDArray
from rasterio.coords import BoundingBox

from geoutils._dispatch import is_dask_dataframe
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
    """Create a small point dataframe with repeated index labels."""

    positions = np.arange(point_count)
    frame = gpd.GeoDataFrame(
        {"height": positions + 10},
        geometry=gpd.points_from_xy(positions, positions + 1),
        crs=32632,
    )
    frame.index = np.resize([0, 1, 0], point_count)
    return frame


class TestPointDataframe:
    """
    Test module for point dataframe operations: metadata, output construction, assignment and row selection.

    The Dask and multiprocessing tests are gathered in TestPointDataframeChunked further below.
    """

    def test_dataframe_attrs__eager_mapping(self) -> None:
        """Checks that metadata is copied from the input dictionary and that later updates add to it."""

        # Start with one metadata entry and save it on the dataframe
        frame = _make_point_frame()
        attrs = {"source": "fixture"}
        _set_dataframe_attrs(frame, attrs)

        # Change the original dictionary after saving it, then add another entry to the dataframe
        attrs["source"] = "changed"
        _set_dataframe_attrs(frame, {"quality": "checked"})

        # The saved source stays "fixture" (changing the original dictionary had no effect)
        # The second call adds quality without deleting source
        assert _get_dataframe_attrs(frame) == {"source": "fixture", "quality": "checked"}

    @pytest.mark.parametrize("as_dataframe", [False, True])
    def test_build_pointcloud_output__eager_metadata(self, as_dataframe: bool) -> None:
        """Checks that eager output metadata matches its rows and that the requested output type is returned."""

        # Start with the wrong point count and old bounds so the result has to replace both
        frame = _make_point_frame()
        source_bounds = BoundingBox(-1, -2, 20, 30)
        attrs = {"source": "fixture", "point_count": 100, "bounds": source_bounds}
        original_attrs = attrs.copy()

        # Return the dataframe itself when requested, otherwise wrap in a PointCloud
        result = _build_pointcloud_output(
            frame,
            data_column="height",
            as_dataframe=as_dataframe,
            attrs=attrs,
        )
        output_frame = result if as_dataframe else result.ds

        # The current rows give the point count, CRS and geometry type, and height is saved as the data column
        # Bounds are reset to None because this call does not say that the point locations stayed the same
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
        """Checks that preserve_locations=True keeps known bounds but updates the point count."""

        # Start with correct bounds for these coordinates but the wrong number of points
        frame = _make_point_frame()
        source_bounds = BoundingBox(0, 1, 5, 6)
        attrs = {"point_count": 100, "bounds": source_bounds}

        # Tell the output builder that the X/Y coordinates stayed in the same order
        result = _build_pointcloud_output(
            frame,
            data_column="height",
            as_dataframe=True,
            attrs=attrs,
            preserve_locations=True,
        )

        # Six rows give the new point count, while the bounds still describe the unchanged coordinates
        output_attrs = _get_dataframe_attrs(result)
        assert output_attrs["point_count"] == len(frame)
        assert output_attrs["bounds"] == source_bounds

    def test_assign_point_values__position_with_duplicate_labels(self) -> None:
        """Checks that eager values follow row positions when dataframe labels repeat."""

        # Keep only the geometry and use values whose row order is easy to see (0, 10, 20, ...)
        frame = _make_point_frame()
        geometry = frame[["geometry"]]
        values = np.arange(len(frame)) * 10

        # Add both columns by row number: the first value goes to the first point, etc.
        result = _assign_point_values(geometry, {"self": values, "other": values + 1})

        # Repeated labels can confuse normal Pandas alignment, so also check the original row order and geometry
        assert result is not geometry
        assert np.array_equal(result.index, frame.index)
        assert np.array_equal(result.geometry, frame.geometry)
        assert np.array_equal(result["self"], values)
        assert np.array_equal(result["other"], values + 1)
        assert list(geometry.columns) == ["geometry"]

    def test_assign_point_values__empty_mapping(self) -> None:
        """Checks the edge case that an empty value dictionary returns the exact same dataframe."""

        # An empty dictionary has no columns to add, so there is no reason to copy the dataframe
        frame = _make_point_frame()
        result = _assign_point_values(frame, {})

        # Check object identity (not just equal rows) to show that no copy was made
        assert result is frame

    @pytest.mark.parametrize(
        ("indices", "expected_positions"),
        [
            (np.array([True, False, True, False, False, True]), [0, 2, 5]),
            (np.array([0, 2, 2, 5]), [0, 2, 2, 5]),
        ],
    )
    def test_select_point_rows__eager_position(self, indices: NDArray[Any], expected_positions: list[int]) -> None:
        """Checks that eager masks and row numbers select positions even when index labels repeat."""

        # Select rows with either a six-value boolean mask or an ordered list of row numbers
        frame = _make_point_frame()
        result = _select_point_rows(frame, indices)

        # Compare with iloc, which also uses row positions: in the integer case, position 2 must appear twice
        expected = frame.iloc[expected_positions]
        assert_geodataframe_equal(result, expected)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestPointDataframeChunked:
    """
    Test module for lazy metadata, output construction, partition alignment, assignment and row selection.

    Every test that computes rows compares them with the same operation on an in-memory GeoDataFrame.
    """

    def test_dataframe_attrs__dask_private_copy(self) -> None:
        """Checks that Dask metadata is copied and can be read without running any tasks."""

        import dask_geopandas as dgpd
        from dask.callbacks import Callback

        # Make a lazy point table and check that it starts with no GeoUtils metadata
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        assert _get_dataframe_attrs(lazy) == {}
        attrs = {"source": "fixture"}

        # Save and read the metadata inside a callback that records every Dask task that runs
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            _set_dataframe_attrs(lazy, attrs)
            output_attrs = _get_dataframe_attrs(lazy)
        attrs["source"] = "changed"

        # No tasks ran, and changing the original dictionary did not change the saved source value
        assert tasks == []
        assert output_attrs == {"source": "fixture"}

    @pytest.mark.parametrize("preserve_locations", [False, True])
    def test_build_pointcloud_output__dask_metadata(self, preserve_locations: bool) -> None:
        """Checks that lazy output metadata changes without reading any Dask partitions."""

        import dask_geopandas as dgpd
        from dask.callbacks import Callback

        # Start with a known point count and bounds for the input coordinates
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        source_bounds = BoundingBox(0, 1, 5, 6)
        attrs = {"source": "fixture", "point_count": len(frame), "bounds": source_bounds}
        original_attrs = attrs.copy()

        # Build the output and read its metadata while recording any Dask tasks that run
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

        # With preserve_locations=True, the same X/Y coordinates keep their count and bounds
        # With False, either may have changed, so both values are reset to None without reading the rows
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
        """Checks that asking for a PointCloud computes Dask rows and returns the same eager data."""

        import dask_geopandas as dgpd

        # Split the rows into three partitions so making an eager PointCloud has work to compute
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)

        # Build a PointCloud (store an eager GeoDataFrame, not Dask)
        result = _build_pointcloud_output(lazy, data_column="height", as_dataframe=False)

        # The PointCloud should have the same ordered rows, height column and point count
        assert isinstance(result, PointCloud)
        assert_geodataframe_equal(result.ds, frame)
        assert result.data_column == "height"
        assert _get_dataframe_attrs(result.ds)["point_count"] == len(frame)

    @pytest.mark.parametrize("partitions", [2, 5])
    @pytest.mark.parametrize("native_series", [False, True])
    def test_point_rows__reuse_partition_layout(self, partitions: int, native_series: bool) -> None:
        """Checks that Dask assignment and row selection stay lazy with repeated labels and known row counts."""

        import dask.array as da
        import dask_geopandas as dgpd
        from dask.callbacks import Callback

        # Repeat index labels and use uneven partitions (matching by label would put values on the wrong rows)
        positions = np.arange(40)
        frame = gpd.GeoDataFrame(
            {"height": positions}, geometry=gpd.points_from_xy(positions, positions + 1), crs=32632
        )
        frame.index = np.tile(["a", "b", "a", "c"], 10)
        lazy = dgpd.from_geopandas(frame, npartitions=partitions, sort=False)
        geometry = lazy[["geometry"]]
        lengths = None if native_series else _point_partition_lengths(geometry)

        # A Series from this dataframe already has matching partitions and needs no row counts
        # Arrays use chunks of 7/9, so both operations reuse the point partition lengths found above
        values = lazy["height"] if native_series else da.from_array(positions, chunks=7)
        mask = lazy["height"] % 3 == 0 if native_series else da.from_array(positions % 3 == 0, chunks=9)

        # Build the assignment + selection graphs while recording any Dask task that actually runs
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            assigned = _assign_point_values(geometry, {"self": values, "other": values * 2}, partition_lengths=lengths)
            result = _select_point_rows(assigned, mask, partition_lengths=lengths)
        assert tasks == []
        assert is_dask_dataframe(result)

        # Compute the final selection and compare it with the same boolean mask on the eager dataframe
        output = result.compute()
        expected = frame.iloc[positions % 3 == 0]
        assert np.array_equal(output.index, expected.index)
        assert np.array_equal(output.geometry, expected.geometry)
        assert np.array_equal(output["self"], expected["height"])
        assert np.array_equal(output["other"], expected["height"] * 2)
        assert is_dask_dataframe(lazy) and is_dask_dataframe(result)

    def test_assign_point_values__dask_values_to_eager_rows(self) -> None:
        """Checks that Dask values are computed before they are added to an eager dataframe."""

        import dask.array as da

        # Put the values in Dask chunks of 4 + 2
        frame = _make_point_frame()
        values = da.from_array(np.arange(len(frame)) * 2, chunks=4)

        # Add the values to rows that are already in memory, which requires an eager output
        result = _assign_point_values(frame[["geometry"]], {"value": values})

        # The result is a GeoDataFrame, and each doubled value stays with its original row number
        assert isinstance(result, gpd.GeoDataFrame)
        assert not is_dask_dataframe(result)
        assert np.array_equal(result["value"], np.arange(len(frame)) * 2)

    def test_assign_point_values__empty_dask_dataframe(self) -> None:
        """Checks that an empty Dask point table accepts an empty value array and keeps its dtype."""

        import dask.array as da
        import dask_geopandas as dgpd

        # Make a zero-row point table that still has geometry/CRS, plus an empty int16 value array
        frame = _make_point_frame(0)
        lazy = dgpd.from_geopandas(frame, npartitions=1, sort=False)
        values = da.from_array(np.array([], dtype=np.int16), chunks=1)

        # Add the empty column and then compute the one empty partition
        result = _assign_point_values(lazy, {"value": values})
        output = result.compute()

        # The result stays lazy until compute, then has the same empty geometry and an int16 value column
        assert is_dask_dataframe(result)
        assert_geodataframe_equal(output.drop(columns="value"), frame)
        assert output["value"].dtype == np.int16
        assert output.empty

    def test_assign_point_values__unknown_array_length(self) -> None:
        """Checks that a Dask array with an unknown length is split to match the point partitions."""

        import dask
        import dask.array as da
        import dask_geopandas as dgpd

        # Put the values behind a delayed task, so Dask reports the array length as unknown
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        delayed_values = dask.delayed(np.arange)(len(frame))
        values = da.from_delayed(delayed_values, shape=(np.nan,), dtype=np.int64)

        # Let the helper find the values and split them across the same three point partitions
        result = _assign_point_values(lazy[["geometry"]], {"value": values})

        # Compare the computed rows with adding 0..5 directly to the eager dataframe
        output = result.compute()
        expected = frame[["geometry"]].copy()
        expected["value"] = np.arange(len(frame))
        assert_geodataframe_equal(output, expected)

    def test_select_point_rows__integer_partition_boundaries(self) -> None:
        """Checks that sorted row numbers cross Dask partitions and keep a row requested twice."""

        import dask_geopandas as dgpd
        from dask.callbacks import Callback

        # Pick rows from all three two-row partitions, and ask for position 2 twice
        frame = _make_point_frame()
        lazy = dgpd.from_geopandas(frame, npartitions=3, sort=False)
        positions = np.array([0, 2, 2, 3, 5])
        lengths = _point_partition_lengths(lazy)

        # Give the known partition lengths and record whether building the selection runs any point tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            result = _select_point_rows(lazy, positions, partition_lengths=lengths)

        # No tasks ran while building the graph; after compute, the rows match frame.iloc[positions]
        assert tasks == []
        assert is_dask_dataframe(result)
        assert_geodataframe_equal(result.compute(), frame.iloc[positions])


class TestPointDataframeErrors:
    """Test module for validation errors raised by eager and chunked point dataframe helpers."""

    @pytest.mark.parametrize("values", [np.arange(5), np.arange(12).reshape(6, 2)])
    def test_assign_point_values__error_invalid_shape(self, values: NDArray[Any]) -> None:
        """Checks that eager assignment rejects five values for six rows and a two-dimensional array."""

        # Try either five values for six points or a 6 x 2 array with two values per point
        frame = _make_point_frame()

        # A new column needs one value per point (six values in one dimension)
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
        """Checks that lazy row selection rejects 2D, unsorted and out-of-range row numbers."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Try a 2D array, decreasing row numbers, -1 or 6 on a dataframe with rows 0..5
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        lengths = _point_partition_lengths(lazy)

        # Valid row numbers must be a sorted 1D array and stay between 0 and 5
        with pytest.raises(error, match=message):
            _select_point_rows(lazy, positions, partition_lengths=lengths)

    @pytest.mark.parametrize("lengths", [(6,), (2, -1, 5)])
    def test_point_array_partitions__error_invalid_partition_lengths(self, lengths: tuple[int, ...]) -> None:
        """Checks that array alignment rejects a missing partition count and a negative row count."""

        pytest.importorskip("dask")
        import dask.array as da

        dgpd = pytest.importorskip("dask_geopandas")

        # Pair six valid values with either one count for three partitions or the counts (2, -1, 5)
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        values = da.from_array(np.arange(6), chunks=2)

        # All three point partitions need their own row count, and no count can be negative
        with pytest.raises(ValueError, match="one nonnegative row count"):
            _point_array_partitions(lazy, [values], partition_lengths=lengths)

    @pytest.mark.parametrize("values", [np.arange(5), np.arange(12).reshape(6, 2)])
    def test_point_array_partitions__error_invalid_value_shape(self, values: NDArray[Any]) -> None:
        """Checks that array alignment rejects five values for six rows and a two-dimensional array."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Try either five values for six points or a 6 x 2 array with two values per point
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)

        # The three point partitions contain 2 + 2 + 2 rows, so the value array must have six scalar values
        with pytest.raises(ValueError, match="one value per dataframe row"):
            _point_array_partitions(lazy, [values], partition_lengths=(2, 2, 2))

    @pytest.mark.parametrize("kind", ["dataframe", "partitions"])
    def test_assign_point_values__error_invalid_dask_series(self, kind: str) -> None:
        """Checks that Dask values must be a Series with the same three partitions as the points."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Pass either a 2D dataframe or a Series split into two partitions instead of three
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        values: Any
        if kind == "dataframe":
            values = lazy[["height"]]
        else:
            values = lazy.repartition(npartitions=2)["height"]

        # A value column must be one-dimensional and have one matching piece for each point partition
        with pytest.raises(ValueError, match="must follow the dataframe's partition layout"):
            _assign_point_values(lazy[["geometry"]], {"value": values})

    def test_select_point_rows__error_invalid_dask_mask(self) -> None:
        """Checks that a Dask boolean mask must have the same partitions as the point dataframe."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Split the mask into two partitions while the point dataframe still has three
        lazy = dgpd.from_geopandas(_make_point_frame(), npartitions=3, sort=False)
        mask = (lazy.repartition(npartitions=2)["height"] > 11).astype(bool)

        # Reject the mask before adding a selection that would pair the wrong pieces of each dataframe
        with pytest.raises(ValueError, match="must follow the dataframe's partition layout"):
            _select_point_rows(lazy, mask)
