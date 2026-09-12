"""Tests for shared point dataframe assignment and row selection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest

from geoutils._misc import import_optional


class TestPointRowsChunked:
    """
    Checks shared row operations on Dask point tables.

    Duplicate labels must remain at their original positions, and known partition lengths should be reused without
    running the Dask graph. Public cosample() behavior is covered in test_cosampling.py.
    """

    @pytest.mark.parametrize("partitions", [2, 5])
    @pytest.mark.parametrize("native_series", [False, True])
    def test_point_rows__reuse_partition_layout(self, partitions: int, native_series: bool) -> None:
        """Checks that duplicate labels and known row counts do not start Dask computation."""

        import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from geoutils.pointcloud.dataframe import (
            _assign_point_values,
            _point_partition_lengths,
            _select_point_rows,
        )

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
        assert isinstance(result, type(lazy))
        assert not lazy.pc.is_loaded and not result.pc.is_loaded

        # Compare original positions and geometry after computing only the requested lazy result
        output = result.compute()
        expected = frame.iloc[positions % 3 == 0]
        assert np.array_equal(output.index, expected.index)
        assert np.array_equal(output.geometry, expected.geometry)
        assert np.array_equal(output["self"], expected["height"])
        assert np.array_equal(output["other"], expected["height"] * 2)
        assert not lazy.pc.is_loaded and not result.pc.is_loaded
