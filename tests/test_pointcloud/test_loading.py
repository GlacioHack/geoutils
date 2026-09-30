"""Test point cloud loading by bounds and row ranges."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.coords import BoundingBox

import geoutils as gu
from geoutils.pointcloud.loading import (
    _concat_point_parts,
    _filter_points_by_bounds,
    _load_pointcloud_bounds,
    _load_pointcloud_rows,
)


class TestPointCloudLoading:
    """Test module for selecting point rows, joining partitions, and reading file slices."""

    def test_filter_points_by_bounds__inclusive_and_independent(self) -> None:
        """Checks that bounds include points on their edges and return values independent of the source."""

        # Three points with two on the requested X boundaries
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            crs=32631,
            index=[10, 11, 12],
        )

        # Select both boundary points, then modify the original dataframe
        selected = _filter_points_by_bounds(frame, BoundingBox(1.0, 0.0, 2.0, 0.0))
        frame.loc[11, "height"] = 20.0

        # Check boundary inclusion, row indexes, and independent values
        assert selected.index.tolist() == [11, 12]
        np.testing.assert_array_equal(selected.geometry.x, [1.0, 2.0])
        np.testing.assert_array_equal(selected["height"], [2.0, 3.0])

    def test_concat_point_parts__values_and_indexes(self) -> None:
        """Checks that joined partitions preserve point order, duplicate indexes, and independent values."""

        # Two partitions with a repeated source index and an empty partition between them
        first = gpd.GeoDataFrame(
            {"height": [1.0, 2.0]},
            geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 0.0]),
            crs=32631,
            index=[5, 5],
        )
        second = gpd.GeoDataFrame({"height": [3.0]}, geometry=gpd.points_from_xy([2.0], [0.0]), crs=32631, index=[8])

        # Join the partitions, then change the first source partition
        joined = _concat_point_parts([first, first.iloc[:0], second], crs=first.crs)
        first["height"] = [10.0, 20.0]

        # Check the original row order, duplicate indexes, coordinates, and values
        assert joined.index.tolist() == [5, 5, 8]
        assert joined.crs == first.crs
        np.testing.assert_array_equal(joined.geometry.x, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(joined["height"], [1.0, 2.0, 3.0])

    def test_concat_point_parts__empty_partitions(self) -> None:
        """Checks that no partitions or only empty partitions produce an empty point table with a CRS."""

        # Empty input with a data column for the all-empty case
        empty = gpd.GeoDataFrame({"height": np.array([], dtype=float)}, geometry=[], crs=32631)

        # Combine no partitions and one empty partition
        no_parts = _concat_point_parts([], crs=empty.crs)
        empty_parts = _concat_point_parts([empty], crs=empty.crs)

        # Check empty geometry, coordinate system, and the existing data column
        assert len(no_parts) == len(empty_parts) == 0
        assert no_parts.crs == empty_parts.crs == empty.crs
        assert "height" in empty_parts.columns

    def test_load_pointcloud_bounds__file_ranges(self, tmp_path: Path) -> None:
        """Checks that a lazy point file reads inclusive bounds or all rows for unbounded support."""

        # Write three points so the file reader can select by its spatial index
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            crs=32631,
        )
        filename = tmp_path / "points.gpkg"
        frame.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="height")

        # Read both boundary points and then the complete file through infinite support
        bounded = _load_pointcloud_bounds(source, BoundingBox(1.0, 0.0, 2.0, 0.0), "height")
        unbounded = _load_pointcloud_bounds(source, BoundingBox(-np.inf, -np.inf, np.inf, np.inf), "height")

        # Check selected values and that neither read loaded the PointCloud
        np.testing.assert_array_equal(bounded["height"], [2.0, 3.0])
        np.testing.assert_array_equal(unbounded["height"], frame["height"])
        assert not source.is_loaded

    @pytest.mark.parametrize("loaded", [True, False], ids=["memory", "file"])
    def test_load_pointcloud_rows__slice_and_empty(self, loaded: bool, tmp_path: Path) -> None:
        """Checks that loaded and file-backed point clouds read the same row slice and zero rows."""

        # Four ordered point rows, available in memory or through a file
        frame = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0, 4.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 2.0, 3.0], [0.0] * 4),
            crs=32631,
        )
        filename = tmp_path / "rows.gpkg"
        frame.to_file(filename, index=False)
        source = gu.PointCloud(frame if loaded else filename, data_column="height")

        # Read two rows after the first and an empty slice after the second
        selected = _load_pointcloud_rows(source, start=1, count=2)
        empty = _load_pointcloud_rows(source, start=2, count=0)

        # Check row positions, values, and the file-backed source's lazy state
        np.testing.assert_array_equal(selected.geometry.x, [1.0, 2.0])
        np.testing.assert_array_equal(selected["height"], [2.0, 3.0])
        assert len(empty) == 0
        assert source.is_loaded == loaded
