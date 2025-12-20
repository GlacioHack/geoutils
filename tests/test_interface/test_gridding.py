"""Test module for gridding functionalities."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from shapely import geometry

from geoutils import Raster
from geoutils.interface.gridding import _grid_pointcloud


class TestPointCloud:
    def test_grid_pc(self) -> None:
        """Test point cloud gridding."""

        # 1/ Check gridding interpolation falls back exactly on original raster

        # Create a point cloud from interpolating a grid, so we can compare back after to check consistency
        rng = np.random.default_rng(42)
        shape = (10, 12)
        rst_arr = np.linspace(0, 10, int(np.prod(shape))).reshape(*shape)
        transform = rio.transform.from_origin(0, shape[0] - 1, 1, 1)
        rst = Raster.from_array(rst_arr, transform=transform, crs=4326, nodata=100)

        # Generate random coordinates to interpolate, to create an irregular point cloud
        points = rng.integers(low=1, high=shape[0] - 1, size=(100, 2)) + rng.normal(0, 0.15, size=(100, 2))
        b1_value = rst.interp_points((points[:, 0], points[:, 1]), as_array=True)
        pc = gpd.GeoDataFrame(data={"b1": b1_value}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]))
        grid_coords = rst.coords(grid=False)

        # Grid the point cloud
        gridded_pc, output_transform = _grid_pointcloud(pc, grid_coords=grid_coords, data_column_name="b1")

        # Compare back to raster, all should be very close (but not exact, some info is lost due to interpolations)
        valids = np.isfinite(gridded_pc)
        assert np.allclose(gridded_pc[valids], rst.data.data[valids], rtol=10e-5)
        # And the transform exactly the same
        assert output_transform == transform

        # 2/ Check the propagation of nodata values

        # 2.1/ Grid points outside the convex hull of all points should always be nodata

        # We convert the full raster to a point cloud, keeping all cells even nodata
        rst_pc = rst.to_pointcloud(skip_nodata=False).ds

        # We define a multi-point geometry from the individual points, and compute its convex hull
        poly = geometry.MultiPoint([[p.x, p.y] for p in pc.geometry])
        chull = poly.convex_hull

        # We compute the index of grid cells intersecting the convex hull
        ind_inters_convhull = rst_pc.intersects(chull)

        # We get corresponding 1D indexes for gridded output
        i, j = rst.xy2ij(x=rst_pc.geometry.x.values, y=rst_pc.geometry.y.values)

        # Check all values outside convex hull are NaNs
        assert all(~np.isfinite(gridded_pc[i[~ind_inters_convhull], j[~ind_inters_convhull]]))

        # 2.2/ For the rest of the points, data should be valid only if a point exists within 1 pixel of their
        # coordinate, that is the closest rounded number
        # TODO: Replace by check with distance, because some pixel not rounded can also be at less than 1 from a point

        # Compute min distance to irregular point cloud for each grid point
        list_min_dist = []
        for p in rst_pc.geometry:
            min_dist = np.min(np.sqrt((p.x - pc.geometry.x.values) ** 2 + (p.y - pc.geometry.y.values) ** 2))
            list_min_dist.append(min_dist)

        ind_close = np.array(list_min_dist) <= 1
        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close to pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # Check for a different distance value
        gridded_pc, output_transform = _grid_pointcloud(
            pc, grid_coords=grid_coords, dist_nodata_pixel=0.5, data_column_name="b1"
        )
        ind_close = np.array(list_min_dist) <= 0.5

        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close  pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # 3/ Errors
        with pytest.raises(TypeError, match="Input grid coordinates must be 1D arrays.*"):
            Raster.from_pointcloud_regular(pc, grid_coords=(1, "lol"))  # type: ignore
        with pytest.raises(ValueError, match="Grid coordinates must be regular*"):
            grid_coords[0][0] += 1
            Raster.from_pointcloud_regular(pc, grid_coords=grid_coords)  # type: ignore
