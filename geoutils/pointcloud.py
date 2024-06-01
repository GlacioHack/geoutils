"""Module for point cloud manipulation."""

import warnings
from typing import Literal

import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata

from geoutils._typing import NDArrayNum


def _grid_pointcloud(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    data_column_name: str = "b1",
    resampling: Literal["nearest", "linear", "cubic"] = "linear",
    dist_nodata_pixel: float = 1.0,
) -> NDArrayNum:
    """
    Grid point cloud (possibly irregular coordinates) to raster (regular grid) using delaunay triangles interpolation.

    Based on scipy.interpolate.griddata combined to a nearest point search to replace values of grid cells further than
    a certain distance (in number of pixels) by nodata values (as griddata interpolates all values in convex hull, no
    matter the distance).

    :param pc: Point cloud.
    :param grid_coords: Grid coordinates for X and Y.
    :param data_column_name: Name of data column for point cloud (only if passed as a geodataframe).
    :param resampling: Resampling method within delauney triangles (defaults to linear).
    :param dist_nodata_pixel: Distance from the point cloud after which grid cells are filled by nodata values,
        expressed in number of pixels.
    """

    # 1/ Interpolate irregular point cloud on a regular grid

    # Get meshgrid coordinates
    xx, yy = np.meshgrid(grid_coords[0], grid_coords[1])

    # Use griddata on all points
    aligned_dem = griddata(
        points=(pc.geometry.x.values, pc.geometry.y.values),
        values=pc[data_column_name].values,
        xi=(xx, yy),
        method=resampling,
        rescale=True,  # Rescale inputs to unit cube to avoid precision issues
    )

    # 2/ Identify which grid points are more than X pixels away from the point cloud, and convert to NaNs
    # (otherwise all grid points in the convex hull of the irregular triangulation are filled, no matter the distance)

    # Get the nearest point for each grid point
    grid_pc = gpd.GeoDataFrame(
        data={"placeholder": np.ones(len(xx.ravel()))}, geometry=gpd.points_from_xy(x=xx.ravel(), y=yy.ravel())
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
        near = gpd.sjoin_nearest(grid_pc, pc)
        # In case there are several points at the same distance, it doesn't matter which one is used to compute the
        # distance, so we keep the first index of closest point
        index_right = near.groupby(by=near.index)["index_right"].min()

    # Compute distance between points as a function of the pixel sizes in X and Y
    res_x = np.abs(grid_coords[0][1] - grid_coords[0][0])
    res_y = np.abs(grid_coords[1][1] - grid_coords[1][0])
    dist = np.sqrt(
        ((pc.geometry.x.values[index_right] - grid_pc.geometry.x.values) / res_x) ** 2
        + ((pc.geometry.y.values[index_right] - grid_pc.geometry.y.values) / res_y) ** 2
    )

    # Replace all points further away than the distance of nodata by NaNs
    aligned_dem[dist.reshape(aligned_dem.shape) > dist_nodata_pixel] = np.nan

    # Flip Y axis of grid
    aligned_dem = np.flip(aligned_dem, axis=0)

    return aligned_dem
