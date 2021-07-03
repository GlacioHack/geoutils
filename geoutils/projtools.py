"""
GeoUtils.projtools provides a toolset for dealing with different coordinate reference systems (CRS).
"""
from __future__ import annotations

from collections import abc

import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rio
import shapely.ops
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from geoutils.georaster import Raster
from geoutils.geovector import Vector


def bounds2poly(
    boundsGeom: list[float] | rio.io.DatasetReader | Raster | Vector,
    in_crs: CRS | None = None,
    out_crs: CRS | None = None,
) -> Polygon:
    """
    Converts self's bounds into a shapely Polygon. Optionally, returns it into a different CRS.

    :param boundsGeom: A geometry with bounds. Can be either a list of coordinates (xmin, ymin, xmax, ymax),\
            a rasterio/Raster object, a geoPandas/Vector object
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Output polygon
    """
    if in_crs is not None:
        raise NotImplementedError

    # If boundsGeom is a GeoPandas or Vector object (warning, has both total_bounds and bounds attributes)
    if hasattr(boundsGeom, "total_bounds"):
        xmin, ymin, xmax, ymax = boundsGeom.total_bounds  # type: ignore
        in_crs = boundsGeom.crs  # type: ignore
    # If boundsGeom is a rasterio or Raster object
    elif hasattr(boundsGeom, "bounds"):
        xmin, ymin, xmax, ymax = boundsGeom.bounds  # type: ignore
        in_crs = boundsGeom.crs  # type: ignore
    # if a list of coordinates
    elif isinstance(boundsGeom, (list, tuple)):
        xmin, ymin, xmax, ymax = boundsGeom
    else:
        raise ValueError(
            "boundsGeom must a list/tuple of coordinates or an object with attributes bounds or total_bounds."
        )

    bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    if out_crs is not None:
        raise NotImplementedError()

    return bbox


def merge_bounds(
    bounds_list: abc.Iterable[list[float] | Raster | rio.io.DatasetReader | Vector | gpd.GeoDataFrame],
    merging_algorithm: str = "union",
) -> tuple[float, ...]:
    """
    Merge a list of bounds into single bounds, using either the union or intersection.

    :param bounds_list: A list of geometries with bounds, i.e. a list of coordinates (xmin, ymin, xmax, ymax), \
a rasterio/Raster object, a geoPandas/Vector object.
    :param merging_algorithm: the algorithm to use for merging, either "union" or "intersection"

    :returns: Output bounds (xmin, ymin, xmax, ymax)
    """
    # Check that bounds_list is a list of bounds objects
    assert isinstance(bounds_list, (list, tuple)), "bounds_list must be a list/tuple"
    for bounds in bounds_list:
        assert hasattr(bounds, "bounds") or hasattr(bounds, "total_bounds") or isinstance(bounds, (list, tuple)), (
            "bounds_list must be a list of lists/tuples of coordinates or an object with attributes bounds "
            "or total_bounds"
        )

    output_poly = bounds2poly(boundsGeom=bounds_list[0])

    for boundsGeom in bounds_list[1:]:
        new_poly = bounds2poly(boundsGeom)

        if merging_algorithm == "union":
            output_poly = output_poly.union(new_poly)
        elif merging_algorithm == "intersection":
            output_poly = output_poly.intersection(new_poly)
        else:
            raise ValueError("merging_algorithm must be 'union' or 'intersection'")

    new_bounds: tuple[float] = output_poly.bounds
    return new_bounds


def reproject_points(pts: list[list[float]] | np.ndarray, in_crs: CRS, out_crs: CRS) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from input_crs to output_crs.

    :param pts: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Reprojected points, of same shape as pts.
    """
    assert np.shape(pts)[0] == 2, "pts must be of shape (2, N)"

    x, y = pts
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs)
    xout, yout = transformer.transform(x, y)
    return (xout, yout)


# Functions to convert from and to latlon

crs_4326 = rio.crs.CRS.from_epsg(4326)


def reproject_to_latlon(
    pts: list[list[float]] | np.ndarray, in_crs: CRS, round_: int = 8
) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from in_crs to lat/lon.

    :param pts: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param in_crs: Input CRS
    :param round_: Output rounding. Default of 8 ensures cm accuracy

    :returns: Reprojected points, of same shape as pts.
    """
    proj_pts = reproject_points(pts, in_crs, crs_4326)
    proj_pts = np.round(proj_pts, round_)
    return proj_pts


def reproject_from_latlon(
    pts: list[list[float]] | tuple[list[float], list[float]] | np.ndarray, out_crs: CRS, round_: int = 2
) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from lat/lon to out_crs.

    :param pts: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param out_crs: Output CRS
    :param round_: Output rounding. Default of 2 ensures cm accuracy

    :returns: Reprojected points, of same shape as pts.
    """
    proj_pts = reproject_points(pts, crs_4326, out_crs)
    proj_pts = np.round(proj_pts, round_)
    return proj_pts


def reproject_shape(inshape: BaseGeometry, in_crs: CRS, out_crs: CRS) -> BaseGeometry:
    """
    Reproject a shapely geometry from one CRS into another CRS.

    :param inshape: Shapely geometry to be reprojected.
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Reprojected geometry
    """
    reproj = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True, skip_equivalent=True).transform
    return shapely.ops.transform(reproj, inshape)


def compare_proj(proj1: CRS, proj2: CRS) -> bool:
    """
    Compare two projections to see if they are the same, using pyproj.CRS.is_exact_same.

    :param proj1: The first projection to compare.
    :param proj2: The first projection to compare.

    :returns: True if the two projections are the same.
    """
    assert all(
        [isinstance(proj1, (pyproj.CRS, CRS)), isinstance(proj2, (pyproj.CRS, CRS))]
    ), "proj1 and proj2 must be rasterio.crs.CRS objects."
    proj1 = pyproj.CRS(proj1.to_string())
    proj2 = pyproj.CRS(proj2.to_string())

    same: bool = proj1.is_exact_same(proj2)
    return same
