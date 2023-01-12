"""
projtools provides a set of tools for dealing with different coordinate reference systems (CRS) and bounds.
"""
from __future__ import annotations

from collections import abc
from math import ceil, floor

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


def latlon_to_utm(lat: float, lon: float) -> str:
    """
    Get UTM zone for a given latitude and longitude coordinates.

    :param lat: Latitude coordinate.
    :param lon: Longitude coordinate.

    :returns: UTM zone.
    """
    if not (isinstance(lat, (float, np.floating, int, np.integer)) and isinstance(lon, (float, np.floating, int, np.integer))):
        raise ValueError('Latitude and longitude must be floats or integers.')
    # The "utm" Python module excludes regions south of 80°S and north of 84°N, unpractical for global vector manipulation
    # utm_all = utm.from_latlon(lat,lon)
    # utm_nb=utm_all[2]

    # Get UTM zone from longitude without exclusions
    if -180 <= lon < 180:
        utm_nb = int(
            np.floor((lon + 180) / 6)) + 1  # lon=-180 refers to UTM zone 1 towards East (West corner convention)
    else:
        raise ValueError('Longitude value is out of range [-180, 180[.')

    if 0 <= lat < 90:  # lat=0 refers to North (South corner convention)
        utm_zone = str(utm_nb).zfill(2) + 'N'
    elif -90 <= lat < 0:
        utm_zone = str(utm_nb).zfill(2) + 'S'
    else:
        raise ValueError('Latitude value is out of range [-90, 90[.')

    return utm_zone


def utm_to_epsg(utm: str) -> int:
    """
    Get EPSG code of UTM zone.

    :param utm: UTM zone.

    :return: EPSG of UTM zone.
    """

    if not (isinstance(utm, str) and 2<=len(utm)<=3 and utm[:-1].isdigit() and 0<int(utm[:-1])<=60 and utm[-1].upper() in ['N', 'S']):
        raise ValueError('UTM zone should be a 3-character string with 2-digit code between 01 and 60, and 1-letter north or south zone, e.g. "18S" or "54N".')

    utm_digits = utm[:-1]
    utm_north_south = utm[-1].upper()

    # Code starts with 326 for North, and 327 for South, to which is added the utm zone number
    if utm_north_south == 'N':
        epsg = int('326' + utm_digits.zfill(2))
    else:
        epsg = int('327' + utm_digits.zfill(2))

    return epsg

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

    corners = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))

    if (in_crs is not None) & (out_crs is not None):
        corners = np.transpose(reproject_points(np.transpose(corners), in_crs, out_crs))

    bbox = Polygon(corners)

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

    :returns: Output bounds (xmin, ymin, xmax, ymax) or empty tuple
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


def align_bounds(
    ref_transform: rio.transform.Affine,
    src_bounds: rio.coords.BoundingBox | tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Aligns the bounds in src_bounds so that it matches the georeferences in ref_transform
    i.e. the distance between the upper-left pixels of ref and src is a multiple of resolution and
    the width/height of the bounds are a multiple of resolution.
    The bounds are padded so that the output bounds always contain the input bounds.

    :param ref_transform: The transform of the dataset to be used as reference
    :param src_bounds: The initial bounds that needs to be aligned to ref_transform. \
    Must be a rasterio BoundingBox or list or tuple with coordinates (left, bottom, right, top).

    :returns: the aligned bounding box (left, bottom, right, top)
    """
    left, bottom, right, top = src_bounds
    xres = ref_transform.a
    yres = ref_transform.e
    ref_left = ref_transform.xoff
    ref_top = ref_transform.yoff

    left = ref_left + floor((left - ref_left) / xres) * xres
    right = left + ceil((right - left) / xres) * xres
    top = ref_top + floor((top - ref_top) / yres) * yres
    bottom = top + ceil((bottom - top) / yres) * yres

    return (left, bottom, right, top)


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
