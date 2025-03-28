# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functionalities to manipulate metadata in different coordinate reference systems (CRS).
"""

from __future__ import annotations

import warnings
from math import ceil, floor
from typing import Iterable, Literal

import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rio
import shapely.geometry
import shapely.ops
from rasterio.crs import CRS
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from geoutils._typing import NDArrayNum, Number


def latlon_to_utm(lat: Number, lon: Number) -> str:
    """
    Get UTM zone for a given latitude and longitude coordinates.

    :param lat: Latitude coordinate.
    :param lon: Longitude coordinate.

    :returns: UTM zone.
    """

    if not (
        isinstance(lat, (float, np.floating, int, np.integer))
        and isinstance(lon, (float, np.floating, int, np.integer))
    ):
        raise TypeError("Latitude and longitude must be floats or integers.")

    if not -180 <= lon < 180:
        raise ValueError("Longitude value is out of range [-180, 180[.")
    if not -90 <= lat < 90:
        raise ValueError("Latitude value is out of range [-90, 90[.")

    # Get UTM zone from name string of crs info
    utm_zone = pyproj.database.query_utm_crs_info(
        "WGS 84", area_of_interest=pyproj.aoi.AreaOfInterest(lon, lat, lon, lat)
    )[0].name.split(" ")[-1]

    return str(utm_zone)


def utm_to_epsg(utm: str) -> int:
    """
    Get EPSG code of UTM zone.

    :param utm: UTM zone.

    :return: EPSG of UTM zone.
    """

    if not isinstance(utm, str):
        raise TypeError("UTM zone must be a str.")

    # Whether UTM is passed as single or double digits, homogenize to single-digit
    utm = str(int(utm[:-1])) + utm[-1].upper()

    # Get corresponding EPSG
    epsg = pyproj.CRS(f"WGS 84 / UTM Zone {utm}").to_epsg()

    return int(epsg)


def _get_utm_ups_crs(df: gpd.GeoDataFrame, method: Literal["centroid"] | Literal["geopandas"] = "centroid") -> CRS:
    """
    Get universal metric coordinate reference system for the vector passed (UTM or UPS).

    :param df: Input geodataframe.
    :param method: Method to choose the zone of the CRS, either based on the centroid of the footprint
       or the extent as implemented in :func:`geopandas.GeoDataFrame.estimate_utm_crs`.
       Forced to centroid if `local_crs="custom"`.
    """
    # Check input
    if method.lower() not in ["centroid", "geopandas"]:
        raise ValueError("Method to get local CRS should be one of 'centroid' and 'geopandas'.")

    # Use geopandas if that is the desired method
    if method == "geopandas":
        crs = df.estimate_utm_crs()

    # Else, compute the centroid of dissolved geometries and get UTM or UPS
    else:
        # Get a rough centroid in geographic coordinates (ignore the warning that it is not the most precise):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            shp_wgs84 = df.to_crs(epsg=4326).dissolve()
            lat, lon = shp_wgs84.centroid.y.values[0], shp_wgs84.centroid.x.values[0]
            del shp_wgs84

        # If absolute latitude is below 80, get the EPSG code of the local UTM
        if -80 <= lat <= 80:
            utm = latlon_to_utm(lat, lon)
            epsg = utm_to_epsg(utm)
            crs = pyproj.CRS.from_epsg(epsg)
        # If latitude is below 80, get UPS South
        elif lat < -80:
            crs = pyproj.CRS.from_epsg(32761)
        # Else, get UPS North
        else:
            crs = pyproj.CRS.from_epsg(32661)

    return crs


def bounds2poly(
    bounds_geom: list[float] | rio.io.DatasetReader,
    in_crs: CRS | None = None,
    out_crs: CRS | None = None,
) -> Polygon:
    """
    Converts self's bounds into a shapely Polygon. Optionally, returns it into a different CRS.

    :param bounds_geom: A geometry with bounds. Can be either a list of coordinates (xmin, ymin, xmax, ymax),\
            a rasterio/Raster object, a geoPandas/Vector object
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Output polygon
    """
    # If boundsGeom is a GeoPandas or Vector object (warning, has both total_bounds and bounds attributes)
    if hasattr(bounds_geom, "total_bounds"):
        xmin, ymin, xmax, ymax = bounds_geom.total_bounds  # type: ignore
        in_crs = bounds_geom.crs  # type: ignore
    # If boundsGeom is a rasterio or Raster object
    elif hasattr(bounds_geom, "bounds"):
        xmin, ymin, xmax, ymax = bounds_geom.bounds  # type: ignore
        in_crs = bounds_geom.crs  # type: ignore
    # if a list of coordinates
    elif isinstance(bounds_geom, (list, tuple)):
        xmin, ymin, xmax, ymax = bounds_geom
    else:
        raise ValueError(
            "boundsGeom must a list/tuple of coordinates or an object with attributes bounds or total_bounds."
        )

    corners = np.array([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])

    if (in_crs is not None) & (out_crs is not None):
        corners = np.transpose(reproject_points(np.transpose(corners), in_crs, out_crs))

    bbox = Polygon(corners)

    return bbox


def merge_bounds(
    bounds_list: Iterable[
        list[float] | tuple[float] | rio.coords.BoundingBox | rio.io.DatasetReader | gpd.GeoDataFrame
    ],
    resolution: float | None = None,
    merging_algorithm: str = "union",
    return_rio_bbox: bool = False,
) -> tuple[float, ...] | rio.coords.BoundingBox:
    """
    Merge a list of bounds into single bounds, using either the union or intersection.

    :param bounds_list: List of geometries with bounds, i.e. list of coordinates (xmin, ymin, xmax, ymax),
        rasterio bounds, a rasterio Dataset (or Raster), a geopandas object (or Vector).
    :param resolution: (For Rasters) Resolution, to make sure extent is a multiple of it.
    :param merging_algorithm: Algorithm to use for merging, either "union" or "intersection".
    :param return_rio_bbox: Whether to return a rio.coords.BoundingBox object instead of a tuple.

    :returns: Output bounds (xmin, ymin, xmax, ymax) or empty tuple
    """
    # Check that bounds_list is a list of bounds objects
    assert isinstance(bounds_list, (list, tuple)), "bounds_list must be a list/tuple"

    for bounds in bounds_list:
        assert hasattr(bounds, "bounds") or hasattr(bounds, "total_bounds") or isinstance(bounds, (list, tuple)), (
            "bounds_list must be a list of lists/tuples of coordinates or an object with attributes bounds "
            "or total_bounds"
        )

    output_poly = bounds2poly(bounds_geom=bounds_list[0])

    # Compute the merging
    for boundsGeom in bounds_list[1:]:
        new_poly = bounds2poly(boundsGeom)

        if merging_algorithm == "union":
            output_poly = output_poly.union(new_poly)
        elif merging_algorithm == "intersection":
            output_poly = output_poly.intersection(new_poly)
        else:
            raise ValueError("merging_algorithm must be 'union' or 'intersection'")

    # Get merged bounds, write as dict to manipulate with resolution in the next step
    new_bounds = output_poly.bounds
    rio_bounds = {"left": new_bounds[0], "bottom": new_bounds[1], "right": new_bounds[2], "top": new_bounds[3]}

    # Make sure that extent is a multiple of resolution
    if resolution is not None:
        for key1, key2 in zip(("left", "bottom"), ("right", "top")):
            modulo = (rio_bounds[key2] - rio_bounds[key1]) % resolution
            rio_bounds[key2] += modulo

    # Format output
    if return_rio_bbox:
        final_bounds = rio.coords.BoundingBox(**rio_bounds)
    else:
        final_bounds = tuple(rio_bounds.values())

    return final_bounds


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


def reproject_points(
    points: list[list[float]] | list[float] | tuple[list[float], list[float]] | NDArrayNum, in_crs: CRS, out_crs: CRS
) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from input_crs to output_crs.

    :param points: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Reprojected points, of same shape as points.
    """
    assert np.shape(points)[0] == 2, "points must be of shape (2, N)"

    x, y = points
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs)
    xout, yout = transformer.transform(x, y)
    return (xout, yout)


# Functions to convert from and to latlon

crs_4326 = rio.crs.CRS.from_epsg(4326)


def reproject_to_latlon(
    points: list[list[float]] | list[float] | NDArrayNum, in_crs: CRS, round_: int = 8
) -> NDArrayNum:
    """
    Reproject a set of point from in_crs to lat/lon.

    :param points: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param in_crs: Input CRS
    :param round_: Output rounding. Default of 8 ensures cm accuracy

    :returns: Reprojected points, of same shape as points.
    """
    proj_points = reproject_points(points, in_crs, crs_4326)
    return np.round(proj_points, round_)


def reproject_from_latlon(
    points: list[list[float]] | tuple[list[float], list[float]] | NDArrayNum, out_crs: CRS, round_: int = 2
) -> NDArrayNum:
    """
    Reproject a set of point from lat/lon to out_crs.

    :param points: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param out_crs: Output CRS
    :param round_: Output rounding. Default of 2 ensures cm accuracy

    :returns: Reprojected points, of same shape as points.
    """
    proj_points = reproject_points(points, crs_4326, out_crs)
    return np.round(proj_points, round_)


def reproject_shape(inshape: BaseGeometry, in_crs: CRS, out_crs: CRS) -> BaseGeometry:
    """
    Reproject a shapely geometry from one CRS into another CRS.

    :param inshape: Shapely geometry to be reprojected.
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Reprojected geometry
    """
    reproj = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
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


def _get_bounds_projected(
    bounds: rio.coords.BoundingBox, in_crs: CRS, out_crs: CRS, densify_points: int = 5000
) -> rio.coords.BoundingBox:
    """
    Get bounds projected in a specified CRS.

    :param in_crs: Input CRS.
    :param out_crs: Output CRS.
    :param densify_points: Maximum points to be added between image corners to account for nonlinear edges.
    Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
    """

    # Calculate new bounds
    left, bottom, right, top = bounds
    new_bounds = rio.warp.transform_bounds(in_crs, out_crs, left, bottom, right, top, densify_points)
    new_bounds = rio.coords.BoundingBox(*new_bounds)

    return new_bounds


def _densify_geometry(
    line_geometry: shapely.geometry.LineString, densify_points: int = 5000
) -> shapely.geometry.LineString:
    """
    Densify a linestring geometry.

    Inspired by: https://gis.stackexchange.com/questions/372912/how-to-densify-linestring-vertices-in-shapely-geopandas.

    :param line_geometry: Linestring.
    :param densify_points: Number of points to densify each line.

    :return: Densified linestring.
    """

    # Get the segments (list of linestrings)
    segments = list(map(shapely.geometry.LineString, zip(line_geometry.coords[:-1], line_geometry.coords[1:])))

    # To store new coordinate tuples
    xy = []

    # For each segment, densify the points
    for i, seg in enumerate(segments):

        # Get the segment length
        length_m = seg.length

        # Loop over a distance on the segment length
        densified_seg = np.linspace(0, length_m, 1 + densify_points)
        # (removing the last point, as it will be the first point of the next segment,
        # except for the last segment)
        if i < len(segments) - 1:
            densified_seg = densified_seg[:-1]

        for distance_along_old_line in densified_seg:
            # Interpolate a point every step along the old line
            point = seg.interpolate(distance_along_old_line)
            # Extract the coordinates and store them in xy list
            xp, yp = point.x, point.y
            xy.append((xp, yp))

    # Recreate a new line with densified points
    densified_line_geometry = shapely.geometry.LineString(xy)

    return densified_line_geometry


def _get_footprint_projected(
    bounds: rio.coords.BoundingBox, in_crs: CRS, out_crs: CRS, densify_points: int = 5000
) -> gpd.GeoDataFrame:
    """
    Get bounding box footprint projected in a specified CRS.

    The polygon points of the vector are densified during reprojection to warp
    the rectangular square footprint of the original projection into the new one.

    :param in_crs: Input CRS.
    :param out_crs: Output CRS.
    :param densify_points: Maximum points to be added between image corners to account for non linear edges.
     Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
    """

    # Get bounds
    left, bottom, right, top = bounds

    # Create linestring
    linestring = shapely.geometry.LineString(
        [[left, bottom], [left, top], [right, top], [right, bottom], [left, bottom]]
    )

    # Densify linestring
    densified_line_geometry = _densify_geometry(linestring, densify_points=densify_points)

    # Get polygon from new linestring
    densified_poly = Polygon(densified_line_geometry)

    # Reproject the polygon
    df = gpd.GeoDataFrame({"geometry": [densified_poly]}, crs=in_crs)
    reproj_df = df.to_crs(crs=out_crs)

    return reproj_df
