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

"""Functionalities to manipulate vector geometries."""

from __future__ import annotations

import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

import geoutils as gu
from geoutils.projtools import _get_utm_ups_crs, bounds2poly


def _buffer_metric(gdf: gpd.GeoDataFrame, buffer_size: float) -> gu.Vector:
    """
    Metric buffering. See Vector.buffer_metric() for details.
    """

    crs_utm_ups = _get_utm_ups_crs(df=gdf)

    # Reproject the shapefile in the local UTM
    ds_utm = gdf.to_crs(crs=crs_utm_ups)

    # Buffer the shapefile
    ds_buffered = ds_utm.buffer(distance=buffer_size)
    del ds_utm

    # Revert-project the shapefile in the original CRS
    ds_buffered_origproj = ds_buffered.to_crs(crs=gdf.crs)
    del ds_buffered

    # Return a Vector object of the buffered GeoDataFrame
    # TODO: Clarify what is conserved in the GeoSeries and what to pass the GeoDataFrame to not lose any attributes
    vector_buffered = gu.Vector(gpd.GeoDataFrame(geometry=ds_buffered_origproj.geometry, crs=gdf.crs))

    return vector_buffered


def _buffer_without_overlap(
    ds: gpd.GeoDataFrame, buffer_size: int | float, metric: bool = True, plot: bool = False
) -> gu.Vector:
    """See Vector.buffer_without_overlap() for details."""

    # Project in local UTM if metric is True
    if metric:
        crs_utm_ups = _get_utm_ups_crs(df=ds)
        gdf = ds.to_crs(crs=crs_utm_ups)
    else:
        gdf = ds

    # Dissolve all geometries into one
    merged = gdf.dissolve()

    # Add buffer around geometries
    merged_buffer = merged.buffer(buffer_size)

    # Extract only the buffered area
    buffer = merged_buffer.difference(merged)

    # Crop Voronoi polygons to bound geometry and add missing polygons
    bound_poly = bounds2poly(gdf)
    bound_poly = bound_poly.buffer(buffer_size)
    voronoi_all = _generate_voronoi_with_bounds(gdf, bound_poly)
    if plot:
        plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(141)
        voronoi_all.plot(ax=ax1)
        gdf.plot(fc="none", ec="k", ax=ax1)
        ax1.set_title("Voronoi polygons, cropped")

    # Extract Voronoi polygons only within the buffer area
    voronoi_diff = voronoi_all.intersection(buffer.geometry[0])

    # Split all polygons, and join attributes of original geometries into the Voronoi polygons
    # Splitting, i.e. explode, is needed when Voronoi generate MultiPolygons that may extend over several features.
    voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_diff.explode(index_parts=True))  # requires geopandas>=0.10
    joined_voronoi = gpd.tools.sjoin(gdf, voronoi_gdf, how="right")

    # Plot results -> some polygons are duplicated
    if plot:
        ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
        joined_voronoi.plot(ax=ax2, column="index_left", alpha=0.5, ec="k")
        gdf.plot(ax=ax2, column=gdf.index.values)
        ax2.set_title("Buffer with duplicated polygons")

    # Find non unique Voronoi polygons, and retain only first one
    _, indexes = np.unique(joined_voronoi.index, return_index=True)
    unique_voronoi = joined_voronoi.iloc[indexes]

    # Plot results -> unique polygons only
    if plot:
        ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
        unique_voronoi.plot(ax=ax3, column="index_left", alpha=0.5, ec="k")
        gdf.plot(ax=ax3, column=gdf.index.values)
        ax3.set_title("Buffer with unique polygons")

    # Dissolve all polygons by original index
    merged_voronoi = unique_voronoi.dissolve(by="index_left")

    # Plot
    if plot:
        ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
        gdf.plot(ax=ax4, column=gdf.index.values)
        merged_voronoi.plot(column=merged_voronoi.index.values, ax=ax4, alpha=0.5)
        ax4.set_title("Final buffer")
        plt.show()

    # Reverse-project to the original CRS if metric is True
    if metric:
        merged_voronoi = merged_voronoi.to_crs(crs=ds.crs)

    return gu.Vector(merged_voronoi)


def _extract_vertices(gdf: gpd.GeoDataFrame) -> list[list[tuple[float, float]]]:
    r"""
    Function to extract the exterior vertices of all shapes within a gpd.GeoDataFrame.

    :param gdf: The GeoDataFrame from which the vertices need to be extracted.

    :returns: A list containing a list of (x, y) positions of the vertices. The length of the primary list is equal
        to the number of geometries inside gdf, and length of each sublist is the number of vertices in the geometry.
    """
    vertices = []
    # Loop on all geometries within gdf
    for geom in gdf.geometry:
        # Extract geometry exterior(s)
        if geom.geom_type == "MultiPolygon":
            exteriors = [p.exterior for p in geom.geoms]
        elif geom.geom_type == "Polygon":
            exteriors = [geom.exterior]
        elif geom.geom_type == "LineString":
            exteriors = [geom]
        elif geom.geom_type == "MultiLineString":
            exteriors = list(geom.geoms)
        else:
            raise NotImplementedError(f"Geometry type {geom.geom_type} not implemented.")

        vertices.extend([list(ext.coords) for ext in exteriors])

    return vertices


def _generate_voronoi_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate Voronoi polygons (tessellation) from the vertices of all geometries in a GeoDataFrame.

    Uses scipy.spatial.voronoi.

    :param: The GeoDataFrame from whose vertices are used for the Voronoi polygons.

    :returns: A GeoDataFrame containing the Voronoi polygons.
    """
    # Extract the coordinates of the vertices of all geometries in gdf
    vertices = _extract_vertices(gdf)
    coords = np.concatenate(vertices)

    # Create the Voronoi diagram and extract ridges
    vor = Voronoi(coords)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    polys = list(shapely.ops.polygonize(lines))
    if len(polys) == 0:
        raise ValueError("Invalid geometry, cannot generate finite Voronoi polygons")

    # Convert into GeoDataFrame
    voronoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    voronoi.crs = gdf.crs

    return voronoi


def _generate_voronoi_with_bounds(gdf: gpd.GeoDataFrame, bound_poly: Polygon) -> gpd.GeoDataFrame:
    """
    Generate Voronoi polygons that are bounded by the polygon bound_poly, to avoid Voronoi polygons that extend \
far beyond the original geometry.

    Voronoi polygons are created using generate_voronoi_polygons, cropped to the extent of bound_poly and gaps \
are filled with new polygons.

    :param: The GeoDataFrame from whose vertices are used for the Voronoi polygons.
    :param: A shapely Polygon to be used for bounding the Voronoi diagrams.

    :returns: A GeoDataFrame containing the Voronoi polygons.
    """
    # Create Voronoi polygons
    voronoi = _generate_voronoi_polygons(gdf)

    # Crop Voronoi polygons to input bound_poly extent
    voronoi_crop = voronoi.intersection(bound_poly)
    voronoi_crop = gpd.GeoDataFrame(geometry=voronoi_crop)  # convert to DataFrame

    # Dissolve all Voronoi polygons and subtract from bounds to get gaps
    voronoi_merged = voronoi_crop.dissolve()
    bound_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bound_poly))
    bound_gdf.crs = gdf.crs
    gaps = bound_gdf.difference(voronoi_merged)

    # Merge cropped Voronoi with gaps, if not empty, otherwise return cropped Voronoi
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS. Results from 'area' are likely incorrect.")
        tot_area = np.sum(gaps.area.values)

    if not tot_area == 0:
        voronoi_all = gpd.GeoDataFrame(geometry=list(voronoi_crop.geometry) + list(gaps.geometry))
        voronoi_all.crs = gdf.crs
        return voronoi_all
    else:
        return voronoi_crop
