# Copyright (c) 2026 GeoUtils developers
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
Module for Vector class.
"""

from __future__ import annotations

import os
import pathlib
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    TypeVar,
    Union,
)

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio as rio
from pandas._typing import WriteBuffer
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils import profiler
from geoutils._misc import copy_doc
from geoutils.vector.base import VectorBase

if TYPE_CHECKING:
    from geoutils.raster.base import RasterType

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
VectorType = TypeVar("VectorType", bound="Vector")
VectorLike = Union["Vector", gpd.GeoDataFrame]


class Vector(VectorBase):
    """
    The georeferenced vector.

     Main attributes:
        ds: :class:`geopandas.GeoDataFrame`
            Geodataframe of the vector.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the vector.
        bounds: :class:`rio.coords.BoundingBox`
            Coordinate bounds of the vector.

    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    @profiler.profile("geoutils.vector.vector.__init__", collect=False)
    def __init__(
        self, filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | dict[str, Any]
    ):
        """
        Instantiate a vector from either a filename, a GeoPandas dataframe or series, or a Shapely geometry.

        :param filename_or_dataset: Path to file, or GeoPandas dataframe or series, or Shapely geometry.
        """

        self._name: str | None = None
        self._ds: gpd.GeoDataFrame | None = None
        self._crs: CRS | None = None
        self._bounds: rio.coords.BoundingBox | None = None
        self._columns: pd.Index | None = None
        self._feature_count: int | None = None
        self._geometry_type: str | None = None

        # If Vector is passed, simply point back to Vector
        if isinstance(filename_or_dataset, Vector):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # If filename is passed
        elif isinstance(filename_or_dataset, (str, pathlib.Path)):
            self._name = os.fspath(filename_or_dataset)
            self._set_metadata_from_file(self._name)
            return
        # If GeoPandas or Shapely object is passed
        elif isinstance(filename_or_dataset, (gpd.GeoDataFrame, gpd.GeoSeries, BaseGeometry)):
            if isinstance(filename_or_dataset, gpd.GeoDataFrame):
                ds = filename_or_dataset
            elif isinstance(filename_or_dataset, gpd.GeoSeries):
                ds = gpd.GeoDataFrame(geometry=filename_or_dataset)
            else:
                ds = gpd.GeoDataFrame({"geometry": [filename_or_dataset]}, crs=None)
        else:
            raise TypeError("Filename argument should be a string, path or geodataframe.")

        # Set geodataframe
        self.ds = ds

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""

        if not self.is_loaded:
            return self._crs  # type: ignore[return-value]
        return self.ds.crs

    @property
    def ds(self) -> gpd.GeoDataFrame:
        """Geodataframe of the vector."""
        if not self.is_loaded:
            self.load()
        return self._ds  # type: ignore[return-value]

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new geodataframe."""

        if isinstance(new_ds, gpd.GeoDataFrame):
            self._ds = new_ds
        elif isinstance(new_ds, gpd.GeoSeries):
            self._ds = gpd.GeoDataFrame(geometry=new_ds)
        else:
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")
        self._set_metadata_from_ds(self._ds)

    def _set_metadata_from_file(self, filename: str) -> None:
        """Read lightweight vector metadata without loading the full GeoDataFrame."""

        info = pyogrio.read_info(filename)
        crs = info.get("crs")
        total_bounds = info.get("total_bounds")

        self._crs = CRS.from_user_input(crs) if crs else None
        if total_bounds is not None:
            self._bounds = rio.coords.BoundingBox(*total_bounds)
        self._columns = pd.Index(list(info.get("fields", [])) + ["geometry"])
        self._feature_count = info.get("features")
        self._geometry_type = info.get("geometry_type")

    def _set_metadata_from_ds(self, ds: gpd.GeoDataFrame) -> None:
        """Update cached vector metadata from an in-memory GeoDataFrame."""

        self._crs = ds.crs
        self._bounds = rio.coords.BoundingBox(*ds.total_bounds)
        self._columns = ds.columns
        self._feature_count = len(ds)
        self._geometry_type = ds.geom_type.iloc[0] if len(ds) > 0 else None

    @property
    def is_loaded(self) -> bool:
        """Whether the vector data are loaded in memory."""

        return self._ds is not None

    def load(self, **kwargs: Any) -> None:
        """
        Load the vector GeoDataFrame from disk.

        :param kwargs: Optional keyword arguments passed to :func:`geopandas.read_file`.
        """

        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.name is None:
            raise AttributeError("Cannot load as name is not set anymore. Did you manually update the name attribute?")

        self.ds = gpd.read_file(self.name, **kwargs)

    @property
    def columns(self) -> pd.Index:
        if not self.is_loaded and self._columns is not None:
            return self._columns
        return self.ds.columns

    def copy(self: VectorType) -> VectorType:
        """Return a copy of the vector."""
        # Utilise the copy method of GeoPandas
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.copy())  # type: ignore
        return new_vector  # type: ignore

    ############################################################################
    # Overridden and wrapped methods from GeoPandas API to logically cast outputs
    ############################################################################

    def _override_gdf_output(
        self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any
    ) -> VectorType | pd.Series:
        """Parse outputs of GeoPandas functions to facilitate object manipulation."""

        # Raise error if output is not treated separately, should appear in tests
        if not isinstance(other, (gpd.GeoDataFrame, pd.Series, BaseGeometry)):
            raise ValueError("Not implemented. This error should only be raised in tests.")

        # If a GeoDataFrame is the output, return it
        if isinstance(other, gpd.GeoDataFrame):
            return Vector(other)
        # If a GeoSeries is the output, re-encapsulate in a GeoDataFrame and return it
        elif isinstance(other, gpd.GeoSeries):
            return Vector(gpd.GeoDataFrame(geometry=other))
        # If a Shapely Geometry is the output, re-encapsulate in a GeoDataFrame and return it
        elif isinstance(other, BaseGeometry):
            return Vector(gpd.GeoDataFrame({"geometry": [other]}, crs=self.crs))
        # If a Pandas Series is the output, append it to that of the GeoDataFrame
        else:
            return other

    # -----------------------------------------------
    # GeoPandasBase - Attributes that return a Series
    # -----------------------------------------------

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def area(self) -> pd.Series:
        return self._override_gdf_output(self.ds.area)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def length(self) -> pd.Series:
        return self._override_gdf_output(self.ds.length)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def interiors(self) -> pd.Series:
        return self._override_gdf_output(self.ds.interiors)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def geom_type(self) -> pd.Series:
        return self._override_gdf_output(self.ds.geom_type)

    # Exception ! bounds is renamed geom_bounds to make Raster and Vector "bounds" the same "total_bounds"
    @property
    def geom_bounds(self) -> pd.Series:
        """Returns or appends to ``Vector`` a ``Series`` with the bounds of each geometry feature."""
        return self.ds.bounds

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_empty(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_empty)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_ring(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_ring)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_simple(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_simple)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_valid(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_valid)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def has_z(self) -> pd.Series:
        return self.ds.has_z

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_ccw(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_ccw)

    @property
    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_closed(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_closed)

    # --------------------------------------------------
    # GeoPandasBase - Attributes that return a GeoSeries
    # --------------------------------------------------

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def boundary(self) -> Vector:
        return self._override_gdf_output(self.ds.boundary)

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def centroid(self) -> Vector:
        return self._override_gdf_output(self.ds.centroid)

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def convex_hull(self) -> Vector:
        return self._override_gdf_output(self.ds.convex_hull)

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def envelope(self) -> Vector:
        return self._override_gdf_output(self.ds.envelope)

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def exterior(self) -> Vector:
        return self._override_gdf_output(self.ds.exterior)

    # ---------------------------------------------------------------------------------
    # GeoPandasBase - Attributes that return a specific value (not Series or GeoSeries)
    # ---------------------------------------------------------------------------------

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def has_sindex(self) -> bool:
        return self.ds.has_sindex

    @property
    @copy_doc(gpd.GeoSeries, "Vector")
    def sindex(self) -> bool:
        return self.ds.sindex

    @property
    def total_bounds(self) -> rio.coords.BoundingBox:
        """Total bounds of the vector."""
        if not self.is_loaded and self._bounds is not None:
            return np.array(self._bounds)
        return self.ds.total_bounds

    # Exception ! Vector.bounds corresponds to the total_bounds
    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """
        Total bounding box of the vector.

        Caution: this is equivalent to ``GeoDataFrame.total_bounds``,
        but not ``GeoDataFrame.bounds`` (per-feature bounds) which is instead defined as
        ``Vector.geom_bounds``.
        """
        if not self.is_loaded and self._bounds is not None:
            return self._bounds
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    # --------------------------------------------
    # GeoPandasBase - Methods that return a Series
    # --------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def contains(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.contains(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def geom_equals(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.geom_equals(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def geom_equals_exact(
        self,
        other: VectorType,
        tolerance: float,
        align: bool = True,
    ) -> pd.Series:
        return self._override_gdf_output(self.ds.geom_equals_exact(other=other.ds, tolerance=tolerance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def crosses(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.crosses(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def disjoint(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.disjoint(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def intersects(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.intersects(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def overlaps(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.overlaps(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def touches(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.touches(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def within(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.within(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def covers(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.covers(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def covered_by(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.covered_by(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def distance(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.distance(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def is_valid_reason(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_valid_reason())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def count_coordinates(self) -> pd.Series:
        return self._override_gdf_output(self.ds.count_coordinates())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def count_geometries(self) -> pd.Series:
        return self._override_gdf_output(self.ds.count_geometries())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def count_interior_rings(self) -> pd.Series:
        return self._override_gdf_output(self.ds.count_interior_rings())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def get_precision(self) -> pd.Series:
        return self._override_gdf_output(self.ds.get_precision())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def minimum_clearance(self) -> pd.Series:
        return self._override_gdf_output(self.ds.minimum_clearance())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def minimum_bounding_radius(self) -> pd.Series:
        return self._override_gdf_output(self.ds.minimum_bounding_radius())

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def contains_properly(self, other: VectorType, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.contains_properly(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def dwithin(self, other: VectorType, distance: float, align: bool = None) -> pd.Series:
        return self._override_gdf_output(self.ds.dwithin(other=other.ds, distance=distance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def hausdorff_distance(self, other: VectorType, align: bool = None, densify: float = None) -> pd.Series:
        return self._override_gdf_output(self.ds.hausdorff_distance(other=other.ds, align=align, densify=densify))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def frechet_distance(self, other: VectorType, align: bool = None, densify: float = None) -> pd.Series:
        return self._override_gdf_output(self.ds.frechet_distance(other=other.ds, align=align, densify=densify))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def hilbert_distance(self, total_bounds: Any = None, level: int = 16) -> pd.Series:
        return self._override_gdf_output(self.ds.hilbert_distance(total_bounds=total_bounds, level=level))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def relate_pattern(self, other: VectorType, pattern: str, align: Any = None) -> pd.Series:
        return self._override_gdf_output(self.ds.relate_pattern(other=other.ds, pattern=pattern, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def relate(self, other: VectorType, align: Any = None) -> VectorType:
        return self._override_gdf_output(self.ds.relate(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def project(self, other: VectorType, normalized: bool = False, align: Any = None) -> VectorType:
        return self._override_gdf_output(self.ds.project(other=other.ds, normalized=normalized, align=align))

    # -----------------------------------------------
    # GeoPandasBase - Methods that return a GeoSeries
    # -----------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector")
    def representative_point(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.representative_point())

    @copy_doc(gpd.GeoSeries, "Vector")
    def normalize(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.normalize())

    @copy_doc(gpd.GeoSeries, "Vector")
    def make_valid(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.make_valid())

    @copy_doc(gpd.GeoSeries, "Vector")
    def difference(self: VectorType, other: VectorType, align: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.difference(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def symmetric_difference(self: VectorType, other: VectorType, align: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.symmetric_difference(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def union(self: VectorType, other: VectorType, align: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.union(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def union_all(self: VectorType, method: str = "unary") -> VectorType:
        return self._override_gdf_output(self.ds.union_all(method=method))

    @copy_doc(gpd.GeoSeries, "Vector")
    def intersection(self: VectorType, other: VectorType, align: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.intersection(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def clip_by_rect(self: VectorType, xmin: float, ymin: float, xmax: float, ymax: float) -> VectorType:
        return self._override_gdf_output(self.ds.clip_by_rect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    @copy_doc(gpd.GeoSeries, "Vector")
    def buffer(
        self: VectorType,
        distance: float,
        resolution: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
        **kwargs: Any,
    ) -> VectorType:
        return self._override_gdf_output(
            self.ds.buffer(
                distance=distance,
                resolution=resolution,
                cap_style=cap_style,
                join_style=join_style,
                mitre_limit=mitre_limit,
                single_sided=single_sided,
                **kwargs,
            )
        )

    @copy_doc(gpd.GeoSeries, "Vector")
    def simplify(self: VectorType, tolerance: float, preserve_topology: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.simplify(tolerance=tolerance, preserve_topology=preserve_topology))

    @copy_doc(gpd.GeoSeries, "Vector")
    def affine_transform(self: VectorType, matrix: tuple[float, ...]) -> VectorType:
        return self._override_gdf_output(self.ds.affine_transform(matrix=matrix))

    @copy_doc(gpd.GeoSeries, "Vector")
    def rotate(self: VectorType, angle: float, origin: str = "center", use_radians: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.rotate(angle=angle, origin=origin, use_radians=use_radians))

    @copy_doc(gpd.GeoSeries, "Vector")
    def scale(
        self: VectorType, xfact: float = 1.0, yfact: float = 1.0, zfact: float = 1.0, origin: str = "center"
    ) -> VectorType:
        return self._override_gdf_output(self.ds.scale(xfact=xfact, yfact=yfact, zfact=zfact, origin=origin))

    @copy_doc(gpd.GeoSeries, "Vector")
    def skew(
        self: VectorType, xs: float = 0.0, ys: float = 0.0, origin: str = "center", use_radians: bool = False
    ) -> VectorType:
        return self._override_gdf_output(self.ds.skew(xs=xs, ys=ys, origin=origin, use_radians=use_radians))

    @copy_doc(gpd.GeoSeries, "Vector")
    def concave_hull(self: VectorType, ratio: float = 0.0, allow_holes: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.concave_hull(ratio=ratio, allow_holes=allow_holes))

    @copy_doc(gpd.GeoSeries, "Vector")
    def delaunay_triangles(self: VectorType, tolerance: float = 0.0, only_edges: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.delaunay_triangles(tolerance=tolerance, only_edges=only_edges))

    @copy_doc(gpd.GeoSeries, "Vector")
    def voronoi_polygons(
        self: VectorType, tolerance: float = 0.0, extend_to: Any = None, only_edges: bool = False
    ) -> VectorType:
        return self._override_gdf_output(
            self.ds.voronoi_polygons(tolerance=tolerance, extend_to=extend_to, only_edges=only_edges)
        )

    @copy_doc(gpd.GeoSeries, "Vector")
    def minimum_rotated_rectangle(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.minimum_rotated_rectangle())

    @copy_doc(gpd.GeoSeries, "Vector")
    def minimum_bounding_circle(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.minimum_bounding_circle())

    @copy_doc(gpd.GeoSeries, "Vector")
    def extract_unique_points(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.extract_unique_points())

    @copy_doc(gpd.GeoSeries, "Vector")
    def offset_curve(
        self: VectorType, distance: float, quad_segs: int = 8, join_style: str = "round", mitre_limit: float = 5.0
    ) -> VectorType:
        return self._override_gdf_output(
            self.ds.offset_curve(distance=distance, quad_segs=quad_segs, join_style=join_style, mitre_limit=mitre_limit)
        )

    @copy_doc(gpd.GeoSeries, "Vector")
    def remove_repeated_points(self: VectorType, tolerance: float = 0.0) -> VectorType:
        return self._override_gdf_output(self.ds.remove_repeated_points(tolerance=tolerance))

    @copy_doc(gpd.GeoSeries, "Vector")
    def reverse(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.reverse())

    @copy_doc(gpd.GeoSeries, "Vector")
    def segmentize(self: VectorType, max_segment_length: float) -> VectorType:
        return self._override_gdf_output(self.ds.segmentize(max_segment_length=max_segment_length))

    @copy_doc(gpd.GeoSeries, "Vector")
    def transform(self: VectorType, transformation: Any, include_z: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.transform(transformation=transformation, include_z=include_z))

    @copy_doc(gpd.GeoSeries, "Vector")
    def force_2d(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.force_2d())

    @copy_doc(gpd.GeoSeries, "Vector")
    def force_3d(self: VectorType, z: Any = 0) -> VectorType:
        return self._override_gdf_output(self.ds.force_3d(z=z))

    @copy_doc(gpd.GeoSeries, "Vector")
    def line_merge(self: VectorType, directed: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.line_merge(directed=directed))

    @copy_doc(gpd.GeoSeries, "Vector")
    def intersection_all(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.intersection_all())

    @copy_doc(gpd.GeoSeries, "Vector")
    def snap(self: VectorType, other: Vector, tolerance: float, align: Any = None) -> VectorType:
        return self._override_gdf_output(self.ds.snap(other=other.ds, tolerance=tolerance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def shared_paths(self: VectorType, other: VectorType, align: Any = None) -> VectorType:
        return self._override_gdf_output(self.ds.shared_paths(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def build_area(self: VectorType, node: bool = True) -> VectorType:
        return self._override_gdf_output(self.ds.build_area(node=node))

    @copy_doc(gpd.GeoSeries, "Vector")
    def polygonize(self: VectorType, node: bool = True, full: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.polygonize(node=node, full=full))

    @copy_doc(gpd.GeoSeries, "Vector")
    def shortest_line(self: VectorType, other: VectorType, align: bool = None) -> VectorType:
        return self._override_gdf_output(self.ds.shortest_line(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def get_geometry(self: VectorType, index: int) -> VectorType:
        return self._override_gdf_output(self.ds.get_geometry(index=index))

    @copy_doc(gpd.GeoSeries, "Vector")
    def interpolate(self: VectorType, distance: float | VectorType, normalized: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.interpolate(distance=distance, normalized=normalized))

    # -----------------------------------------------
    # GeoPandasBase - Methods that return other types
    # -----------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector")
    def get_coordinates(
        self, include_z: bool = False, ignore_index: bool = False, index_parts: bool = False
    ) -> pd.DataFrame:
        return self.ds.get_coordinates(include_z=include_z, ignore_index=ignore_index, index_parts=index_parts)

    # ----------------------------------------------
    # GeoDataFrame - Methods that return a GeoSeries
    # ----------------------------------------------

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def dissolve(
        self: VectorType,
        by: Any = None,
        aggfunc: Any = "first",
        as_index: bool = True,
        level: Any = None,
        sort: bool = True,
        observed: bool = False,
        dropna: bool = True,
        method: str = "unary",
        **kwargs: Any,
    ) -> VectorType:
        return self._override_gdf_output(
            self.ds.dissolve(
                by=by,
                aggfunc=aggfunc,
                as_index=as_index,
                level=level,
                sort=sort,
                observed=observed,
                dropna=dropna,
                method=method,
                **kwargs,
            )
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def explode(
        self: VectorType,
        column: str | None = None,
        ignore_index: bool = False,
        index_parts: bool | None = None,
        **kwargs: Any,
    ) -> VectorType:
        return self._override_gdf_output(
            self.ds.explode(column=column, ignore_index=ignore_index, index_parts=index_parts, **kwargs)
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def clip(self: VectorType, mask: Any, keep_geom_type: bool = False, sort: bool = False) -> VectorType:
        return self._override_gdf_output(self.ds.clip(mask=mask, keep_geom_type=keep_geom_type, sort=sort))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def sjoin(self: VectorType, df: VectorType | gpd.GeoDataFrame, *args: Any, **kwargs: Any) -> VectorType:
        # Ensure input is a geodataframe
        if isinstance(df, Vector):
            gdf = df.ds
        else:
            gdf = df

        return self._override_gdf_output(self.ds.sjoin(gdf, *args, **kwargs))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def sjoin_nearest(
        self: VectorType,
        right: VectorType | gpd.GeoDataFrame,
        how: str = "inner",
        max_distance: float | None = None,
        lsuffix: str = "left",
        rsuffix: str = "right",
        distance_col: str | None = None,
        exclusive: bool = False,
    ) -> VectorType:
        # Ensure input is a geodataframe
        if isinstance(right, Vector):
            gdf = right.ds
        else:
            gdf = right

        return self._override_gdf_output(
            self.ds.sjoin_nearest(
                right=gdf,
                how=how,
                max_distance=max_distance,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                distance_col=distance_col,
                exclusive=exclusive,
            )
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def overlay(
        self: VectorType,
        right: VectorType | gpd.GeoDataFrame,
        how: str = "intersection",
        keep_geom_type: bool | None = None,
        make_valid: bool = True,
    ) -> VectorType:
        # Ensure input is a geodataframe
        if isinstance(right, Vector):
            gdf = right.ds
        else:
            gdf = right

        return self._override_gdf_output(
            self.ds.overlay(right=gdf, how=how, keep_geom_type=keep_geom_type, make_valid=make_valid)
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def set_geometry(
        self: VectorType, col: str, drop: bool = False, inplace: bool = False, crs: CRS = None
    ) -> VectorType | None:

        if inplace:
            self.ds = self.ds.set_geometry(col=col, drop=drop, crs=crs)
            return None
        else:
            return self._override_gdf_output(self.ds.set_geometry(col=col, drop=drop, crs=crs))

    # Subsection of methods that shouldn't override the output for Vector subclasses

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_crs(
        self: VectorType, crs: CRS | None = None, epsg: int | None = None, inplace: bool = False
    ) -> VectorType | None:

        if inplace:
            self.ds = self.ds.to_crs(crs=crs, epsg=epsg)
            return None
        else:
            copy = self.copy()
            copy.ds = self.ds.to_crs(crs=crs, epsg=epsg)
            return copy

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def set_crs(
        self: VectorType,
        crs: CRS | None = None,
        epsg: int | None = None,
        inplace: bool = False,
        allow_override: bool = False,
    ) -> VectorType | None:

        if inplace:
            self.ds = self.ds.set_crs(crs=crs, epsg=epsg, allow_override=allow_override)
            return None
        else:
            copy = self.copy()
            copy.ds = self.ds.set_crs(crs=crs, epsg=epsg, allow_override=allow_override)
            return copy

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def set_precision(
        self: VectorType,
        grid_size: float = 0.0,
        mode: str = "valid_output",
        inplace: bool = False,
    ) -> VectorType | None:

        if inplace:
            self.ds = self.ds.set_precision(grid_size=grid_size, mode=mode)
            return None
        else:
            copy = self.copy()
            copy.ds = self.ds.set_precision(grid_size=grid_size, mode=mode)
            return copy

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def rename_geometry(self, col: str, inplace: bool = False) -> Vector | None:

        if inplace:
            self.ds = self.ds.set_geometry(col=col)
            return None
        else:
            copy = self.copy()
            copy.ds = self.ds.rename_geometry(col=col)
            return copy

    # -----------------------------------
    # GeoDataFrame: other functionalities
    # -----------------------------------

    def __getitem__(self, key: RasterType | VectorType | list[float] | tuple[float, ...] | Any) -> Any:
        """
        Index the geodataframe.
        """

        return self._override_gdf_output(self.ds.__getitem__(key))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def __setitem__(self, key: Any, value: Any) -> None:
        self.ds.__setitem__(key, value)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def cx(self: VectorType) -> VectorType:
        return self._override_gdf_output(self.ds.cx)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def estimate_utm_crs(self, datum_name: str = "WGS 84") -> CRS:

        return self.ds.estimate_utm_crs(datum_name=datum_name)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def iterfeatures(
        self, na: str | None = "null", show_bbox: bool = False, drop_id: bool = False
    ) -> Generator[dict[str, str | dict[str, Any] | None | dict[str, Any]], Any, Any]:

        return self.ds.iterfeatures(na=na, show_bbox=show_bbox, drop_id=drop_id)

    @classmethod
    @copy_doc(gpd.GeoDataFrame, "Vector")
    def from_file(cls, filename: str, **kwargs: Any) -> Vector:

        return cls(gpd.GeoDataFrame.from_file(filename=filename, **kwargs))

    @classmethod
    @copy_doc(gpd.GeoDataFrame, "Vector")
    def from_arrow(cls, table: Any, geometry: Any = None) -> Vector:

        return cls(gpd.GeoDataFrame.from_arrow(table=table, geometry=geometry))

    @classmethod
    @copy_doc(gpd.GeoDataFrame, "Vector")
    def from_features(cls, features: Iterable[dict[str, Any]], crs: CRS, columns: list[str]) -> Vector:

        return cls(gpd.GeoDataFrame.from_features(features=features, crs=crs, columns=columns))

    @classmethod
    @copy_doc(gpd.GeoDataFrame, "Vector")
    def from_postgis(
        cls,
        sql: str,
        con: Any,
        geom_col: str = "geom",
        crs: CRS | None = None,
        index_col: str | None = None,
        coerce_float: bool = True,
        parse_dates: Any = None,
        params: Any = None,
        chunksize: Any = None,
    ) -> Vector:

        return cls(
            gpd.GeoDataFrame.from_postgis(
                sql=sql,
                con=con,
                geom_col=geom_col,
                crs=crs,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                params=params,
                chunksize=chunksize,
            )
        )

    @classmethod
    @copy_doc(gpd.GeoDataFrame, "Vector")
    def from_dict(cls, data: dict[str, Any], geometry: Any = None, crs: CRS | None = None, **kwargs: Any) -> Vector:

        return cls(gpd.GeoDataFrame.from_dict(data=data, geometry=geometry, crs=crs, **kwargs))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_file(self, filename: str, driver: Any = None, schema: Any = None, index: Any = None, **kwargs: Any) -> None:

        return self.ds.to_file(filename=filename, driver=driver, schema=schema, index=index, **kwargs)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_feather(
        self, path: Any, index: Any = None, compression: Any = None, schema_version: Any = None, **kwargs: Any
    ) -> None:

        return self.ds.to_feather(
            path=path, index=index, compression=compression, schema_version=schema_version, **kwargs
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_parquet(
        self, path: Any, index: Any = None, compression: Any = "snappy", schema_version: Any = None, **kwargs: Any
    ) -> None:

        return self.ds.to_parquet(
            path=path, index=index, compression=compression, schema_version=schema_version, **kwargs
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_arrow(
        self, index: Any = None, geometry_encoding: Any = "WKB", interleaved: Any = True, include_z: Any = None
    ) -> Any:

        return self.ds.to_arrow(
            index=index, geometry_encoding=geometry_encoding, interleaved=interleaved, include_z=include_z
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_geo_dict(self, na: Any = "null", show_bbox: bool = False, drop_id: bool = False) -> Any:

        return self.ds.to_geo_dict(na=na, show_bbox=show_bbox, drop_id=drop_id)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_wkt(self, **kwargs: Any) -> pd.DataFrame:

        return self.ds.to_wkt(**kwargs)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_wkb(self, hex: bool = False, **kwargs: Any) -> pd.DataFrame:

        return self.ds.to_wkb(hex=hex, **kwargs)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_json(self, na: Any = "null", show_bbox: bool = False, drop_id: bool = False, **kwargs: Any) -> str | None:

        return self.ds.to_json(na=na, show_bbox=show_bbox, drop_id=drop_id, **kwargs)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_postgis(
        self,
        name: str,
        con: Any,
        schema: Any = None,
        if_exists: Any = "fail",
        index: Any = False,
        index_label: Any = None,
        chunksize: Any = None,
        dtype: Any = None,
    ) -> None:

        return self.ds.to_postgis(
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def to_csv(
        self,
        path_or_buf: str | PathLike[str] | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        sep: str = ",",
        na_rep: str = "",
        float_format: Any = None,
        columns: Sequence[Hashable] | None = None,
        header: bool | list[str] = True,
        index: bool = True,
        index_label: Hashable | Sequence[Hashable] | None = None,
        mode: str = "w",
        encoding: str | None = None,
        compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd", "tar"] | dict[str, Any] | None = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        decimal: str = ".",
        errors: str = "strict",
        storage_options: dict[str, Any] | None = None,
    ) -> str | None:

        return self.ds.to_csv(
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            mode=mode,
            encoding=encoding,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
            errors=errors,
            storage_options=storage_options,
        )

    # --------------------------------
    # End of GeoPandas functionalities
    # --------------------------------
