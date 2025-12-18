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
Module for Vector class.
"""

from __future__ import annotations

import pathlib
from collections import abc
from os import PathLike
from typing import (
    Any,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    TypeVar,
    overload,
)

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from geopandas.testing import assert_geodataframe_equal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import Version
from pandas._typing import WriteBuffer
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

import geoutils as gu
from geoutils import profiler
from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.interface.distance import _proximity_from_vector_or_raster
from geoutils.interface.raster_vector import _create_mask, _rasterize
from geoutils.misc import copy_doc, deprecate
from geoutils.projtools import (
    _get_bounds_projected,
    _get_footprint_projected,
    _get_utm_ups_crs,
)
from geoutils.vector.geometric import _buffer_metric, _buffer_without_overlap
from geoutils.vector.geotransformations import _reproject

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
VectorType = TypeVar("VectorType", bound="Vector")


class Vector:
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

    @profiler.profile("geoutils.vector.vector.__init__", memprof=True)  # type: ignore
    def __init__(
        self, filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | dict[str, Any]
    ):
        """
        Instantiate a vector from either a filename, a GeoPandas dataframe or series, or a Shapely geometry.

        :param filename_or_dataset: Path to file, or GeoPandas dataframe or series, or Shapely geometry.
        """

        self._name: str | None = None
        self._ds: gpd.GeoDataFrame | None = None

        # If Vector is passed, simply point back to Vector
        if isinstance(filename_or_dataset, Vector):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # If filename is passed
        elif isinstance(filename_or_dataset, (str, pathlib.Path)):
            ds = gpd.read_file(filename_or_dataset)
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

        # Write name attribute
        if isinstance(filename_or_dataset, str):
            self._name = filename_or_dataset
        if isinstance(filename_or_dataset, pathlib.Path):
            self._name = filename_or_dataset.name

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""
        return self.ds.crs

    @property
    def ds(self) -> gpd.GeoDataFrame:
        """Geodataframe of the vector."""
        return self._ds

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new geodataframe."""

        if isinstance(new_ds, gpd.GeoDataFrame):
            self._ds = new_ds
        elif isinstance(new_ds, gpd.GeoSeries):
            self._ds = gpd.GeoDataFrame(geometry=new_ds)
        else:
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")

    def vector_equal(self, other: gu.Vector, **kwargs: Any) -> bool:
        """
        Check if two vectors are equal.

        Keyword arguments are passed to geopandas.assert_geodataframe_equal.
        """

        try:
            assert_geodataframe_equal(self.ds, other.ds, **kwargs)
            vector_eq = True
        except AssertionError:
            vector_eq = False

        return vector_eq

    @property
    def name(self) -> str | None:
        """Name on disk, if it exists."""
        return self._name

    @property
    def geometry(self) -> gpd.GeoSeries:
        return self.ds.geometry

    @property
    def columns(self) -> pd.Index:
        return self.ds.columns

    @property
    def index(self) -> pd.Index:
        return self.ds.index

    def copy(self: VectorType) -> VectorType:
        """Return a copy of the vector."""
        # Utilise the copy method of GeoPandas
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.copy())  # type: ignore
        return new_vector  # type: ignore

    def __repr__(self) -> str:
        """Convert vector to string representation."""

        # Get the representation of ds
        str_ds = "\n       ".join(self.__str__().split("\n"))

        s = str(
            self.__class__.__name__
            + "(\n"
            + "  ds="
            + str_ds
            + "\n  crs="
            + self.crs.__str__()
            + "\n  bounds="
            + self.bounds.__str__()
            + ")"
        )

        return s

    def _repr_html_(self) -> str:
        """Convert vector to HTML string representation for documentation."""

        str_ds = "\n       ".join(self.ds.__str__().split("\n"))

        # Over-ride Raster's method to remove nodata value (always None)
        # Use <pre> to keep white spaces, <span> to keep line breaks
        s = str(
            '<pre><span style="white-space: pre-wrap"><b><em>'
            + self.__class__.__name__
            + "</em></b>(\n"
            + "  <b>ds=</b>"
            + str_ds
            + "\n  <b>crs=</b>"
            + self.crs.__str__()
            + "\n  <b>bounds=</b>"
            + self.bounds.__repr__()
            + ")</span></pre>"
        )

        return s

    def __str__(self) -> str:
        """Provide simplified vector string representation for print()."""

        return str(self.ds.__str__())

    @overload
    def info(self, verbose: Literal[True] = ...) -> None: ...

    @overload
    def info(self, verbose: Literal[False]) -> str: ...

    def info(self, verbose: bool = True) -> str | None:
        """
        Summarize information about the vector.

        :param verbose: If set to True (default) will directly print to screen and return None

        :returns: Information about vector attributes.
        """
        as_str = [  # 'Driver:             {} \n'.format(self.driver),
            f"Filename:           {self.name} \n",
            f"Coordinate system:  EPSG:{self.ds.crs.to_epsg()}\n",
            f"Extent:             {self.ds.total_bounds.tolist()} \n",
            f"Number of features: {len(self.ds)} \n",
            f"Attributes:         {self.ds.columns.tolist()}",
        ]

        if verbose:
            print("".join(as_str))
            return None
        else:
            return "".join(as_str)

    def plot(
        self,
        ref_crs: gu.Raster | rio.io.DatasetReader | VectorType | gpd.GeoDataFrame | str | CRS | int | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        return_axes: bool = False,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        r"""
        Plot the vector.

        This method is a wrapper to geopandas.GeoDataFrame.plot. Any \*\*kwargs which
        you give this method will be passed to it.

        :param ref_crs: Coordinate reference system to match when plotting.
        :param cmap: Colormap to use. Default is plt.rcParams['image.cmap'].
        :param vmin: Colorbar minimum value. Default is data min.
        :param vmax: Colorbar maximum value. Default is data max.
        :param alpha: Transparency of raster and colorbar.
        :param cbar_title: Colorbar label. Default is None.
        :param add_cbar: Set to True to display a colorbar. Default is True if a "column" argument is passed.
        :param ax: A figure ax to be used for plotting. If None, will plot on current axes. If "new",
            will create a new axis.
        :param return_axes: Whether to return axes.

        :returns: None, or (ax, caxes) if return_axes is True
        """

        # Ensure that the vector is in the same crs as a reference
        if isinstance(ref_crs, (gu.Raster, rio.io.DatasetReader, Vector, gpd.GeoDataFrame, str)):
            vect_reproj = self.reproject(ref=ref_crs)
        elif isinstance(ref_crs, (CRS, int)):
            vect_reproj = self.reproject(crs=ref_crs)
        else:
            vect_reproj = self

        # Create axes, or get current ones by default (like in matplotlib)
        if ax is None:
            ax0 = plt.gca()
        elif isinstance(ax, str) and ax.lower() == "new":
            _, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")

        # Set add_cbar depending on column argument
        if "column" in kwargs.keys() and add_cbar:
            add_cbar = True
        else:
            add_cbar = False

        # Update with this function's arguments
        if add_cbar:
            legend = True
        else:
            legend = False

        if "legend" in list(kwargs.keys()):
            legend = kwargs.pop("legend")

        # Get colormap arguments that might have been passed in the keyword args
        if "legend_kwds" in list(kwargs.keys()) and legend:
            legend_kwds = kwargs.pop("legend_kwds")
            if cbar_title is not None:
                legend_kwds.update({"label": cbar_title})  # Pad updates depending on figsize during plot,
        else:
            if cbar_title is not None:
                legend_kwds = {"label": cbar_title}
            else:
                legend_kwds = None

        # Add colorbar
        if add_cbar or cbar_title:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad="2%")
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=norm
            )  # , orientation="horizontal", ticklocation="top")
            cbar.solids.set_alpha(alpha)
        else:
            cax = None
            cbar = None

        # Plot
        vect_reproj.ds.plot(
            ax=ax0,
            cax=cax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs,
        )
        plt.sca(ax0)

        # If returning axes
        if return_axes:
            return ax0, cax
        else:
            return None

    @deprecate(
        removal_version=Version("0.3.0"),
        details="The function .save() will be soon deprecated, use .to_file() instead.",
    )  # type: ignore
    def save(
        self,
        filename: str | pathlib.Path,
        driver: str | None = None,
        schema: dict[str, Any] | None = None,
        index: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the vector to file.

        This function is a simple wrapper of :func:`geopandas.GeoDataFrame.to_file`. See there for details.

        :param filename: Filename to write the file to.
        :param driver: Driver to write file with.
        :param schema: Dictionary passed to Fiona to better control how the file is written.
        :param index: Whether to write the index or not.

        :returns: None.
        """

        self.ds.to_file(filename=filename, driver=driver, schema=schema, index=index, **kwargs)

    ############################################################################
    # Overridden and wrapped methods from GeoPandas API to logically cast outputs
    ############################################################################

    def _override_gdf_output(
        self, other: gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | pd.Series | Any
    ) -> Vector | pd.Series:
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

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def area(self) -> pd.Series:
        return self._override_gdf_output(self.ds.area)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def length(self) -> pd.Series:
        return self._override_gdf_output(self.ds.length)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def interiors(self) -> pd.Series:
        return self._override_gdf_output(self.ds.interiors)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def geom_type(self) -> pd.Series:
        return self._override_gdf_output(self.ds.geom_type)

    # Exception ! bounds is renamed geom_bounds to make Raster and Vector "bounds" the same "total_bounds"
    @property
    def geom_bounds(self) -> pd.Series:
        """Returns or appends to ``Vector`` a ``Series`` with the bounds of each geometry feature."""
        return self.ds.bounds

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_empty(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_empty)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_ring(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_ring)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_simple(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_simple)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_valid(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_valid)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def has_z(self) -> pd.Series:
        return self.ds.has_z

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_ccw(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_ccw)

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)  # type: ignore
    @property
    def is_closed(self) -> pd.Series:
        return self._override_gdf_output(self.ds.is_closed)

    # --------------------------------------------------
    # GeoPandasBase - Attributes that return a GeoSeries
    # --------------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def boundary(self) -> Vector:
        return self._override_gdf_output(self.ds.boundary)

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def centroid(self) -> Vector:
        return self._override_gdf_output(self.ds.centroid)

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def convex_hull(self) -> Vector:
        return self._override_gdf_output(self.ds.convex_hull)

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def envelope(self) -> Vector:
        return self._override_gdf_output(self.ds.envelope)

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def exterior(self) -> Vector:
        return self._override_gdf_output(self.ds.exterior)

    # ---------------------------------------------------------------------------------
    # GeoPandasBase - Attributes that return a specific value (not Series or GeoSeries)
    # ---------------------------------------------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def has_sindex(self) -> bool:
        return self.ds.has_sindex

    @copy_doc(gpd.GeoSeries, "Vector")  # type: ignore
    @property
    def sindex(self) -> bool:
        return self.ds.sindex

    @property
    def total_bounds(self) -> rio.coords.BoundingBox:
        """Total bounds of the vector."""
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
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    @property
    def footprint(self) -> Vector:
        """Footprint of the raster."""
        return self.get_footprint_projected(self.crs)

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
        other: Vector,
        tolerance: float,
        align: bool = True,
    ) -> pd.Series:
        return self._override_gdf_output(self.ds.geom_equals_exact(other=other.ds, tolerance=tolerance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def crosses(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.crosses(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def disjoint(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.disjoint(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def intersects(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.intersects(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def overlaps(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.overlaps(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def touches(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.touches(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def within(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.within(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def covers(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.covers(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def covered_by(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.covered_by(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def distance(self, other: Vector, align: bool = True) -> pd.Series:
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
    def contains_properly(self, other: Vector, align: bool = True) -> pd.Series:
        return self._override_gdf_output(self.ds.contains_properly(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def dwithin(self, other: Vector, distance: float, align: bool = None) -> pd.Series:
        return self._override_gdf_output(self.ds.dwithin(other=other.ds, distance=distance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def hausdorff_distance(self, other: Vector, align: bool = None, densify: float = None) -> pd.Series:
        return self._override_gdf_output(self.ds.hausdorff_distance(other=other.ds, align=align, densify=densify))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def frechet_distance(self, other: Vector, align: bool = None, densify: float = None) -> pd.Series:
        return self._override_gdf_output(self.ds.frechet_distance(other=other.ds, align=align, densify=densify))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def hilbert_distance(self, total_bounds: Any = None, level: int = 16) -> pd.Series:
        return self._override_gdf_output(self.ds.hilbert_distance(total_bounds=total_bounds, level=level))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def relate_pattern(self, other: Vector, pattern: str, align: Any = None) -> pd.Series:
        return self._override_gdf_output(self.ds.relate_pattern(other=other.ds, pattern=pattern, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def relate(self, other: Vector, align: Any = None) -> Vector:
        return self._override_gdf_output(self.ds.relate(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector", replace_return_series_statement=True)
    def project(self, other: Vector, normalized: bool = False, align: Any = None) -> Vector:
        return self._override_gdf_output(self.ds.project(other=other.ds, normalized=normalized, align=align))

    # -----------------------------------------------
    # GeoPandasBase - Methods that return a GeoSeries
    # -----------------------------------------------

    @copy_doc(gpd.GeoSeries, "Vector")
    def representative_point(self) -> Vector:
        return self._override_gdf_output(self.ds.representative_point())

    @copy_doc(gpd.GeoSeries, "Vector")
    def normalize(self) -> Vector:
        return self._override_gdf_output(self.ds.normalize())

    @copy_doc(gpd.GeoSeries, "Vector")
    def make_valid(self) -> Vector:
        return self._override_gdf_output(self.ds.make_valid())

    @copy_doc(gpd.GeoSeries, "Vector")
    def difference(self, other: Vector, align: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.difference(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def symmetric_difference(self, other: Vector, align: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.symmetric_difference(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def union(self, other: Vector, align: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.union(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def union_all(self, method: str = "unary") -> Vector:
        return self._override_gdf_output(self.ds.union_all(method=method))

    @copy_doc(gpd.GeoSeries, "Vector")
    def intersection(self, other: Vector, align: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.intersection(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def clip_by_rect(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Vector:
        return self._override_gdf_output(self.ds.clip_by_rect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    @copy_doc(gpd.GeoSeries, "Vector")
    def buffer(
        self,
        distance: float,
        resolution: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False,
        **kwargs: Any,
    ) -> Vector:
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
    def simplify(self, tolerance: float, preserve_topology: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.simplify(tolerance=tolerance, preserve_topology=preserve_topology))

    @copy_doc(gpd.GeoSeries, "Vector")
    def affine_transform(self, matrix: tuple[float, ...]) -> Vector:
        return self._override_gdf_output(self.ds.affine_transform(matrix=matrix))

    @copy_doc(gpd.GeoSeries, "Vector")
    def rotate(self, angle: float, origin: str = "center", use_radians: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.rotate(angle=angle, origin=origin, use_radians=use_radians))

    @copy_doc(gpd.GeoSeries, "Vector")
    def scale(self, xfact: float = 1.0, yfact: float = 1.0, zfact: float = 1.0, origin: str = "center") -> Vector:
        return self._override_gdf_output(self.ds.scale(xfact=xfact, yfact=yfact, zfact=zfact, origin=origin))

    @copy_doc(gpd.GeoSeries, "Vector")
    def skew(self, xs: float = 0.0, ys: float = 0.0, origin: str = "center", use_radians: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.skew(xs=xs, ys=ys, origin=origin, use_radians=use_radians))

    @copy_doc(gpd.GeoSeries, "Vector")
    def concave_hull(self, ratio: float = 0.0, allow_holes: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.concave_hull(ratio=ratio, allow_holes=allow_holes))

    @copy_doc(gpd.GeoSeries, "Vector")
    def delaunay_triangles(self, tolerance: float = 0.0, only_edges: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.delaunay_triangles(tolerance=tolerance, only_edges=only_edges))

    @copy_doc(gpd.GeoSeries, "Vector")
    def voronoi_polygons(self, tolerance: float = 0.0, extend_to: Any = None, only_edges: bool = False) -> Vector:
        return self._override_gdf_output(
            self.ds.voronoi_polygons(tolerance=tolerance, extend_to=extend_to, only_edges=only_edges)
        )

    @copy_doc(gpd.GeoSeries, "Vector")
    def minimum_rotated_rectangle(self) -> Vector:
        return self._override_gdf_output(self.ds.minimum_rotated_rectangle())

    @copy_doc(gpd.GeoSeries, "Vector")
    def minimum_bounding_circle(self) -> Vector:
        return self._override_gdf_output(self.ds.minimum_bounding_circle())

    @copy_doc(gpd.GeoSeries, "Vector")
    def extract_unique_points(self) -> Vector:
        return self._override_gdf_output(self.ds.extract_unique_points())

    @copy_doc(gpd.GeoSeries, "Vector")
    def offset_curve(
        self, distance: float, quad_segs: int = 8, join_style: str = "round", mitre_limit: float = 5.0
    ) -> Vector:
        return self._override_gdf_output(
            self.ds.offset_curve(distance=distance, quad_segs=quad_segs, join_style=join_style, mitre_limit=mitre_limit)
        )

    @copy_doc(gpd.GeoSeries, "Vector")
    def remove_repeated_points(self, tolerance: float = 0.0) -> Vector:
        return self._override_gdf_output(self.ds.remove_repeated_points(tolerance=tolerance))

    @copy_doc(gpd.GeoSeries, "Vector")
    def reverse(self) -> Vector:
        return self._override_gdf_output(self.ds.reverse())

    @copy_doc(gpd.GeoSeries, "Vector")
    def segmentize(self, max_segment_length: float) -> Vector:
        return self._override_gdf_output(self.ds.segmentize(max_segment_length=max_segment_length))

    @copy_doc(gpd.GeoSeries, "Vector")
    def transform(self, transformation: Any, include_z: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.transform(transformation=transformation, include_z=include_z))

    @copy_doc(gpd.GeoSeries, "Vector")
    def force_2d(self) -> Vector:
        return self._override_gdf_output(self.ds.force_2d())

    @copy_doc(gpd.GeoSeries, "Vector")
    def force_3d(self, z: Any = 0) -> Vector:
        return self._override_gdf_output(self.ds.force_3d(z=z))

    @copy_doc(gpd.GeoSeries, "Vector")
    def line_merge(self, directed: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.line_merge(directed=directed))

    @copy_doc(gpd.GeoSeries, "Vector")
    def intersection_all(self) -> Vector:
        return self._override_gdf_output(self.ds.intersection_all())

    @copy_doc(gpd.GeoSeries, "Vector")
    def snap(self, other: Vector, tolerance: float, align: Any = None) -> Vector:
        return self._override_gdf_output(self.ds.snap(other=other.ds, tolerance=tolerance, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def shared_paths(self, other: Vector, align: Any = None) -> Vector:
        return self._override_gdf_output(self.ds.shared_paths(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def build_area(self, node: bool = True) -> Vector:
        return self._override_gdf_output(self.ds.build_area(node=node))

    @copy_doc(gpd.GeoSeries, "Vector")
    def polygonize(self, node: bool = True, full: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.polygonize(node=node, full=full))

    @copy_doc(gpd.GeoSeries, "Vector")
    def shortest_line(self, other: Vector, align: bool = None) -> Vector:
        return self._override_gdf_output(self.ds.shortest_line(other=other.ds, align=align))

    @copy_doc(gpd.GeoSeries, "Vector")
    def get_geometry(self, index: int) -> Vector:
        return self._override_gdf_output(self.ds.get_geometry(index=index))

    @copy_doc(gpd.GeoSeries, "Vector")
    def interpolate(self, distance: float | Vector, normalized: bool = False) -> Vector:
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
        self,
        by: Any = None,
        aggfunc: Any = "first",
        as_index: bool = True,
        level: Any = None,
        sort: bool = True,
        observed: bool = False,
        dropna: bool = True,
        method: str = "unary",
        **kwargs: Any,
    ) -> Vector:
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
        self, column: str | None = None, ignore_index: bool = False, index_parts: bool | None = None, **kwargs: Any
    ) -> Vector:
        return self._override_gdf_output(
            self.ds.explode(column=column, ignore_index=ignore_index, index_parts=index_parts, **kwargs)
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def clip(self, mask: Any, keep_geom_type: bool = False, sort: bool = False) -> Vector:
        return self._override_gdf_output(self.ds.clip(mask=mask, keep_geom_type=keep_geom_type, sort=sort))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def sjoin(self, df: Vector | gpd.GeoDataFrame, *args: Any, **kwargs: Any) -> Vector:
        # Ensure input is a geodataframe
        if isinstance(df, Vector):
            gdf = df.ds
        else:
            gdf = df

        return self._override_gdf_output(self.ds.sjoin(gdf, *args, **kwargs))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def sjoin_nearest(
        self,
        right: Vector | gpd.GeoDataFrame,
        how: str = "inner",
        max_distance: float | None = None,
        lsuffix: str = "left",
        rsuffix: str = "right",
        distance_col: str | None = None,
        exclusive: bool = False,
    ) -> Vector:
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
        self,
        right: Vector | gpd.GeoDataFrame,
        how: str = "intersection",
        keep_geom_type: bool | None = None,
        make_valid: bool = True,
    ) -> Vector:
        # Ensure input is a geodataframe
        if isinstance(right, Vector):
            gdf = right.ds
        else:
            gdf = right

        return self._override_gdf_output(
            self.ds.overlay(right=gdf, how=how, keep_geom_type=keep_geom_type, make_valid=make_valid)
        )

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def set_geometry(self, col: str, drop: bool = False, inplace: bool = False, crs: CRS = None) -> Vector | None:

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

    def __getitem__(self, key: gu.Raster | Vector | list[float] | tuple[float, ...] | Any) -> Any:
        """
        Index the geodataframe.
        """

        return self._override_gdf_output(self.ds.__getitem__(key))

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def __setitem__(self, key: Any, value: Any) -> None:
        self.ds.__setitem__(key, value)

    @copy_doc(gpd.GeoDataFrame, "Vector")
    def cx(self) -> Vector:
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

    @property
    def active_geometry_name(self) -> str:
        return self.ds.active_geometry_name

    @overload
    def crop(
        self: VectorType,
        crop_geom: gu.Raster | Vector | list[float] | tuple[float, ...],
        clip: bool,
        *,
        inplace: Literal[False] = False,
    ) -> VectorType: ...

    @overload
    def crop(
        self: VectorType,
        crop_geom: gu.Raster | Vector | list[float] | tuple[float, ...],
        clip: bool,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def crop(
        self: VectorType,
        crop_geom: gu.Raster | Vector | list[float] | tuple[float, ...],
        clip: bool,
        *,
        inplace: bool = False,
    ) -> VectorType | None: ...

    @profiler.profile("geoutils.vector.vector.crop", memprof=True)  # type: ignore
    def crop(
        self: VectorType,
        crop_geom: gu.Raster | Vector | list[float] | tuple[float, ...],
        clip: bool = False,
        *,
        inplace: bool = False,
    ) -> VectorType | None:
        """
        Crop the vector to given extent.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.

        Optionally, clip geometries to that extent (by default keeps all intersecting).

        Reprojection is done on the fly if georeferenced objects have different projections.

        :param crop_geom: Geometry to crop vector to, as either a Raster object, a Vector object, or a list of
            coordinates. If ``crop_geom`` is a raster or a vector, will crop to the bounds. If ``crop_geom`` is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param clip: Whether to clip the geometry to the given extent (by default keeps all intersecting).
        :param inplace: Whether to update the vector in-place.

        :returns: Cropped vector (or None if inplace).
        """
        if isinstance(crop_geom, (gu.Raster, Vector)):
            # For another Vector or Raster, we reproject the bounding box in the same CRS as self
            xmin, ymin, xmax, ymax = crop_geom.get_bounds_projected(out_crs=self.crs)
        elif isinstance(crop_geom, (list, tuple)):
            xmin, ymin, xmax, ymax = crop_geom
        else:
            raise TypeError("Crop geometry must be a Raster, Vector, or list of coordinates.")

        # Need to separate the two options, inplace update
        if inplace:
            self._ds = self.ds.cx[xmin:xmax, ymin:ymax]
            if clip:
                self._ds = self.ds.clip(mask=(xmin, ymin, xmax, ymax))
            return None
        # Or create a copy otherwise
        else:
            new_vector = self.copy()
            new_vector._ds = new_vector.ds.cx[xmin:xmax, ymin:ymax]
            if clip:
                new_vector._ds = new_vector.ds.clip(mask=(xmin, ymin, xmax, ymax))
            return new_vector

    @overload
    def reproject(
        self: Vector,
        ref: gu.Raster | rio.io.DatasetReader | VectorType | gpd.GeoDataFrame | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[False] = False,
    ) -> Vector: ...

    @overload
    def reproject(
        self: Vector,
        ref: gu.Raster | rio.io.DatasetReader | VectorType | gpd.GeoDataFrame | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def reproject(
        self: Vector,
        ref: gu.Raster | rio.io.DatasetReader | VectorType | gpd.GeoDataFrame | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: bool = False,
    ) -> Vector | None: ...

    @profiler.profile("geoutils.vector.vector.reproject", memprof=True)  # type: ignore
    def reproject(
        self: Vector,
        ref: gu.Raster | rio.io.DatasetReader | VectorType | gpd.GeoDataFrame | None = None,
        crs: CRS | str | int | None = None,
        inplace: bool = False,
    ) -> Vector | None:
        """
        Reproject vector to a specified coordinate reference system.

        **Match-reference:** a reference raster or vector can be passed to match CRS during reprojection.

        Alternatively, a CRS can be passed in many formats (string, EPSG integer, or CRS).

        To reproject a Vector with different source bounds, first run Vector.crop().

        :param ref: Reference raster or vector whose CRS to use as a reference for reprojection.
            Can be provided as a raster, vector, Rasterio dataset, or GeoPandas dataframe.
        :param crs: Specify the coordinate reference system or EPSG to reproject to. If dst_ref not set,
            defaults to self.crs.
        :param inplace: Whether to update the vector in-place.

        :returns: Reprojected vector (or None if inplace).
        """

        new_ds = _reproject(gdf=self.ds, ref=ref, crs=crs)

        if inplace:
            self.ds = new_ds
            return None
        else:
            copy = self.copy()
            copy.ds = new_ds
            return copy

    @overload
    def translate(
        self: VectorType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: Literal[False] = False,
    ) -> VectorType: ...

    @overload
    def translate(
        self: VectorType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def translate(
        self: VectorType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: bool = False,
    ) -> VectorType | None: ...

    def translate(
        self: VectorType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        inplace: bool = False,
    ) -> VectorType | None:
        """
        Shift a vector by a (x,y) offset, and optionally a z offset.

        The shifting only updates the coordinates (data is untouched).

        :param xoff: Translation x offset.
        :param yoff: Translation y offset.
        :param zoff: Translation z offset.
        :param inplace: Whether to modify the raster in-place.

        :returns: Shifted vector (or None if inplace).
        """

        translated_geoseries = self.geometry.translate(xoff=xoff, yoff=yoff, zoff=zoff)

        if inplace:
            # Overwrite transform by shifted transform
            self.ds.geometry = translated_geoseries
            return None
        else:
            vector_copy = self.copy()
            vector_copy.ds.geometry = translated_geoseries
            return vector_copy

    @overload
    def create_mask(
        self,
        ref: gu.PointCloud | gu.Raster | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] = None,
        *,
        as_array: Literal[False] = False,
    ) -> gu.PointCloudMask | gu.Raster: ...

    @overload
    def create_mask(
        self,
        ref: gu.Raster | gu.PointCloud | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] = None,
        *,
        as_array: Literal[True],
    ) -> NDArrayBool: ...

    def create_mask(
        self,
        ref: gu.Raster | gu.PointCloud | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] = None,
        as_array: bool = False,
    ) -> gu.Raster | gu.PointCloudMask | NDArrayBool:
        """
        Create a raster or point cloud mask from the vector features (True if pixel/point contained by any vector
        feature, False if not).

        For a raster reference, creates a raster mask with the resolution, bounds and CRS of the reference raster.
        For a point cloud reference, creates a point mask with the coordinates and CRS of the reference point cloud.

        Alternatively, for a raster mask, one can specify a grid to rasterize on based on bounds, resolution and CRS.
        For a point mask, one can specify the points coordinates and CRS.

        :param ref: Reference raster or pointcloud to use during masking.
        :param res: (Only for raster masking) Spatial resolution of mask. Required if no reference is passed.
        :param points: (Only for point cloud masking) Point X/Y coordinates of mask. Required if no reference is passed.
        :param bounds: (Only for raster masking) Bounds of mask (left, bottom, right, top). Optional, defaults to this
            vector's bounds. Only used if no reference is passed.
        :param crs: Coordinate reference system for output mask. Optional, defaults to this vector's crs. Only used if
            no reference is passed.
        :param as_array: Whether to return mask as a boolean array.

        :returns: A raster or point cloud mask.
        """

        # Create mask
        mask, transform, crs, pts = _create_mask(gdf=self.ds, ref=ref, crs=crs, res=res, points=points, bounds=bounds)

        # Return output as mask or as array
        if as_array:
            return mask.squeeze()
        else:
            # If pts is None, the output is a point cloud mask
            if pts is not None:
                return gu.PointCloud.from_xyz(x=pts.x.values, y=pts.y.values, z=mask, crs=crs)
            # Otherwise, the transform is not None
            else:
                assert transform is not None  # For mypy
                return gu.Raster.from_array(data=mask, transform=transform, crs=crs, nodata=None)

    @profiler.profile("geoutils.vector.vector.rasterize", memprof=True)  # type: ignore
    def rasterize(
        self,
        raster: gu.Raster | None = None,
        crs: CRS | int | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        in_value: int | float | abc.Iterable[int | float] | None = None,
        out_value: int | float = 0,
    ) -> gu.Raster:
        """
        Rasterize vector to a raster or mask, with input geometries burned in.

        **Match-reference:** a raster can be passed to match its resolution, bounds and CRS when rasterizing the vector.

        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Burn value is set by user and can be either a single number, or an iterable of same length as self.ds.
        Default is an index from 1 to len(self.ds).

        :param raster: Reference raster to match during rasterization.
        :param crs: Coordinate reference system as string or EPSG code
            (Default to raster.crs if not None then self.crs).
        :param xres: Output raster spatial resolution in x. Only if raster is None.
            Must be in units of crs, if set.
        :param yres: Output raster spatial resolution in y. Only if raster is None.
            Must be in units of crs, if set. (Default to xres).
        :param bounds: Output raster bounds (left, bottom, right, top). Only if raster is None.
            Must be in same system as crs, if set. (Default to self bounds).
        :param in_value: Value(s) to be burned inside the polygons (Default is self.ds.index + 1).
        :param out_value: Value to be burned outside the polygons (Default is 0).

        :returns: Raster or mask containing the burned geometries.
        """

        return _rasterize(
            gdf=self.ds,
            raster=raster,
            crs=crs,
            xres=xres,
            yres=yres,
            bounds=bounds,
            in_value=in_value,
            out_value=out_value,
        )

    @classmethod
    def from_bounds_projected(
        cls, raster_or_vector: gu.Raster | VectorType, out_crs: CRS | None = None, densify_points: int = 5000
    ) -> VectorType:
        """Create a vector polygon from projected bounds of a raster or vector.

        :param raster_or_vector: A raster or vector
        :param out_crs: In which CRS to compute the bounds
        :param densify_points: Maximum points to be added between image corners to account for nonlinear edges.
            Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        if out_crs is None:
            out_crs = raster_or_vector.crs

        df = _get_footprint_projected(
            raster_or_vector.bounds, in_crs=raster_or_vector.crs, out_crs=out_crs, densify_points=densify_points
        )

        return cls(df)  # type: ignore

    def query(self: Vector, expression: str, inplace: bool = False) -> Vector | None:
        """
        Query the vector with a valid Pandas expression.

        :param expression: A python-like expression to evaluate. Example: "col1 > col2".
        :param inplace: Whether the query should modify the data in place or return a modified copy.

        :returns: Vector resulting from the provided query expression or itself if inplace=True.
        """
        # Modify inplace if wanted and return the self instance.
        if inplace:
            self._ds = self.ds.query(expression, inplace=True)
            return None

        # Otherwise, create a new Vector from the queried dataset.
        new_vector = Vector(self.ds.query(expression))
        return new_vector

    def proximity(
        self,
        raster: gu.Raster | None = None,
        size: tuple[int, int] = (1000, 1000),
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> gu.Raster:
        """
        Compute proximity distances to this vector's geometry.

        **Match-reference**: a raster can be passed to match its resolution, bounds and CRS for computing
        proximity distances.

        Alternatively, a grid size can be passed to create a georeferenced grid with the bounds and CRS of this vector.

        By default, the boundary of the Vector's geometry will be used. The full geometry can be used by
        passing "geometry",
        or any lower dimensional geometry attribute such as "centroid", "envelope" or "convex_hull".
        See all geometry attributes in the Shapely documentation at https://shapely.readthedocs.io/.

        :param raster: Raster to burn the proximity grid on.
        :param size: If no Raster is provided, grid size to use with this Vector's extent and CRS
            (defaults to 1000 x 1000).
        :param geometry_type: Type of geometry to use for the proximity, defaults to 'boundary'.
        :param in_or_out: Compute proximity only 'in' or 'out'-side the polygon, or 'both'.
        :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.

        :return: Proximity raster.
        """

        # 0/ If no Raster is passed, create one on the Vector bounds of size 1000 x 1000
        if raster is None:
            # TODO: this bit of code is common in several vector functions (rasterize, etc): move out as common code?
            # By default, use self's bounds
            if self.bounds is None:
                raise ValueError("To automatically rasterize on the vector, bounds need to be defined.")

            # Calculate raster shape
            left, bottom, right, top = self.bounds

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, size[0], size[1])

            raster = gu.Raster.from_array(data=np.zeros((1000, 1000)), transform=transform, crs=self.crs)

        proximity = _proximity_from_vector_or_raster(
            raster=raster, vector=self, geometry_type=geometry_type, in_or_out=in_or_out, distance_unit=distance_unit
        )

        out_nodata = gu.raster.raster._default_nodata(proximity.dtype)
        return gu.Raster.from_array(
            data=proximity,
            transform=raster.transform,
            crs=raster.crs,
            nodata=out_nodata,
            area_or_point=raster.area_or_point,
            tags=raster.tags,
        )

    def buffer_metric(self, buffer_size: float) -> Vector:
        """
        Buffer the vector features in a local metric system (UTM or UPS).

        The outlines are projected to the local UTM or UPS, then reverted to the original projection after buffering.

        :param buffer_size: Buffering distance in meters.

        :return: Buffered shapefile.
        """

        return _buffer_metric(gdf=self.ds, buffer_size=buffer_size)

    def get_bounds_projected(self, out_crs: CRS, densify_points: int = 5000) -> rio.coords.BoundingBox:
        """
        Get vector bounds projected in a specified CRS.

        :param out_crs: Output CRS.
        :param densify_points: Maximum points to be added between image corners to account for nonlinear edges.
            Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points)

        return new_bounds

    def get_footprint_projected(self, out_crs: CRS, densify_points: int = 5000) -> Vector:
        """
        Get vector footprint projected in a specified CRS.

        The polygon points of the vector are densified during reprojection to warp
        the rectangular square footprint of the original projection into the new one.

        :param out_crs: Output CRS.
        :param densify_points: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        return Vector(
            _get_footprint_projected(
                bounds=self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points
            )
        )

    def get_metric_crs(
        self,
        local_crs_type: Literal["universal"] | Literal["custom"] = "universal",
        method: Literal["centroid"] | Literal["geopandas"] = "centroid",
    ) -> CRS:
        """
        Get local metric coordinate reference system for the vector (UTM, UPS, or custom Mercator or Polar).

        :param local_crs_type: Whether to get a "universal" local CRS (UTM or UPS) or a "custom" local CRS
            (Mercator or Polar centered on centroid).
        :param method: Method to choose the zone of the CRS, either based on the centroid of the footprint
            or the extent as implemented in :func:`geopandas.GeoDataFrame.estimate_utm_crs`.
            Forced to centroid if `local_crs="custom"`.
        """

        # For universal CRS (UTM or UPS)
        if local_crs_type == "universal":
            return _get_utm_ups_crs(self.ds, method=method)
        # For a custom CRS
        else:
            raise NotImplementedError("This is not implemented yet.")

    def buffer_without_overlap(self, buffer_size: int | float, metric: bool = True, plot: bool = False) -> Vector:
        """
        Buffer the vector geometries without overlapping each other.

        The algorithm is based upon this tutorial: https://statnmap.com/2020-07-31-buffer-area-for-nearest-neighbour/.
        The buffered polygons are created using Voronoi polygons in order to delineate the "area of influence"
        of each geometry.
        The buffer is slightly inaccurate where two geometries touch, due to the nature of the Voronoi polygons,
        hence one geometry "steps" slightly on the neighbor buffer in some cases.
        The algorithm may also yield unexpected results on very simple geometries.

        Note: A similar functionality is provided by momepy (http://docs.momepy.org) and is probably more robust.
        It could be implemented in GeoPandas in the future: https://github.com/geopandas/geopandas/issues/2015.

        :param buffer_size: Buffer size in self's coordinate system units.
        :param metric: Whether to perform the buffering in a local metric system (defaults to ``True``).
        :param plot: Whether to show intermediate plots.

        :returns: A Vector containing the buffered geometries.

        :examples: On glacier outlines.
            >>> outlines = Vector(gu.examples.get_path('everest_rgi_outlines'))
            >>> outlines = Vector(outlines.ds.to_crs('EPSG:32645'))
            >>> buffer = outlines.buffer_without_overlap(500)
            >>> ax = buffer.ds.plot()  # doctest: +SKIP
            >>> outlines.ds.plot(ax=ax, ec='k', fc='none')  # doctest: +SKIP
            >>> plt.plot()  # doctest: +SKIP
        """

        return _buffer_without_overlap(self.ds, buffer_size=buffer_size, metric=metric, plot=plot)
