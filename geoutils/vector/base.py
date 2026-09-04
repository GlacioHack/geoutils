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

"""Base class for vector object and the ``vct`` Pandas accessor."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from packaging.version import Version
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils import profiler
from geoutils._dispatch import (
    get_geo_attr,
    has_geo_attr,
    is_dask_dataframe,
)
from geoutils._misc import deprecate, import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface.distance import _proximity_from_vector_or_raster
from geoutils.interface.rasterization import _create_mask, _rasterize
from geoutils.multiproc import MultiprocConfig
from geoutils.projtools import (
    _get_bounds_projected,
    _get_footprint_projected,
    _get_utm_ups_crs,
)
from geoutils.vector.geometric import _buffer_metric, _buffer_without_overlap
from geoutils.vector.testing import _vector_allclose, _vector_equal
from geoutils.vector.transformation import _crop, _reproject

if TYPE_CHECKING:
    import matplotlib

    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterLike, RasterType


VectorBaseType = TypeVar("VectorBaseType", bound="VectorBase")
VectorBaseLike = Union["VectorBase", gpd.GeoDataFrame]


def _as_geodataframe(obj: Any) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from a Vector-like object."""

    ds = obj if isinstance(obj, gpd.GeoDataFrame) else get_geo_attr(obj, "ds")
    if is_dask_dataframe(ds):
        ds = ds.compute()
    if not isinstance(ds, gpd.GeoDataFrame):
        raise TypeError(f"Expected a Vector or GeoDataFrame, received {type(obj).__name__}.")
    return ds


def _as_vector(obj: Any) -> Any | None:
    """Return a GeoUtils vector interface when the input is vector like."""

    # Wrap GeoDataFrames so spatial operations follow the GeoUtils Vector API
    if isinstance(obj, gpd.GeoDataFrame):
        from geoutils.vector.vector import Vector

        return Vector(obj)

    # Reuse Vector objects directly and normalize dataframe accessors otherwise
    if hasattr(obj, "create_mask") and not has_geo_attr(obj, "shape"):
        return obj
    return getattr(obj, "vct", None)


class VectorBase(ABC):
    """
    Shared implementation for :class:`~geoutils.Vector` and the ``vct`` Pandas accessor.

    GeoPandas API wrappers stay implemented on ``Vector`` itself. This base class only contains GeoUtils-specific
    behavior that can be expressed through a ``GeoDataFrame`` backend.
    """

    _ACCESSOR_OUTPUT = False

    def __init__(self) -> None:
        """Initialize shared accessor state without assigning a concrete dataframe."""

        self._obj: gpd.GeoDataFrame | None = None
        self._name: str | None = None

    @property
    def _is_pd(self) -> bool:
        """Whether the object is backed by a Pandas/GeoPandas accessor."""

        return getattr(self, "_obj", None) is not None

    def _cast_raster_output(self, raster: Any) -> Any:
        """Return an accessor-backed raster when this vector is accessor-backed."""

        if not self._is_pd:
            return raster
        if hasattr(raster, "rst"):
            return raster

        from geoutils.raster.xr_accessor import RasterAccessor, open_raster

        if raster.name is not None and not raster.is_loaded:
            return open_raster(raster.name, is_mask=raster.is_mask)
        return RasterAccessor.from_array(
            data=raster.data,
            transform=raster.transform,
            crs=raster.crs,
            nodata=raster.nodata,
            area_or_point=raster.area_or_point,
            tags=raster.tags,
        )

    def _cast_pointcloud_output(self, pointcloud: Any) -> Any:
        """Return an accessor-backed point cloud when this vector is accessor-backed."""

        if is_dask_dataframe(pointcloud):
            return pointcloud

        if self._is_pd:
            ds = pointcloud.ds
            ds.attrs["data_column"] = pointcloud.data_column
            return ds
        return pointcloud

    @property
    @abstractmethod
    def ds(self) -> gpd.GeoDataFrame:
        """GeoDataFrame of the vector."""
        ...

    @ds.setter
    @abstractmethod
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new GeoDataFrame."""
        ...

    @abstractmethod
    def copy(self: VectorBaseType) -> VectorBaseType | gpd.GeoDataFrame:
        """Return a copy of the vector-like object."""
        ...

    def _override_gdf_output(self, other: gpd.GeoDataFrame | gpd.GeoSeries | pd.Series | Any) -> Any:
        """Cast a GeoPandas output to the correct public type."""

        if is_dask_dataframe(other):
            return other
        if not isinstance(other, (gpd.GeoDataFrame, gpd.GeoSeries, pd.Series, BaseGeometry)):
            raise ValueError("Not implemented. This error should only be raised in tests.")

        if isinstance(other, gpd.GeoSeries):
            other = gpd.GeoDataFrame(geometry=other)
        elif isinstance(other, BaseGeometry):
            other = gpd.GeoDataFrame({"geometry": [other]}, crs=self.crs)

        if isinstance(other, gpd.GeoDataFrame) and not self._ACCESSOR_OUTPUT:
            from geoutils.vector.vector import Vector

            return Vector(other)
        return other

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""

        return self.ds.crs

    @property
    def name(self) -> str | None:
        """Name on disk, if it exists."""

        return self._name

    @property
    def is_loaded(self) -> bool:
        """Whether the vector data are loaded in memory."""

        return not is_dask_dataframe(self.ds)

    @property
    def geometry(self) -> gpd.GeoSeries:
        """Active geometry column of the vector."""

        return self.ds.geometry

    @property
    def columns(self) -> pd.Index:
        """Column names available on the vector dataframe."""

        return self.ds.columns

    @property
    def index(self) -> pd.Index:
        """Row index of the vector dataframe."""

        return self.ds.index

    def vector_equal(self, other: Any, **kwargs: Any) -> bool:
        """
        Check if two vectors are equal.

        :param other: Vector, vector accessor or GeoDataFrame to compare.
        :param kwargs: Keyword arguments passed to :func:`geopandas.testing.assert_geodataframe_equal`.
        :returns: True if geometry, data and metadata are equal.
        """

        return _vector_equal(self, other, **kwargs)

    def vector_allclose(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, **kwargs: Any) -> bool:
        """
        Check that two vectors have equal metadata and numerically close coordinates and columns.

        :param other: Vector, vector accessor or GeoDataFrame to compare.
        :param rtol: Relative tolerance for geometry coordinates and numeric columns.
        :param atol: Absolute tolerance for geometry coordinates and numeric columns.
        :param kwargs: Additional comparison options. ``check_dtype=False`` allows numeric dtypes to differ.
        :returns: True if metadata are equal and numeric values are within tolerance.
        """

        return _vector_allclose(self, other, rtol=rtol, atol=atol, **kwargs)

    def __repr__(self) -> str:
        """Convert vector to string representation."""

        str_ds = "\n       ".join(self.__str__().split("\n"))

        return str(
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

    def _repr_html_(self) -> str:
        """Convert vector to HTML string representation for documentation."""

        str_ds = "\n       ".join(self.ds.__str__().split("\n"))

        return str(
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

        :param verbose: If True, print to screen and return None.
        """

        as_str = [
            f"Filename:           {self.name} \n",
            f"Coordinate system:  EPSG:{self.ds.crs.to_epsg()}\n",
            f"Extent:             {self.ds.total_bounds.tolist()} \n",
            f"Number of features: {len(self.ds)} \n",
            f"Attributes:         {self.ds.columns.tolist()}",
        ]

        if verbose:
            print("".join(as_str))
            return None
        return "".join(as_str)

    def plot(
        self,
        ref_crs: RasterLike | VectorBaseLike | CRS | int | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        return_axes: bool = False,
        savefig_fname: str | None = None,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        r"""
        Plot the vector.

        This method is a wrapper to geopandas.GeoDataFrame.plot. Any \*\*kwargs are passed to it.
        """

        matplotlib = import_optional("matplotlib")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if has_geo_attr(ref_crs, "crs"):
            crs = get_geo_attr(ref_crs, "crs")
            vect_reproj = self.reproject(crs=crs)
        elif isinstance(ref_crs, (CRS, int)):
            vect_reproj = self.reproject(crs=ref_crs)
        else:
            vect_reproj = self

        if ax is None:
            ax0 = plt.gca()
        elif isinstance(ax, str) and ax.lower() == "new":
            _, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")

        if "column" in kwargs.keys() and add_cbar:
            add_cbar = True
        else:
            add_cbar = False

        legend = bool(add_cbar)
        if "legend" in list(kwargs.keys()):
            legend = kwargs.pop("legend")

        if "legend_kwds" in list(kwargs.keys()) and legend:
            legend_kwds = kwargs.pop("legend_kwds")
            if cbar_title is not None:
                legend_kwds.update({"label": cbar_title})
        elif cbar_title is not None:
            legend_kwds = {"label": cbar_title}
        else:
            legend_kwds = None

        if add_cbar or cbar_title:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad="2%")
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            cbar.solids.set_alpha(alpha)
        else:
            cax = None

        plot_ds = _as_geodataframe(vect_reproj)
        plot_ds.plot(
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

        if savefig_fname:
            plt.savefig(savefig_fname)

        if return_axes:
            return ax0, cax
        return None

    @property
    def total_bounds(self) -> rio.coords.BoundingBox:
        """Total bounds of the vector."""

        return self.ds.total_bounds

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Total bounding box of the vector."""

        return rio.coords.BoundingBox(*self.ds.total_bounds)

    @property
    def footprint(self) -> Any:
        """Footprint of the vector."""

        return self.get_footprint_projected(self.crs)

    @property
    def active_geometry_name(self) -> str:
        """Name of the active geometry column."""

        return self.ds.active_geometry_name

    @overload
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorBaseLike | tuple[float, float, float, float],
        clip: bool,
        *,
        inplace: Literal[False] = False,
        crop_geom: Any = None,
    ) -> VectorBaseType | gpd.GeoDataFrame: ...

    @overload
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorBaseLike | tuple[float, float, float, float],
        clip: bool,
        *,
        inplace: Literal[True],
        crop_geom: Any = None,
    ) -> None: ...

    @overload
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorBaseLike | tuple[float, float, float, float],
        clip: bool,
        *,
        inplace: bool = False,
        crop_geom: Any = None,
    ) -> VectorBaseType | gpd.GeoDataFrame | None: ...

    @profiler.profile("geoutils.vector.base.crop", memprof=True)
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorBaseLike | tuple[float, float, float, float] = None,
        clip: bool = False,
        *,
        inplace: bool = False,
        crop_geom: Any = None,
    ) -> VectorBaseType | gpd.GeoDataFrame | None:
        """
        Crop the vector to given extent.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.
        """

        if crop_geom is not None:
            warnings.warn(DeprecationWarning("Argument 'crop_geom' is deprecated, use 'bbox' instead."))
            bbox = crop_geom
        if bbox is None:
            raise ValueError("Argument 'bbox' must be passed.")
        if inplace and is_dask_dataframe(self.ds):
            raise ValueError("Dask-backed vectors cannot be modified in place; use the returned dataframe instead.")

        new_ds = _crop(self, bbox=bbox, clip=clip)

        if inplace:
            self.ds = new_ds
            return None
        return self._override_gdf_output(new_ds)

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorBaseLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[False] = False,
    ) -> VectorBaseType | gpd.GeoDataFrame: ...

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorBaseLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorBaseLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: bool = False,
    ) -> VectorBaseType | gpd.GeoDataFrame | None: ...

    @profiler.profile("geoutils.vector.base.reproject", memprof=True)
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorBaseLike | None = None,
        crs: CRS | str | int | None = None,
        inplace: bool = False,
    ) -> VectorBaseType | gpd.GeoDataFrame | None:
        """Reproject vector to a specified coordinate reference system."""

        if inplace and is_dask_dataframe(self.ds):
            raise ValueError("Dask-backed vectors cannot be modified in place; use the returned dataframe instead.")

        new_ds = _reproject(self, ref=ref, crs=crs)

        if inplace:
            self.ds = new_ds
            return None
        return self._override_gdf_output(new_ds)

    @overload
    def translate(
        self: VectorBaseType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: Literal[False] = False,
    ) -> VectorBaseType | gpd.GeoDataFrame: ...

    @overload
    def translate(
        self: VectorBaseType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def translate(
        self: VectorBaseType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        *,
        inplace: bool = False,
    ) -> VectorBaseType | gpd.GeoDataFrame | None: ...

    def translate(
        self: VectorBaseType,
        xoff: float = 0.0,
        yoff: float = 0.0,
        zoff: float = 0.0,
        inplace: bool = False,
    ) -> VectorBaseType | gpd.GeoDataFrame | None:
        """Shift a vector by a coordinate offset."""

        if inplace and is_dask_dataframe(self.ds):
            raise ValueError("Dask-backed vectors cannot be modified in place; use the returned dataframe instead.")

        new_ds = self.ds.copy()
        new_ds.geometry = self.geometry.translate(xoff=xoff, yoff=yoff, zoff=zoff)

        if inplace:
            self.ds = new_ds
            return None
        return self._override_gdf_output(new_ds)

    @overload
    def create_mask(
        self,
        ref: RasterLike | PointCloudLike | None = None,
        all_touched: bool = False,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        shape: tuple[int, int] | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] | None = None,
        *,
        as_array: Literal[False] = False,
        chunksizes: tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        dask: bool = False,
    ) -> RasterType | PointCloudLike: ...

    @overload
    def create_mask(
        self,
        ref: RasterLike | PointCloudLike | None = None,
        all_touched: bool = False,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        shape: tuple[int, int] | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] | None = None,
        *,
        as_array: Literal[True],
        chunksizes: tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        dask: bool = False,
    ) -> NDArrayBool: ...

    def create_mask(
        self,
        ref: RasterLike | PointCloudLike | None = None,
        all_touched: bool = False,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        shape: tuple[int, int] | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        points: tuple[NDArrayNum, NDArrayNum] | None = None,
        *,
        as_array: bool = False,
        chunksizes: tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        dask: bool = False,
    ) -> RasterType | PointCloudLike | NDArrayBool:
        """Create a raster or point cloud mask from the vector features."""

        # Functional interfaces operate on Vector while outputs follow the caller type
        source_vector = self.to_geoutils() if self._is_pd else self
        output = _create_mask(
            source_vector=source_vector,
            ref=ref,
            all_touched=all_touched,
            crs=crs,
            res=res,
            shape=shape,
            grid_coords=grid_coords,
            points=points,
            bounds=bounds,
            as_array=as_array,
            chunksizes=chunksizes,
            mp_config=mp_config,
            dask=dask,
        )
        # Preserve plain arrays and cast geospatial results to their matching accessor
        if as_array:
            return output
        if has_geo_attr(output, "data_column"):
            return self._cast_pointcloud_output(output)
        if has_geo_attr(output, "transform") and has_geo_attr(output, "shape"):
            return self._cast_raster_output(output)
        return output

    @profiler.profile("geoutils.vector.base.rasterize", memprof=True)
    def rasterize(
        self,
        ref: RasterType | None = None,
        in_value: int | float | list[int | float] | tuple[int | float, ...] | None = None,
        out_value: int | float = 0,
        all_touched: bool = False,
        out_dtype: DTypeLike | None = None,
        res: tuple[Number, Number] | Number | None = None,
        shape: tuple[int, int] | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        crs: CRS | int | None = None,
        *,
        chunksizes: tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        dask: bool = False,
        **kwargs: Any,
    ) -> RasterType:
        """Rasterize vector to a raster or mask, with input geometries burned in."""

        if "xres" in kwargs.keys() or "yres" in kwargs.keys():
            warnings.warn(
                message="Argument 'xres' and 'yres' are deprecrated in favour of 'res'.",
                category=DeprecationWarning,
            )
        xres = kwargs.get("xres", None)
        yres = kwargs.get("yres", None)
        if xres is not None:
            if yres is not None:
                res = (xres, yres)
            else:
                res = xres
        if "raster" in kwargs.keys():
            warnings.warn(message="Argument 'raster' is deprecrated in favour of 'ref'.", category=DeprecationWarning)
            ref = kwargs.get("raster", None)

        # Run the common implementation and cast its Raster output for accessors
        source_vector = self.to_geoutils() if self._is_pd else self
        raster = _rasterize(
            source_vector=source_vector,
            ref=ref,
            in_value=in_value,
            out_value=out_value,
            all_touched=all_touched,
            out_dtype=out_dtype,
            res=res,
            shape=shape,
            grid_coords=grid_coords,
            bounds=bounds,
            crs=crs,
            chunksizes=chunksizes,
            mp_config=mp_config,
            dask=dask,
        )
        return self._cast_raster_output(raster)

    @classmethod
    def from_bounds_projected(
        cls, raster_or_vector: RasterType | VectorBaseLike, out_crs: CRS | None = None, densify_points: int = 5000
    ) -> VectorBaseType | gpd.GeoDataFrame:
        """Create a vector polygon from projected bounds of a raster or vector.

        :param raster_or_vector: A raster or vector
        :param out_crs: In which CRS to compute the bounds
        :param densify_points: Maximum points to be added between image corners to account for nonlinear edges.
            Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        if out_crs is None:
            out_crs = get_geo_attr(raster_or_vector, "crs")

        df = _get_footprint_projected(
            get_geo_attr(raster_or_vector, "bounds"),
            in_crs=get_geo_attr(raster_or_vector, "crs"),
            out_crs=out_crs,
            densify_points=densify_points,
        )

        if cls._ACCESSOR_OUTPUT:
            return df
        return cls(df)  # type: ignore

    def query(self: VectorBaseType, expression: str, inplace: bool = False) -> VectorBaseType | gpd.GeoDataFrame | None:
        """Query the vector with a valid Pandas expression."""

        if inplace and is_dask_dataframe(self.ds):
            raise ValueError("Dask-backed vectors cannot be modified in place; use the returned dataframe instead.")
        new_ds = self.ds.query(expression)
        if inplace:
            self.ds = new_ds
            return None
        return self._override_gdf_output(new_ds)

    def proximity(
        self,
        raster: RasterType | None = None,
        size: tuple[int, int] = (1000, 1000),
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> RasterType:
        """Compute proximity distances to this vector's geometry."""

        from geoutils.raster.raster import Raster, _default_nodata

        if raster is None:
            if self.bounds is None:
                raise ValueError("To automatically rasterize on the vector, bounds need to be defined.")

            left, bottom, right, top = self.bounds
            transform = rio.transform.from_bounds(left, bottom, right, top, size[0], size[1])
            raster = Raster.from_array(data=np.zeros((1000, 1000)), transform=transform, crs=self.crs)

        source_vector = self.to_geoutils() if self._is_pd else self
        proximity = _proximity_from_vector_or_raster(
            raster=raster,
            vector=source_vector,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

        out_nodata = _default_nodata(proximity.dtype)
        raster_out = Raster.from_array(
            data=proximity,
            transform=raster.transform,
            crs=raster.crs,
            nodata=out_nodata,
            area_or_point=raster.area_or_point,
            tags=raster.tags,
        )
        return self._cast_raster_output(raster_out)

    def buffer_metric(self: VectorBaseType, buffer_size: float) -> VectorBaseType | gpd.GeoDataFrame:
        """Buffer the vector features in a local metric system."""

        new_ds = _buffer_metric(gdf=self.ds, buffer_size=buffer_size)
        return self._override_gdf_output(new_ds)

    def get_bounds_projected(self, out_crs: CRS, densify_points: int = 5000) -> rio.coords.BoundingBox:
        """Get vector bounds projected in a specified CRS."""

        return _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points)

    def get_footprint_projected(
        self: VectorBaseType, out_crs: CRS, densify_points: int = 5000
    ) -> VectorBaseType | gpd.GeoDataFrame:
        """Get vector footprint projected in a specified CRS."""

        new_ds = _get_footprint_projected(
            bounds=self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points
        )
        return self._override_gdf_output(new_ds)

    def get_metric_crs(
        self,
        local_crs_type: Literal["universal"] | Literal["custom"] = "universal",
        method: Literal["centroid"] | Literal["geopandas"] = "centroid",
    ) -> CRS:
        """Get local metric coordinate reference system for the vector."""

        if local_crs_type == "universal":
            return _get_utm_ups_crs(self.ds, method=method)
        raise NotImplementedError("This is not implemented yet.")

    def buffer_without_overlap(
        self: VectorBaseType, buffer_size: int | float, metric: bool = True, plot: bool = False
    ) -> VectorBaseType | gpd.GeoDataFrame:
        """Buffer the vector geometries without overlapping each other."""

        new_ds = _buffer_without_overlap(self.ds, buffer_size=buffer_size, metric=metric, plot=plot)
        return self._override_gdf_output(new_ds)

    def to_geoutils(self) -> Any:
        """Convert to a GeoUtils Vector object."""

        from geoutils.vector.vector import Vector

        return Vector(self.ds)

    @deprecate(
        removal_version=Version("0.3.0"),
        details="The function .save() will be soon deprecated, use .to_file() instead.",
    )
    def save(self, *args: Any, **kwargs: Any) -> None:
        """Write the vector to file."""

        return self.to_file(*args, **kwargs)
