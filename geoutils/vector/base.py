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

import copy
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from packaging.version import Version
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from geoutils import profiler
from geoutils._dispatch import (
    _check_match_bbox,
    get_geo_attr,
    has_geo_attr,
    is_dask_dataframe,
)
from geoutils._misc import deprecate
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
from geoutils.vector.transformation import _clip, _crop, _reproject

if TYPE_CHECKING:
    import matplotlib

    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterLike, RasterType


VectorBaseType = TypeVar("VectorBaseType", bound="VectorBase")
_UNSET = object()
# Accept Vector subclasses and accessors, as well as GeoDataFrames
VectorLike = Union["VectorBase", gpd.GeoDataFrame]


def _as_geodataframe(obj: Any) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame from a Vector-like object."""

    ds = obj if isinstance(obj, gpd.GeoDataFrame) else get_geo_attr(obj, "ds")
    if is_dask_dataframe(ds):
        ds = ds.compute()
    if not isinstance(ds, gpd.GeoDataFrame):
        raise TypeError(f"Expected a Vector or GeoDataFrame, received {type(obj).__name__}.")
    return ds


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
            + "\n  bbox="
            + self.bbox.__str__()
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
            + "\n  <b>bbox=</b>"
            + self.bbox.__repr__()
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
            f"Coordinate system:  {[CRS(self.crs).name if self.crs is not None else None]}\n",
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
        ref: RasterLike | VectorLike | CRS | str | int | None = None,
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
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
        r"""
        Plot the vector.

        This method is a wrapper to geopandas.GeoDataFrame.plot. Any \*\*kwargs are passed to it.
        """

        import matplotlib.pyplot as plt

        from geoutils.vector.plotting import (
            _create_axes,
            _get_reference_bbox,
            _plot_geodataframe,
        )

        # REMOVE AFTER DEPRECATION: Delete this block when ref_crs compatibility is removed
        if "ref_crs" in kwargs:
            if ref is not None:
                raise TypeError("plot() received both 'ref' and deprecated 'ref_crs'; use only 'ref'.")
            deprecated_ref = kwargs.pop("ref_crs")
            warnings.warn(
                "Argument 'ref_crs' is deprecated; use 'ref' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Preserve the old behavior, which matched only the CRS and did not use reference bounds
            if deprecated_ref is not None:
                if has_geo_attr(deprecated_ref, "crs"):
                    deprecated_ref = get_geo_attr(deprecated_ref, "crs")
                ref = CRS.from_user_input(deprecated_ref)

        reference_bbox = None
        if has_geo_attr(ref, "crs"):
            crs = get_geo_attr(ref, "crs")
            vect_reproj = self.reproject(crs=crs)
            reference_bbox = _get_reference_bbox(ref)
        elif isinstance(ref, (CRS, str, int)):
            vect_reproj = self.reproject(crs=ref)
        else:
            vect_reproj = self

        ax0 = _create_axes(ax)
        column = kwargs.pop("column", None)
        plot_ds = _as_geodataframe(vect_reproj)
        cax = _plot_geodataframe(
            dataframe=plot_ds,
            ax=ax0,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            cbar_title=cbar_title,
            add_cbar=add_cbar,
            **kwargs,
        )
        if reference_bbox is not None:
            ax0.set_xlim(reference_bbox.left, reference_bbox.right)
            ax0.set_ylim(reference_bbox.bottom, reference_bbox.top)
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
    def bbox(self) -> rio.coords.BoundingBox:
        """Total bounding box of the vector."""

        # Reduce lazy partitions to four coordinates without replacing the Dask collection
        dataframe = self.ds
        total_bounds = dataframe.total_bounds
        if is_dask_dataframe(dataframe):
            total_bounds = total_bounds.compute()
        return rio.coords.BoundingBox(*total_bounds)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Total bounding box of the vector, provided as an alias of bbox."""

        return self.bbox

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
        bbox: RasterLike | VectorLike | tuple[float, float, float, float],
        mode: Literal["intersects", "within"] = "intersects",
        *,
        inplace: Literal[False] = False,
        crop_geom: Any = None,
        **kwargs: Any,
    ) -> VectorBaseType | gpd.GeoDataFrame: ...

    @overload
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorLike | tuple[float, float, float, float],
        mode: Literal["intersects", "within"] = "intersects",
        *,
        inplace: Literal[True],
        crop_geom: Any = None,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorLike | tuple[float, float, float, float],
        mode: Literal["intersects", "within"] = "intersects",
        *,
        inplace: bool = False,
        crop_geom: Any = None,
        **kwargs: Any,
    ) -> VectorBaseType | gpd.GeoDataFrame | None: ...

    @profiler.profile("geoutils.vector.base.crop", memprof=True)
    def crop(
        self: VectorBaseType,
        bbox: RasterLike | VectorLike | tuple[float, float, float, float] = None,
        mode: Literal["intersects", "within"] = "intersects",
        *,
        inplace: bool = False,
        crop_geom: Any = None,
        **kwargs: Any,
    ) -> VectorBaseType | gpd.GeoDataFrame | None:
        """
        Crop vector to a bounding box, without modifying geometric features.

        Equivalent to selecting geometries intersecting or contained within a bounding box.
        To cut each feature with another geometry or bounding box, use ``clip()``.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.

        :param bbox: Bounding box or georeferenced reference object defining the selected extent.
        :param mode: ``"intersects"`` keeps every geometry touching the extent. ``"within"`` keeps only geometries
            entirely inside it.
        :param inplace: Whether to replace this vector's selected rows. Dask-backed vectors cannot be changed in place.
        :param crop_geom: Deprecated alias of ``bbox``.
        :param kwargs: Deprecated ``clip`` argument. Use clip() for exact clipping.
        :returns: A vector or dataframe containing unchanged selected geometries, or None when ``inplace=True``.
        """

        deprecated_clip = kwargs.pop("clip", _UNSET)
        if len(kwargs) > 0:
            unexpected = next(iter(kwargs))
            raise TypeError(f"crop() got an unexpected keyword argument {unexpected!r}")
        if deprecated_clip is not _UNSET:
            warnings.warn(
                "Argument 'clip' is deprecated; use crop() without it or call clip() separately.",
                DeprecationWarning,
                stacklevel=2,
            )

        if crop_geom is not None:
            warnings.warn(DeprecationWarning("Argument 'crop_geom' is deprecated, use 'bbox' instead."))
            bbox = crop_geom
        if bbox is None:
            raise ValueError("Argument 'bbox' must be passed.")
        if mode not in ("intersects", "within"):
            raise ValueError("Argument 'mode' must be either 'intersects' or 'within'.")
        if inplace and self._is_pd and is_dask_dataframe(self.ds):
            raise ValueError("Dask-backed vectors cannot be modified in place; use the returned dataframe instead.")

        # Preserve crop(..., clip=True) by clipping to the normalized rectangular crop extent
        if deprecated_clip is not _UNSET and deprecated_clip:
            normalized_bbox = tuple(float(value) for value in _check_match_bbox(self, bbox))
            new_ds = _clip(self, mask=normalized_bbox, keep_geom_type=False, sort=False)
            if inplace:
                self.ds = new_ds
                return None
            return self._override_gdf_output(new_ds)

        # Store file filters without reading any feature geometry or attribute values
        if not self._is_pd and not self.is_loaded and self.name is not None:
            normalized_bbox = tuple(float(value) for value in _check_match_bbox(self, bbox))
            output = copy.copy(self)
            output._crop_filters = list(getattr(self, "_crop_filters", [])) + [(normalized_bbox, mode)]
            output._bounds = None
            output._feature_count = None
            if hasattr(output, "_nb_points"):
                output._nb_points = -1

            if inplace:
                self.__dict__.update(output.__dict__)
                return None
            return output

        new_ds = _crop(self, bbox=bbox, mode=mode)

        if inplace:
            self.ds = new_ds
            return None
        return self._override_gdf_output(new_ds)

    def clip(
        self: VectorBaseType,
        mask: Any,
        keep_geom_type: bool = False,
        sort: bool = False,
        mp_config: MultiprocConfig | None = None,
    ) -> VectorBaseType | gpd.GeoDataFrame:
        """
        Clip geometries exactly to a mask.

        Vector geometries are intersected with the mask. Point cloud rows outside the mask are removed. Unlike
        crop(), this operation can change geometry data and therefore reads an unloaded source.

        :param mask: Clipping geometry, vector, point cloud, raster or bounding box. Georeferenced masks are
            reprojected to this object's CRS.
        :param keep_geom_type: Whether to remove intersections with a different geometry type than the input.
        :param sort: Whether to sort clipped geometries by their original index.
        :param mp_config: Worker configuration with an integer number of features per chunk. Multiprocessing writes
            an unloaded GeoPackage result and cannot be combined with a Dask input.
        :returns: A vector, point cloud or dataframe clipped to the mask.
        """

        if mp_config is not None:
            if self._is_pd and is_dask_dataframe(self.ds):
                raise ValueError("Argument ``mp_config`` cannot be combined with a Dask vector.")

            # Keep file-backed vectors unloaded while workers clip independent source row ranges
            from geoutils.vector.transformation import _clip_vector_multiproc

            clipped_vector = _clip_vector_multiproc(
                self,
                mask=mask,
                keep_geom_type=keep_geom_type,
                sort=sort,
                mp_config=mp_config,
            )
            if self._is_pd:
                clipped_vector.load()
                return clipped_vector.ds
            return cast(VectorBaseType, clipped_vector)

        clipped = _clip(self, mask=mask, keep_geom_type=keep_geom_type, sort=sort)
        return self._override_gdf_output(clipped)

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[False] = False,
    ) -> VectorBaseType | gpd.GeoDataFrame: ...

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: bool = False,
    ) -> VectorBaseType | gpd.GeoDataFrame | None: ...

    @profiler.profile("geoutils.vector.base.reproject", memprof=True)
    def reproject(
        self: VectorBaseType,
        ref: RasterLike | VectorLike | None = None,
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
        """Create a raster or point cloud mask from the vector geometry features."""

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

    # Keep the Rasterio-style name as an equivalent public alias
    geometry_mask = create_mask

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
        cls, raster_or_vector: RasterType | VectorLike, out_crs: CRS | None = None, densify_points: int = 5000
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
            get_geo_attr(raster_or_vector, "bbox"),
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
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
        max_distance: float | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:
        """
        Compute proximity distances to this vector's current geometry.

        Apply a geometry operation before proximity() to use a derived geometry, for example
        ``vector.boundary.proximity(raster)``.

        :param raster: Raster whose grid is used for the proximity output.
        :param size: Output width and height when raster is not provided.
        :param distance_unit: Calculate distance in georeferenced or pixel units.
        :param max_distance: Largest distance to return, with farther cells set to nodata. This value is required for
            Dask and multiprocessing execution because it defines the overlap between chunks.
        :param mp_config: Multiprocessing parameters. Cannot be combined with Dask input.

        :returns: Raster of proximity distances on the selected grid.
        """

        from geoutils.raster.raster import Raster

        if raster is None:
            if self.bbox is None:
                raise ValueError("To automatically rasterize on the vector, bounds need to be defined.")

            left, bottom, right, top = self.bbox
            transform = rio.transform.from_bounds(left, bottom, right, top, size[0], size[1])
            raster = Raster.from_array(data=np.zeros((1000, 1000)), transform=transform, crs=self.crs)

        source_vector = self.to_geoutils() if self._is_pd else self
        output = _proximity_from_vector_or_raster(
            raster=raster,
            vector=source_vector,
            distance_unit=distance_unit,
            max_distance=max_distance,
            mp_config=mp_config,
        )
        return self._cast_raster_output(output)

    def buffer_metric(self: VectorBaseType, buffer_size: float) -> VectorBaseType | gpd.GeoDataFrame:
        """Buffer the vector features in a local metric system."""

        new_ds = _buffer_metric(gdf=self.ds, buffer_size=buffer_size)
        return self._override_gdf_output(new_ds)

    def get_bounds_projected(self, out_crs: CRS, densify_points: int = 5000) -> rio.coords.BoundingBox:
        """Get vector bounds projected in a specified CRS."""

        return _get_bounds_projected(self.bbox, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points)

    def get_footprint_projected(
        self: VectorBaseType, out_crs: CRS, densify_points: int = 5000
    ) -> VectorBaseType | gpd.GeoDataFrame:
        """Get vector footprint projected in a specified CRS."""

        new_ds = _get_footprint_projected(
            bounds=self.bbox, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points
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
        details="Use .to_file() instead.",
    )
    def save(self, *args: Any, **kwargs: Any) -> None:
        """Write the vector to file."""

        return self.to_file(*args, **kwargs)
