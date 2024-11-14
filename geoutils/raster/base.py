"""Module for the RasterBase class, parent of both the Raster class and the 'rst' Xarray accessor."""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Iterable, Literal, TypeVar, overload

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr
from packaging.version import Version
from rasterio.crs import CRS
from rasterio.enums import Resampling

from geoutils import projtools
from geoutils._config import config
from geoutils._typing import ArrayLike, DTypeLike, MArrayNum, NDArrayNum, Number
from geoutils.interface.distance import _proximity_from_vector_or_raster
from geoutils.interface.interpolate import _interp_points
from geoutils.interface.raster_point import (
    _raster_to_pointcloud,
    _regular_pointcloud_to_raster,
)
from geoutils.interface.raster_vector import _polygonize
from geoutils.misc import deprecate
from geoutils.projtools import (
    _get_bounds_projected,
    _get_footprint_projected,
    _get_utm_ups_crs,
    reproject_from_latlon,
)
from geoutils.raster.georeferencing import (
    _bounds,
    _cast_nodata,
    _cast_pixel_interpretation,
    _coords,
    _default_nodata,
    _ij2xy,
    _outside_image,
    _res,
    _xy2ij,
)
from geoutils.raster.geotransformations import _crop, _reproject, _translate
from geoutils.raster.sampling import subsample_array
from geoutils.vector.vector import Vector

RasterType = TypeVar("RasterType", bound="RasterBase")


class RasterBase:
    """
    This class is non-public and made to be subclassed.

    It gathers all the functions shared by the Raster class and the 'rst' Xarray accessor.
    """

    _obj: None | xr.DataArray

    def __init__(self):
        # Main attributes of a raster
        self._data: MArrayNum | xr.DataArray
        self._transform: affine.Affine | None = None
        self._crs: CRS | None = None
        self._nodata: int | float | None
        self._area_or_point: Literal["Area", "Point"] | None = None

        # Other non-derivatives attributes
        self._driver: str | None = None
        self._name: str | None = None
        self.filename: str | None = None
        self.tags: dict[str, Any] = {}

    @property
    def is_xr(self) -> bool:
        return self._obj is not None

    @property
    def data(self) -> MArrayNum | xr.DataArray:
        if self.is_xr:
            return self._obj.rio.data
        else:
            return self._data

    @property
    def transform(self) -> affine.Affine:
        if self.is_xr:
            return self._obj.rio.transform(recalc=True)
        else:
            return self._transform

    @property
    def crs(self) -> CRS:
        if self.is_xr:
            return self._obj.rio.crs
        else:
            return self._crs

    @property
    def nodata(self) -> int | float | None:
        if self.is_xr:
            return self._obj.rio.nodata
        else:
            return self._nodata

    @property
    def area_or_point(self):
        return self._area_or_point

    @property
    def res(self) -> tuple[float | int, float | int]:
        """Resolution (X, Y) of the raster in georeferenced units."""
        return _res(self.transform)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Bounding coordinates of the raster."""
        return _bounds(transform=self.transform, shape=self.shape)

    @property
    def footprint(self) -> Vector:
        """Footprint of the raster."""
        return self.get_footprint_projected(self.crs)

    @property
    def count_on_disk(self) -> None | int:
        """Count of bands on disk if it exists."""
        if self._disk_shape is not None:
            return self._disk_shape[0]
        return None

    @property
    def count(self) -> int:
        """Count of bands loaded in memory if they are, otherwise the one on disk."""
        if self.is_loaded:
            if self.data.ndim == 2:
                return 1
            else:
                return int(self.data.shape[0])
        #  This can only happen if data is not loaded, with a DatasetReader on disk is open, never returns None
        return self.count_on_disk  # type: ignore

    @property
    def height(self) -> int:
        """Height of the raster in pixels."""
        if not self.is_loaded:
            if self._out_shape is not None:
                return self._out_shape[0]
            else:
                return self._disk_shape[1]  # type: ignore
        else:
            # If the raster is single-band
            if self.data.ndim == 2:
                return int(self.data.shape[0])
            # Or multi-band
            else:
                return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Width of the raster in pixels."""
        if not self.is_loaded:
            if self._out_shape is not None:
                return self._out_shape[1]
            else:
                return self._disk_shape[2]  # type: ignore
        else:
            # If the raster is single-band
            if self.data.ndim == 2:
                return int(self.data.shape[1])
            # Or multi-band
            else:
                return int(self.data.shape[2])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (i.e., height, width) of the raster in pixels."""
        # If a downsampling argument was defined but data not loaded yet
        if self._out_shape is not None and not self.is_loaded:
            return self._out_shape
        # If data loaded or not, pass the disk/data shape through height and width
        return self.height, self.width

    @property
    def is_loaded(self) -> bool:
        """Whether the raster array is loaded."""
        return self._data is not None

    @property
    def dtype(self) -> str:
        """Data type of the raster (string representation)."""
        if not self.is_loaded and self._disk_dtype is not None:
            return self._disk_dtype
        return str(self.data.dtype)

    @property
    def bands_on_disk(self) -> None | tuple[int, ...]:
        """Band indexes on disk if a file exists."""
        if self._disk_bands is not None:
            return self._disk_bands
        return None

    @property
    def bands(self) -> tuple[int, ...]:
        """Band indexes loaded in memory if they are, otherwise on disk."""
        if self._bands is not None and not self.is_loaded:
            if isinstance(self._bands, int):
                return (self._bands,)
            return tuple(self._bands)
        # if self._indexes_loaded is not None:
        #     if isinstance(self._indexes_loaded, int):
        #         return (self._indexes_loaded, )
        #     return tuple(self._indexes_loaded)
        if self.is_loaded:
            return tuple(range(1, self.count + 1))
        return self.bands_on_disk  # type: ignore

    @property
    def indexes(self) -> tuple[int, ...]:
        """
        Band indexes (duplicate of .bands attribute, mirroring Rasterio naming "indexes").
        Loaded in memory if they are, otherwise on disk.
        """
        return self.bands

    @property
    def driver(self) -> str | None:
        """Driver used to read a file on disk."""
        return self._driver

    @property
    def name(self) -> str | None:
        """Name of the file on disk, if it exists."""
        return self._name

    def set_area_or_point(
        self, new_area_or_point: Literal["Area", "Point"] | None, shift_area_or_point: bool | None = None
    ) -> None:
        """
        Set new pixel interpretation of the raster.

        Overwrites the `area_or_point` attribute and updates "AREA_OR_POINT" in raster metadata tags.

        Optionally, shifts the raster to correct value coordinates in relation to interpretation:

        - By half a pixel (right and downwards) if old interpretation was "Area" and new is "Point",
        - By half a pixel (left and upwards) if old interpretration was "Point" and new is "Area",
        - No shift for all other cases.

        :param new_area_or_point: New pixel interpretation "Area", "Point" or None.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            indexes if self.area_or_point is "Point" and maintains corner pixel indexes if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].

        :return: None.
        """

        # If undefined, default to the global system config
        if shift_area_or_point is None:
            shift_area_or_point = config["shift_area_or_point"]

        # Check input
        if new_area_or_point is not None and not (
            isinstance(new_area_or_point, str) and new_area_or_point.lower() in ["area", "point"]
        ):
            raise ValueError("New pixel interpretation must be 'Area', 'Point' or None.")

        # Update string input as exactly "Area" or "Point"
        if new_area_or_point is not None:
            if new_area_or_point.lower() == "area":
                new_area_or_point = "Area"
            else:
                new_area_or_point = "Point"

        # Save old area or point
        old_area_or_point = self.area_or_point

        # Set new interpretation
        self._area_or_point = new_area_or_point
        # Update tag only if not None
        if new_area_or_point is not None:
            self.tags.update({"AREA_OR_POINT": new_area_or_point})
        else:
            if "AREA_OR_POINT" in self.tags:
                self.tags.pop("AREA_OR_POINT")

        # If shift is True, and both interpretation were different strings, a change is needed
        if (
            shift_area_or_point
            and isinstance(old_area_or_point, str)
            and isinstance(new_area_or_point, str)
            and old_area_or_point != new_area_or_point
        ):
            # The shift below represents +0.5/+0.5 or opposite in indexes (as done in xy2ij), but because
            # the Y axis is inverted, a minus signs is added to shift the coordinate (even if the unit is in pixel)

            # If the new one is Point, we shift back by half a pixel
            if new_area_or_point == "Point":
                xoff = 0.5
                yoff = -0.5
            # Otherwise we shift forward half a pixel
            else:
                xoff = -0.5
                yoff = 0.5
            # We perform the shift in place
            self.translate(xoff=xoff, yoff=yoff, distance_unit="pixel", inplace=True)

    @property
    def area_or_point(self) -> Literal["Area", "Point"] | None:
        """
        Pixel interpretation of the raster.

        Based on the "AREA_OR_POINT" raster metadata:

        - If pixel interpretation is "Area", the value of the pixel is associated with its upper left corner.
        - If pixel interpretation is "Point", the value of the pixel is associated with its center.

        When setting with self.area_or_point = new_area_or_point, uses the default arguments of
        self.set_area_or_point().
        """
        return self._area_or_point

    @area_or_point.setter
    def area_or_point(self, new_area_or_point: Literal["Area", "Point"] | None) -> None:
        """
        Setter for pixel interpretation.

        Uses default arguments of self.set_area_or_point(): shifts by half a pixel going from "Area" to "Point",
        or the opposite.

        :param new_area_or_point: New pixel interpretation "Area", "Point" or None.

        :return: None.
        """
        self.set_area_or_point(new_area_or_point=new_area_or_point)

    @overload
    def info(self, stats: bool = False, *, verbose: Literal[True] = ...) -> None: ...

    @overload
    def info(self, stats: bool = False, *, verbose: Literal[False]) -> str: ...

    def info(self, stats: bool = False, verbose: bool = True) -> None | str:
        """
        Print summary information about the raster.

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.
        :param verbose: If set to True (default) will directly print to screen and return None

        :returns: Summary string or None.
        """
        as_str = [
            f"Driver:               {self.driver} \n",
            f"Opened from file:     {self.filename} \n",
            f"Filename:             {self.name} \n",
            f"Loaded?               {self.is_loaded} \n",
            f"Modified since load?  {self.is_modified} \n",
            f"Grid size:            {self.width}, {self.height}\n",
            f"Number of bands:      {self.count:d}\n",
            f"Data types:           {self.dtype}\n",
            f"Coordinate system:    {[self.crs.to_string() if self.crs is not None else None]}\n",
            f"Nodata value:         {self.nodata}\n",
            f"Pixel interpretation: {self.area_or_point}\n",
            "Pixel size:           {}, {}\n".format(*self.res),
            "Upper left corner:    {}, {}\n".format(*self.bounds[:2]),
            "Lower right corner:   {}, {}\n".format(*self.bounds[2:]),
        ]

        if stats:
            if not self.is_loaded:
                self.load()

            if self.count == 1:
                as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data):.2f}\n")
                as_str.append(f"[MINIMUM]:          {np.nanmin(self.data):.2f}\n")
                as_str.append(f"[MEDIAN]:           {np.ma.median(self.data):.2f}\n")
                as_str.append(f"[MEAN]:             {np.nanmean(self.data):.2f}\n")
                as_str.append(f"[STD DEV]:          {np.nanstd(self.data):.2f}\n")
            else:
                for b in range(self.count):
                    # try to keep with rasterio convention.
                    as_str.append(f"Band {b + 1}:\n")
                    as_str.append(f"[MAXIMUM]:          {np.nanmax(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MINIMUM]:          {np.nanmin(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MEDIAN]:           {np.ma.median(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[MEAN]:             {np.nanmean(self.data[b, :, :]):.2f}\n")
                    as_str.append(f"[STD DEV]:          {np.nanstd(self.data[b, :, :]):.2f}\n")

        if verbose:
            print("".join(as_str))
            return None
        else:
            return "".join(as_str)

    def raster_equal(self, other: RasterType, strict_masked: bool = True, warn_failure_reason: bool = False) -> bool:
        """
        Check if two rasters are equal.

        This means that are equal:
        - The raster's masked array's data (including masked values), mask, fill_value and dtype,
        - The raster's transform, crs and nodata values.

        :param other: Other raster.
        :param strict_masked: Whether to check if masked cells (in .data.mask) have the same value (in .data.data).
        :param warn_failure_reason: Whether to warn for the reason of failure if the check does not pass.
        """

        if not isinstance(other, RasterBase):  # TODO: Possibly add equals to SatelliteImage?
            raise NotImplementedError("Equality with other object than Raster not supported by raster_equal.")

        if strict_masked:
            names = ["data.data", "data.mask", "data.fill_value", "dtype", "transform", "crs", "nodata"]
            equalities = [
                np.array_equal(self.data.data, other.data.data, equal_nan=True),
                # Use getmaskarray to avoid comparing boolean with array when mask=False
                np.array_equal(np.ma.getmaskarray(self.data), np.ma.getmaskarray(other.data)),
                self.data.fill_value == other.data.fill_value,
                self.data.dtype == other.data.dtype,
                self.transform == other.transform,
                self.crs == other.crs,
                self.nodata == other.nodata,
            ]
        else:
            names = ["data", "data.fill_value", "dtype", "transform", "crs", "nodata"]
            equalities = [
                np.ma.allequal(self.data, other.data),
                self.data.fill_value == other.data.fill_value,
                self.data.dtype == other.data.dtype,
                self.transform == other.transform,
                self.crs == other.crs,
                self.nodata == other.nodata,
            ]

        complete_equality = all(equalities)

        if not complete_equality and warn_failure_reason:
            where_fail = np.nonzero(~np.array(equalities))[0]
            warnings.warn(
                category=UserWarning, message=f"Equality failed for: {', '.join([names[w] for w in where_fail])}."
            )

        return complete_equality

    def georeferenced_grid_equal(self: RasterType, raster: RasterType) -> bool:
        """
        Check that raster shape, geotransform and CRS are equal.

        :param raster: Another raster.

        :return: Whether the two objects have the same georeferenced grid.
        """

        return all([self.shape == raster.shape, self.transform == raster.transform, self.crs == raster.crs])

    def get_bounds_projected(self, out_crs: CRS, densify_points: int = 5000) -> rio.coords.BoundingBox:
        """
        Get raster bounds projected in a specified CRS.

        :param out_crs: Output CRS.
        :param densify_points: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.

        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_points = min(max(self.width, self.height), densify_points)

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_points=densify_points)

        return new_bounds

    def get_footprint_projected(self, out_crs: CRS, densify_points: int = 5000) -> Vector:
        """
        Get raster footprint projected in a specified CRS.

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
        Get local metric coordinate reference system for the raster (UTM, UPS, or custom Mercator or Polar).

        :param local_crs_type: Whether to get a "universal" local CRS (UTM or UPS) or a "custom" local CRS
            (Mercator or Polar centered on centroid).
        :param method: Method to choose the zone of the CRS, either based on the centroid of the footprint
            or the extent as implemented in :func:`geopandas.GeoDataFrame.estimate_utm_crs`.
            Forced to centroid if `local_crs="custom"`.
        """

        # For universal CRS (UTM or UPS)
        if local_crs_type == "universal":
            return _get_utm_ups_crs(self.get_footprint_projected(out_crs=self.crs).ds, method=method)
        # For a custom CRS
        else:
            raise NotImplementedError("This is not implemented yet.")

    def intersection(self, raster: RasterType, match_ref: bool = True) -> tuple[float, float, float, float]:
        """
        Returns the bounding box of intersection between this image and another.

        If the rasters have different projections, the intersection extent is given in self's projection system.

        :param rst : path to the second image (or another Raster instance)
        :param match_ref: if set to True, returns the smallest intersection that aligns with that of self, i.e. same \
        resolution and offset with self's origin is a multiple of the resolution
        :returns: extent of the intersection between the 2 images \
        (xmin, ymin, xmax, ymax) in self's coordinate system.

        """

        # Reproject the bounds of raster to self's
        raster_bounds_sameproj = raster.get_bounds_projected(self.crs)

        # Calculate intersection of bounding boxes
        intersection = projtools.merge_bounds([self.bounds, raster_bounds_sameproj], merging_algorithm="intersection")

        # Check that intersection is not void (changed to NaN instead of empty tuple end 2022)
        if intersection == () or all(math.isnan(i) for i in intersection):
            warnings.warn("Intersection is void")
            return (0.0, 0.0, 0.0, 0.0)

        # if required, ensure the intersection is aligned with self's georeferences
        if match_ref:
            intersection = projtools.align_bounds(self.transform, intersection)

        # mypy raises a type issue, not sure how to address the fact that output of merge_bounds can be ()
        return intersection  # type: ignore

    # Note the star is needed because of the default argument 'mode' preceding non default arg 'inplace'
    # Then the final overload must be duplicated
    # Also note that in the first overload, only "inplace: Literal[False]" does not work. The ellipsis is
    # essential, otherwise MyPy gives incompatible return type Optional[Raster].
    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[False] = False,
    ) -> RasterType: ...

    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
    ) -> RasterType | None: ...

    def crop(
        self: RasterType,
        crop_geom: RasterType | Vector | list[float] | tuple[float, ...],
        mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
        *,
        inplace: bool = False,
    ) -> RasterType | None:
        """
        Crop the raster to a given extent.

        **Match-reference:** a reference raster or vector can be passed to match bounds during cropping.

        Reprojection is done on the fly if georeferenced objects have different projections.

        :param crop_geom: Geometry to crop raster to. Can use either a raster or vector as match-reference, or a list of
            coordinates. If ``crop_geom`` is a raster or vector, will crop to the bounds. If ``crop_geom`` is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param mode: Whether to match within pixels or exact extent. ``'match_pixel'`` will preserve the original pixel
            resolution, cropping to the extent that most closely aligns with the current coordinates. ``'match_extent'``
            will match the extent exactly, adjusting the pixel resolution to fit the extent.
        :param inplace: Whether to update the raster in-place.

        :returns: A new raster (or None if inplace).
        """

        crop_img, tfm = _crop(source_raster=self, crop_geom=crop_geom, mode=mode)

        if inplace:
            self._data = crop_img
            self.transform = tfm
            return None
        else:
            newraster = self.from_array(crop_img, tfm, self.crs, self.nodata, self.area_or_point)
            return newraster

    @overload
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[False] = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> RasterType: ...

    @overload
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        *,
        inplace: Literal[True],
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> None: ...

    @overload
    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        *,
        inplace: bool = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> RasterType | None: ...

    def reproject(
        self: RasterType,
        ref: RasterType | str | None = None,
        crs: CRS | str | int | None = None,
        res: float | Iterable[float] | None = None,
        grid_size: tuple[int, int] | None = None,
        bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
        nodata: int | float | None = None,
        dtype: DTypeLike | None = None,
        resampling: Resampling | str = Resampling.bilinear,
        force_source_nodata: int | float | None = None,
        inplace: bool = False,
        silent: bool = False,
        n_threads: int = 0,
        memory_limit: int = 64,
    ) -> RasterType | None:
        """
        Reproject raster to a different geotransform (resolution, bounds) and/or coordinate reference system (CRS).

        **Match-reference**: a reference raster can be passed to match resolution, bounds and CRS during reprojection.

        Alternatively, the destination resolution, bounds and CRS can be passed individually.

        Any resampling algorithm implemented in Rasterio can be passed as a string.

        :param ref: Reference raster to match resolution, bounds and CRS.
        :param crs: Destination coordinate reference system as a string or EPSG. If ``ref`` not set,
            defaults to this raster's CRS.
        :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
            Do not use with ``grid_size``.
        :param grid_size: Destination grid size as (x, y). Do not use with ``res``.
        :param bounds: Destination bounds as a Rasterio bounding box, or a dictionary containing left, bottom,
            right, top bounds in the destination CRS.
        :param nodata: Destination nodata value. If set to ``None``, will use the same as source. If source does
            not exist, will use GDAL's default.
        :param dtype: Destination data type of array.
        :param resampling: A Rasterio resampling method, can be passed as a string.
            See https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
            for the full list.
        :param inplace: Whether to update the raster in-place.
        :param force_source_nodata: Force a source nodata value (read from the metadata by default).
        :param silent: Whether to print warning statements.
        :param n_threads: Number of threads. Defaults to (os.cpu_count() - 1).
        :param memory_limit: Memory limit in MB for warp operations. Larger values may perform better.

        :returns: Reprojected raster (or None if inplace).

        """

        # Reproject
        return_copy, data, transformed, crs, nodata = _reproject(
            source_raster=self,
            ref=ref,
            crs=crs,
            res=res,
            grid_size=grid_size,
            bounds=bounds,
            nodata=nodata,
            dtype=dtype,
            resampling=resampling,
            force_source_nodata=force_source_nodata,
            silent=silent,
            n_threads=n_threads,
            memory_limit=memory_limit,
        )

        # If return copy is True (target georeferenced grid was the same as input)
        if return_copy:
            if inplace:
                return None
            else:
                return self

        # To make MyPy happy without overload for _reproject (as it might re-structured soon anyway)
        assert data is not None
        assert transformed is not None
        assert crs is not None

        # Write results to a new Raster.
        if inplace:
            # Order is important here, because calling self.data will use nodata to mask the array properly
            self._crs = crs
            self._nodata = nodata
            self._transform = transformed
            # A little trick to force the right shape of data in, then update the mask properly through the data setter
            self._data = data.squeeze()
            self.data = data
            return None
        else:
            return self.from_array(data, transformed, crs, nodata, self.area_or_point)

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: Literal[False] = False,
    ) -> RasterType: ...

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced"] | Literal["pixel"] = "georeferenced",
        *,
        inplace: bool = False,
    ) -> RasterType | None: ...

    def translate(
        self: RasterType,
        xoff: float,
        yoff: float,
        distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
        inplace: bool = False,
    ) -> RasterType | None:
        """
        Translate a raster by a (x,y) offset.

        The translation only updates the geotransform (no resampling is performed).

        :param xoff: Translation x offset.
        :param yoff: Translation y offset.
        :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.
        :param inplace: Whether to modify the raster in-place.

        :returns: Translated raster (or None if inplace).
        """

        translated_transform = _translate(self.transform, xoff=xoff, yoff=yoff, distance_unit=distance_unit)

        if inplace:
            # Overwrite transform by translated transform
            self.transform = translated_transform
            return None
        else:
            raster_copy = self.copy()
            raster_copy.transform = translated_transform
            return raster_copy

    def reduce_points(
        self,
        points: tuple[ArrayLike, ArrayLike],
        reducer_function: Callable[[NDArrayNum], float] = np.ma.mean,
        window: int | None = None,
        input_latlon: bool = False,
        band: int | None = None,
        masked: bool = False,
        return_window: bool = False,
        boundless: bool = True,
    ) -> Any:
        """
        Reduce raster values around point coordinates.


        By default, samples pixel value of each band. Can be passed a band index to sample from.

        Uses Rasterio's windowed reading to keep memory usage low (for a raster not loaded).

        :param points: Point(s) at which to interpolate raster value (tuple of X/Y array-likes). If points fall
            outside of image, value returned is nan.
        :param reducer_function: Reducer function to apply to the values in window (defaults to np.mean).
        :param window: Window size to read around coordinates. Must be odd.
        :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS
        :param band: Band number to extract from (from 1 to self.count).
        :param masked: Whether to return a masked array, or classic array.
        :param return_window: Whether to return the windows (in addition to the reduced value).
        :param boundless: Whether to allow windows that extend beyond the extent.

        :returns: When called on a raster or with a specific band set, return value of pixel.
        :returns: If multiple band raster and the band is not specified, a
            dictionary containing the value of the pixel in each band.
        :returns: In addition, if return_window=True, return tuple of (values, arrays)

        :examples:

            >>> self.value_at_coords(-48.125, 67.8901, window=3)  # doctest: +SKIP
            Returns mean of a 3*3 window:
                v v v \
                v c v  | = float(mean)
                v v v /
            (c = provided coordinate, v= value of surrounding coordinate)

        """

        x, y = points

        # Check for array-like inputs
        if (
            not isinstance(x, (float, np.floating, int, np.integer))
            and isinstance(y, (float, np.floating, int, np.integer))
            or isinstance(x, (float, np.floating, int, np.integer))
            and not isinstance(y, (float, np.floating, int, np.integer))
        ):
            raise TypeError("Coordinates must be both numbers or both array-like.")

        # If for a single value, wrap in a list
        if isinstance(x, (float, np.floating, int, np.integer)):
            x = [x]  # type: ignore
            y = [y]  # type: ignore
            # For the end of the function
            unwrap = True
        else:
            unwrap = False
            # Check that array-like objects are the same length
            if len(x) != len(y):  # type: ignore
                raise ValueError("Coordinates must be of the same length.")

        # Check window parameter
        if window is not None:
            if not float(window).is_integer():
                raise ValueError("Window must be a whole number.")
            if window % 2 != 1:
                raise ValueError("Window must be an odd number.")
            window = int(window)

        # Define subfunction for reducing the window array
        def format_value(value: Any) -> Any:
            """Check if valid value has been extracted"""
            if type(value) in [np.ndarray, np.ma.core.MaskedArray]:
                if window is not None:
                    value = reducer_function(value.flatten())
                else:
                    value = value[0, 0]
            else:
                value = None
            return value

        # Initiate output lists
        list_values = []
        if return_window:
            list_windows = []

        # Convert to latlon if asked
        if input_latlon:
            x, y = reproject_from_latlon((y, x), self.crs)  # type: ignore

        # Convert coordinates to pixel space
        rows, cols = rio.transform.rowcol(self.transform, x, y, op=math.floor)

        # Loop over all coordinates passed
        for k in range(len(rows)):  # type: ignore
            value: float | dict[int, float] | tuple[float | dict[int, float] | tuple[list[float], NDArrayNum] | Any]

            row = rows[k]  # type: ignore
            col = cols[k]  # type: ignore

            # Decide what pixel coordinates to read:
            if window is not None:
                half_win = (window - 1) / 2
                # Subtract start coordinates back to top left of window
                col = col - half_win
                row = row - half_win
                # Offset to read to == window
                width = window
                height = window
            else:
                # Start reading at col,row and read 1px each way
                width = 1
                height = 1

            # Make sure coordinates are int
            col = int(col)
            row = int(row)

            # Create rasterio's window for reading
            rio_window = rio.windows.Window(col, row, width, height)

            if self.is_loaded:
                if self.count == 1:
                    data = self.data[row : row + height, col : col + width]
                else:
                    data = self.data[slice(None) if band is None else band - 1, row : row + height, col : col + width]
                if not masked:
                    data = data.astype(np.float32).filled(np.nan)
                value = format_value(data)
                win: NDArrayNum | dict[int, NDArrayNum] = data

            else:
                # TODO: if we want to allow sampling multiple bands, need to do it also when data is loaded
                # if self.count == 1:
                with rio.open(self.filename) as raster:
                    data = raster.read(
                        window=rio_window,
                        fill_value=self.nodata,
                        boundless=boundless,
                        masked=masked,
                        indexes=band,
                    )
                value = format_value(data)
                win = data
                # else:
                #     value = {}
                #     win = {}
                #     with rio.open(self.filename) as raster:
                #         for b in self.indexes:
                #             data = raster.read(
                #                 window=rio_window, fill_value=self.nodata, boundless=boundless,
                #                 masked=masked, indexes=b
                #             )
                #             val = format_value(data)
                #             value[b] = val
                #             win[b] = data  # type: ignore

            list_values.append(value)
            if return_window:
                list_windows.append(win)

        # If for a single value, unwrap output list
        if unwrap:
            output_val = list_values[0]
            if return_window:
                output_win = list_windows[0]
        else:
            output_val = np.array(list_values)  # type: ignore
            if return_window:
                output_win = list_windows  # type: ignore

        if return_window:
            return (output_val, output_win)
        else:
            return output_val

    def xy2ij(
        self,
        x: ArrayLike,
        y: ArrayLike,
        op: type = np.float32,
        precision: float | None = None,
        shift_area_or_point: bool | None = None,
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get indexes (row,column) of coordinates (x,y).

        By default, the indexes are shifted with the interpretation of pixel coordinates "AREA_OR_POINT" of the raster,
        to ensure that the indexes of points represent the right location. See parameter description of
        shift_area_or_point for more details.

        This function is reversible with ij2xy for any pixel interpretation.

        :param x: X coordinates.
        :param y: Y coordinates.
        :param op: Operator to compute index.
        :param precision: Precision passed to :func:`rasterio.transform.rowcol`.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            indexes if self.area_or_point is "Point" and maintains corner pixel indexes if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].

        :returns i, j: Indices of (x,y) in the image.
        """

        return _xy2ij(
            x=x,
            y=y,
            transform=self.transform,
            area_or_point=self.area_or_point,
            op=op,
            precision=precision,
            shift_area_or_point=shift_area_or_point,
        )

    def ij2xy(
        self, i: ArrayLike, j: ArrayLike, shift_area_or_point: bool | None = None, force_offset: str | None = None
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get coordinates (x,y) of indexes (row,column).

        By default, the indexes are shifted with the interpretation of pixel coordinates "AREA_OR_POINT" of the
        raster, to ensure that the indexes of points represent the right location. See parameter description of
        shift_area_or_point for more details.

        This function is reversible with xy2ij for any pixel interpretation.

        :param i: Row (i) index of pixel.
        :param j: Column (j) index of pixel.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_offset: Ignore pixel interpretation and force coordinate to a certain offset: "center" of pixel, or
            any corner (upper-left "ul", "ur", "ll", lr"). Default coordinate of a raster is upper-left.

        :returns x, y: x,y coordinates of i,j in reference system.
        """

        return _ij2xy(
            i=i,
            j=j,
            transform=self.transform,
            area_or_point=self.area_or_point,
            shift_area_or_point=shift_area_or_point,
            force_offset=force_offset,
        )

    def coords(
        self, grid: bool = True, shift_area_or_point: bool | None = None, force_offset: str | None = None
    ) -> tuple[NDArrayNum, NDArrayNum]:
        """
        Get coordinates (x,y) of all pixels in the raster.

        :param grid: Whether to return mesh grids of coordinates matrices.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_offset: Ignore pixel interpretation and force coordinate to a certain offset: "center" of pixel, or
            any corner (upper-left "ul", "ur", "ll", lr"). Default coordinate of a raster is upper-left.

        :returns x,y: Arrays of the (x,y) coordinates.
        """

        return _coords(
            transform=self.transform,
            shape=self.shape,
            area_or_point=self.area_or_point,
            grid=grid,
            shift_area_or_point=shift_area_or_point,
            force_offset=force_offset,
        )

    def outside_image(self, xi: ArrayLike, yj: ArrayLike, index: bool = True) -> bool:
        """
        Check whether a given point falls outside the raster.

        :param xi: Indices (or coordinates) of x direction to check.
        :param yj: Indices (or coordinates) of y direction to check.
        :param index: Interpret ij as raster indices (default is ``True``). If False, assumes ij is coordinates.

        :returns is_outside: ``True`` if ij is outside the image.
        """

        return _outside_image(
            xi=xi, yj=yj, transform=self.transform, shape=self.shape, area_or_point=self.area_or_point, index=index
        )

    def interp_points(
        self,
        points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
        dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int = "half_order_up",
        band: int = 1,
        input_latlon: bool = False,
        shift_area_or_point: bool | None = None,
        force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
        **kwargs: Any,
    ) -> NDArrayNum:
        """
         Interpolate raster values at a set of points.

         Uses scipy.ndimage.map_coordinates if the Raster is on an equal grid using "nearest" or "linear" (for speed),
         otherwise uses scipy.interpn on a regular grid.

         Optionally, user can enforce the interpretation of pixel coordinates in self.tags['AREA_OR_POINT']
         to ensure that the interpolation of points is done at the right location. See parameter description
         of shift_area_or_point for more details.

        :param points: Point(s) at which to interpolate raster value (tuple of X/Y array-likes). If points fall
            outside of image, value returned is nan.
        :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', 'quintic', 'slinear', 'pchip' or
            'splinef2d'. For more information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn.
            Default is linear.
        :param dist_nodata_spread: Distance of nodata spreading during interpolation, either half-interpolation order
            rounded up (default; equivalent to 0 for nearest, 1 for linear methods, 2 for cubic methods and 3 for
            quintic method), or rounded down, or a fixed integer.
        :param band: Band to use (from 1 to self.count).
        :param input_latlon: Whether the input is in latlon, unregarding of Raster CRS.
        :param shift_area_or_point: Whether to shift with pixel interpretation, which shifts to center of pixel
            coordinates if self.area_or_point is "Point" and maintains corner pixel coordinate if it is "Area" or None.
            Defaults to True. Can be configured with the global setting geoutils.config["shift_area_or_point"].
        :param force_scipy_function: Force to use either map_coordinates or interpn. Mainly for testing purposes.

        :returns rpoints: Array of raster value(s) for the given points.
        """

        # Extract array supporting NaNs
        array = self.get_nanarray()
        if self.count != 1:
            array = array[band - 1, :, :]

        # If those are in latlon, convert to Raster CRS
        if input_latlon:
            points = reproject_from_latlon(points, out_crs=self.crs)  # type: ignore

        return _interp_points(
            array,
            transform=self.transform,
            area_or_point=self.area_or_point,
            points=points,
            method=method,
            shift_area_or_point=shift_area_or_point,
            dist_nodata_spread=dist_nodata_spread,
            force_scipy_function=force_scipy_function,
            **kwargs,
        )

    @deprecate(
        Version("0.3.0"),
        "Raster.to_points() is deprecated in favor of Raster.to_pointcloud() and " "will be removed in v0.3.",
    )
    def to_points(self, **kwargs):  # type: ignore

        self.to_pointcloud(**kwargs)  # type: ignore

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: Literal[False] = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum: ...

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: Literal[True],
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> Vector: ...

    @overload
    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        *,
        as_array: bool = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum | Vector: ...

    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        as_array: bool = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum | Vector:
        """
        Convert raster to point cloud.

        A point cloud is a vector of point geometries associated to a data column, and possibly other auxiliary data
        columns, see geoutils.PointCloud.

        For a single band raster, the main data column name of the point cloud defaults to "b1" and stores values of
        that single band.
        For a multi-band raster, the main data column name of the point cloud defaults to "bX" where X is the data band
        index chosen by the user (defaults to 1, the first band).
        Optionally, all other bands can also be stored in columns "b1", "b2", etc. For more specific band selection,
        use Raster.split_bands previous to converting to point cloud.

        Optionally, randomly subsample valid pixels for the data band (nodata values can be skipped, but only for the
        band that will be used as data column of the point cloud).
        If 'subsample' is either 1, or is equal to the pixel count, all (valid) points are returned.
        If 'subsample' is smaller than 1 (for fractions), or smaller than the pixel count, a random subsample
        of (valid) points is returned.

        If the raster is not loaded, sampling will be done from disk using rasterio.sample after loading only the masks
        of the dataset.

        Formats:
            * `as_array` == False: A vector with dataframe columns ["b1", "b2", ..., "geometry"],
            * `as_array` == True: A numpy ndarray of shape (N, 2 + count) with the columns [x, y, b1, b2..].

        :param data_column_name: Name to use for point cloud data column, defaults to "bX" where X is the data band
            number.
        :param data_band: (Only for multi-band rasters) Band to use for data column, defaults to first. Band counting
            starts at 1.
        :param auxiliary_data_bands: (Only for multi-band rasters) Whether to save other band numbers as auxiliary data
            columns, defaults to none.
        :param auxiliary_column_names: (Only for multi-band rasters) Names to use for auxiliary data bands, only if
            auxiliary data bands is not none, defaults to "b1", "b2", etc.
        :param subsample: Subsample size. If > 1, parsed as a count, otherwise a fraction.
        :param skip_nodata: Whether to skip nodata values.
        :param as_array: Return an array instead of a vector.
        :param random_state: Random state or seed number.
        :param force_pixel_offset: Force offset to derive point coordinate with. Raster coordinates normally only
            associate to upper-left corner "ul" ("Area" definition) or center ("Point" definition).

        :raises ValueError: If the sample count or fraction is poorly formatted.

        :returns: A point cloud, or array of the shape (N, 2 + count) where N is the sample count.
        """

        return _raster_to_pointcloud(
            source_raster=self,
            data_column_name=data_column_name,
            data_band=data_band,
            auxiliary_data_bands=auxiliary_data_bands,
            auxiliary_column_names=auxiliary_column_names,
            subsample=subsample,
            skip_nodata=skip_nodata,
            as_array=as_array,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
        )

    @classmethod
    def from_pointcloud_regular(
        cls: type[RasterType],
        pointcloud: gpd.GeoDataFrame,
        grid_coords: tuple[NDArrayNum, NDArrayNum] = None,
        transform: rio.transform.Affine = None,
        shape: tuple[int, int] = None,
        nodata: int | float | None = None,
        data_column_name: str = "b1",
        area_or_point: Literal["Area", "Point"] = "Point",
    ) -> RasterType:
        """
        Create a raster from a point cloud with coordinates on a regular grid.

        To inform on what grid to create the raster, either pass a tuple of X/Y grid coordinates, or the expected
        transform and shape. All point cloud coordinates must fall exactly at one of the coordinates of this grid.

        :param pointcloud: Point cloud.
        :param grid_coords: Regular coordinate vectors for the raster, from which the geotransform and shape are
            deduced.
        :param transform: Geotransform of the raster.
        :param shape: Shape of the raster.
        :param nodata: Nodata value of the raster.
        :param data_column_name: Name to use for point cloud data column, defaults to "bX" where X is the data band
            number.
        :param area_or_point: Whether to set the pixel interpretation of the raster to "Area" or "Point".
        """

        arr, transform, crs, nodata, aop = _regular_pointcloud_to_raster(
            pointcloud=pointcloud,
            grid_coords=grid_coords,
            transform=transform,
            shape=shape,
            nodata=nodata,
            data_column_name=data_column_name,
            area_or_point=area_or_point,
        )

        return cls.from_array(data=arr, transform=transform, crs=crs, nodata=nodata, area_or_point=area_or_point)

    def polygonize(
        self,
        target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"] = "all",
        data_column_name: str = "id",
    ) -> Vector:
        """
        Polygonize the raster into a vector.

        :param target_values: Value or range of values of the raster from which to
          create geometries (defaults to "all", for which all unique pixel values of the raster are used).
        :param data_column_name: Data column name to be associated with target values in the output vector
            (defaults to "id").

        :returns: Vector containing the polygonized geometries associated to target values.
        """

        return _polygonize(source_raster=self, target_values=target_values, data_column_name=data_column_name)

    def proximity(
        self,
        vector: Vector | None = None,
        target_values: list[float] | None = None,
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> RasterBase:
        """
        Compute proximity distances to the raster target pixels, or to a vector geometry on the raster grid.

        **Match-reference**: a raster can be passed to match its resolution, bounds and CRS for computing
        proximity distances.

        When passing a vector, by default, the boundary of the geometry will be used. The full geometry can be used by
        passing "geometry", or any lower dimensional geometry attribute such as "centroid", "envelope" or "convex_hull".
        See all geometry attributes in the Shapely documentation at https://shapely.readthedocs.io/.

        :param vector: Vector for which to compute the proximity to geometry,
            if not provided computed on this raster target pixels.
        :param target_values: (Only with raster) List of target values to use for the proximity,
            defaults to all non-zero values.
        :param geometry_type: (Only with a vector) Type of geometry to use for the proximity, defaults to 'boundary'.
        :param in_or_out: (Only with a vector) Compute proximity only 'in' or 'out'-side the geometry, or 'both'.
        :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.

        :return: Proximity distances raster.
        """

        proximity = _proximity_from_vector_or_raster(
            raster=self,
            vector=vector,
            target_values=target_values,
            geometry_type=geometry_type,
            in_or_out=in_or_out,
            distance_unit=distance_unit,
        )

        out_nodata = _default_nodata(proximity.dtype)
        return self.from_array(
            data=proximity,
            transform=self.transform,
            crs=self.crs,
            nodata=out_nodata,
            area_or_point=self.area_or_point,
            tags=self.tags,
        )

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum: ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[NDArrayNum, ...]: ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]: ...

    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly sample the raster. Only valid values are considered.

        :param subsample: Subsample size. If <= 1, a fraction of the total pixels to extract.
            If > 1, the number of pixels.
        :param return_indices: Whether to return the extracted indices only.
        :param random_state: Random state or seed number.

        :return: Array of sampled valid values, or array of sampled indices.
        """

        return subsample_array(
            array=self.data, subsample=subsample, return_indices=return_indices, random_state=random_state
        )
