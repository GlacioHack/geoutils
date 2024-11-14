"""
Functionalities for geotransformations of raster objects.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterable, Literal

import affine
import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling

import geoutils as gu
from geoutils._typing import DTypeLike, MArrayNum
from geoutils.raster.georeferencing import (
    _cast_pixel_interpretation,
    _default_nodata,
    _res,
)


def _resampling_method_from_str(method_str: str) -> rio.enums.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.enums.Resampling:
        if method.name == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.enums.Resampling method. "
            f"Valid methods: {[method.name for method in rio.enums.Resampling]}"
        )
    return resampling_method


##############
# 1/ REPROJECT
##############


def _user_input_reproject(
    source_raster: gu.Raster,
    ref: gu.Raster,
    crs: CRS | str | int | None,
    res: float | Iterable[float] | None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[
    CRS, DTypeLike, int | float | None, int | float | None, float | Iterable[float] | None, rio.coords.BoundingBox
]:
    """Check all user inputs of reproject."""

    # --- Sanity checks on inputs and defaults -- #
    # Check that either ref or crs is provided
    if ref is not None and crs is not None:
        raise ValueError("Either of `ref` or `crs` must be set. Not both.")
    # If none are provided, simply preserve the CRS
    elif ref is None and crs is None:
        crs = source_raster.crs

    # Set output dtype
    if dtype is None:
        # Warning: this will not work for multiple bands with different dtypes
        dtype = source_raster.dtype

    # --- Set source nodata if provided -- #
    if force_source_nodata is None:
        src_nodata = source_raster.nodata
    else:
        src_nodata = force_source_nodata
        # Raise warning if a different nodata value exists for this raster than the forced one (not None)
        if source_raster.nodata is not None:
            warnings.warn(
                "Forcing source nodata value of {} despite an existing nodata value of {} in the raster. "
                "To silence this warning, use self.set_nodata() before reprojection instead of forcing.".format(
                    force_source_nodata, source_raster.nodata
                )
            )

    # --- Set destination nodata if provided -- #
    # This is needed in areas not covered by the input data.
    # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
    if nodata is None:
        nodata = source_raster.nodata
        if nodata is None:
            nodata = _default_nodata(dtype)
            # If nodata is already being used, raise a warning.
            # TODO: for uint8, if all values are used, apply rio.warp to mask to identify invalid values
            if not source_raster.is_loaded:
                warnings.warn(
                    f"For reprojection, nodata must be set. Setting default nodata to {nodata}. You may "
                    f"set a different nodata with `nodata`."
                )

            elif nodata in source_raster.data:
                warnings.warn(
                    f"For reprojection, nodata must be set. Default chosen value {nodata} exists in "
                    f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                    f"self.set_nodata()."
                )

    # Create a BoundingBox if required
    if bounds is not None:
        if not isinstance(bounds, rio.coords.BoundingBox):
            bounds = rio.coords.BoundingBox(
                bounds["left"],
                bounds["bottom"],
                bounds["right"],
                bounds["top"],
            )

    # Case a raster is provided as reference
    if ref is not None:
        # Check that ref type is either str, Raster or rasterio data set
        # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
        if isinstance(ref, gu.Raster):
            # Raise a warning if the reference is a raster that has a different pixel interpretation
            _cast_pixel_interpretation(source_raster.area_or_point, ref.area_or_point)
            ds_ref = ref
        elif isinstance(ref, str):
            if not os.path.exists(ref):
                raise ValueError("Reference raster does not exist.")
            ds_ref = gu.Raster(ref, load_data=False)
        else:
            raise TypeError("Type of ref not understood, must be path to file (str), Raster.")

        # Read reprojecting params from ref raster
        crs = ds_ref.crs
        res = ds_ref.res
        bounds = ds_ref.bounds
    else:
        # Determine target CRS
        crs = CRS.from_user_input(crs)
        res = res

    return crs, dtype, src_nodata, nodata, res, bounds


def _get_target_georeferenced_grid(
    raster: gu.Raster,
    crs: CRS | str | int | None = None,
    grid_size: tuple[int, int] | None = None,
    res: int | float | Iterable[float] | None = None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
) -> tuple[affine.Affine, tuple[int, int]]:
    """
    Derive the georeferencing parameters (transform, size) for the target grid.

    Needed to reproject a raster to a different grid (resolution or size, bounds) and/or
    coordinate reference system (CRS).

    If requested bounds are incompatible with output resolution (would result in non integer number of pixels),
    the bounds are rounded up to the nearest compatible value.

    :param crs: Destination coordinate reference system as a string or EPSG. Defaults to this raster's CRS.
    :param grid_size: Destination size as (ncol, nrow). Mutually exclusive with ``res``.
    :param res: Destination resolution (pixel size) in units of destination CRS. Single value or (xres, yres).
        Mutually exclusive with ``size``.
    :param bounds: Destination bounds as a Rasterio bounding box, or a dictionary containing left, bottom,
        right, top bounds in the destination CRS.

    :returns: Calculated transform and size.
    """
    # --- Input sanity checks --- #
    # check size and res are not both set
    if (grid_size is not None) and (res is not None):
        raise ValueError("size and res both specified. Specify only one.")

    # Set CRS to input CRS by default
    if crs is None:
        crs = raster.crs

    if grid_size is None:
        width, height = None, None
    else:
        width, height = grid_size

    # Convert bounds to BoundingBox
    if bounds is not None:
        if not isinstance(bounds, rio.coords.BoundingBox):
            bounds = rio.coords.BoundingBox(
                bounds["left"],
                bounds["bottom"],
                bounds["right"],
                bounds["top"],
            )

    # If all georeferences are the same as input, skip calculating because of issue in
    # rio.warp.calculate_default_transform (https://github.com/rasterio/rasterio/issues/3010)
    if (
        (crs == raster.crs)
        & ((grid_size is None) | ((height == raster.shape[0]) & (width == raster.shape[1])))
        & ((res is None) | np.all(np.array(res) == raster.res))
        & ((bounds is None) | (bounds == raster.bounds))
    ):
        return raster.transform, raster.shape[::-1]

    # --- First, calculate default transform ignoring any change in bounds --- #
    tmp_transform, tmp_width, tmp_height = rio.warp.calculate_default_transform(
        raster.crs,
        crs,
        raster.width,
        raster.height,
        left=raster.bounds.left,
        right=raster.bounds.right,
        top=raster.bounds.top,
        bottom=raster.bounds.bottom,
        resolution=res,
        dst_width=width,
        dst_height=height,
    )

    # If no bounds specified, can directly use output of rio.warp.calculate_default_transform
    if bounds is None:
        dst_size = (tmp_width, tmp_height)
        dst_transform = tmp_transform

    # --- Second, crop to requested bounds --- #
    else:
        # If output size and bounds are known, can use rio.transform.from_bounds to get dst_transform
        if grid_size is not None:
            dst_transform = rio.transform.from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top, grid_size[0], grid_size[1]
            )
            dst_size = grid_size

        else:
            # Otherwise, need to calculate the new output size, rounded to nearest integer
            ref_win = rio.windows.from_bounds(*list(bounds), tmp_transform).round_lengths()
            dst_size = (int(ref_win.width), int(ref_win.height))

            if res is not None:
                # In this case, we force output resolution
                if isinstance(res, tuple):
                    dst_transform = rio.transform.from_origin(bounds.left, bounds.top, res[0], res[1])
                else:
                    dst_transform = rio.transform.from_origin(bounds.left, bounds.top, res, res)
            else:
                # In this case, we force output bounds
                dst_transform = rio.transform.from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top, dst_size[0], dst_size[1]
                )

    return dst_transform, dst_size


def _get_reproj_params(
    source_raster: gu.Raster,
    crs: CRS,
    res: float | Iterable[float] | None,
    grid_size: tuple[int, int] | None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None,
    dtype: DTypeLike,
    src_nodata: int | float | None,
    nodata: int | float | None,
    resampling: Resampling | str,
) -> dict[str, Any]:
    """Get all reprojection parameters."""

    # First, set basic reprojection options
    reproj_kwargs = {
        "src_transform": source_raster.transform,
        "src_crs": source_raster.crs,
        "resampling": resampling if isinstance(resampling, Resampling) else _resampling_method_from_str(resampling),
        "src_nodata": src_nodata,
        "dst_nodata": nodata,
    }

    # Second, determine target transform and grid size
    transform, grid_size = _get_target_georeferenced_grid(
        source_raster, crs=crs, grid_size=grid_size, res=res, bounds=bounds
    )

    # Finally, update reprojection options accordingly
    reproj_kwargs.update({"dst_transform": transform})
    data = np.ones((source_raster.count, grid_size[1], grid_size[0]), dtype=dtype)
    reproj_kwargs.update({"destination": data})
    reproj_kwargs.update({"dst_crs": crs})

    return reproj_kwargs


def _is_reproj_needed(src_shape: tuple[int, int], reproj_kwargs: dict[str, Any]) -> bool:
    """Check if reprojection is actually needed based on transformation parameters."""

    src_transform = reproj_kwargs["src_transform"]
    transform = reproj_kwargs["dst_transform"]
    src_crs = reproj_kwargs["src_crs"]
    crs = reproj_kwargs["dst_crs"]
    grid_size = reproj_kwargs["destination"].shape[1:][::-1]
    src_res = _res(src_transform)
    res = _res(transform)

    # Caution, grid_size is (width, height) while shape is (height, width)
    return all(
        [
            (transform == src_transform) or (transform is None),
            (crs == src_crs) or (crs is None),
            (grid_size == src_shape[::-1]) or (grid_size is None),
            np.all(np.array(res) == src_res) or (res is None),
        ]
    )


def _reproject(
    source_raster: gu.Raster,
    ref: gu.Raster,
    crs: CRS | str | int | None = None,
    res: float | Iterable[float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: dict[str, float] | rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
    n_threads: int = 0,
    memory_limit: int = 64,
) -> tuple[bool, MArrayNum | None, affine.Affine | None, CRS | None, int | float | None]:
    """
    Reproject raster. See Raster.reproject() for details.
    """

    # 1/ Process user input
    crs, dtype, src_nodata, nodata, res, bounds = _user_input_reproject(
        source_raster=source_raster,
        ref=ref,
        crs=crs,
        bounds=bounds,
        res=res,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # 2/ Derive georeferencing parameters for reprojection (transform, grid size)
    reproj_kwargs = _get_reproj_params(
        source_raster=source_raster,
        crs=crs,
        res=res,
        grid_size=grid_size,
        bounds=bounds,
        dtype=dtype,
        src_nodata=src_nodata,
        nodata=nodata,
        resampling=resampling,
    )

    # 3/ Check if reprojection is needed, otherwise return source raster with warning
    if _is_reproj_needed(src_shape=source_raster.shape, reproj_kwargs=reproj_kwargs):
        if (nodata == src_nodata) or (nodata is None):
            if not silent:
                warnings.warn("Output projection, bounds and grid size are identical -> returning self (not a copy!)")
            return True, None, None, None, None

        elif nodata is not None:
            if not silent:
                warnings.warn(
                    "Only nodata is different, consider using the 'set_nodata()' method instead'\
                ' -> returning self (not a copy!)"
                )
            return True, None, None, None, None

    # 4/ Perform reprojection

    # --- Set the performance keywords --- #
    if n_threads == 0:
        # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
        cpu_count = os.cpu_count() or 2
        num_threads = cpu_count - 1
    else:
        num_threads = n_threads
    reproj_kwargs.update({"num_threads": num_threads, "warp_mem_limit": memory_limit})

    # --- Run the reprojection of data --- #
    # If data is loaded, reproject the numpy array directly
    if source_raster.is_loaded:
        # All masked values must be set to a nodata value for rasterio's reproject to work properly
        # TODO: another option is to apply rio.warp.reproject to the mask to identify invalid pixels
        if src_nodata is None and np.sum(source_raster.data.mask) > 0:
            raise ValueError(
                "No nodata set, set one for the raster with self.set_nodata() or use a temporary one "
                "with `force_source_nodata`."
            )

        # Mask not taken into account by rasterio, need to fill with src_nodata
        data, transformed = rio.warp.reproject(source_raster.data.filled(src_nodata), **reproj_kwargs)

    # If not, uses the dataset instead
    else:
        data = []  # type: ignore
        for k in range(source_raster.count):
            with rio.open(source_raster.filename) as ds:
                band = rio.band(ds, k + 1)
                band, transformed = rio.warp.reproject(band, **reproj_kwargs)
                data.append(band.squeeze())

        data = np.array(data)

    # Enforce output type
    data = np.ma.masked_array(data.astype(dtype), fill_value=nodata)

    if nodata is not None:
        data.mask = data == nodata

    # Check for funny business.
    if reproj_kwargs["dst_transform"] is not None:
        assert reproj_kwargs["dst_transform"] == transformed

    return False, data, transformed, crs, nodata


#########
# 2/ CROP
#########


def _crop(
    source_raster: gu.Raster,
    crop_geom: gu.Raster | gu.Vector | list[float] | tuple[float, ...],
    mode: Literal["match_pixel"] | Literal["match_extent"] = "match_pixel",
) -> tuple[MArrayNum, affine.Affine]:
    """Crop raster. See details in Raster.crop()."""

    assert mode in [
        "match_extent",
        "match_pixel",
    ], "mode must be one of 'match_pixel', 'match_extent'"

    if isinstance(crop_geom, (gu.Raster, gu.Vector)):
        # For another Vector or Raster, we reproject the bounding box in the same CRS as self
        xmin, ymin, xmax, ymax = crop_geom.get_bounds_projected(out_crs=source_raster.crs)
        if isinstance(crop_geom, gu.Raster):
            # Raise a warning if the reference is a raster that has a different pixel interpretation
            _cast_pixel_interpretation(source_raster.area_or_point, crop_geom.area_or_point)
    elif isinstance(crop_geom, (list, tuple)):
        xmin, ymin, xmax, ymax = crop_geom
    else:
        raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

    if mode == "match_pixel":
        # Finding the intersection of requested bounds and original bounds, cropped to image shape
        ref_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=source_raster.transform)
        self_win = rio.windows.from_bounds(*source_raster.bounds, transform=source_raster.transform).crop(
            *source_raster.shape
        )
        final_window = ref_win.intersection(self_win).round_lengths().round_offsets()

        # Update bounds and transform accordingly
        new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=source_raster.transform)
        tfm = rio.transform.from_origin(new_xmin, new_ymax, *source_raster.res)

        if source_raster.is_loaded:
            # In case data is loaded on disk, can extract directly from np array
            (rowmin, rowmax), (colmin, colmax) = final_window.toranges()

            if source_raster.count == 1:
                crop_img = source_raster.data[rowmin:rowmax, colmin:colmax]
            else:
                crop_img = source_raster.data[:, rowmin:rowmax, colmin:colmax]
        else:

            assert source_raster._disk_shape is not None  # This should not be the case, sanity check to make mypy happy

            # If data was not loaded, and self's transform was updated (e.g. due to downsampling) need to
            # get the Window corresponding to on disk data
            ref_win_disk = rio.windows.from_bounds(
                new_xmin, new_ymin, new_xmax, new_ymax, transform=source_raster._disk_transform
            )
            self_win_disk = rio.windows.from_bounds(
                *source_raster.bounds, transform=source_raster._disk_transform
            ).crop(*source_raster._disk_shape[1:])
            final_window_disk = ref_win_disk.intersection(self_win_disk).round_lengths().round_offsets()

            # Round up to downsampling size, to match __init__
            final_window_disk = rio.windows.round_window_to_full_blocks(
                final_window_disk, ((source_raster._downsample, source_raster._downsample),)
            )

            # Load data for "on_disk" window but out_shape matching in-memory transform -> enforce downsampling
            # AD (24/04/24): Note that the same issue as #447 occurs here when final_window_disk extends beyond
            # self's bounds. Using option `boundless=True` solves the issue but causes other tests to fail
            # This should be fixed with #447 and previous line would be obsolete.
            with rio.open(source_raster.filename) as raster:
                crop_img = raster.read(
                    indexes=source_raster._bands,
                    masked=source_raster._masked,
                    window=final_window_disk,
                    out_shape=(final_window.height, final_window.width),
                )

            # Squeeze first axis for single-band
            if crop_img.ndim == 3 and crop_img.shape[0] == 1:
                crop_img = crop_img.squeeze(axis=0)

    else:
        bbox = rio.coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)
        out_rst = source_raster.reproject(bounds=bbox)  # should we instead raise an issue and point to reproject?
        crop_img = out_rst.data
        tfm = out_rst.transform

    return crop_img, tfm


##############
# 3/ TRANSLATE
##############


def _translate(
    transform: affine.Affine,
    xoff: float,
    yoff: float,
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> affine.Affine:
    """
    Translate geotransform horizontally, either in pixels or georeferenced units.

    :param transform: Input geotransform.
    :param xoff: Translation x offset.
    :param yoff: Translation y offset.
    :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.

    :return: Translated transform.
    """

    if distance_unit not in ["georeferenced", "pixel"]:
        raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

    # Get transform
    dx, b, xmin, d, dy, ymax = list(transform)[:6]

    # Convert pixel offsets to georeferenced units
    if distance_unit == "pixel":
        xoff *= dx
        yoff *= abs(dy)  # dy is negative

    return rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)
