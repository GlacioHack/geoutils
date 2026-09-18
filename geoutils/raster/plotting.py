# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raster plotting functionalities with automatic downsampling for performance."""

from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import rasterio as rio
from pyproj import CRS
from rasterio.enums import Resampling

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array
from geoutils._misc import import_optional
from geoutils.raster.referencing import _default_nodata

if TYPE_CHECKING:
    import matplotlib

    from geoutils.raster.base import RasterBase


def _create_axes(ax: matplotlib.axes.Axes | Literal["new"] | None) -> matplotlib.axes.Axes:
    """Return the requested Matplotlib axes, creating them when needed."""

    matplotlib = import_optional("matplotlib")
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.gca()
    if isinstance(ax, str) and ax.lower() == "new":
        _, new_axes = plt.subplots()
        return new_axes
    if isinstance(ax, matplotlib.axes.Axes):
        return ax
    raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")


def _resolve_target_crs(source: RasterBase, ref_crs: Any) -> CRS | None:
    """Resolve the display CRS from an explicit CRS or georeferenced reference object."""

    if ref_crs is None:
        if source.crs is None:
            return None
        return CRS.from_user_input(source.crs)
    if has_geo_attr(ref_crs, "crs"):
        return CRS.from_user_input(get_geo_attr(ref_crs, "crs"))
    return CRS.from_user_input(ref_crs)


def _default_grid_size(source: RasterBase, target_crs: CRS | None) -> tuple[int, int]:
    """Return the native-sized grid width and height in the display CRS."""

    if target_crs is None:
        return source.width, source.height
    if source.crs is None:
        raise ValueError("A raster without a CRS cannot be plotted in a reference CRS.")
    if CRS.from_user_input(source.crs) == target_crs:
        return source.width, source.height

    _, width, height = rio.warp.calculate_default_transform(
        source.crs,
        target_crs,
        source.width,
        source.height,
        *source.bounds,
    )
    return width, height


def _display_grid_size(
    width: int,
    height: int,
    max_pixels: Literal["auto"] | int | None,
    ax: matplotlib.axes.Axes,
) -> tuple[int, int]:
    """Limit a raster grid to the axes dimensions or an explicit total pixel count."""

    if max_pixels is None:
        return width, height

    if max_pixels == "auto":
        axes_bounds = ax.get_window_extent()
        scale = min(1.0, axes_bounds.width / width, axes_bounds.height / height)
    elif isinstance(max_pixels, int) and not isinstance(max_pixels, bool) and max_pixels > 0:
        scale = min(1.0, np.sqrt(max_pixels / (width * height)))
    else:
        raise ValueError("max_pixels must be 'auto', a strictly positive integer or None.")

    target_width = max(1, int(np.floor(width * scale)))
    target_height = max(1, int(np.floor(height * scale)))

    # Keep rounding from exceeding an explicit total budget
    if isinstance(max_pixels, int):
        while target_width * target_height > max_pixels:
            if target_width >= target_height and target_width > 1:
                target_width -= 1
            elif target_height > 1:
                target_height -= 1
            else:
                break
    return target_width, target_height


def _prepare_display_raster(
    source: RasterBase,
    ax: matplotlib.axes.Axes,
    max_pixels: Literal["auto"] | int | None,
    ref_crs: Any,
    resampling: Resampling | str | None,
) -> RasterBase:
    """
    Reproject a raster to the requested display CRS and pixel budget.

    _resolve_target_crs() selects the plot CRS, _default_grid_size() derives the equivalent native grid, and
    _display_grid_size() limits that grid to the rendered axes. The existing reproject() implementation then keeps
    Dask-backed inputs lazy until the small display array is requested.
    """

    target_crs = _resolve_target_crs(source, ref_crs)
    native_width, native_height = _default_grid_size(source, target_crs)
    target_width, target_height = _display_grid_size(native_width, native_height, max_pixels, ax)

    # Keep the source object when its grid already matches the requested display
    same_crs = source.crs is None and target_crs is None
    if source.crs is not None and target_crs is not None:
        same_crs = CRS.from_user_input(source.crs) == target_crs
    if same_crs and (target_width, target_height) == (source.width, source.height):
        return source
    if target_crs is None:
        return source

    # Use a floating display grid when integer accessors need NaN to represent missing output cells
    display_dtype = "float32" if np.issubdtype(source.dtype, np.integer) and not source.is_mask else None
    output_dtype = source.dtype if display_dtype is None else display_dtype
    display_nodata = source.nodata if source.nodata is not None else _default_nodata(output_dtype)
    display_resampling = Resampling.nearest if source.is_mask and resampling is None else resampling
    reprojected = source.reproject(
        crs=target_crs,
        grid_size=(target_width, target_height),
        nodata=display_nodata,
        resampling=display_resampling,
        dtype=display_dtype,
        silent=True,
    )
    if hasattr(reprojected, "rst"):
        return reprojected.rst
    return reprojected


def _plot_raster(
    source: RasterBase,
    bands: int | tuple[int, ...] | None = None,
    cmap: matplotlib.colors.Colormap | str | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
    alpha: float | int | None = None,
    title: str | None = None,
    cbar_title: str | None = None,
    add_cbar: bool = True,
    ax: matplotlib.axes.Axes | Literal["new"] | None = None,
    ref_crs: Any = None,
    max_pixels: Literal["auto"] | int | None = "auto",
    resampling: Resampling | str | None = None,
    return_axes: bool = False,
    savefig_fname: str | None = None,
    **kwargs: Any,
) -> None | tuple[matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
    """
    Prepare a bounded raster array and draw it with Matplotlib.

    _create_axes() establishes the rendering dimensions before _prepare_display_raster() reprojects to that bounded
    grid. The selected display bands are then computed, arranged for single-band or RGB(A) display, and passed to
    Matplotlib together with the projected extent and optional colorbar.
    """

    matplotlib = import_optional("matplotlib")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create the axes before calculating their display pixel budget
    ax0 = _create_axes(ax)

    # Check the requested bands before performing any reprojection work
    if bands is None or isinstance(bands, tuple):
        if bands is None:
            bands = tuple(range(1, source.count + 1))
        if len(bands) not in [1, 3, 4]:
            raise ValueError(
                f"Only single-band or 3/4-band (RGB-A) plotting is supported. "
                f"Found {len(bands)} bands. Use the `bands` argument to specify bands."
            )
        if len(bands) == 1:
            bands = bands[0]
    elif isinstance(bands, int):
        if bands < 1 or bands > source.count:
            raise ValueError(f"Index must be in range 1-{source.count:d}")
    else:
        raise ValueError("Index must be int, tuple or None")

    # Reproject only to the resolution that can be displayed
    display = _prepare_display_raster(source, ax0, max_pixels, ref_crs, resampling)
    data = display.data if display.count == 1 else display.data[np.array(bands) - 1, :, :]
    if is_dask_array(data):
        data = cast(Any, data).compute()

    # Reorder RGB(A) bands for imshow and omit their colorbar
    if isinstance(bands, abc.Sequence):
        if len(bands) > 1:
            add_cbar = False
            source_dtype = getattr(source, "_disk_dtype", None)
            if source._is_xr and source._obj is not None:
                source_dtype = source._obj.encoding.get("rasterio_dtype", source_dtype)
            if source_dtype is None:
                source_dtype = source.dtype
            if source_dtype is not None and np.issubdtype(np.dtype(source_dtype), np.integer):
                data = np.ma.masked_invalid(data).astype(source_dtype)
            elif np.nanmin(data) >= 0 and 1 < np.nanmax(data) <= 255:
                # Convert byte-scaled floating RGB(A) values to Matplotlib's zero-to-one convention
                data = data / 255
        data = np.moveaxis(data, 0, -1)

    # Use Matplotlib defaults unless the caller selected a colormap
    if cmap is None:
        cmap = plt.get_cmap(plt.rcParams["image.cmap"])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Derive color limits from the bounded array that is actually drawn
    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    try:
        vmin = float(vmin)
        vmax = float(vmax)
    except (TypeError, ValueError):
        raise ValueError("vmin or vmax cannot be converted to float") from None

    # Draw the display array in projected coordinates
    if "interpolation" not in kwargs:
        kwargs["interpolation"] = None
    extent = [display.bounds.left, display.bounds.right, display.bounds.bottom, display.bounds.top]
    ax0.imshow(
        np.flip(data, axis=0),
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        **kwargs,
    )
    if title is not None:
        ax0.set_title(title)

    # Add a colorbar beside single-band plots
    cax = None
    if add_cbar:
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad="2%")
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.solids.set_alpha(alpha)
        if cbar_title is not None:
            cbar.set_label(cbar_title)

    plt.sca(ax0)
    plt.tight_layout()
    if savefig_fname:
        plt.savefig(savefig_fname)
    if return_axes:
        return ax0, cax
    return None
