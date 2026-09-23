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

"""Point cloud plotting functionalities with automated downsampling for performance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pyproj import CRS

from geoutils._dispatch import (
    _get_pointcloud_interface,
    get_geo_attr,
    has_geo_attr,
    is_dask_dataframe,
)
from geoutils.projtools import _get_bounds_projected
from geoutils.vector.plotting import (
    _create_axes,
    _get_reference_bbox,
    _plot_geodataframe,
    _reduce_tick_label_overlap,
)

if TYPE_CHECKING:
    import matplotlib

    from geoutils.pointcloud.base import PointCloudBase


_AUTO_MAX_POINTS = 100_000


def _display_point_count(
    max_points: Literal["auto"] | int | None,
    ax: matplotlib.axes.Axes,
) -> int | None:
    """Resolve an automatic or explicit maximum number of displayed points."""

    if max_points is None:
        return None
    if max_points == "auto":
        axes_bounds = ax.get_window_extent()
        return max(1, min(int(axes_bounds.width * axes_bounds.height), _AUTO_MAX_POINTS))
    if isinstance(max_points, int) and not isinstance(max_points, bool) and max_points > 0:
        return max_points
    raise ValueError("max_points must be 'auto', a strictly positive integer or None.")


def _subsample_request(point_count: int, max_points: int) -> int | float:
    """Convert a point limit to the existing subsample() count or fraction convention."""

    if max_points > 1:
        return max_points
    return 1 / point_count


def _prepare_display_pointcloud(
    source: PointCloudBase,
    ax: matplotlib.axes.Axes,
    max_points: Literal["auto"] | int | None,
    random_state: int | np.random.Generator | None,
    ref: Any,
) -> tuple[PointCloudBase, Any, bool]:
    """
    Select a point subsample and reproject it for plotting.

    Internal behaviour:
    - _display_point_count() translates the axes size to a point limit,
    - subsample() selects the same rows for eager and chunked inputs, and
    - reproject() changes only the CRS of the subsample, if necessary.

    The returned bounding box is the one from the full source before subsampling to avoid over-cropping.
    """

    point_limit = _display_point_count(max_points, ax)
    display: Any = source
    if point_limit is not None and source.point_count > point_limit:
        display = source.subsample(
            _subsample_request(source.point_count, point_limit),
            random_state=random_state,
        )
    display = _get_pointcloud_interface(display)

    source_crs = None if source.crs is None else CRS.from_user_input(source.crs)
    target_crs = source_crs
    reference_bbox = None
    match_reference_extent = False
    if ref is not None:
        if source_crs is None:
            raise ValueError("A point cloud without a CRS cannot be plotted in a reference CRS.")
        if has_geo_attr(ref, "crs"):
            target_crs = CRS.from_user_input(get_geo_attr(ref, "crs"))
            reprojected = display.reproject(ref=ref)
            reference_bbox = _get_reference_bbox(ref)
            match_reference_extent = reference_bbox is not None
        else:
            target_crs = CRS.from_user_input(ref)
            reprojected = display.reproject(crs=target_crs)
        display = _get_pointcloud_interface(reprojected)

    if reference_bbox is not None:
        display_bbox = reference_bbox
    elif source_crs != target_crs:
        display_bbox = _get_bounds_projected(source.bbox, source.crs, target_crs)
    else:
        display_bbox = source.bbox
    return display, display_bbox, match_reference_extent


def _plot_pointcloud(
    source: PointCloudBase,
    column: str | None = None,
    ref: Any = None,
    cmap: matplotlib.colors.Colormap | str | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
    alpha: float | int | None = None,
    cbar_title: str | None = None,
    add_cbar: bool = True,
    ax: matplotlib.axes.Axes | Literal["new"] | None = None,
    max_points: Literal["auto"] | int | None = "auto",
    random_state: int | np.random.Generator | None = 0,
    return_axes: bool = False,
    savefig_fname: str | None = None,
    **kwargs: Any,
) -> None | tuple[matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
    """
    Prepare an optionally downsampled raster array and draw it with GeoPandas/Matplotlib.

    _create_axes() establishes the rendering dimensions before _prepare_display_pointcloud() selects and reprojects
    the temporary point rows. Only that sample is computed for GeoPandas plotting, while the full source bounding box
    keeps the axes extent representative of every input point.
    """

    import matplotlib.pyplot as plt

    # Create axes, then we estimate the number of points displayed from their size
    ax0 = _create_axes(ax)
    display, display_bbox, match_reference_extent = _prepare_display_pointcloud(
        source, ax0, max_points, random_state, ref
    )
    dataframe = display.ds.compute() if is_dask_dataframe(display.ds) else display.ds

    # Use the main data column unless the caller supplied one fixed color for every point
    if column is None and "color" not in kwargs:
        column = source.data_column

    # We plot after GeoPandas sets the map aspect so geographic colorbars remain next to the data axes
    cax = _plot_geodataframe(
        dataframe=dataframe,
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

    # Use the source bounding box by default (for downsampled points), or use the complete
    # reference extent when one was passed as input
    if match_reference_extent:
        ax0.set_xlim(display_bbox.left, display_bbox.right)
        ax0.set_ylim(display_bbox.bottom, display_bbox.top)
    else:
        ax0.update_datalim(
            np.array(
                [
                    [display_bbox.left, display_bbox.bottom],
                    [display_bbox.right, display_bbox.top],
                ]
            )
        )
        ax0.autoscale_view()
    plt.sca(ax0)
    _reduce_tick_label_overlap(ax0)

    if savefig_fname:
        plt.savefig(savefig_fname)
    if return_axes:
        return ax0, cax
    return None
