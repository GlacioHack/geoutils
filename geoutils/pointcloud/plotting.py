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
from geoutils._misc import import_optional
from geoutils.projtools import _get_bounds_projected

if TYPE_CHECKING:
    import matplotlib

    from geoutils.pointcloud.base import PointCloudBase


_AUTO_MAX_POINTS = 100_000


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
    ref_crs: Any,
) -> tuple[PointCloudBase, Any]:
    """
    Select a bounded point sample and reproject it for plotting.

    _display_point_count() translates the axes size to a point limit, subsample() selects the same rows for eager and
    chunked inputs, and reproject() changes only the temporary sample. The returned bounds cover the complete source
    so sampling does not crop sparse edge points from the axes.
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
    if ref_crs is not None:
        if source_crs is None:
            raise ValueError("A point cloud without a CRS cannot be plotted in a reference CRS.")
        if has_geo_attr(ref_crs, "crs"):
            target_crs = CRS.from_user_input(get_geo_attr(ref_crs, "crs"))
            reprojected = display.reproject(ref=ref_crs)
        else:
            target_crs = CRS.from_user_input(ref_crs)
            reprojected = display.reproject(crs=target_crs)
        display = _get_pointcloud_interface(reprojected)

    display_bounds = source.bounds
    if source_crs != target_crs:
        display_bounds = _get_bounds_projected(source.bounds, source.crs, target_crs)
    return display, display_bounds


def _plot_pointcloud(
    source: PointCloudBase,
    column: str | None = None,
    ref_crs: Any = None,
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
    Prepare a bounded point sample and draw it with GeoPandas and Matplotlib.

    _create_axes() establishes the rendering dimensions before _prepare_display_pointcloud() selects and reprojects
    the temporary point rows. Only that sample is computed for GeoPandas plotting, while the full source bounds keep
    the axes extent representative of every input point.
    """

    matplotlib = import_optional("matplotlib")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create axes before translating their size to an automatic point limit
    ax0 = _create_axes(ax)
    display, display_bounds = _prepare_display_pointcloud(source, ax0, max_points, random_state, ref_crs)
    dataframe = display.ds.compute() if is_dask_dataframe(display.ds) else display.ds

    if column is None:
        column = source.data_column

    # Keep the existing colorbar controls used by PointCloud.plot()
    legend = bool(add_cbar)
    if "legend" in kwargs:
        legend = kwargs.pop("legend")
    if "legend_kwds" in kwargs and legend:
        legend_kwds = kwargs.pop("legend_kwds")
        if cbar_title is not None:
            legend_kwds.update({"label": cbar_title})
    elif cbar_title is not None:
        legend_kwds = {"label": cbar_title}
    else:
        legend_kwds = None

    # Add the separate GeoUtils colorbar when requested
    cax = None
    if add_cbar or cbar_title:
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="5%", pad="2%")
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cbar.solids.set_alpha(alpha)

    # Plot the selected points and retain the complete source extent
    dataframe.plot(
        ax=ax0,
        cax=cax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        legend=legend,
        legend_kwds=legend_kwds,
        **kwargs,
    )
    ax0.update_datalim(
        np.array(
            [
                [display_bounds.left, display_bounds.bottom],
                [display_bounds.right, display_bounds.top],
            ]
        )
    )
    ax0.autoscale_view()
    plt.sca(ax0)

    if savefig_fname:
        plt.savefig(savefig_fname)
    if return_axes:
        return ax0, cax
    return None
