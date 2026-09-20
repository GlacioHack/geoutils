# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draw GeoDataFrames with consistent axes and colorbar placement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import rasterio as rio

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_dataframe
from geoutils._misc import import_optional

if TYPE_CHECKING:
    import matplotlib


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


def _get_reference_bounds(reference: Any) -> rio.coords.BoundingBox | None:
    """Return the total bounds of a georeferenced plotting reference."""

    if reference is None or not has_geo_attr(reference, "bounds"):
        return None
    if isinstance(reference, (gpd.GeoDataFrame, gpd.GeoSeries)):
        bounds = reference.total_bounds
    else:
        bounds = get_geo_attr(reference, "bounds")
    if is_dask_dataframe(bounds):
        bounds = bounds.compute()
    if isinstance(bounds, pd.DataFrame):
        bounds = (bounds.minx.min(), bounds.miny.min(), bounds.maxx.max(), bounds.maxy.max())
    return rio.coords.BoundingBox(*bounds)


def _plot_geodataframe(
    dataframe: gpd.GeoDataFrame,
    ax: matplotlib.axes.Axes,
    column: str | None,
    cmap: matplotlib.colors.Colormap | str | None,
    vmin: float | int | None,
    vmax: float | int | None,
    alpha: float | int | None,
    cbar_title: str | None,
    add_cbar: bool,
    **kwargs: Any,
) -> matplotlib.axes.Axes | None:
    """
    Plot a GeoDataFrame and return the continuous colorbar axes when one is created.

    GeoPandas creates its colorbar after setting the map aspect, which keeps the colorbar next to geographic plots.
    The new figure axes are captured so Vector.plot() and PointCloud.plot() can retain their return-axes behavior.
    """

    # Prepare GeoPandas legend options without changing a dictionary supplied by the caller
    legend = bool(add_cbar and column is not None)
    if "legend" in kwargs:
        legend = bool(kwargs.pop("legend"))
    supplied_legend_kwds = kwargs.pop("legend_kwds", None)
    legend_kwds = dict(supplied_legend_kwds) if supplied_legend_kwds is not None else {}
    if cbar_title is not None:
        legend_kwds["label"] = cbar_title

    # Use a two-percent continuous colorbar gap while leaving categorical legend options untouched
    continuous = (
        legend
        and column is not None
        and column in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
        and not kwargs.get("categorical", False)
        and kwargs.get("scheme") is None
    )
    if continuous:
        legend_kwds.setdefault("pad", 0.02)

    # Let GeoPandas create the colorbar against the final map aspect unless explicit axes were supplied
    cax = kwargs.pop("cax", None)
    previous_axes = list(ax.figure.axes)
    dataframe.plot(
        ax=ax,
        cax=cax,
        column=column,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        legend=legend,
        legend_kwds=legend_kwds if legend else None,
        **kwargs,
    )
    if cax is None and continuous:
        added_axes = [figure_axes for figure_axes in ax.figure.axes if figure_axes not in previous_axes]
        cax = added_axes[-1] if added_axes else None

    # Apply the requested transparency to a continuous colorbar created by GeoPandas
    if cax is not None and alpha is not None:
        for collection in cax.collections:
            collection.set_alpha(alpha)
    return cax
