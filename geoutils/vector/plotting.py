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


def _create_colorbar_axes(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    """Create a colorbar axes beside a map without changing the map axes size."""

    return ax.inset_axes((1.02, 0, 0.05, 1))


def _reduce_tick_label_overlap(ax: matplotlib.axes.Axes) -> None:
    """Limit automatic axes to a few evenly spaced ticks."""

    from matplotlib.ticker import AutoLocator, MaxNLocator

    # Use sparse ticks while preserving caller-supplied locators
    for axis in (ax.xaxis, ax.yaxis):
        if isinstance(axis.get_major_locator(), AutoLocator):
            axis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))


def _get_reference_bbox(reference: Any) -> rio.coords.BoundingBox | None:
    """Return the total bounding box of a georeferenced plotting reference."""

    if reference is None:
        return None
    if isinstance(reference, (gpd.GeoDataFrame, gpd.GeoSeries)):
        bbox = reference.total_bounds
    else:
        if not (has_geo_attr(reference, "bbox") or has_geo_attr(reference, "bounds")):
            return None
        bbox_attr = "bbox" if has_geo_attr(reference, "bbox") else "bounds"
        bbox = get_geo_attr(reference, bbox_attr)
    if is_dask_dataframe(bbox):
        bbox = bbox.compute()
    if isinstance(bbox, pd.DataFrame):
        bbox = (bbox.minx.min(), bbox.miny.min(), bbox.maxx.max(), bbox.maxy.max())
    return rio.coords.BoundingBox(*bbox)


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

    The default continuous colorbar uses axes-relative bounds so its height and gap follow the map axes when their
    extent changes. It does not resize the map axes. An explicitly supplied colorbar axes is kept unchanged.
    """

    # Prepare GeoPandas legend options without changing a dictionary supplied by the caller
    legend = bool(add_cbar and column is not None)
    if "legend" in kwargs:
        legend = bool(kwargs.pop("legend"))
    supplied_legend_kwds = kwargs.pop("legend_kwds", None)
    legend_kwds = dict(supplied_legend_kwds) if supplied_legend_kwds is not None else {}
    if cbar_title is not None:
        legend_kwds["label"] = cbar_title

    # Identify continuous colorbars while leaving categorical legend options untouched
    continuous = (
        legend
        and column is not None
        and column in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[column])
        and not kwargs.get("categorical", False)
        and kwargs.get("scheme") is None
    )

    # Attach the bar to the map without changing the space allocated to the map axes
    cax = kwargs.pop("cax", None)
    if cax is None and continuous:
        cax = _create_colorbar_axes(ax)

    # Keep boolean data continuous for plotting so masks use a colorbar instead of GeoPandas' categorical legend
    plot_dataframe = dataframe
    if column is not None and column in dataframe.columns and pd.api.types.is_bool_dtype(dataframe[column]):
        plot_dataframe = dataframe.assign(**{column: dataframe[column].astype("uint8")})

    # Draw the geometries and let GeoPandas populate the prepared colorbar or a categorical legend
    plot_dataframe.plot(
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

    # Apply the requested transparency to a continuous colorbar created by GeoPandas
    if cax is not None and alpha is not None:
        for collection in cax.collections:
            collection.set_alpha(alpha)
    return cax
