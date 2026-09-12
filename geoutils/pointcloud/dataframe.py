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

"""Module for dataframe operations: managing values, row selection, metadata and output construction."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import geopandas as gpd
import numpy as np
import pandas as pd

from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloudLike


DataFrameType = TypeVar("DataFrameType")


############################################
# 1/ METADATA AND POINT CLOUD OUTPUTS
############################################


def _get_dataframe_attrs(ds: Any) -> dict[str, Any]:
    """Get GeoUtils metadata from Pandas or Dask dataframes."""

    # Dask does not carry Pandas ``attrs`` reliably through graph operations
    if is_dask_dataframe(ds):
        try:
            return object.__getattribute__(ds, "_geoutils_attrs")
        except AttributeError:
            return {}
    return getattr(ds, "attrs", {})


def _set_dataframe_attrs(ds: Any, attrs: dict[str, Any]) -> None:
    """Set GeoUtils metadata on Pandas or Dask dataframes."""

    # Keep a private copy on Dask collections and use the public mapping for Pandas
    if is_dask_dataframe(ds):
        object.__setattr__(ds, "_geoutils_attrs", attrs.copy())
    elif hasattr(ds, "attrs"):
        ds.attrs.update(attrs)


@overload
def _build_pointcloud_output(
    dataframe: DataFrameType,
    *,
    data_column: str | None,
    as_dataframe: Literal[True],
    attrs: Mapping[str, Any] | None = None,
    preserve_locations: bool = False,
) -> DataFrameType: ...


@overload
def _build_pointcloud_output(
    dataframe: Any,
    *,
    data_column: str | None,
    as_dataframe: bool,
    attrs: Mapping[str, Any] | None = None,
    preserve_locations: bool = False,
) -> PointCloudLike: ...


def _build_pointcloud_output(
    dataframe: Any,
    *,
    data_column: str | None,
    as_dataframe: bool,
    attrs: Mapping[str, Any] | None = None,
    preserve_locations: bool = False,
) -> PointCloudLike:
    """
    Build a point cloud result with metadata matching its current rows and coordinate system.

    _set_dataframe_attrs() records the active value column, CRS and point geometry type without changing supplied
    metadata. Row selections may change counts and bounds, so bounds and lazy counts are cleared unless
    preserve_locations is True, meaning the result has the same ordered points and unchanged X/Y coordinates.
    Eager results always receive a fresh count. Return the eager or Dask dataframe when as_dataframe is True;
    otherwise compute Dask rows before constructing a PointCloud.
    """

    # Copy supplied metadata and load Dask rows only when the caller needs a PointCloud object
    metadata = {} if attrs is None else dict(attrs)
    if not as_dataframe and is_dask_dataframe(dataframe):
        dataframe = dataframe.compute()

    # Reuse spatial metadata only when the caller guarantees unchanged point rows and X/Y coordinates
    point_count = metadata.get("point_count") if preserve_locations else None
    if not is_dask_dataframe(dataframe):
        point_count = len(dataframe)
    bounds = metadata.get("bounds") if preserve_locations else None
    metadata.update(
        data_column=data_column,
        geometry_type="Point",
        crs=dataframe.crs,
        point_count=point_count,
        bounds=bounds,
    )
    _set_dataframe_attrs(dataframe, metadata)

    # Accessors return dataframes; object callers receive an eager PointCloud with the selected value column
    if as_dataframe:
        return dataframe

    from geoutils.pointcloud.pointcloud import PointCloud

    return PointCloud(dataframe, data_column=data_column)


############################################
# 2/ POINT DATAFRAME PARTITIONS
############################################


def _import_dask_dataframe() -> Any:
    """Import Dask DataFrame while suppressing optional dask-expr warnings from older environments."""

    # Delay the optional import until a lazy dataframe operation is requested
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        import dask.dataframe as dd

    return dd


def _point_partition_lengths(dataframe: Any) -> tuple[int, ...]:
    """
    Compute the row count of each point partition without collecting its rows or attached values.

    Dataframe follows _assign_point_values(). The caller can reuse these counts while the row layout stays unchanged.

    :returns: One integer row count per partition, in partition order.
    """

    return tuple(int(length) for length in dataframe.geometry.map_partitions(len).compute())


def _point_array_partitions(
    dataframe: Any, arrays: Sequence[Any], *, partition_lengths: Sequence[int] | None = None
) -> list[Any]:
    """
    Align one-dimensional value arrays with the row partitions of a lazy point dataframe.

    Dataframe, arrays and partition_lengths follow _assign_point_values(). Only partition lengths are collected here;
    the point rows and corresponding values remain lazy. Dask Series keep each partition's index, including duplicates.
    """

    import_optional("dask")
    import dask.array as da

    # Count rows per partition so array chunks can be matched by position, independently of index labels
    lengths = _point_partition_lengths(dataframe) if partition_lengths is None else tuple(partition_lengths)
    if len(lengths) != dataframe.npartitions or any(length < 0 for length in lengths):
        raise ValueError("Point partition lengths must give one nonnegative row count per dataframe partition.")
    total = sum(lengths)
    dd = _import_dask_dataframe()
    columns = []
    for values in arrays:
        array = da.asarray(values)
        if any(np.isnan(length) for length in array.shape):
            array = array.compute_chunk_sizes()
        if array.ndim != 1 or array.shape[0] != total:
            raise ValueError("Point values must contain one value per dataframe row.")

        # Keep every empty point partition; Dask otherwise collapses a zero-length array to one block
        if total == 0:
            columns.append(dataframe.index.to_series().astype(array.dtype))
            continue

        # Give each value partition the exact index of its corresponding point partition
        aligned = array.rechunk((lengths,))
        columns.append(dd.from_dask_array(aligned, index=dataframe.index))
    return columns


############################################
# 3/ ASSIGN VALUES BY POINT POSITION
############################################


def _assign_point_partition(dataframe: gpd.GeoDataFrame, *columns: Any, names: Sequence[str]) -> gpd.GeoDataFrame:
    """Assign the matching value partitions prepared by _assign_point_values(), ignoring their index labels."""

    # Assign directly so user column names such as self do not collide with dataframe.assign() arguments
    result = dataframe.copy()
    for name, column in zip(names, columns):
        array = np.asarray(column)
        if array.ndim != 1 or len(array) != len(dataframe):
            raise ValueError("Point value partitions must contain one value per dataframe row.")
        result[name] = array
    return result


def _assign_point_values(
    dataframe: Any, values: Mapping[str, Any], *, partition_lengths: Sequence[int] | None = None
) -> Any:
    """
    Attach named values to point rows by position while keeping the dataframe's eager or lazy backend.

    _point_array_partitions() matches lazy values to point partitions. _assign_point_partition() assigns each
    partition by position, avoiding index alignment when several points have the same row label.
    Dask Series already following the point partitions pass through without computing their lengths.

    :param dataframe: GeoDataFrame or Dask-GeoPandas dataframe whose ordered point rows define the value locations.
    :param values: Mapping of column names to one-dimensional NumPy or Dask arrays, with one value per point row.
        Dask Series may also be passed when their partitions contain the same ordered rows as the dataframe.
    :param partition_lengths: Optional known row counts for the current dataframe partitions, reused when aligning
        array inputs. These counts must be updated after selecting rows. Matching Dask Series do not need them.
    :returns: A dataframe with the named columns assigned and its existing columns, point order and labels preserved.
    """

    # Keep the original table when there are no additional values to attach
    if not values:
        return dataframe

    # Preserve eager output even when a selected value array was computed lazily
    if not is_dask_dataframe(dataframe):
        columns = list(values.values())
        if any(is_dask_array(array) or is_dask_dataframe(array) for array in columns):
            import_optional("dask")
            import dask

            columns = list(dask.compute(*columns))
        return _assign_point_partition(dataframe, *columns, names=list(values))
    # Use point Series directly and align only array inputs, sharing one partition count for every array
    array_names = [name for name, value in values.items() if not is_dask_dataframe(value)]
    aligned = {}
    if array_names:
        arrays = _point_array_partitions(
            dataframe, [values[name] for name in array_names], partition_lengths=partition_lengths
        )
        aligned = dict(zip(array_names, arrays))
    columns = []
    for name, value in values.items():
        if is_dask_dataframe(value):
            if value.ndim != 1 or value.npartitions != dataframe.npartitions:
                raise ValueError("Point value Series must follow the dataframe's partition layout.")
            columns.append(value)
        else:
            columns.append(aligned[name])

    # Attach columns positionally within each corresponding partition and preserve their numeric metadata
    meta = dataframe._meta.copy()
    for name, column in zip(values, columns):
        meta[name] = pd.Series([], dtype=column.dtype)
    return dataframe.map_partitions(_assign_point_partition, *columns, names=list(values), meta=meta)


############################################
# 4/ SELECT POINT ROWS
############################################


def _select_point_partition(
    dataframe: gpd.GeoDataFrame,
    indices: Any,
    *,
    starts: Any = None,
    partition_info: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """
    Select point rows from one partition using the mask or global positions supplied to _select_point_rows().

    :param starts: Cumulative partition row counts for integer selection, or None for a matching boolean mask.
    :param partition_info: Dask partition metadata identifying which global row offset to use.
    """

    # Boolean masks already match this partition; integer indices refer to the full ordered dataframe
    if starts is None:
        mask = np.asarray(indices)
        if mask.ndim != 1 or len(mask) != len(dataframe):
            raise ValueError("Point mask partitions must contain one value per dataframe row.")
        return dataframe.iloc[mask]
    assert partition_info is not None
    start, stop = starts[partition_info["number"] : partition_info["number"] + 2]
    lower, upper = np.searchsorted(indices, (start, stop))
    return dataframe.iloc[indices[lower:upper] - start]


def _select_point_rows(dataframe: Any, indices: Any, *, partition_lengths: Sequence[int] | None = None) -> Any:
    """
    Select point rows by position without collecting a complete lazy point dataframe.

    _point_array_partitions() aligns boolean masks with point partitions. For integer positions, partition lengths
    locate the requested rows and _select_point_partition() selects them with iloc(), independently of row labels.

    Dataframe and partition_lengths follow _assign_point_values().

    :param indices: One-dimensional boolean NumPy or Dask mask, or sorted integer NumPy positions to keep.
        A boolean Dask Series must follow the dataframe's partition layout and avoids computing partition lengths.
    :returns: Selected rows in their original order, preserving the dataframe backend, geometry and index labels.
    """

    # Eager point supports keep eager results, including when their eligibility mask was generated lazily
    if not is_dask_dataframe(dataframe):
        selected = indices.compute() if is_dask_array(indices) or is_dask_dataframe(indices) else indices
        return dataframe.iloc[np.asarray(selected)]

    # Match boolean mask partitions without collecting the full eligibility array
    if indices.dtype == np.bool_:
        if is_dask_dataframe(indices):
            if indices.ndim != 1 or indices.npartitions != dataframe.npartitions:
                raise ValueError("Point mask Series must follow the dataframe's partition layout.")
            mask = indices
        else:
            (mask,) = _point_array_partitions(dataframe, [indices], partition_lengths=partition_lengths)
        return dataframe.map_partitions(_select_point_partition, mask, meta=dataframe._meta)

    # Translate global positions into local rows using only small partition-length summaries
    positions = np.asarray(indices, dtype=np.int64)
    if positions.ndim != 1 or np.any(positions[1:] < positions[:-1]):
        raise ValueError("Point row positions must be a sorted one-dimensional integer array.")
    lengths = _point_partition_lengths(dataframe) if partition_lengths is None else tuple(partition_lengths)
    if len(lengths) != dataframe.npartitions or any(length < 0 for length in lengths):
        raise ValueError("Point partition lengths must give one nonnegative row count per dataframe partition.")
    starts = np.concatenate(([0], np.cumsum(lengths)))
    if len(positions) and (positions[0] < 0 or positions[-1] >= starts[-1]):
        raise IndexError("Point row position is outside the dataframe.")
    return dataframe.map_partitions(_select_point_partition, positions, starts=starts, meta=dataframe._meta)
