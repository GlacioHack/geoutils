# Copyright (c) 2025 GeoUtils developers
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

"""Functionalities for geotransformations of vectors."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
import pyogrio
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from geoutils._dispatch import (
    _check_match_bbox,
    _clip_geometry,
    _get_reproject_crs,
    is_dask_dataframe,
)

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterLike
    from geoutils.vector.base import VectorBase
    from geoutils.vector.vector import Vector, VectorLike


#########
# 1/ CROP
#########


def _crop_geodataframe(
    ds: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    mode: Literal["intersects", "within"],
) -> gpd.GeoDataFrame:
    """Select unchanged geometries that intersect or lie within bounding coordinates."""

    xmin, ymin, xmax, ymax = bounds
    cropped = ds.cx[xmin:xmax, ymin:ymax]  # type: ignore[misc]
    if mode == "within":
        cropped = cropped[cropped.geometry.within(box(*bounds))]
    return cropped


def _crop(source_vector: Any, bbox: Any, mode: Literal["intersects", "within"]) -> Any:
    """Crop vector data, dispatching to eager or partitioned execution."""

    if mode not in ("intersects", "within"):
        raise ValueError("Argument 'mode' must be either 'intersects' or 'within'.")

    # Convert references and coordinate sequences to one box in the source CRS
    xmin, ymin, xmax, ymax = (float(value) for value in _check_match_bbox(source_vector, bbox))
    bounds = (xmin, ymin, xmax, ymax)
    if is_dask_dataframe(source_vector.ds):
        # Spatial partitions can discard unrelated partitions before reading their rows
        if getattr(source_vector.ds, "spatial_partitions", None) is not None:
            cropped = source_vector.ds.cx[xmin:xmax, ymin:ymax]  # type: ignore[misc]
            if mode == "within":
                cropped = cropped[cropped.geometry.within(box(*bounds))]
            return cropped

        # Otherwise apply the same eager selection independently inside each partition
        return source_vector.ds.map_partitions(_crop_geodataframe, bounds, mode, meta=source_vector.ds._meta)
    return _crop_geodataframe(source_vector.ds, bounds=bounds, mode=mode)


def _crop_read_bbox(
    filters: Sequence[tuple[tuple[float, float, float, float], Literal["intersects", "within"]]],
) -> tuple[float, float, float, float] | None:
    """Return one conservative read box containing every deferred crop filter."""

    if len(filters) == 0:
        return None

    # Reading the union envelope preserves geometries that span several sequential crop boxes
    boxes = [bounds for bounds, _ in filters]
    return (
        min(bounds[0] for bounds in boxes),
        min(bounds[1] for bounds in boxes),
        max(bounds[2] for bounds in boxes),
        max(bounds[3] for bounds in boxes),
    )


def _apply_crop_filters(
    ds: gpd.GeoDataFrame,
    filters: Sequence[tuple[tuple[float, float, float, float], Literal["intersects", "within"]]],
) -> gpd.GeoDataFrame:
    """Apply deferred crop filters without changing the selected geometries."""

    cropped = ds
    for bounds, mode in filters:
        cropped = _crop_geodataframe(cropped, bounds=bounds, mode=mode)
    return cropped


#########
# 2/ CLIP
#########


def _clip_geodataframe(
    ds: gpd.GeoDataFrame,
    geometry: BaseGeometry,
    keep_geom_type: bool,
    sort: bool,
) -> gpd.GeoDataFrame:
    """Clip one GeoDataFrame or Dask partition to an exact geometry."""

    return ds.clip(mask=geometry, keep_geom_type=keep_geom_type, sort=sort)


def _clip(source_vector: Any, mask: Any, keep_geom_type: bool, sort: bool) -> Any:
    """Clip vector or point geometries exactly, preserving lazy source partitions."""

    target_crs = None if source_vector.crs is None else CRS.from_user_input(source_vector.crs)
    geometry = _clip_geometry(mask, target_crs=target_crs)

    # Build one clipping task per partition without computing any source rows
    if is_dask_dataframe(source_vector.ds):
        return source_vector.ds.map_partitions(
            _clip_geodataframe,
            geometry,
            keep_geom_type,
            sort,
            meta=source_vector.ds._meta,
        )
    return _clip_geodataframe(source_vector.ds, geometry=geometry, keep_geom_type=keep_geom_type, sort=sort)


def _vector_partition_size(mp_config: MultiprocConfig) -> int:
    """Return the number of vector features assigned to each multiprocessing task."""

    if not isinstance(mp_config.chunks, int):
        raise ValueError("Vector multiprocessing requires an integer chunk size.")
    return mp_config.chunks


def _clip_vector_partition(
    source: gpd.GeoDataFrame | pathlib.Path,
    start: int,
    count: int,
    crop_filters: Sequence[tuple[tuple[float, float, float, float], Literal["intersects", "within"]]],
    geometry: BaseGeometry,
    keep_geom_type: bool,
    sort: bool,
    filename: pathlib.Path,
) -> pathlib.Path:
    """Read and clip one vector row partition, then save it for ordered output construction."""

    # Read independent file ranges so an unloaded source stays outside the parent process
    if isinstance(source, pathlib.Path):
        dataframe = pyogrio.read_dataframe(source, skip_features=start, max_features=count)
        dataframe = _apply_crop_filters(dataframe, crop_filters)
    else:
        dataframe = source

    # Save exact GeoPandas values for the parent to append after workers finish
    clipped = _clip_geodataframe(
        dataframe,
        geometry=geometry,
        keep_geom_type=keep_geom_type,
        sort=sort,
    )
    clipped.to_pickle(filename)
    return filename


def _write_vector_partitions(
    filename: pathlib.Path,
    partition_filenames: Sequence[pathlib.Path],
) -> Vector:
    """Append saved vector partitions to one GeoPackage and return an unloaded Vector."""

    from geoutils.vector.vector import Vector

    if len(partition_filenames) == 0:
        raise ValueError("Vector output requires at least one saved partition.")
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Build the complete destination beside the requested path before replacing any existing file
    with tempfile.TemporaryDirectory(prefix=".geoutils-vector-output-", dir=filename.parent) as directory:
        temporary_output = pathlib.Path(directory) / "output.gpkg"
        output_created = False
        layer_options: dict[str, str] | None = None

        for partition_filename in partition_filenames:
            dataframe = pd.read_pickle(partition_filename)
            if layer_options is None:
                column_names = {str(column).lower() for column in dataframe.columns}
                layer_options = {"FID": "fid", "GEOMETRY_NAME": "geom"}
                for option, field_name in layer_options.items():
                    while field_name.lower() in column_names:
                        field_name = "_" + field_name
                    layer_options[option] = field_name

            # Create the layer from the first partition even when every clipped partition is empty
            if not output_created:
                pyogrio.write_dataframe(
                    dataframe,
                    temporary_output,
                    layer="features",
                    driver="GPKG",
                    append=False,
                    geometry_type="Unknown",
                    layer_options=layer_options,
                )
                output_created = True
            elif len(dataframe) > 0:
                pyogrio.write_dataframe(
                    dataframe,
                    temporary_output,
                    layer="features",
                    driver="GPKG",
                    append=True,
                    layer_options=layer_options,
                )

        os.replace(temporary_output, filename)

    return Vector(filename)


def _clip_vector_multiproc(
    source_vector: VectorBase,
    mask: Any,
    keep_geom_type: bool,
    sort: bool,
    mp_config: MultiprocConfig,
) -> Vector:
    """
    Clip independent vector row partitions and return an unloaded GeoPackage.

    _clip_vector_partition() reads and clips each source range in a worker. _write_vector_partitions() then appends
    the saved results in source order without holding the complete clipped dataset in memory.
    """

    # Validate the one file format whose append behavior preserves arbitrary vector geometry types
    chunks = _vector_partition_size(mp_config)
    if chunks <= 0:
        raise ValueError("Argument ``chunks`` must be a strictly positive integer.")
    driver = "GPKG" if mp_config.driver is None else mp_config.driver.upper()
    output_filename = pathlib.Path(mp_config.outfile)
    if driver != "GPKG":
        raise ValueError("Vector multiprocessing clip() supports only the 'GPKG' output driver.")
    if output_filename.suffix and output_filename.suffix.lower() != ".gpkg":
        raise ValueError("Arguments ``driver`` and ``outfile`` must both select GeoPackage vector output.")

    # Normalize the clipping mask before sending one immutable geometry to every worker
    target_crs = None if source_vector.crs is None else CRS.from_user_input(source_vector.crs)
    geometry = _clip_geometry(mask, target_crs=target_crs)

    # Plan raw file ranges from metadata, preserving any deferred crop filters inside each worker
    source_filename = pathlib.Path(source_vector.name) if not source_vector.is_loaded and source_vector.name else None
    if source_filename is None:
        dataframe = source_vector.ds
        feature_count = len(dataframe)
    else:
        dataframe = None
        feature_count = int(pyogrio.read_info(source_filename, force_feature_count=True)["features"])
        if feature_count < 0:
            raise RuntimeError("Could not determine the number of vector features from the file metadata.")
    crop_filters = list(getattr(source_vector, "_crop_filters", []))

    # Complete all worker reads in temporary files before atomically replacing the destination
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".geoutils-clip-", dir=output_filename.parent) as directory:
        temporary_directory = pathlib.Path(directory)
        futures = []
        for index, start in enumerate(range(0, max(feature_count, 1), chunks)):
            count = min(chunks, feature_count - start)
            partition_source = source_filename if dataframe is None else dataframe.iloc[start : start + count]
            futures.append(
                mp_config.cluster.submit(
                    _clip_vector_partition,
                    partition_source,
                    start,
                    count,
                    crop_filters,
                    geometry,
                    keep_geom_type,
                    sort,
                    temporary_directory / f"clipped_{index}.pkl",
                )
            )
        clipped_parts = mp_config.cluster.gather(futures)
        return _write_vector_partitions(output_filename, clipped_parts)


##############
# 3/ REPROJECT
##############


def _reproject(
    source_vector: Any,
    ref: RasterLike | VectorLike | None = None,
    crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Reproject a vector. See Vector.reproject() for more details."""

    target_crs = _get_reproject_crs(ref=ref, crs=crs)
    new_ds = source_vector.ds.to_crs(crs=target_crs)

    return new_ds
