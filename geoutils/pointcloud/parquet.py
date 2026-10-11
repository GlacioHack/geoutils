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

"""Module to read and write GeoParquet point arrays in row partitions with PyArrow."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from collections.abc import Iterable
from contextlib import ExitStack
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from rasterio.coords import BoundingBox

from geoutils._misc import import_optional

#################################
# Read metadata and row groups
#################################


def _parquet_files(filename: str | pathlib.Path) -> list[pathlib.Path]:
    """List files in a partitioned dataset in deterministic filename order."""
    path = pathlib.Path(filename)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError("The GeoParquet dataset contains no Parquet files.")
    return files


def _point_parquet_metadata(filename: str | pathlib.Path) -> dict[str, Any]:
    """Read point count, CRS, schema and available bounds from Parquet footers."""
    import_optional("pyarrow")
    import pyarrow.parquet as pq

    # Inspect footers only, so counts, schemas and bounds stay available for unloaded inputs
    count = 0
    extents = []
    row_groups = []
    metadata = None
    schema = None
    for path in _parquet_files(filename):
        # Check point encoding and geometry types before reading any rows
        file = pq.ParquetFile(path)
        encoded = (file.schema_arrow.metadata or {}).get(b"geo")
        if encoded is None:
            raise ValueError("Point Parquet input must contain GeoParquet geometry metadata.")
        geo = json.loads(encoded)
        name = geo["primary_column"]
        geometry = geo["columns"][name]
        if geometry["encoding"] not in ("point", "WKB"):
            raise ValueError("GeoParquet point clouds require point or WKB geometry encoding.")
        types = geometry.get("geometry_types", [])
        if any(value not in ("Point", "Point Z") for value in types):
            raise ValueError("GeoParquet point clouds require point geometries.")

        # A missing CRS means "CRS84" under GeoParquet, while explicit null means unspecified coordinates
        crs = geometry.get("crs", "OGC:CRS84")
        crs = None if crs is None else CRS.from_user_input(crs)
        # Separate stored Pandas row labels from point attributes
        pandas_metadata = json.loads((file.schema_arrow.metadata or {}).get(b"pandas", b"{}"))
        index_columns = [column for column in pandas_metadata.get("index_columns", []) if isinstance(column, str)]
        columns = [column for column in file.schema_arrow.names if column not in (name, *index_columns)]
        attrs = json.loads((file.schema_arrow.metadata or {}).get(b"geoutils", b"{}"))
        current = {
            "crs": crs,
            "columns": columns,
            "geometry_name": name,
            "encoding": geometry["encoding"],
            "geometry_type": "Point Z" if "Point Z" in types else "Point",
            "attrs": attrs,
            "index_columns": index_columns,
            "pandas_metadata": pandas_metadata,
        }
        # Check every partition before combining its rows with the rest of the dataset
        if metadata is not None:
            matching_metadata = all(
                current[key] == metadata[key]
                for key in ("crs", "columns", "geometry_name", "encoding", "geometry_type", "attrs")
            )
            if not matching_metadata or not file.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError("GeoParquet partitions must have matching coordinate systems, schemas and attributes.")
        metadata = current
        schema = file.schema_arrow
        count += file.metadata.num_rows

        # Save row  group sizes to construct delayed arrays without opening the footers again
        row_groups.extend(
            (str(path), group, file.metadata.row_group(group).num_rows) for group in range(file.metadata.num_row_groups)
        )

        # Native coords stats give a complete extent even when the optional file bbox is absent
        if "bbox" not in geometry and geometry["encoding"] == "point":
            axis_statistics = {"x": [], "y": []}
            for group in range(file.metadata.num_row_groups):
                row_group = file.metadata.row_group(group)
                for column_index in range(row_group.num_columns):
                    # Read coordinate extrema from statistics rather than loading point arrays
                    column = row_group.column(column_index)
                    axis = column.path_in_schema.removeprefix(name + ".")
                    statistics = column.statistics
                    if axis in axis_statistics and statistics is not None and statistics.has_min_max:
                        axis_statistics[axis].append((statistics.min, statistics.max))
            if all(axis_statistics.values()):
                geometry["bbox"] = [
                    min(value[0] for value in axis_statistics["x"]),
                    min(value[0] for value in axis_statistics["y"]),
                    max(value[1] for value in axis_statistics["x"]),
                    max(value[1] for value in axis_statistics["y"]),
                ]
        if "bbox" in geometry:
            # Select horizontal bounds from either a 2D or 3D bounding box
            bbox = geometry["bbox"]
            extents.append([bbox[0], bbox[1], bbox[len(bbox) // 2], bbox[len(bbox) // 2 + 1]])

    # Combine file extents into the bounds of the complete point cloud
    assert metadata is not None
    if extents:
        bounds = np.asarray(extents)
        extent = [bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()]
    else:
        extent = [np.nan] * 4
    metadata.update(point_count=count, bounds=BoundingBox(*extent), schema=schema, row_groups=row_groups)
    return metadata


def _read_point_parquet_group(filename: str, group: int, geometry_name: str, columns: list[str]) -> dict[str, Any]:
    """Decode one Parquet row group, reading native coordinate fields directly."""

    import_optional("pyarrow")
    import pyarrow.parquet as pq

    # Read the selected attributes and geometry together for this row group
    table = pq.ParquetFile(filename).read_row_group(group, columns=[geometry_name, *columns])
    geometry = table[geometry_name].combine_chunks()
    arrays = {name: table[name].to_numpy() for name in columns}
    geo = json.loads(table.schema.metadata[b"geo"])
    if geo["columns"][geometry_name]["encoding"] == "point":
        # Native point fields give numeric coordinates without creating Shapely objects
        if geometry.null_count:
            raise ValueError("Point arrays cannot represent missing geometries.")
        arrays["x"] = geometry.field("x").to_numpy()
        arrays["y"] = geometry.field("y").to_numpy()
        if "z" in [field.name for field in geometry.type]:
            arrays["_geometry_z"] = geometry.field("z").to_numpy()
    else:
        import shapely

        # Existing WKB files need geometry decoding once at the input boundary
        points = shapely.from_wkb(geometry.to_numpy(zero_copy_only=False))
        if np.any(shapely.get_type_id(points) != 0):
            raise ValueError("GeoParquet point clouds require non-missing point geometries.")
        arrays["x"], arrays["y"] = shapely.get_x(points), shapely.get_y(points)
        if shapely.has_z(points).all() and len(points):
            arrays["_geometry_z"] = shapely.get_z(points)
    return arrays


def _read_point_parquet_frame(filename: str, group: int, columns: list[str]) -> gpd.GeoDataFrame:
    """Decode one GeoParquet row group without resetting its stored Pandas index."""
    import_optional("pyarrow")
    import pyarrow.parquet as pq
    from geopandas.io.arrow import _arrow_to_geopandas

    table = pq.ParquetFile(filename).read_row_group(group, columns=columns)
    return _arrow_to_geopandas(table)


def _point_parquet_columns(
    metadata: dict[str, Any], data_name: str | None, columns: Literal["all", "main"] | list[str]
) -> tuple[str, list[str], bool]:
    """Resolve the active values, selected attributes and whether values come from geometry Z."""
    # Prefer the saved active attribute, then native Z or the first available value column
    saved_column = metadata["attrs"].get("data_name", metadata["attrs"].get("data_column"))
    data_name = data_name or saved_column
    if data_name is None:
        candidates = metadata["columns"]
        if "Z" in candidates:
            data_name = "Z"
        elif candidates:
            data_name = candidates[0]
        else:
            data_name = "_geometry_z"

    # Geometry elevations are selectable only when the file contains 3D points
    available = [*metadata["columns"]]
    if metadata["geometry_type"] == "Point Z":
        available.append("_geometry_z")
    if data_name not in available:
        raise ValueError(f"Point data column {data_name!r} does not exist.")
    if columns == "all":
        selected = metadata["columns"]
    elif columns == "main":
        selected = [data_name]
    else:
        selected = list(columns)

    # Read the active values even when a supplied list selects only auxiliary attributes
    selected = list(dict.fromkeys([data_name, *selected]))
    if any(name not in available for name in selected):
        raise ValueError("Requested point attributes are absent from the GeoParquet file.")
    use_z = data_name == "_geometry_z" or bool(metadata["attrs"].get("geometry_z") and data_name == saved_column)
    return data_name, selected, use_z


def _set_parquet_point_values(
    frame: Any,
    data_name: str | None,
    *,
    row_starts: tuple[int, ...] | None = None,
    partition_info: dict[str, Any] | None = None,
) -> Any:
    """Select point values and assign ordered row numbers to native GeoParquet partitions."""
    frame = frame.copy()

    # Dask resets unstored indexes in each row group; footer row counts give their global positions
    if row_starts is not None and partition_info is not None:
        start = row_starts[partition_info["number"]]
        frame.index = pd.RangeIndex(start, start + len(frame))

    # Shared column metadata also survives Pandas concatenation after explicit computation
    if data_name is not None and data_name not in frame.columns:
        frame[data_name] = frame.geometry.z
    frame.attrs["data_name"] = data_name
    return frame


#################################
# Open arrays or GeoDataFrames
#################################


def _open_point_parquet(
    filename: str | pathlib.Path,
    *,
    data_name: str | None = None,
    columns: Literal["all", "main"] | list[str] = "all",
    chunks: int | None = None,
) -> xr.DataArray:
    """
    Open point arrays in-memory or with Dask from independently readable Parquet row groups.
    """
    from geoutils.pointcloud.xr_accessor import DataArrayPointCloudAccessor

    # Select values and auxiliary attributes from footer metadata alone
    metadata = _point_parquet_metadata(filename)
    data_name, selected, use_z = _point_parquet_columns(metadata, data_name, columns)
    index_columns = metadata["index_columns"]
    if len(index_columns) > 1:
        raise ValueError("Array point clouds require a single row index rather than a Pandas MultiIndex.")
    stored = [name for name in selected if name != "_geometry_z"]
    stored.extend(index_columns)
    arrays: dict[str, list[Any]] = {name: [] for name in ("x", "y", *selected)}
    arrays.update({name: [] for name in index_columns})
    if metadata["geometry_type"] == "Point Z":
        arrays.setdefault("_geometry_z", [])

    # Coordinate fields are float64; each attribute uses its own Arrow schema dtype
    pa = import_optional("pyarrow")
    dtypes: dict[str, Any] = {}
    for name in arrays:
        if name in ("x", "y", "_geometry_z"):
            dtypes[name] = np.float64
        else:
            field_type = metadata["schema"].field(name).type
            # Dictionary attributes contain labels in arrays rather than a Pandas categorical dtype
            dtypes[name] = object if pa.types.is_dictionary(field_type) else field_type.to_pandas_dtype()
    if chunks is not None:
        dask = import_optional("dask")
        import dask.array as da

    # Share one physical row-group read across coordinates and selected attributes
    for path, group, count in metadata["row_groups"]:
        if chunks is None:
            values = _read_point_parquet_group(path, group, metadata["geometry_name"], stored)
            for name in arrays:
                arrays[name].append(values[name])
        else:
            values = dask.delayed(_read_point_parquet_group)(path, group, metadata["geometry_name"], stored)
            for name in arrays:
                arrays[name].append(da.from_delayed(values[name], shape=(count,), dtype=dtypes[name]))

    # Join ordered row groups, then split lazy arrays into the requested chunk size
    combined = {}
    for name, parts in arrays.items():
        if chunks is None:
            combined[name] = np.concatenate(parts) if parts else np.empty(0, dtype=dtypes[name])
        elif parts:
            combined[name] = da.concatenate(parts).rechunk(chunks)
        else:
            combined[name] = da.from_array(np.empty(0, dtype=dtypes[name]), chunks=chunks)

    # Selecting another active attribute preserves geometry Z in a separate coordinate
    if use_z and data_name != "_geometry_z":
        combined.pop("_geometry_z", None)
    row_labels = combined.pop(index_columns[0]) if index_columns else None
    result = DataArrayPointCloudAccessor.from_xyz(
        combined.pop("x"),
        combined.pop("y"),
        combined.pop(data_name),
        metadata["crs"],
        data_name="z" if data_name == "_geometry_z" else data_name,
        auxiliary=combined,
        use_z=use_z,
    )
    result.attrs.update({key: value for key, value in metadata["attrs"].items() if key.startswith("dataframe_")})
    if row_labels is not None:
        # Leave stored labels lazy without asking Xarray to create an eager Pandas index
        coordinates = xr.Coordinates({"point": ("point", row_labels)}, indexes={})
        result = result.assign_coords(coordinates)
        descriptors = metadata["pandas_metadata"].get("columns", [])
        index_descriptor = next(item for item in descriptors if item["field_name"] == index_columns[0])
        result.attrs["dataframe_index_name"] = index_descriptor["name"]
    result.encoding["source"] = str(filename)
    return result


def _open_parquet_geodataframe(
    filename: str | pathlib.Path,
    *,
    data_name: str | None = None,
    columns: Literal["all", "main"] | list[str] = "all",
    chunks: int | None = None,
) -> gpd.GeoDataFrame | Any:
    """Use GeoPandas to decode point geometry while preserving the selected values and lazy metadata."""
    from geoutils.pointcloud.dataframe import _build_pointcloud_output
    from geoutils.vector.pd_accessor import _import_dask_geopandas

    # Use the same active attribute and schema checks as the numeric array reader
    metadata = _point_parquet_metadata(filename)
    data_name, selected, use_z = _point_parquet_columns(metadata, data_name, columns)
    stored = [name for name in selected if name in metadata["columns"] and not (use_z and name == data_name)]
    read_columns = [*stored, metadata["geometry_name"]]
    files = [str(path) for path in _parquet_files(filename)]
    active_column = None if use_z else data_name

    # GeoPandas handles native point and WKB decoding; Dask-GeoPandas delays the same geometry work
    if chunks is None:
        frame = gpd.read_parquet(files, columns=read_columns)
        frame = _set_parquet_point_values(frame, active_column)
    else:
        descriptors = metadata["pandas_metadata"].get("columns", [])
        index_names = [item["name"] for item in descriptors if item["field_name"] in metadata["index_columns"]]
        # Dask-GeoPandas assumes 2D bounds, so read 3D points with the existing row group reader
        if metadata["geometry_type"] == "Point Z" or any(name in metadata["columns"] for name in index_names):
            # Dask's Parquet reader resets named indexes, which fails when a value uses the same name
            # Decode bounded row groups with GeoPandas instead, preserving that valid Pandas layout
            dask = import_optional("dask")
            import dask.dataframe as dd
            from geopandas.io.arrow import _arrow_to_geopandas

            group_columns = list(dict.fromkeys([*read_columns, *metadata["index_columns"]]))
            meta = _arrow_to_geopandas(metadata["schema"].empty_table().select(group_columns))
            parts = [
                dask.delayed(_read_point_parquet_frame)(path, group, group_columns)
                for path, group, _ in metadata["row_groups"]
            ]
            frame = _import_dask_geopandas().from_dask_dataframe(dd.from_delayed(parts, meta=meta))
        else:
            frame = _import_dask_geopandas().read_parquet(files, columns=read_columns, split_row_groups=True)

        # Native point files have no stored Pandas index, so assign unique positions across their row groups
        row_starts = None
        if b"pandas" not in (metadata["schema"].metadata or {}):
            lengths = [count for _, _, count in metadata["row_groups"]]
            row_starts = tuple(int(start) for start in np.cumsum((0, *lengths[:-1])))
        frame = frame.map_partitions(
            _set_parquet_point_values,
            active_column,
            row_starts=row_starts,
            meta=_set_parquet_point_values(frame._meta, active_column),
        )

    # Cache counts and extents from footers, so lazy metadata queries do not run row readers
    attrs = {key: metadata[key] for key in ("point_count", "crs", "bounds", "geometry_type")}
    return _build_pointcloud_output(
        frame, data_name=active_column, as_dataframe=True, attrs=attrs, preserve_locations=True
    )


#################################
# Construct tables and write partitions
#################################


def _point_parquet_table(points: xr.DataArray, *, write_bbox: bool = True) -> Any:
    """Encode one eager point array as a GeoParquet 1.1 table with native coordinate fields."""
    pa = import_optional("pyarrow")

    # Build native coordinate fields directly from the numeric arrays
    x, y, _ = points.pc.to_xyz()
    axes = [pa.array(x, type=pa.float64()), pa.array(y, type=pa.float64())]
    axis_names = ["x", "y"]
    geometry_z = points.attrs.get("geometry_z", False)
    if geometry_z or "_geometry_z" in points.coords:
        elevation = points.data if geometry_z else points.coords["_geometry_z"].data
        axes.append(pa.array(elevation, type=pa.float64()))
        axis_names.append("z")
    geometry = pa.StructArray.from_arrays(axes, names=axis_names)

    # Store auxiliary attributes separately to preserve integer, string and timestamp dtypes
    values = {
        name: pa.array(points.data if name == points.name else points.coords[name].data)
        for name in points.pc.columns
        if name != "_geometry_z"
    }
    values["geometry"] = geometry

    # Store standard geospatial metadata and the active value name for lossless GeoUtils reopening
    geo_column: dict[str, Any] = {
        "encoding": "point",
        "geometry_types": ["Point Z" if len(axes) == 3 else "Point"],
        "crs": None if points.pc.crs is None else points.pc.crs.to_json_dict(),
    }
    # A partition bbox is useful for separate files; a shared footer must describe every row group
    if write_bbox and points.size and np.isfinite(np.stack([x, y])).all():
        coordinates = [np.asarray(axis) for axis in (x, y)]
        if len(axes) == 3:
            coordinates.append(np.asarray(elevation))
        geo_column["bbox"] = [float(axis.min()) for axis in coordinates] + [float(axis.max()) for axis in coordinates]
    geo = {"version": "1.1.0", "primary_column": "geometry", "columns": {"geometry": geo_column}}
    # GeoUtils records the active values and geometry elevations for exact reopening
    attrs = {"data_name": points.pc.data_name, "geometry_z": bool(points.attrs.get("geometry_z"))}
    attrs.update({key: value for key, value in points.attrs.items() if key.startswith("dataframe_")})
    table = pa.table(values)
    metadata = {b"geo": json.dumps(geo).encode(), b"geoutils": json.dumps(attrs).encode()}

    # Explicit point labels are an index, rather than another value column
    # Let Arrow describe that index without converting the point values through Pandas
    dimension = points.dims[0]
    if dimension in points.coords:
        labels = pd.Index(points.coords[dimension].data, name=dimension)
        frame = pd.DataFrame(index=labels, columns=pd.Index([], dtype=object))
        index = pa.Table.from_pandas(frame, preserve_index=True)
        table = table.append_column(dimension, index[dimension])
        pandas_metadata = json.loads(index.schema.metadata[b"pandas"])
        # The dimension name avoids collisions with value fields; Pandas restores the original index name
        pandas_metadata["columns"][0]["name"] = points.attrs.get("dataframe_index_name", dimension)
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode()
    return table.replace_schema_metadata(metadata)


def _write_parquet_row_groups(
    partitions: Iterable[xr.DataArray | gpd.GeoDataFrame],
    destination: pathlib.Path,
    *,
    compression: str = "zstd",
    data_name: str | None = None,
) -> None:
    """Append bounded point partitions to one Parquet stream and close its footer on success or failure."""
    import_optional("pyarrow")
    import pyarrow.parquet as pq

    # Open one writer after the first partition has supplied the shared Arrow schema
    with ExitStack() as stack:
        writer = None
        for part in partitions:
            if isinstance(part, gpd.GeoDataFrame):
                # Staged multiprocessing frames store the active column alongside their geometry
                part.attrs["data_name"] = data_name
                part = part.pc.to_xarray()
            table = _point_parquet_table(part, write_bbox=False)

            # Omit partition-specific bboxes; row-group coordinate statistics supply complete file bounds
            if writer is None:
                writer = stack.enter_context(pq.ParquetWriter(destination, table.schema, compression=compression))
            writer.write_table(table)


def _write_point_parquet(
    points: xr.DataArray,
    filename: str | pathlib.Path,
    *,
    chunks: int | None = None,
    partitioned: bool = False,
    compression: str = "zstd",
) -> None:
    """
    Write native point row groups or ordered files without collecting a lazy point cloud.

    _array_partitions() computes one bounded row partition. _point_parquet_table() supplies an Arrow table
    for each separate file, while _write_parquet_row_groups() writes a shared stream. We finish the temporary
    output before replacing the destination, so a failed partition leaves an existing file intact.
    """
    import_optional("pyarrow")
    import pyarrow.parquet as pq

    from geoutils.pointcloud.writing import _array_partitions

    # Check the destination before computing point rows
    destination = pathlib.Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if partitioned and destination.exists():
        raise FileExistsError("A partitioned GeoParquet destination must not already exist.")

    # Stage the complete output beside the destination before replacing it
    with tempfile.TemporaryDirectory(prefix=".geoutils-parquet-", dir=destination.parent) as directory:
        temporary = pathlib.Path(directory)
        partitions = _array_partitions(points, chunks)
        if partitioned:
            # Ordered names preserve row order when the directory is reopened
            output = temporary / "dataset"
            output.mkdir()
            for index, part in enumerate(partitions):
                pq.write_table(
                    _point_parquet_table(part), output / f"part-{index:08d}.parquet", compression=compression
                )
        else:
            output = temporary / "points.parquet"
            _write_parquet_row_groups(partitions, output, compression=compression)
        os.replace(output, destination)
