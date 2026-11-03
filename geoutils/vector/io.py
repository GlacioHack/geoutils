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

"""Read/write vector files, from local or remote files."""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
import pyogrio

from geoutils._dispatch import is_dask_dataframe
from geoutils._io import _filesystem_file_options, _FilesystemOpener, _path_suffix, _resolve_filesystem
from geoutils._misc import import_optional

if TYPE_CHECKING:
    from geoutils.vector.base import VectorBase
    from geoutils.vector.vector import Vector


##############
# 1/ OPENING
##############


def _import_dask_geopandas() -> Any:
    """Import Dask-GeoPandas as an optional dependency."""

    import_optional("dask_geopandas", package_name="dask-geopandas")
    import dask_geopandas as dgpd

    return dgpd


def open_vector(
    filename: str | Path,
    chunks: int | None = None,
    *,
    load: bool = False,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame | Any:
    """
    Open a vector file as a GeoDataFrame.

    Passing ``chunks`` opens the vector data lazily using Dask-GeoPandas.

    To open a remote file, pass an HTTPS URL or S3 address, such as ``"https://example.org/data.gpkg"`` or
    ``"s3://bucket/data.gpkg"``.

    Use ``storage_options`` to pass a dictionary of connection settings, such as credentials or a custom
    server address. GeoUtils creates a Fsspec filesystem for you, which handles access to files on that service.
    For example, ``storage_options={"anon": True}`` reads a public S3 bucket without login credentials.

    Use ``filesystem`` if you already have a Fsspec filesystem object and want to reuse its connection
    settings across files. Both options are optional and cannot be combined.

    These settings are preserved for deferred and chunked reads.
    Using these options requires the optional Fiona backend.
    See :ref:`scalability-formats` for supported formats and authentication.

    :param filename: Path to the vector file to open.
    :param chunks: Number of features per Dask partition. None uses one partition, or in-memory GeoPandas
        with load=True.
    :param load: Whether to read features immediately when chunks is None. The default False opens one Dask partition
        and requires Dask-GeoPandas. Explicit chunks always stay lazy.
    :param filesystem: An existing Fsspec filesystem object whose connection settings are used to open files.
        Cannot be combined with ``storage_options``.
    :param storage_options: Dictionary of connection settings used to create a Fsspec filesystem from the file URL,
        such as credentials or a custom server address. Cannot be combined with ``filesystem``.
    :param kwargs: Keyword arguments passed to :func:`geopandas.read_file`.

    :returns: A lazy Dask-GeoPandas GeoDataFrame, or an in-memory GeoDataFrame when load=True without chunks.
    """

    # 1/ Check inputs
    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)

    # 2/ Open depending on format

    # 2.1/ Parquet readers use Arrow, through Dask Geopandas
    if _path_suffix(filename) in (".parquet", ".pq"):
        if load and chunks is None:
            return _read_vector_file(filename, filesystem=filesystem, **kwargs)
        from geoutils.vector.pd_accessor import _register_dask_vector_accessor

        _register_dask_vector_accessor()
        if chunks is not None and chunks <= 0:
            raise ValueError("Argument 'chunks' must be a strictly positive integer.")
        frame = _import_dask_geopandas().read_parquet(filename, filesystem=filesystem, **kwargs)
        if chunks is not None:
            parquet_info = _read_vector_info(filename, filesystem=filesystem)
            frame = frame.repartition(npartitions=max(1, (parquet_info["features"] + chunks - 1) // chunks))
        return frame

    # 2.2/ Other vector formats, such as GeoPackage, Shapefile and GeoJSON

    # 2.2.1/ Default to one lazy partition unless load=True
    info: dict[str, Any] | None = None
    if not load and chunks is None:
        info = _read_vector_info(filename, filesystem=filesystem, layer=kwargs.get("layer"))
        chunks = max(1, info["features"])

    # 2.2.2/ Read lazily whenever chunks are set
    if chunks is not None:
        # Open directly with Dask-GeoPandas
        if chunks <= 0:
            raise ValueError("Argument 'chunks' must be a strictly positive integer.")
        from geoutils.vector.pd_accessor import _register_dask_vector_accessor

        dgpd = _import_dask_geopandas()
        _register_dask_vector_accessor()
        if filesystem is None and not any(key in kwargs for key in ("bbox", "mask", "where", "rows", "fid_as_index")):
            return dgpd.read_file(filename, chunksize=chunks, **kwargs)

        # Select row ranges
        dask = import_optional("dask")
        import dask.dataframe as dd

        if info is None:
            info = _read_vector_info(filename, filesystem=filesystem, layer=kwargs.get("layer"))
        read_options = dict(kwargs)
        rows = read_options.pop("rows", None)
        if isinstance(rows, int) and rows < 0:
            raise ValueError("Lazy vector reads require non-negative row bounds.")
        if isinstance(rows, slice) and (
            (rows.start is not None and rows.start < 0)
            or (rows.stop is not None and rows.stop < 0)
            or (rows.step is not None and rows.step <= 0)
        ):
            raise ValueError("Lazy vector reads require non-negative row bounds and a positive step.")
        selection = range(info["features"])
        if isinstance(rows, int):
            selection = selection[:rows]
        elif isinstance(rows, slice):
            selection = selection[rows]
        elif rows is not None:
            raise TypeError("Argument 'rows' must be an integer or a slice.")

        # Read schema and index settings
        empty = _read_vector_file(filename, filesystem=filesystem, rows=0, info=info, **read_options)
        filtered = any(read_options.get(key) is not None for key in ("bbox", "mask", "where"))
        preserve_index = (
            read_options.get("fid_as_index", False)
            and filesystem is None
            and read_options.get("engine", gpd.options.io_engine or "pyogrio") == "pyogrio"
        )

        # Build deferred partitions
        parts = []
        offset = 0
        for start in range(0, len(selection), chunks):
            selected = selection[start : start + chunks]
            row_slice = slice(selected.start, selected.stop, None if selected.step == 1 else selected.step)
            part = dask.delayed(_read_vector_file)(
                filename, filesystem=filesystem, rows=row_slice, meta=empty, **read_options
            )
            if preserve_index:
                parts.append(part)
                continue
            if not filtered:
                parts.append(part.set_axis(pd.RangeIndex(start, start + len(selected))))
                continue
            # Filters can shorten each partition, so build indices from the returned rows
            length = dask.delayed(len)(part)
            index = dask.delayed(pd.RangeIndex)(offset, offset + length)
            parts.append(part.set_axis(index))
            offset = offset + length

        # Assemble the lazy GeoDataFrame
        if not parts:
            parts = [dask.delayed(empty)]
        with dask.config.set({"dataframe.convert-string": False}):
            frame = dd.from_delayed(parts, meta=empty)
        return dgpd.from_dask_dataframe(frame, geometry=empty.geometry.name)

    # 2.2.3/ If chunks are not set and load=True, we read eagerly

    # Without chunks, use GeoPandas' regular behavior
    return _read_vector_file(filename, filesystem=filesystem, **kwargs)


##############
# 2/ LOADING
##############


def _vector_file_options(filename: str | Path, filesystem: Any, *, mode: str = "r") -> tuple[str, _FilesystemOpener]:
    """Pass GDAL a relative filename while preserving URL queries on filesystem requests."""

    filename = os.fspath(filename)
    import_optional("fiona")

    # Fiona cannot reliably read Shapefiles through a filesystem
    if mode == "r" and _path_suffix(filename) == ".shp":
        raise NotImplementedError(
            "Fiona cannot reliably read Shapefiles through a filesystem opener. "
            "Use another vector format or a GDAL URL without filesystem settings."
        )
    return _filesystem_file_options(filename, filesystem)


def _read_vector_file(
    filename: str | Path,
    *,
    filesystem: Any = None,
    info: dict[str, Any] | None = None,
    meta: gpd.GeoDataFrame | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Read a vector file, using Fiona when a filesystem is provided.

    Shared by Vector, PointCloud and their accessors.
    """

    import geopandas as gpd

    source_filename = filename
    rows = kwargs.get("rows")

    # 1/ Read Parquet files
    if _path_suffix(filename) in (".parquet", ".pq"):
        kwargs.pop("rows", None)
        dataframe = gpd.read_parquet(filename, filesystem=filesystem, **kwargs)
        if rows is not None:
            return dataframe.iloc[:rows] if isinstance(rows, int) else dataframe.iloc[rows]
        return dataframe

    # 2/ Read other vector formats

    # 2.1/ Prepare row selection
    empty_rows = rows == 0 or (isinstance(rows, slice) and rows.stop is not None and rows.stop == (rows.start or 0))
    if empty_rows:
        # Pyogrio interprets zero rows as unlimited, so read only the schema row
        kwargs["rows"] = 1

    # 2.2/ Use Fiona with filesystem settings for remote files
    if filesystem is not None:
        import_optional("fiona")
        if kwargs.get("engine", "fiona") != "fiona":
            raise ValueError("A filesystem requires engine='fiona' for vector files.")
        if "opener" in kwargs:
            raise ValueError("Pass either filesystem or opener, not both.")
        # GeoPandas tries URLs before using the supplied filesystem
        # Pass a relative filename so Fiona handles credentials and "companion" files (like .shx/.prj in a Shapefile)
        filename, opener = _vector_file_options(filename, filesystem)
        kwargs.update(engine="fiona", opener=opener)

    # 2.3/ Read features
    dataframe = gpd.read_file(filename, **kwargs)
    if empty_rows and len(dataframe):
        return dataframe.iloc[:0]
    if empty_rows:
        dataframe = dataframe.iloc[:0]

    # 2.4/ Restore column types for empty results
    if len(dataframe) == 0:
        if meta is not None:
            return meta.copy()
        if info is None:
            info = _read_vector_info(source_filename, filesystem=filesystem, layer=kwargs.get("layer"))
        field_types = dict(zip(info["fields"], info["dtypes"]))
        dataframe = dataframe.astype({name: field_types[name] for name in dataframe.columns if name in field_types})
    return dataframe


def _read_vector_info(
    filename: str | Path, *, filesystem: Any = None, layer: str | int | None = None, force_feature_count: bool = False
) -> dict[str, Any]:
    """
    Read a vector layer info: feature count, bounds and column types.

    Shared by Vector, PointCloud and their accessors.
    """

    # 1/ Read Parquet metadata
    if _path_suffix(filename) in (".parquet", ".pq"):
        pa = import_optional("pyarrow")
        import pyarrow.dataset as ds

        # 1.1/ Read column and geometry metadata

        # Parquet footers provide counts and geometry metadata
        dataset = ds.dataset(str(filename), filesystem=filesystem, format="parquet")
        schema = dataset.schema
        geo = json.loads(schema.metadata[b"geo"])
        geometry_name = geo["primary_column"]
        geometry = geo["columns"][geometry_name]
        pandas_metadata = json.loads((schema.metadata or {}).get(b"pandas", b"{}"))
        indexes = pandas_metadata.get("index_columns", [])
        fields = [field for field in schema if field.name != geometry_name and field.name not in indexes]

        # 1.2/ Combine partition bounds

        # Combine saved bounds (without reading geometries from partition files)
        extents = []
        for fragment in dataset.get_fragments():
            fragment_geo = json.loads(fragment.physical_schema.metadata[b"geo"])
            bounds = fragment_geo["columns"][geometry_name].get("bbox")
            if bounds is None:
                extents = []
                break
            extents.append([bounds[0], bounds[1], bounds[3], bounds[4]] if len(bounds) == 6 else bounds)
        bounds = None
        if extents:
            bounds = [
                min(box[0] for box in extents),
                min(box[1] for box in extents),
                max(box[2] for box in extents),
                max(box[3] for box in extents),
            ]

        # 1.3/ Return layer info
        types = geometry.get("geometry_types", [])
        return {
            "features": dataset.count_rows(),
            "crs": geometry.get("crs", "OGC:CRS84"),
            "total_bounds": bounds,
            "geometry_type": types[0] if len(types) == 1 else "Unknown",
            "driver": "Parquet",
            "fields": [field.name for field in fields],
            "dtypes": [
                object if pa.types.is_dictionary(field.type) else field.type.to_pandas_dtype() for field in fields
            ],
        }

    # 2/ Read metadata for other vector formats

    # 2.1/ Use Pyogrio without filesystem settings
    if filesystem is None:
        import pyogrio

        return pyogrio.read_info(filename, layer=layer, force_feature_count=force_feature_count)

    # 2.2/ Use Fiona with filesystem settings

    # Read metadata with the filesystem access settings
    fiona = import_optional("fiona")
    filename, opener = _vector_file_options(filename, filesystem)
    with fiona.open(filename, layer=layer, opener=opener) as source:
        info = {
            "crs": source.crs_wkt,
            "features": len(source),
            "total_bounds": source.bounds if len(source) else None,
            "geometry_type": source.schema["geometry"],
            "driver": source.driver,
        }

        # Convert column types
        fields = source.schema["properties"]
        dtypes = []
        for field_type in fields.values():
            value_type = fiona.prop_type(field_type)
            if value_type in (int, float, bool):
                dtypes.append(value_type)
            elif field_type == "datetime":
                dtypes.append("datetime64[ms]")
            else:
                dtypes.append(object)
        info.update(fields=list(fields), dtypes=dtypes)
    return info


def _vector_row_source(source_vector: VectorBase) -> tuple[VectorBase | gpd.GeoDataFrame, int]:
    """Describe raw vector row ranges without loading a file-backed source."""

    # Count rows directly when the data are already in memory
    if source_vector.is_loaded or source_vector.name is None:
        dataframe = source_vector._dataset
        return dataframe, len(dataframe)

    # If not in memory, count all rows in the selected layer without loading its features
    info = _read_vector_info(
        source_vector.name,
        filesystem=getattr(source_vector, "_filesystem", None),
        layer=getattr(source_vector, "_layer", None),
        force_feature_count=True,
    )
    feature_count = int(info["features"])
    if feature_count < 0:
        raise RuntimeError("Could not determine the number of vector features from the file metadata.")
    return source_vector, feature_count


def _load_vector_rows(source_vector: VectorBase, start: int, count: int) -> gpd.GeoDataFrame:
    """Read a vector row range using the source filesystem and selected layer."""

    assert source_vector.name is not None

    # Read one row for empty requests to preserve column names and types
    dataframe = _read_vector_file(
        source_vector.name,
        filesystem=getattr(source_vector, "_filesystem", None),
        layer=getattr(source_vector, "_layer", None),
        rows=slice(start, start + max(1, count)),
    )

    return dataframe.iloc[:0] if count == 0 else dataframe


##############
# 3/ WRITING
##############


def _write_vector_file(
    dataframe: gpd.GeoDataFrame | Any,
    filename: str,
    driver: Any = None,
    schema: Any = None,
    index: Any = None,
    *,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Write a GeoDataFrame to a vector file or a Dask GeoParquet dataset.

    Shared by Vector, PointCloud and their accessors.
    """

    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)
    suffix = _path_suffix(filename) if isinstance(filename, (str, os.PathLike)) else ""
    parquet = suffix in (".parquet", ".pq") or str(driver).lower() in ("parquet", "geoparquet")
    if parquet and not is_dask_dataframe(dataframe):
        if schema is not None:
            raise ValueError("Argument 'schema' is not supported for GeoParquet output.")
        dataframe.to_parquet(filename, index=index, filesystem=filesystem, **kwargs)
        return

    if is_dask_dataframe(dataframe):
        # GeoParquet supports direct partitioned output from Dask-GeoPandas
        if parquet:
            if schema is not None:
                raise ValueError("Argument 'schema' is not supported for Dask GeoParquet output.")
            if driver is not None:
                kwargs.pop("driver", None)
            if index is not None:
                kwargs["write_index"] = index
            else:
                kwargs.setdefault("write_index", False)
            if filesystem is not None:
                kwargs["filesystem"] = filesystem
            dataframe.to_parquet(filename, **kwargs)
            return

        # Other vector formats require one in-memory GeoDataFrame for GeoPandas writing
        ds = dataframe.compute()
    else:
        ds = dataframe

    if filesystem is not None:
        if kwargs.get("engine", "fiona") != "fiona":
            raise ValueError("A filesystem requires engine='fiona' for vector files.")
        if "opener" in kwargs:
            raise ValueError("Pass either filesystem or opener, not both.")
        import_optional("fiona")
        filename, opener = _vector_file_options(filename, filesystem, mode="w")
        kwargs.update(engine="fiona", opener=opener)
    ds.to_file(filename=filename, driver=driver, schema=schema, index=index, **kwargs)


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
