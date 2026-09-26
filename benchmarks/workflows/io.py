"""Open and write inputs and outputs (I/O) for benchmark tests."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

from benchmarks.workflows.config import ExecutionMode, RuntimeConfig
from geoutils._misc import _trim_process_memory, import_optional

############################
# Shared size and I/O helpers
############################


def logical_raster_size_mb(config: RuntimeConfig) -> float:
    """Return the uncompressed float32 raster size in decimal megabytes."""

    return config.shape[0] * config.shape[1] * np.dtype("float32").itemsize / 1_000_000


def memory_limit_mb(memory_limit: str) -> float:
    """Convert the worker-memory formats used by the benchmark suite to decimal megabytes."""

    # Match the common decimal and binary units accepted by Dask
    value = memory_limit.strip().lower()
    if value.endswith("gib"):
        return float(value[:-3]) * 1024**3 / 1_000_000
    if value.endswith("gb"):
        return float(value[:-2]) * 1000
    if value.endswith("mib"):
        return float(value[:-3]) * 1024**2 / 1_000_000
    if value.endswith("mb"):
        return float(value[:-2])
    return float(value) / 1_000_000


def tiff_block_size(size: int, requested: int) -> int:
    """Return a valid tiled-GeoTIFF block size no larger than one raster axis."""

    # GeoTIFF tile dimensions must be divisible by sixteen
    block_size = min(size, requested, 512)
    return max(16, block_size // 16 * 16)


def read_raster_center(filename: str) -> float:
    """Read one central output pixel without loading the complete raster."""

    with rio.open(filename) as dataset:
        row = dataset.height // 2
        col = dataset.width // 2
        return float(dataset.read(1, window=rio.windows.Window(col, row, 1, 1))[0, 0])


def read_point_file_sample(filename: str, column: str) -> tuple[int, float]:
    """Read the feature count and one value without loading a complete point file."""

    if pathlib.Path(filename).suffix.lower() in (".las", ".laz"):
        laspy = import_optional("laspy")
        with laspy.open(filename) as reader:
            count = int(reader.header.point_count)
            sample = reader.read_points(1)
        return count, float(sample.z[0])

    import pyogrio

    # Use file metadata for the complete row count and read only one feature for the constant-value check
    info = pyogrio.read_info(filename, force_feature_count=True)
    sample = pyogrio.read_dataframe(filename, columns=[column], max_features=1)
    return int(info["features"]), float(sample[column].iloc[0])


def open_raster_input(filename: str, backend: ExecutionMode, config: RuntimeConfig) -> Any:
    """Open one prepared raster through the interface used by an execution backend."""

    if backend == "dask":
        from geoutils.raster.xr_accessor import open_raster

        # Raster chunks remain lazy until the output values are computed
        return open_raster(filename, chunks={"y": config.chunks[0], "x": config.chunks[1]})

    from geoutils import Raster

    # In-memory comparisons load the complete input while multiprocessing reads windows
    return Raster(filename, load_data=backend == "inmem")


def prepare_output_file(
    directory: str,
    backend: ExecutionMode,
    operation: str,
    suffix: str = ".tif",
) -> str:
    """Return an empty reusable output path for one benchmark operation."""

    filename = os.path.join(directory, f"output-{backend}-{operation}{suffix}")
    if os.path.isfile(filename):
        os.remove(filename)
    return filename


def _write_dask_raster(
    raster: Any,
    filename: str,
    config: RuntimeConfig,
    client: Any | None,
) -> None:
    """Compute and write one lazy raster in bounded groups of blocks."""

    block_y = tiff_block_size(config.shape[0], config.chunks[0])
    block_x = tiff_block_size(config.shape[1], config.chunks[1])

    # GeoTIFF has no boolean sample type, so masks use the equivalent byte values
    if np.issubdtype(raster.dtype, np.bool_):
        raster = raster.astype("uint8")

    # Read georeferencing from metadata without evaluating any raster value
    data = raster.data
    if data.chunks is None:
        raise ValueError("Dask benchmark output must remain chunked before writing")
    if config.dask_write_batch_size < 1:
        raise ValueError("Dask write batch size must be strictly positive")
    nodata = raster.rio.nodata

    # Computing a few blocks together reduces scheduling overhead while retaining a fixed memory bound
    dask = import_optional("dask", extra_name="benchmark")
    pending_blocks = []
    pending_windows: list[rio.windows.Window] = []

    def write_pending_blocks(destination: rio.io.DatasetWriter) -> None:
        """Compute and write the current bounded group of output blocks."""

        if not pending_blocks:
            return

        # One scheduler request computes the independent blocks as a group
        computed_blocks = dask.compute(*pending_blocks)
        for block, window in zip(computed_blocks, pending_windows):
            destination.write(np.asarray(block), indexes=1, window=window)
        pending_blocks.clear()
        pending_windows.clear()

        # Large data contracts may release native workspaces between bounded groups
        if config.trim_dask_memory:
            _trim_process_memory()
            if client is not None:
                client.run(_trim_process_memory)

    # Open one tiled destination shared by all bounded block groups
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype=np.dtype(data.dtype),
        crs=raster.rio.crs,
        transform=raster.rio.transform(),
        nodata=nodata,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        BIGTIFF="IF_NEEDED",
    ) as destination:
        row_offset = 0
        for row_index, row_size in enumerate(data.chunks[0]):
            col_offset = 0
            for col_index, col_size in enumerate(data.chunks[1]):
                # Retain lazy blocks only until the fixed batch is ready to compute
                pending_blocks.append(data.blocks[row_index, col_index])
                window = rio.windows.Window(col_offset, row_offset, col_size, row_size)
                pending_windows.append(window)
                if len(pending_blocks) == config.dask_write_batch_size:
                    write_pending_blocks(destination)
                col_offset += col_size
            row_offset += row_size

        # Write a final partial group at the edge of the output raster
        write_pending_blocks(destination)


def materialize_raster_output(
    raster: Any,
    backend: ExecutionMode,
    filename: str | None,
    config: RuntimeConfig,
    client: Any | None,
) -> str:
    """Write a complete raster output and return its file path."""

    if backend == "multiprocessing":
        # Multiprocessing operations already wrote their returned Raster to disk
        return str(raster.name)
    if filename is None:
        raise ValueError("A raster output path is required outside multiprocessing")
    if backend == "dask":
        _write_dask_raster(raster, filename, config, client)
        return filename

    # In-memory results use the same tiled output contract
    block_y = tiff_block_size(config.shape[0], config.chunks[0])
    block_x = tiff_block_size(config.shape[1], config.chunks[1])
    raster.to_file(
        filename,
        co_opts={
            "TILED": "YES",
            "BLOCKYSIZE": str(block_y),
            "BLOCKXSIZE": str(block_x),
            "COMPRESS": "NONE",
        },
    )
    return filename


##############################
# Deterministic source files
##############################


def write_constant_raster(filename: str, config: RuntimeConfig) -> None:
    """Write a deterministic constant raster one storage block at a time."""

    if os.path.exists(filename):
        return

    # Use a real WGS84 extent so reprojection exercises a coordinate transform
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = tiff_block_size(height, config.chunks[0])
    block_x = tiff_block_size(width, config.chunks[1])

    # Compression keeps the deterministic constant fixture compact on disk
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-99999,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        compress="DEFLATE",
        BIGTIFF="IF_NEEDED",
    ) as destination:
        # Allocate only the current storage block instead of the complete raster
        for _, window in destination.block_windows(1):
            block = np.full((int(window.height), int(window.width)), config.raster_value, dtype=np.float32)
            destination.write(block, indexes=1, window=window)


def write_vector_source(filename: str, features_per_axis: int = 1) -> None:
    """Write regularly spaced polygons used by rasterization and mask scenarios."""

    if os.path.exists(filename):
        return
    if features_per_axis < 1:
        raise ValueError("Vector features per axis must be strictly positive")

    # Leave a regular gap around every feature while retaining one central feature
    x_edges = np.linspace(7.05, 7.95, features_per_axis + 1)
    y_edges = np.linspace(45.05, 45.95, features_per_axis + 1)
    geometries = []
    for x_start, x_stop in zip(x_edges[:-1], x_edges[1:]):
        for y_start, y_stop in zip(y_edges[:-1], y_edges[1:]):
            x_margin = (x_stop - x_start) * 0.2
            y_margin = (y_stop - y_start) * 0.2
            geometries.append(box(x_start + x_margin, y_start + y_margin, x_stop - x_margin, y_stop - y_margin))

    # Constant burn values give every engine the same binary output
    vector = gpd.GeoDataFrame({"value": np.ones(len(geometries), dtype=np.uint8)}, geometry=geometries, crs=4326)
    vector.to_file(filename, driver="GPKG")
