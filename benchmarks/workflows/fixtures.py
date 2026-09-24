"""Create deterministic benchmark inputs and inspect bounded output samples."""

from __future__ import annotations

import os
import pathlib

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

from benchmarks.workflows.config import BenchmarkConfig
from geoutils._misc import import_optional

############################
# Size and output helpers
############################


def logical_raster_size_mb(config: BenchmarkConfig) -> float:
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


##############################
# Deterministic source files
##############################


def write_constant_raster(filename: str, config: BenchmarkConfig) -> None:
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


def write_polygon_raster(filename: str, config: BenchmarkConfig) -> None:
    """Write regularly spaced connected regions for polygonization scenarios."""

    if os.path.exists(filename):
        return
    if config.polygon_regions_per_axis < 1:
        raise ValueError("Polygon regions per axis must be strictly positive")

    # Separate value-one rectangles with nodata so every rectangle is one region
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = tiff_block_size(height, config.chunks[0])
    block_x = tiff_block_size(width, config.chunks[1])

    # Stream the patterned raster without allocating the complete benchmark input
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
        regions = config.polygon_regions_per_axis
        for _, window in destination.block_windows(1):
            # Pixel phases locate the inner half of every regular grid cell
            row_start = int(window.row_off)
            col_start = int(window.col_off)
            rows = np.arange(row_start, row_start + int(window.height))
            cols = np.arange(col_start, col_start + int(window.width))
            row_phase = ((rows + 0.5) * regions / height) % 1
            col_phase = ((cols + 0.5) * regions / width) % 1
            inside_rows = (row_phase >= 0.25) & (row_phase <= 0.75)
            inside_cols = (col_phase >= 0.25) & (col_phase <= 0.75)
            inside = inside_rows[:, None] & inside_cols[None, :]

            # Nodata gaps keep neighboring rectangles disconnected for both engines
            block = np.full(inside.shape, -99999, dtype=np.float32)
            block[inside] = config.raster_value
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


def write_point_source(filename: str, points_per_axis: int = 5) -> None:
    """Write a regular constant-valued point cloud for gridding scenarios."""

    if os.path.exists(filename):
        return
    if points_per_axis < 1:
        raise ValueError("Points per axis must be strictly positive")

    # Keep points away from the exact border so every geometry is unambiguous
    coords_x = np.linspace(7.05, 7.95, points_per_axis)
    coords_y = np.linspace(45.05, 45.95, points_per_axis)
    xx, yy = np.meshgrid(coords_x, coords_y)
    points = gpd.GeoDataFrame(
        {"z": np.ones(xx.size, dtype=np.float64)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=4326,
    )
    points.to_file(filename, driver="GPKG")
