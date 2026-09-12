"""Prepare deterministic raster, point and pair inputs for variography benchmarks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import xarray as xr
from rasterio.transform import from_origin

import geoutils as gu
from geoutils._misc import import_optional


def prepare_variogram_pairs(n_pairs: int) -> xr.Dataset:
    """Create complete endpoint values at log-uniform distances, independently of spatial pair sampling.

    A fixed seed gives every estimator identical float64 inputs. The increasing difference amplitude creates a
    nonconstant variogram, while random endpoint offsets avoid a special case with one constant endpoint.
    """

    # Spread observations across short and long distances without enumerating a spatial distance matrix
    rng = np.random.default_rng(42)
    distances = np.exp(rng.uniform(0, np.log(1024), n_pairs))
    first = rng.normal(size=n_pairs)
    differences = rng.normal(size=n_pairs) * np.sqrt(1 - np.exp(-distances / 100))

    # Keep only the public pair layout consumed by Variogram.from_pairs()
    return xr.Dataset(
        {
            "distance": ("pair", distances),
            "value": (("pair", "endpoint"), np.column_stack((first, first + differences))),
        },
        attrs={"min_distance": 1.0, "max_distance": 1024.0},
    )


def prepare_pair_raster(size: int, execution_mode: Literal["eager", "dask"]) -> Any:
    """Create a smooth projected raster with scattered missing cells and 256 by 256 Dask chunks.

    Both modes start from the same prepared float32 values. Dask measures selected chunk reads and task scheduling
    from memory; this fixture does not measure disk throughput or claim a larger-than-memory contract.
    """

    # Vary values in both directions and remove a known fraction of cells to require finite endpoint checks
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    values = (np.sin(columns / 31) + np.cos(rows / 53)).astype(np.float32)
    values[(rows * size + columns) % 17 == 0] = np.nan
    transform = from_origin(0, size, 1, 1)

    # Expose the public object method in both modes while leaving lazy values uncomputed
    if execution_mode == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        array = da.from_array(values, chunks=(256, 256))
        return gu.RasterAccessor.from_array(array, transform, 32633, nodata=-99999).rst
    return gu.Raster.from_array(values, transform, 32633, nodata=-99999)


def prepare_pair_pointcloud(n_points: int) -> gu.PointCloud:
    """Create irregular projected points at roughly unit spacing with finite, smoothly varying values."""

    # Keep average point density constant so increasing point count increases the search extent
    rng = np.random.default_rng(42)
    coordinates = rng.uniform(0, np.sqrt(n_points), size=(n_points, 2))
    x, y = coordinates.T
    values = np.sin(x / 31) + np.cos(y / 53)
    return gu.PointCloud.from_xyz(x, y, values, crs=32633)
