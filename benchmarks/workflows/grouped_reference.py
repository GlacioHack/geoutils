"""Prepare equivalent grouped statistics workloads for GeoUtils and optional Flox comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from benchmarks.workflows.registry import ExecutionMode
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig


def prepare_grouped_reference(
    size: int,
    groups_per_axis: int,
    layout: Literal["local", "interleaved"],
    execution_mode: ExecutionMode,
) -> tuple[Any, Any, Any, NDArrayNum]:
    """Create two float64 value arrays with distinct gaps, declared integer groups and a shared mask.

    Local groups occupy rectangles; interleaved groups repeat throughout the raster. Values have shape
    (2, size, size), groups and mask have shape (size, size), and Dask uses 256 by 256 spatial chunks.
    Prepare these inputs before timing so both libraries measure only masking, reduction and result construction.
    """

    # 1/ Prepare two bounded signals with different missing observations
    positions = np.arange(size * size).reshape(size, size)
    first = (positions % 997).astype(np.float64) * 0.125
    values = np.stack((first, first * 2 + 10))
    values[0, positions % 17 == 0] = np.nan
    values[1, positions % 29 == 0] = np.nan

    # 2/ Keep group membership and the common selection independent of missing values
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    if layout == "local":
        groups = (rows * groups_per_axis // size) * groups_per_axis + columns * groups_per_axis // size
    else:
        groups = (rows % groups_per_axis) * groups_per_axis + columns % groups_per_axis
    groups = groups.astype(np.int32)
    mask = positions % 13 != 0
    categories = np.arange(groups_per_axis**2)

    # 3/ Give both implementations the same lazy arrays and spatial partitions
    if execution_mode == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        values = da.from_array(values, chunks=(1, 256, 256))
        groups = da.from_array(groups, chunks=(256, 256))
        mask = da.from_array(mask, chunks=(256, 256))
    return values, groups, mask, categories


def compute_grouped_reference(
    values: Any,
    groups: Any,
    mask: Any,
    categories: NDArrayNum,
    *,
    implementation: Literal["geoutils", "flox"],
    statistics: Sequence[str] = ("mean", "std"),
    mp_config: MultiprocConfig | None = None,
) -> pd.DataFrame:
    """Compute the same finite counts and selected statistics with the GeoUtils or Flox backend.

    Both backends return all declared groups, ignore missing and masked observations, and use population standard
    deviation (ddof=0). Flox uses its default eager engine and map-reduce for lazy labels. All lazy results are
    computed together, and both paths include construction of the same typed dataframe in the measured call.
    An optional multiprocessing configuration sends GeoUtils tiles to an already initialized worker pool; tile
    serialization, dispatch and result merging stay inside this call.
    """

    # Measure the selected reduction backend through the same complete GeoUtils public API
    from geoutils.stats import stats

    return stats(
        {"first": values[0], "second": values[1]},
        statistics,
        by={"zone": groups},
        categories={"zone": categories},
        mask=mask,
        observed=False,
        backend=implementation,
        mp_config=mp_config,
    )
