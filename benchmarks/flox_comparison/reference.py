"""Calculate grouped statistics directly with Flox for external comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def grouped_stats(
    values: Any,
    groups: Any,
    mask: Any,
    categories: np.ndarray[Any, Any],
    *,
    use_dask: bool,
) -> pd.DataFrame:
    """Return finite count, mean and population standard deviation from direct Flox reductions."""

    import flox

    # Apply the common selection and each value's missing data before reducing both arrays together
    if use_dask:
        import dask.array as da

        selected = da.where(mask[None, ...] & da.isfinite(values), values, np.nan)
    else:
        selected = np.where(mask[None, ...] & np.isfinite(values), values, np.nan)

    # Ask Flox for the three public results used by the equivalent GeoUtils stats() call
    axes = tuple(range(-groups.ndim, 0))
    method = "map-reduce" if use_dask else None
    reductions = []
    for function, fill_value, dtype, finalize_kwargs in (
        ("count", 0, np.int64, None),
        ("nanmean", np.nan, np.float64, None),
        ("nanstd", np.nan, np.float64, {"ddof": 0}),
    ):
        result = flox.groupby_reduce(
            selected,
            groups,
            func=function,
            expected_groups=categories,
            sort=False,
            axis=axes,
            fill_value=fill_value,
            dtype=dtype,
            method=method,
            finalize_kwargs=finalize_kwargs,
        )[0]
        reductions.append(result)
    if use_dask:
        import dask

        reductions = list(dask.compute(*reductions))

    # Match the complete typed table returned by GeoUtils without calling any GeoUtils formatter
    count, mean, std = reductions
    data = np.stack((count, mean, std), axis=-1).transpose(1, 0, 2).reshape(len(categories), -1)
    columns = pd.MultiIndex.from_product(
        (("first", "second"), ("count", "mean", "std")),
        names=("value", "statistic"),
    )
    table = pd.DataFrame(data, index=pd.Index(categories, name="zone"), columns=columns)
    for name in ("first", "second"):
        table[(name, "count")] = table[(name, "count")].astype(np.int64)
    return table
