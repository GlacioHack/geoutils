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

"""Functionalities at the interface of rasters and vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Callable, Generic, TypeVar, Any
import warnings

import geopandas as gpd
import numpy as np
import rasterio as rio
import pandas as pd
from rasterio import features
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from geoutils._misc import import_optional, silence_rasterio_message
from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.multiproc.chunked import GeoGrid, ChunkedGeoGrid, _chunks2d_from_chunksizes_shape
from geoutils.multiproc.mparray import MultiprocConfig

if TYPE_CHECKING:
    from geoutils.raster.base import Raster, RasterBase, RasterType
    from geoutils.vector.vector import Vector
    import dask.array as da

###################
# 1/ POLYGONIZATION
###################

T = TypeVar("T")


# Base polygonize (using Rasterio/GeoPandas) + a custom canonicalize for consistent float32 behaviour
#####################################################################################################

def _canon_values(values, atol: float, *, out_dtype=None):
    """Canonicalize array values for equality-based float grouping."""

    # Integers or zero tolerances: do nothing
    v = np.asarray(values)
    if not np.issubdtype(v.dtype, np.floating) or atol <= 0:
        return v if out_dtype is None else v.astype(out_dtype, copy=False)

    # Convert to float64 before snapped mapping, to ensure enough precision
    v64 = v.astype(np.float64, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.rint(v64 / float(atol)).astype(np.int64)

    # Snap to tolerance grid, preserve dtype or follow user-input
    snapped64 = q.astype(np.float64) * float(atol)
    if out_dtype is None:
        out_dtype = v.dtype
    return snapped64.astype(out_dtype, copy=False)


def _canon_scalar(v, *, atol: float, out_dtype=None):
    """Canonicalize scalar values for equality-based float grouping."""

    if v is None:
        return None
    vv = np.asarray(v)

    # Integers or zero tolerances: do nothing
    if not np.issubdtype(vv.dtype, np.floating) or atol <= 0:
        x = vv.item()
        return x if out_dtype is None else np.asarray(x).astype(out_dtype).item()

    # Convert to float64 before snapped mapping, to ensure enough precision
    x64 = np.float64(vv)
    q = np.int64(np.rint(x64 / float(atol)))
    snapped64 = np.float64(q) * float(atol)

    # Preserve dtype or follow user-input
    if out_dtype is None:
        out_dtype = vv.dtype
    return np.asarray(snapped64).astype(out_dtype).item()

def _polygonize_base(
    values: NDArrayNum,
    mask: NDArrayBool,
    *,
    transform: rio.Affine,
    crs: Any,
    data_column_name: str,
    connectivity: Literal[4, 8] = 4,
    value_column: str = "raster_value",
    float_tol: float = 0.001,
) -> gpd.GeoDataFrame:
    """
    Polygonize from arrays.

    Backend-agnostic core function that we use both for eager base polygonize, and per-block polygonize in chunked
    backends.

    :param values: 2D array of values to polygonize.
    :param mask: 2D boolean-like array selecting pixels to polygonize.
    :param transform: Affine transform of the array.
    :param crs: CRS of the output.
    :param data_column_name: Name of the feature id column.
    :param value_column: Name of the value attribute column.
    :param float_tol: Tolerance threshold for float32 value inputs (overrides GDAL internal logic, to keep chunk
        consistent).
    :param connectivity: Connectivity level.
    """

    # Safety checks (should never be triggered at this stage)
    if values.ndim != 2 or mask.ndim != 2:
        raise ValueError("values and mask must be 2D arrays.")
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shape.")

    # If mask is empty, return empty but well-formed GeoDataFrame (important for chunked behaviour)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return gpd.GeoDataFrame({data_column_name: [], value_column: []}, geometry=[], crs=crs)

    # Now, we let rasterio.features.shapes yield (geometry, value) pairs
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    with silence_rasterio_message(param_name="MEM", warn_code="CPLE_AppDefined"):
        # We canonicalize values for float32 input (override Rasterio internal behaviour that is not exposed),
        # to ensure similar grouping behaviour in full-array or chunks
        v_arr = _canon_values(values, atol=float_tol)
        out_dtype = np.asarray(values).dtype
        results = (
            {"properties": {value_column: _canon_scalar(v, atol=float_tol, out_dtype=out_dtype)}, "geometry": s}
            for s, v in features.shapes(v_arr, mask=m, transform=transform, connectivity=connectivity)
        )
        gdf = gpd.GeoDataFrame.from_features(list(results))

    if len(gdf) == 0:
        return gpd.GeoDataFrame({data_column_name: [], value_column: []}, geometry=[], crs=crs)

    # Insert under the data column name
    gdf.insert(0, data_column_name, range(0, len(gdf)))
    gdf = gdf.set_geometry(col="geometry")
    gdf = gdf.set_crs(crs)

    return gdf


# Input check helper
# ##################

def _choose_polygonize_dtype(dtype: Any) -> str:
    """Choose a GeoPandas-compatible dtype for polygonize values."""

    # Compatibles data types
    dtype = np.dtype(dtype)
    gpd_dtypes = ["uint8", "uint16", "int16", "int32", "float32"]

    # Return if compatible, otherwise float32
    for candidate in gpd_dtypes:
        cand_dtype = np.dtype(candidate)
        if np.can_cast(dtype, cand_dtype, casting="safe"):
            return candidate

    return "float32"


@dataclass(frozen=True)
class _PolygonizePrepared:
    """
    Polygonize preparation object, compatible with all backends (and reducing the passing of volatile arguments).

    It only stores metadata and normalized parameters used by core function/block readers/tasks.
    """

    # Normalized target values
    target_values: Any

    # If True, the selection is effectively boolean (single class),
    # so labeling can run on the mask only (computational speed up)
    use_boolean_labeling: bool

    # GeoPandas-compatible dtype for values passed to rasterio.features.shapes
    final_dtype: str

    # Nodata behavior for "all" selection (valid pixels)
    nodata: Any

    # Whether the source is a mask raster
    is_mask_raster: bool

    # Connectivity and strategy are stored so readers can behave consistently if needed
    connectivity: Literal[4, 8]
    strategy: Literal["label_union", "label_stitch", "geometry_stitch"]

    # Column names (kept here to avoid drifting defaults across backends)
    value_column: str
    id_column: str
    data_column_name: str

    # Halo (i.e. required overlap) derived from strategy/connectivity (computed once)
    halo: int

    # FLoating tolerance
    float_tol: float


def _polygonize_prepare(
    source_raster: RasterBase,
    target_values: Any,
    data_column_name: str,
    *,
    connectivity: Literal[4, 8],
    strategy: Literal["label_union", "label_stitch", "geometry_stitch"],
    value_column: str = "raster_value",
    id_column: str = "component_id",
    float_tol: float = 0.001,
) -> _PolygonizePrepared:
    """
    Prepare polygonize inputs.

    This function performs metadata-level normalization without touching the array (safe for Dask).

    Preparation includes:
      1. Normalize `target_values` for mask rasters (force True, warn on non-boolean).
      2. Decide labeling mode:
         - boolean labeling (fast) when selection is a single class,
         - per-value labeling (exact) when selection is a set of discrete values.
      3. Choose `final_dtype` that `GeoPandas.from_features` can represent efficiently.
      4. Compute `halo` once from (strategy, connectivity) so backends are consistent.
      5. Store column naming (id/value/data_column_name) to avoid drift across backends.
      6. Store floating tolerance.
    """

    # 1/ Normalize target spec for mask rasters (cheap)
    eff = target_values
    is_mask_raster = bool(getattr(source_raster, "is_mask", False))
    if is_mask_raster:
        if target_values != "all" and (
            not isinstance(target_values, (int, np.integer, float, np.floating)) or target_values not in [0, 1]
        ):
            import warnings
            warnings.warn("Raster mask (boolean type) passed, using target value of 1 (True) and ignoring target "
                          f"values input {target_values!r}.")
        eff = True

    # 2/ Decide whether labeling can be boolean-only (faster computations)

    # Fast path is valid whenever all selected pixels are a single “class”:
    # - mask raster (eff=True)
    # - scalar target
    use_boolean_labeling = (
            is_mask_raster
            or isinstance(eff, (int, float, np.integer, np.floating))
    )

    # 3/ Choose dtype for polygonize values (GeoPandas compatibility)
    final_dtype = _choose_polygonize_dtype(getattr(source_raster, "dtype", np.float32))

    # 4/ Compute halo once (strategy-specific)
    if strategy == "label_union":
        halo = 0  # Not required for label_union
    elif strategy == "label_stitch":
        halo = 0 if connectivity == 4 else 1
    elif strategy == "geometry_stitch":
        halo = 1
    else:
        raise ValueError(
            f"Unsupported chunked strategy '{strategy}'. "
            "Should be one of 'label_union', 'label_stitch', or 'geometry_stitch'."
        )

    # 5/ Keep nodata in metadata (used later to interpret 'all' lazily)
    nodata = getattr(source_raster, "nodata", None)

    return _PolygonizePrepared(
        target_values=eff,
        use_boolean_labeling=use_boolean_labeling,
        final_dtype=final_dtype,
        nodata=nodata,
        is_mask_raster=is_mask_raster,
        connectivity=connectivity,
        strategy=strategy,
        value_column=value_column,
        id_column=id_column,
        data_column_name=data_column_name,
        halo=halo,
        float_tol=float_tol
    )

def _build_selection_mask(
    values: Any,
    prepared: _PolygonizePrepared,
) -> Any:
    """
    Build the uint8 selection mask for polygonize.

    Works with NumPy or Dask arrays. Compute eager mask for the former, and delayed mask for the latter.
    """
    eff = prepared.target_values

    # Helper to pick NumPy or Dask ufunc/module based on the array type
    def _xp(x):
        try:
            import dask.array as da  # type: ignore
            if isinstance(x, da.Array):
                return da
        except Exception:
            pass
        return np
    xp = _xp(values)

    v_dtype = np.dtype(getattr(values, "dtype", np.asarray(values).dtype))

    if eff == "all":
        nodata = prepared.nodata

        # If nodata is NaN, treat as "float validity" (not NaN)
        if nodata is not None:
            try:
                if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
                    nodata = None
            except Exception:
                pass

        # For float: validity is "not NaN" (regardless of nodata)
        if np.issubdtype(v_dtype, np.floating):
            return (~xp.isnan(values)).astype(np.uint8)

        # For int/bool: if nodata is set, exclude it; else everything valid
        if nodata is not None:
            return (values != nodata).astype(np.uint8)

        return xp.ones_like(values, dtype=np.uint8)

    # If mask, use directly
    if prepared.is_mask_raster:
        return values.astype(bool).astype(np.uint8)

    # If scalar, perform equality
    if isinstance(eff, (int, float, np.integer, np.floating)):
        return (values == eff).astype(np.uint8)

    # If range, perform logical comparison
    if isinstance(eff, tuple):
        return ((values >= eff[0]) & (values <= eff[1])).astype(np.uint8)

    raise ValueError("target_values must be a number, a tuple, a sequence, or 'all'.")


# Chunked execution abstractions (runner + reader)
##################################################

class _ChunkedRunner(Generic[T]):
    """
    Minimal execution interface for chunked backends.

    We define this class (and the _DaskRunner below) so that the same chunked strategies can run on:
      - Dask delayed + compute.
      - Multiprocessing cluster submit + gather.
    """

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def gather(self, handles: list[Any]) -> list[T]:
        raise NotImplementedError


class _DaskRunner(_ChunkedRunner[T]):
    """Runner implementation using dask.delayed + dask.compute. See _ChunkedRunner for details."""

    def __init__(self) -> None:
        import_optional("dask")
        self._dask = __import__("dask")

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        delayed = self._dask.delayed(func)
        return delayed(*args, **kwargs)

    def gather(self, handles: list[Any]) -> list[T]:
        if len(handles) == 0:
            return []
        return list(self._dask.compute(*handles))


class _MultiprocRunner(_ChunkedRunner[T]):
    """Runner implementation MultiprocConfig.cluster. See _ChunkedRunner for details."""

    def __init__(self, mp_config: MultiprocConfig) -> None:
        self._cluster = mp_config.cluster

    def submit(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        return self._cluster.launch_task(func, [*args], **kwargs)

    def gather(self, handles: list[Any]) -> list[T]:
        out: list[T] = []
        for h in handles:
            res = self._cluster.get_res(h)
            out.append(res)
        return out

@dataclass(frozen=True)
class _ChunkedDaskReader:
    """
    Reader for Dask-backed arrays.

    Details:
      - The values/mask/labels are Dask arrays with identical chunking.
      - The function read_block returns Dask slices (materialized inside tasks via np.asarray()).
      - The seam strip reads return label/value/mask strips sliced from the SAME arrays, guaranteeing consistent
      label ids between seam building and polygonization.
    """
    values: Any   # Dask array
    mask: Any     # Dask array (uint8-ish)
    labels: Any   # Dask array (int32)
    prepared: _PolygonizePrepared

    def read_block(
        self,
        b: dict[str, int],
        *,
        tiling_transform: rio.Affine | None = None,  # To maintain same signature with RasterReader
    ):
        ys, ye, xs, xe = b["ys"], b["ye"], b["xs"], b["xe"]
        return (
            self.values[ys:ye, xs:xe],
            self.mask[ys:ye, xs:xe],
            self.labels[ys:ye, xs:xe],
        )

    def read_vseam_strips(self, bL, bR, *, halo: int, shape, tiling_transform=None):
        """Read vertical seam for labels, values and mask."""

        # Left and right indexes
        ys = bL["ys"]
        ye = bL["ye"]

        xL0, xL1 = bL["xe"] - 1, bL["xe"]
        xR0, xR1 = bR["xs"], bR["xs"] + 1

        return (
            self.labels[ys:ye, xL0:xL1],
            self.labels[ys:ye, xR0:xR1],
            self.values[ys:ye, xL0:xL1],
            self.values[ys:ye, xR0:xR1],
            self.mask[ys:ye, xL0:xL1],
            self.mask[ys:ye, xR0:xR1],
        )

    def read_hseam_strips(self, bT, bB, *, halo: int, shape, tiling_transform=None):
        """Read horizontal seam for labels, values and mask."""

        # Top and bottom indexes
        xs = bT["xs"]
        xe = bT["xe"]

        yT0, yT1 = bT["ye"] - 1, bT["ye"]
        yB0, yB1 = bB["ys"], bB["ys"] + 1

        return (
            self.labels[yT0:yT1, xs:xe],
            self.labels[yB0:yB1, xs:xe],
            self.values[yT0:yT1, xs:xe],
            self.values[yB0:yB1, xs:xe],
            self.mask[yT0:yT1, xs:xe],
            self.mask[yB0:yB1, xs:xe],
        )

    def read_diag_corners(self, bTL, bBR):
        """Read diagonal corners (8-connectivity only) for labels, values and mask."""

        # Bottom-right of TL and top-left of BR
        yTL = bTL["ye"] - 1
        xTL = bTL["xe"] - 1
        yBR = bBR["ys"]
        xBR = bBR["xs"]
        return (
            self.labels[yTL:yTL + 1, xTL:xTL + 1],
            self.labels[yBR:yBR + 1, xBR:xBR + 1],
            self.values[yTL:yTL + 1, xTL:xTL + 1],
            self.values[yBR:yBR + 1, xBR:xBR + 1],
            self.mask[yTL:yTL + 1, xTL:xTL + 1],
            self.mask[yBR:yBR + 1, xBR:xBR + 1],
        )

    def read_antidiag_corners(self, bTR, bBL):
        """Read anti-diagonal corners (8-connectivity only) for labels, values and mask."""

        # Bottom-left of TR vs top-right of BL
        yTR = bTR["ye"] - 1
        xTR = bTR["xs"]
        yBL = bBL["ys"]
        xBL = bBL["xe"] - 1
        return (
            self.labels[yTR:yTR + 1, xTR:xTR + 1],
            self.labels[yBL:yBL + 1, xBL:xBL + 1],
            self.values[yTR:yTR + 1, xTR:xTR + 1],
            self.values[yBL:yBL + 1, xBL:xBL + 1],
            self.mask[yTR:yTR + 1, xTR:xTR + 1],
            self.mask[yBL:yBL + 1, xBL:xBL + 1],
        )


def _chunked_structure_from_connectivity(connectivity: Literal[4, 8]) -> NDArrayBool:
    """Helper to get structuring element depending on connectivity level."""
    if connectivity == 4:
        return np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=bool)
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("connectivity must be 4 or 8")


def _chunked_label_block_boolean(
    mask: NDArrayBool,
    *,
    connectivity: Literal[4, 8],
) -> NDArrayNum:
    """Label connected components of a boolean mask (faster logic than per value)."""
    from scipy.ndimage import label as ndi_label

    # Normalize mask and preallocate int32 label array (0 = background)
    m = np.asarray(mask, dtype=bool)

    out = np.zeros(m.shape, dtype=np.int32)

    # Fast exit if no selected pixels
    if not m.any():
        return out

    # Build structuring element from connectivity (4/8)
    structure = _chunked_structure_from_connectivity(connectivity)

    # Label connected components on the boolean mask
    lab, _ = ndi_label(m, structure=structure)

    return lab.astype(np.int32, copy=False)


def _chunked_label_block_per_value(
    values: NDArrayNum,
    mask: NDArrayBool,
    *,
    connectivity: Literal[4, 8],
    float_tol: float = 0.001,
) -> NDArrayNum:
    """Label connected components per exact value (slow but exact)."""
    from scipy.ndimage import label as ndi_label

    # Normalize mask and preallocate label array
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(values.shape, dtype=np.int32)

    # Fast exit if no selected pixels
    if not m.any():
        return out

    # Structuring element derived from connectivity
    structure = _chunked_structure_from_connectivity(connectivity)

    # Canonicalize values to enforce float tolerance consistency
    vQ = _canon_values(values, atol=float_tol)

    # Process each unique value independently
    uniq = np.unique(vQ[m])
    next_id = 1  # Label counter within block

    for v in uniq:

        # Select mask pixels of this value
        sel = m & (vQ == v)
        if not sel.any():
            continue

        # Label connected components for this value
        lab, n = ndi_label(sel, structure=structure)
        if n == 0:
            continue

        # Shift local labels to global block label space
        lab = lab.astype(np.int32, copy=False)
        lab[lab > 0] += (next_id - 1)

        # Write labeled pixels into output array
        out[lab > 0] = lab[lab > 0]

        next_id += n

    return out

@dataclass(frozen=True)
class _ChunkedRasterReader:
    """
    Reader for on-disk rasters using Raster.icrop(bounds) that works lazily with rio.windows.

    This reader mirrors Dask selection/label semantics via `_build_selection_mask` and `prepared.use_boolean_labeling`.
    """
    raster: Raster
    prepared: _PolygonizePrepared

    _cache: dict[tuple[int,int,int,int], tuple[NDArrayNum, NDArrayBool, NDArrayNum]] = field(
        default_factory=dict
    )

    def _get_block_np(self, b: dict[str, int], tiling_transform: rio.Affine) -> tuple[NDArrayNum, NDArrayBool, NDArrayNum]:

        # Get coordinates of window
        key = (b["ys"], b["ye"], b["xs"], b["xe"])
        if key in self._cache:
            return self._cache[key]

        # Convert to bounds
        w = rio.windows.Window(
            col_off=b["xs"], row_off=b["ys"],
            width=b["xe"] - b["xs"], height=b["ye"] - b["ys"]
        )
        bb = rio.coords.BoundingBox(*rio.windows.bounds(w, tiling_transform))

        # Get values, mask and labels for this block
        values, mask, labels = self._read_block_bounds(bb)
        self._cache[key] = (values, mask, labels)
        return values, mask, labels

    def _read_values_window(self, bounds: rio.coords.BoundingBox) -> NDArrayNum:
        """Read a window as numeric values (cast to prepared.final_dtype)."""

        # Crop raster lazily to the bounds of this block
        blk = self.raster.crop((bounds.left, bounds.bottom, bounds.right, bounds.top))

        # Convert masked-array to NaN array (if floatting only), without copy using filled()
        if np.issubdtype(self.prepared.final_dtype, np.integer):
            fill_value = self.raster.nodata
        else:
            fill_value = np.nan
        arr = np.asarray(blk.data.filled(fill_value))
        return arr.astype(self.prepared.final_dtype, copy=False)

    def _read_block_bounds(self, bounds: rio.coords.BoundingBox) -> tuple[NDArrayNum, NDArrayBool, NDArrayNum]:
        """Core window read: values + mask + labels."""

        # Read values
        values = self._read_values_window(bounds)

        # Build selection mask
        mask = _build_selection_mask(values, self.prepared)

        # Compute labels for this block
        if self.prepared.use_boolean_labeling:
            labels = _chunked_label_block_boolean(mask, connectivity=self.prepared.connectivity)
        else:
            labels = _chunked_label_block_per_value(values, mask, connectivity=self.prepared.connectivity,
                                                    float_tol=self.prepared.float_tol)

        return values, mask, labels

    def read_block(self, b: dict[str, int], *, tiling_transform: rio.Affine):
        return self._get_block_np(b, tiling_transform)

    def read_strip_bounds(self, bounds: rio.coords.BoundingBox) -> tuple[NDArrayNum, NDArrayBool, NDArrayNum]:
        """Read a thin strip by bounds (used by seam helpers)."""
        return self._read_block_bounds(bounds)

    def read_vseam_strips(self, bL, bR, *, halo: int, shape, tiling_transform: rio.Affine):
        """Read vertical seam for labels, values and mask."""

        ys = max(bL["ys"], bR["ys"])
        ye = min(bL["ye"], bR["ye"])
        if ye <= ys:
            # If empty overlap, return empty strips
            z = np.zeros((0, 1), dtype=np.int32)
            zv = np.zeros((0, 1), dtype=self.prepared.final_dtype)
            zm = np.zeros((0, 1), dtype=np.uint8)
            return z, z, zv, zv, zm, zm

        vL, mL, labL = self._get_block_np(bL, tiling_transform)
        vR, mR, labR = self._get_block_np(bR, tiling_transform)

        slL0, slL1 = ys - bL["ys"], ye - bL["ys"]
        slR0, slR1 = ys - bR["ys"], ye - bR["ys"]

        return (
            labL[slL0:slL1, -1:],  # Last col of left block labels
            labR[slR0:slR1, :1],  # First col of right block labels
            vL[slL0:slL1, -1:],
            vR[slR0:slR1, :1],
            mL[slL0:slL1, -1:],
            mR[slR0:slR1, :1],
        )

    def read_hseam_strips(self, bT, bB, *, halo: int, shape, tiling_transform: rio.Affine):
        """Read horizontal seam for labels, values and mask."""

        xs = max(bT["xs"], bB["xs"])
        xe = min(bT["xe"], bB["xe"])
        if xe <= xs:
            z = np.zeros((1, 0), dtype=np.int32)
            zv = np.zeros((1, 0), dtype=self.prepared.final_dtype)
            zm = np.zeros((1, 0), dtype=np.uint8)
            return z, z, zv, zv, zm, zm

        vT, mT, labT = self._get_block_np(bT, tiling_transform)
        vB, mB, labB = self._get_block_np(bB, tiling_transform)

        slT0, slT1 = xs - bT["xs"], xe - bT["xs"]
        slB0, slB1 = xs - bB["xs"], xe - bB["xs"]

        return (
            labT[-1:, slT0:slT1],  # Last row of top block labels
            labB[:1, slB0:slB1],  # First row of bottom block labels
            vT[-1:, slT0:slT1],
            vB[:1, slB0:slB1],
            mT[-1:, slT0:slT1],
            mB[:1, slB0:slB1],
        )

    def read_diag_corners(self, bTL, bBR, *, tiling_transform: rio.Affine):
        """Read diagonal corners (8-connectivity only) for labels, values and mask."""

        vTL, mTL, labTL = self._get_block_np(bTL, tiling_transform)
        vBR, mBR, labBR = self._get_block_np(bBR, tiling_transform)

        return (
            labTL[-1:, -1:],  # Bottom-right pixel label of TL
            labBR[:1, :1],  # Top-left pixel label of BR
            vTL[-1:, -1:],
            vBR[:1, :1],
            mTL[-1:, -1:],
            mBR[:1, :1],
        )

    def read_antidiag_corners(self, bTR, bBL, *, tiling_transform: rio.Affine):
        """Read anti-diagonal corners (8-connectivity only) for labels, values and mask."""

        vTR, mTR, labTR = self._get_block_np(bTR, tiling_transform)
        vBL, mBL, labBL = self._get_block_np(bBL, tiling_transform)

        return (
            labTR[-1:, :1],  # Bottom-left pixel label of TR
            labBL[:1, -1:],  # Top-right pixel label of BL
            vTR[-1:, :1],
            vBL[:1, -1:],
            mTR[-1:, :1],
            mBL[:1, -1:],
        )


# Chunked tiling helper (based on GeoGrid/ChunkedGeoGrid)
#########################################################

def _chunked_build_dst_geotiling(
    *,
    shape: tuple[int, int],
    transform: rio.Affine,
    crs: Any,
    chunks: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[ChunkedGeoGrid, list[GeoGrid], list[dict[str, int]]]:
    """
    Build tiling objects for a chunk scheme.

    :return: (tiling, block_geogrids, block_ids)
    """
    dst_geogrid = GeoGrid(transform=transform, shape=shape, crs=crs)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=chunks)

    # Block order is deterministic and matches other chunked operations
    block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    block_ids = dst_geotiling.get_block_locations()

    return dst_geotiling, block_geogrids, block_ids

# Step 1: Build global seam
###########################

def _chunked_seam_pairs_from_strips(
    left_labels: NDArrayNum,
    right_labels: NDArrayNum,
    left_values: NDArrayNum,
    right_values: NDArrayNum,
    left_mask: NDArrayBool,
    right_mask: NDArrayBool,
    *,
    left_block_id: int,
    right_block_id: int,
    connectivity: Literal[4, 8] = 4,
    axis: Literal["v", "h"] = "v",
    float_tol: float = 0.001,
) -> list[tuple[int, int]]:
    """
    Build union pairs across a seam.

    For 4-connectivity:
      - vertical seam: match aligned rows
      - horizontal seam: match aligned cols

    For 8-connectivity:
      - also match diagonals across the seam:
        vertical seam: row shifts +/-1
        horizontal seam: col shifts +/-1
    """
    l_lab = np.asarray(left_labels)
    r_lab = np.asarray(right_labels)
    l_val = _canon_values(left_values, atol=float_tol)
    r_val = _canon_values(right_values, atol=float_tol)
    l_m = np.asarray(left_mask, dtype=bool)
    r_m = np.asarray(right_mask, dtype=bool)

    # Ensure 2D strips
    if l_lab.ndim != 2 or r_lab.ndim != 2:
        raise ValueError("Seam strips must be 2D arrays.")
    if l_lab.shape != r_lab.shape:
        raise ValueError("Left/right seam strips must have identical shape.")

    pairs: list[tuple[int, int]] = []

    def _pairs_from_ok(ok: NDArrayBool, ll: NDArrayNum, rr: NDArrayNum) -> None:
        if not np.any(ok):
            return
        llv = ll[ok].astype(np.int64, copy=False)
        rrv = rr[ok].astype(np.int64, copy=False)
        left_nodes = (np.int64(left_block_id) << 32) | llv
        right_nodes = (np.int64(right_block_id) << 32) | rrv
        pairs.extend(zip(left_nodes.tolist(), right_nodes.tolist()))

    # Aligned adjacency (always, both 4 and 8)
    ok0 = (
        l_m & r_m &
        (l_lab > 0) & (r_lab > 0) &
        (l_val == r_val)
    )

    _pairs_from_ok(ok0, l_lab, r_lab)

    if connectivity == 4:
        return pairs

    # Diagonal adjacency (8-connectivity only)

    # Vertical seam strips are shape (H, 1): allow row shifts +/-1
    # Horizontal seam strips are shape (1, W): allow col shifts +/-1
    if axis == "v":
        # Right shifted up: left[r] adjacent to right[r-1]  (exclude first row of left)
        ok_up = (
            l_m[1:, :] & r_m[:-1, :] &
            (l_lab[1:, :] > 0) & (r_lab[:-1, :] > 0) &
            (l_val[1:, :] == r_val[:-1, :])
        )
        _pairs_from_ok(ok_up, l_lab[1:, :], r_lab[:-1, :])

        # Right shifted down: left[r] adjacent to right[r+1] (exclude last row of left)
        ok_down = (
            l_m[:-1, :] & r_m[1:, :] &
            (l_lab[:-1, :] > 0) & (r_lab[1:, :] > 0) &
            (l_val[:-1, :] == r_val[1:, :])
        )
        _pairs_from_ok(ok_down, l_lab[:-1, :], r_lab[1:, :])

    elif axis == "h":
        # Horizontal strips are (1, W): allow col shifts +/-1
        ok_left = (
            l_m[:, 1:] & r_m[:, :-1] &
            (l_lab[:, 1:] > 0) & (r_lab[:, :-1] > 0) &
            (l_val[:, 1:] == r_val[:, :-1])
        )
        _pairs_from_ok(ok_left, l_lab[:, 1:], r_lab[:, :-1])

        ok_right = (
            l_m[:, :-1] & r_m[:, 1:] &
            (l_lab[:, :-1] > 0) & (r_lab[:, 1:] > 0) &
            (l_val[:, :-1] == r_val[:, 1:])
        )
        _pairs_from_ok(ok_right, l_lab[:, :-1], r_lab[:, 1:])

    else:
        raise ValueError("axis must be 'v' or 'h'.")

    return pairs

# Union-find for seam stitching
class _UnionFind:
    """Union-find structure for integer node ids."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        p = self._parent.get(x, x)
        if p != x:
            self._parent[x] = self.find(p)
        return self._parent.get(x, x)

    def union(self, a: int, b: int) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            self._parent[ra] = rb
        elif self._rank[ra] > self._rank[rb]:
            self._parent[rb] = ra
        else:
            self._parent[rb] = ra
            self._rank[ra] += 1

    def compact_root_ids(self) -> dict[int, int]:
        roots = sorted({self.find(x) for x in self._parent})
        return {r: i + 1 for i, r in enumerate(roots)}

    @property
    def nodes(self) -> list[int]:
        return list(self._parent.keys())

def _chunked_union_mapping_from_pairs(pair_lists: list[list[tuple[int, int]]]) -> dict[int, int]:
    """
    Convert seam union pairs into a mapping node -> global component id.
    """
    uf = _UnionFind()
    for pairs in pair_lists:
        for a, b in pairs:
            uf.union(int(a), int(b))

    root_to_gid = uf.compact_root_ids()

    mapping: dict[int, int] = {}
    for node in uf.nodes:
        mapping[node] = root_to_gid[uf.find(node)]
    return mapping

def _build_seam_mapping(
    runner: _ChunkedRunner,
    reader: Any,
    *,
    tiling: ChunkedGeoGrid,
    block_ids: list[dict[str, int]],
    shape: tuple[int, int],
    connectivity: Literal[4, 8],
    float_tol: float = 0.001,
) -> dict[int, int]:
    """
    Build seam union mapping for label_union.

    Robustness guarantees:
      - does NOT assume block_ids are ordered row-major
      - computes neighbor relationships from (ys, xs) starts
      - uses overlap windows for seams (safe for irregular chunk sizes)
      - adds diagonal corner unions for 8-connectivity
      - halo is intentionally ignored for label_union correctness
    """
    t = tiling.grid.transform

    # Build a stable 2D indexing of blocks from their (ys, xs) starts
    ys_starts = sorted({b["ys"] for b in block_ids})
    xs_starts = sorted({b["xs"] for b in block_ids})

    y_index = {y: i for i, y in enumerate(ys_starts)}
    x_index = {x: i for i, x in enumerate(xs_starts)}

    ny = len(ys_starts)
    nx = len(xs_starts)

    # Map (iy, ix) -> bid, and also keep the block dict for each bid
    pos2bid: dict[tuple[int, int], int] = {}
    for bid, b in enumerate(block_ids):
        iy = y_index[b["ys"]]
        ix = x_index[b["xs"]]
        pos2bid[(iy, ix)] = bid

    seam_tasks: list[Any] = []

    def _seam_task(labA, labB, valA, valB, mA, mB, bidA, bidB, axis):
        return _chunked_seam_pairs_from_strips(
            np.asarray(labA),
            np.asarray(labB),
            np.asarray(valA),
            np.asarray(valB),
            np.asarray(mA, dtype=bool),
            np.asarray(mB, dtype=bool),
            left_block_id=int(bidA),
            right_block_id=int(bidB),
            connectivity=connectivity,
            axis=axis,
            float_tol=float_tol,
        )

    # Vertical seams (ix -> ix+1)
    for iy in range(ny):
        for ix in range(nx - 1):
            bidL = pos2bid.get((iy, ix))
            bidR = pos2bid.get((iy, ix + 1))
            if bidL is None or bidR is None:
                continue

            bL = block_ids[bidL]
            bR = block_ids[bidR]

            # Overlap in Y (safe for irregular chunking)
            ys = max(bL["ys"], bR["ys"])
            ye = min(bL["ye"], bR["ye"])
            if ye <= ys:
                continue

            labL, labR, valL, valR, mL, mR = reader.read_vseam_strips(
                {**bL, "ys": ys, "ye": ye},
                {**bR, "ys": ys, "ye": ye},
                halo=0,
                shape=shape,
                tiling_transform=t,
            )
            seam_tasks.append(runner.submit(_seam_task, labL, labR, valL, valR, mL, mR, bidL, bidR, "v"))

    # Horizontal seams (iy -> iy+1)
    for iy in range(ny - 1):
        for ix in range(nx):
            bidT = pos2bid.get((iy, ix))
            bidB = pos2bid.get((iy + 1, ix))
            if bidT is None or bidB is None:
                continue

            bT = block_ids[bidT]
            bB = block_ids[bidB]

            # Overlap in X (safe for irregular chunking)
            xs = max(bT["xs"], bB["xs"])
            xe = min(bT["xe"], bB["xe"])
            if xe <= xs:
                continue

            labT, labB_, valT, valB, mT, mB = reader.read_hseam_strips(
                {**bT, "xs": xs, "xe": xe},
                {**bB, "xs": xs, "xe": xe},
                halo=0,
                shape=shape,
                tiling_transform=t,
            )
            seam_tasks.append(runner.submit(_seam_task, labT, labB_, valT, valB, mT, mB, bidT, bidB, "h"))

    # Diagonal corners for 8-connectivity
    if connectivity == 8:
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # TL -> BR
                bidTL = pos2bid.get((iy, ix))
                bidBR = pos2bid.get((iy + 1, ix + 1))
                if bidTL is not None and bidBR is not None:
                    bTL = block_ids[bidTL]
                    bBR = block_ids[bidBR]
                    try:
                        labA, labB, valA, valB, mA, mB = reader.read_diag_corners(bTL, bBR, tiling_transform=t)
                    except TypeError:
                        labA, labB, valA, valB, mA, mB = reader.read_diag_corners(bTL, bBR)
                    seam_tasks.append(runner.submit(_seam_task, labA, labB, valA, valB, mA, mB, bidTL, bidBR, "v"))

                # TR -> BL
                bidTR = pos2bid.get((iy, ix + 1))
                bidBL = pos2bid.get((iy + 1, ix))
                if bidTR is not None and bidBL is not None:
                    bTR = block_ids[bidTR]
                    bBL = block_ids[bidBL]
                    try:
                        labA, labB, valA, valB, mA, mB = reader.read_antidiag_corners(bTR, bBL, tiling_transform=t)
                    except TypeError:
                        labA, labB, valA, valB, mA, mB = reader.read_antidiag_corners(bTR, bBL)
                    seam_tasks.append(runner.submit(_seam_task, labA, labB, valA, valB, mA, mB, bidTR, bidBL, "v"))

    seam_pairs_lists = runner.gather(seam_tasks)
    return _chunked_union_mapping_from_pairs(seam_pairs_lists)

# Step 2: Polygonize per block
##############################

def _touches_block_fast(
    gdf: gpd.GeoDataFrame,
    *,
    block_bounds: rio.coords.BoundingBox,
) -> np.ndarray:
    """
    Fast conservative test for polygons touching the block boundary.

    Uses geometry bounds and isclose against block bbox.
    """
    if len(gdf) == 0:
        return np.zeros((0,), dtype=bool)

    minx, miny, maxx, maxy = gdf.geometry.bounds.T.values
    bminx, bminy, bmaxx, bmaxy = block_bounds

    return (
        np.isclose(minx, bminx) | np.isclose(maxx, bmaxx) |
        np.isclose(miny, bminy) | np.isclose(maxy, bmaxy)
    )

def _chunked_polygonize_block_labels(
    labels: NDArrayNum,
    values: NDArrayNum,
    mask: NDArrayBool,
    *,
    transform: rio.Affine,
    value_column: str,
    connectivity: Literal[4, 8],
    float_tol: float = 0.001,
    local_id_column: str = "local_id",
) -> gpd.GeoDataFrame:
    """
    Polygonize a label image, attaching raster value per polygon.

    (Performance note: we avoid np.argwhere per polygon by precomputing label to value lookup once)
    """
    mm = np.asarray(mask, dtype=bool) & (labels > 0)
    if not mm.any():
        return gpd.GeoDataFrame({local_id_column: [], value_column: []}, geometry=[])

    # 1/ Precompute representative value per label in O(N) over selected pixels.
    # We take the first occurrence of each label in the mask.
    lab_flat = np.asarray(labels, dtype=np.int32).ravel()
    val_flat = _canon_values(np.asarray(values), atol=float_tol).ravel()
    mm_flat = np.asarray(mm, dtype=bool).ravel()

    lab_sel = lab_flat[mm_flat]
    val_sel = val_flat[mm_flat]

    # Get first index per label among selected pixels
    uniq_labs, first_idx = np.unique(lab_sel, return_index=True)
    rep_vals = val_sel[first_idx]

    # Map label -> value using dict; dict is typically fine (uniq_labs count ~= polygons)
    out_dtype = np.asarray(values).dtype
    lab2val = {int(l): _canon_scalar(rep_vals[i], atol=float_tol, out_dtype=out_dtype) for i, l in enumerate(uniq_labs)}

    # 2/ Polygonize label raster and attach values via lookup.
    feats: list[dict[str, Any]] = []
    lab_raster = np.asarray(labels, dtype=np.int32, order="C")

    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
    with silence_rasterio_message(param_name="MEM", warn_code="CPLE_AppDefined"):
        for geom, lab in features.shapes(lab_raster, mask=mm, transform=transform, connectivity=connectivity):
            lab_i = int(lab)
            if lab_i == 0:
                continue
            # Lookup representative value (should exist for all labels in mm)
            v = lab2val.get(lab_i)
            if v is None:
                continue
            feats.append({"properties": {local_id_column: lab_i, value_column: v}, "geometry": geom})
        gdf = gpd.GeoDataFrame.from_features(feats)

    return gdf


def _polygonize_block_from_labels(
    reader: Any,
    *,
    block_id: int,
    b: dict[str, int],
    tiling_transform: rio.Affine,
    block_transform: rio.Affine,
    block_bounds: rio.coords.BoundingBox,
    connectivity: Literal[4, 8],
    float_tol: float,
    value_column: str,
) -> gpd.GeoDataFrame:
    """
    Polygonize one block from its label image.

    Expects the reader to provide labels for the block.
    Returns columns: [local_id, value_column, geometry] + metadata [_block_id, _touches_block].
    """
    values, mask, labels = reader.read_block(b, tiling_transform=tiling_transform)
    assert labels is not None  # label strategies only

    v_np = np.asarray(values)
    m_np = np.asarray(mask)
    l_np = np.asarray(labels)

    g = _chunked_polygonize_block_labels(
        labels=l_np,
        values=v_np,
        mask=m_np,
        transform=block_transform,
        value_column=value_column,
        connectivity=connectivity,
        local_id_column="local_id",
        float_tol=float_tol,
    )
    if len(g) == 0:
        return g

    g["_block_id"] = block_id
    g["_touches_block"] = _touches_block_fast(g, block_bounds=block_bounds)
    return g


def _attach_label_union_ids(
    g: gpd.GeoDataFrame,
    *,
    block_id: int,
    id_column: str,
    seam_mapping: dict[int, int],
) -> gpd.GeoDataFrame:
    if len(g) == 0:
        return g

    # Extract local component ids (per-block labeling)
    local_ids = g["local_id"].astype(np.int64).to_numpy()

    # Build 64-bit global node id
    nodes = (np.int64(block_id) << 32) | local_ids

    # Allocate output global ids
    # Positive ids = merged via seam_mapping
    # Negative ids = unique (unmerged) components
    gids = np.empty(nodes.shape[0], dtype=np.int64)

    for i, n in enumerate(nodes):
        n_int = int(n)
        # Lookup seam-merged id if present
        gid = seam_mapping.get(n_int)
        # Use merged id or assign stable negative fallback (avoid -0)
        gids[i] = gid if gid is not None else -(n_int + 1)

    # Attach global id column
    g[id_column] = gids
    return g


def _polygonal_only(g):
    """
    Return a Polygon/MultiPolygon containing only polygonal parts of `g`.
    Returns None if no polygonal area exists.
    """
    # Guard against null or empty geometry
    if g is None:
        return None
    if g.is_empty:
        return None

    # Fast path: already purely polygonal
    gt = g.geom_type
    if gt in ("Polygon", "MultiPolygon"):
        return g

    # Extract polygonal components from collections or mixed geometries
    parts = []
    # Attempt to iterate over sub-geometries (collections/multiparts)
    try:
        geoms = list(g.geoms)  # works for collections/multiparts
    except Exception:
        geoms = [g]

    for x in geoms:
        # Skip null or empty parts
        if x is None or x.is_empty:
            continue
        if x.geom_type == "Polygon":
            parts.append(x)
        elif x.geom_type == "MultiPolygon":
            parts.extend(list(x.geoms))
        elif x.geom_type == "GeometryCollection":
            # Recurse one level into nested collections
            for y in x.geoms:
                py = _polygonal_only(y)
                if py is None:
                    continue
                if py.geom_type == "Polygon":
                    parts.append(py)
                elif py.geom_type == "MultiPolygon":
                    parts.extend(list(py.geoms))

    # No polygonal area found
    if not parts:
        return None

    # Merge polygonal pieces into single geometry
    out = unary_union(parts)
    if out is None or out.is_empty:
        return None

    # If result is already polygonal, return directly
    if out.geom_type in ("Polygon", "MultiPolygon"):
        return out

    # Rare case: union still returns GeometryCollection, we recurse once more
    out2 = _polygonal_only(out)
    return out2

def _chunked_clip_gdf_to_bounds_polygonal(
    gdf: gpd.GeoDataFrame,
    bounds: rio.coords.BoundingBox,
    *,
    keep_border: bool = True,
    area_eps: float = 0.0,
) -> gpd.GeoDataFrame:
    """
    Clip to bounds and keep only polygonal area parts.

    Optionally, keep_border (polygons that touch the clip box boundary).
    """
    if len(gdf) == 0:
        return gdf

    clip_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Filter fast
    sel = gdf.geometry.intersects(clip_geom)
    if not np.any(sel):
        return gdf.iloc[0:0].copy()

    out = gdf.loc[sel].copy()

    # Intersection can produce None in some cases, so guard it
    def _safe_intersection(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            return geom.intersection(clip_geom)
        except Exception:
            return None

    out["geometry"] = out.geometry.apply(_safe_intersection)

    # Clean + polygonal-only extraction
    def _clean_poly(geom):
        if geom is None or geom.is_empty:
            return None

        # fix invalids (prefer make_valid if available)
        try:
            mv = getattr(shapely, "make_valid", None)
            if mv is not None:
                geom = mv(geom)
            else:
                geom = geom.buffer(0)
        except Exception:
            pass

        geom = _polygonal_only(geom)
        if geom is None:
            return None

        # Optional: drop zero-area slivers
        if area_eps > 0.0 and float(geom.area) <= area_eps:
            return None

        if not keep_border:
            # Drop polygons that only touch the boundary (optional stricter mode)
            try:
                if geom.boundary.intersection(clip_geom.boundary).length > 0 and geom.within(clip_geom):
                    pass
            except Exception:
                pass

        return geom

    out["geometry"] = out.geometry.apply(_clean_poly)
    out = out[out.geometry.notna()].copy()
    out = out[~out.geometry.is_empty]

    return out

def _polygonize_block_geometry_halo(
    reader: Any,
    *,
    b: dict[str, int],
    tiling_transform: rio.Affine,
    crs: Any,
    data_column_name: str,
    value_column: str,
    halo: int,
    connectivity: Literal[4, 8],
    float_tol: float,
    shape: tuple[int, int],
) -> gpd.GeoDataFrame:
    """
    Polygonize one block by running shapes() on a halo window, then clipping to the interior.

    Returns polygons for the interior part only (halo duplicates removed).
    """
    ys0, ye0, xs0, xe0 = b["ys"], b["ye"], b["xs"], b["xe"]

    ys = max(0, ys0 - halo)
    ye = min(shape[0], ye0 + halo)
    xs = max(0, xs0 - halo)
    xe = min(shape[1], xe0 + halo)

    # Halo transform: global -> halo local origin
    t_halo = tiling_transform * rio.Affine.translation(xs, ys)
    b_halo = {"ys": ys, "ye": ye, "xs": xs, "xe": xe}

    values, mask, _ = reader.read_block(b_halo, tiling_transform=tiling_transform)
    v_np = np.asarray(values)
    m_np = np.asarray(mask)

    g = _polygonize_base(
        v_np,
        m_np,
        transform=t_halo,
        crs=crs,
        data_column_name=data_column_name,
        value_column=value_column,
        connectivity=connectivity,
        float_tol=float_tol,
    )
    if len(g) == 0:
        return g

    # Clip back to interior window
    win_interior = rio.windows.Window(
        col_off=xs0 - xs,
        row_off=ys0 - ys,
        width=xe0 - xs0,
        height=ye0 - ys0,
    )
    bb_interior = rio.coords.BoundingBox(*rio.windows.bounds(win_interior, t_halo))

    # Clip to bounds respecting border and keeping only polygonal types (clipping often creates points/linestrings)
    g = _chunked_clip_gdf_to_bounds_polygonal(
        g,
        bb_interior,
        keep_border=True,
        area_eps=0,
    )

    # Drop per-call id; re-index later
    return g.drop(columns=[data_column_name], errors="ignore")

def _concat_nonempty(parts: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Concatenate GeoDataFrames, skipping empty/None entries."""
    keep = [p for p in parts if p is not None and len(p) > 0]
    return pd.concat(keep, ignore_index=True) if keep else gpd.GeoDataFrame()


# Step 3: Stitching or union of chunked polygons
################################################

def _chunked_stitch_geometries_by_value_graph_connectivity(
    gdf: gpd.GeoDataFrame,
    *,
    value_column: str,
    connectivity: Literal[4, 8],
    predicate: Literal["touches", "intersects"] | None = None,
) -> gpd.GeoDataFrame:
    """
    Stitch geometries within each value group using a geometry adjacency graph.

    Connectivity handling:
      - 8-connectivity: polygons connected if they touch at edge OR corner -> predicate="touches"
      - 4-connectivity: polygons connected only if they share an edge (not just a corner).
        Implemented by filtering candidate pairs where boundary intersection has non-zero length.

    :param gdf: Input GeoDataFrame with at least [value_column, geometry].
    :param value_column: Column holding raster values.
    :param connectivity: 4 or 8.
    :param predicate: Optional override for spatial join predicate. If None, chosen from connectivity.
    """
    if len(gdf) == 0:
        return gdf

    # For 4-connectivity we start from "intersects" and then filter to edge-sharing only
    # "touches" is too strict because polygons that overlap due to numeric issues could be missed
    if predicate is None:
        predicate = "touches" if connectivity == 8 else "intersects"

    out_parts: list[gpd.GeoDataFrame] = []

    for val, sub in gdf.groupby(value_column, sort=False):
        sub = sub.reset_index(drop=True)
        if len(sub) <= 1:
            out_parts.append(sub)
            continue

        # Spatial self-join to get candidate adjacency pairs
        sj = gpd.sjoin(sub, sub, predicate=predicate)
        sj = sj[sj.index != sj["index_right"]]

        if len(sj) == 0:
            out_parts.append(sub)
            continue

        # For 4-connectivity, remove corner-touch-only pairs:
        # keep pairs that share a boundary segment (intersection length > 0).
        if connectivity == 4:
            left_geom = sub.geometry.iloc[sj.index.values].values
            right_geom = sub.geometry.iloc[sj["index_right"].values].values

            # Compute boundary intersection length; vectorized enough for candidate set sizes.
            keep = []
            for g1, g2 in zip(left_geom, right_geom):
                inter = g1.boundary.intersection(g2.boundary)
                keep.append(inter.length > 0)

            keep = np.asarray(keep, dtype=bool)
            sj = sj.iloc[keep.nonzero()[0]]
            if len(sj) == 0:
                out_parts.append(sub)
                continue

        # Union-find connected components in adjacency graph
        uf = _UnionFind()
        for i in range(len(sub)):
            uf.add(i)

        for a, b in zip(sj.index.values, sj["index_right"].values):
            uf.union(int(a), int(b))

        root_to_gid = uf.compact_root_ids()
        comp = np.array([root_to_gid[uf.find(i)] for i in range(len(sub))], dtype=np.int64)

        sub = sub.assign(_comp_id=comp)
        dissolved = sub.dissolve(by="_comp_id", as_index=False, aggfunc="first")
        dissolved[value_column] = val
        dissolved = dissolved.drop(columns=["_comp_id"], errors="ignore")
        out_parts.append(dissolved)

    return pd.concat(out_parts, ignore_index=True)


def _chunked_stitch_by_value_neighbor_blocks(
    gdf: gpd.GeoDataFrame,
    *,
    value_column: str,
    connectivity: Literal[4, 8],
    block_ids: list[dict[str, int]],
    chunks: tuple[tuple[int, ...], tuple[int, ...]],
) -> gpd.GeoDataFrame:
    """
    Stitch by geometry adjacency, but only for polygons that (1) touch block boundary
    and (2) are in neighboring blocks. Much faster than full self-join.
    """
    if len(gdf) == 0:
        return gdf

    if "_block_id" not in gdf.columns or "_touches_block" not in gdf.columns:
        # Fallback to global method if metadata is missing
        return _chunked_stitch_geometries_by_value_graph_connectivity(
            gdf, value_column=value_column, connectivity=connectivity
        )

    ny, nx = len(chunks[0]), len(chunks[1])

    def _neighbors(bid: int) -> list[int]:
        iy, ix = divmod(bid, nx)
        nbs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                if connectivity == 4 and (abs(dx) + abs(dy) != 1):
                    continue
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < ny and 0 <= jx < nx:
                    nbs.append(jy * nx + jx)
        return nbs

    # Build index of candidate polygons per block per value
    cand = gdf[gdf["_touches_block"].astype(bool)].copy()
    if len(cand) == 0:
        return gdf  # nothing to stitch

    # Prepare UF over row indices in full gdf
    uf = _UnionFind()
    for i in range(len(gdf)):
        uf.add(i)

    # For each value group, stitch only neighbor-block pairs
    # We only compare candidates, but unions are applied to the full gdf indices.
    for val, sub in cand.groupby(value_column, sort=False):
        # map block -> indices in original gdf
        by_block: dict[int, np.ndarray] = {}
        for bid, sbb in sub.groupby("_block_id", sort=False):
            by_block[int(bid)] = sbb.index.to_numpy()

        # For each block, join with neighbor blocks (only once: bid < nb)
        for bid, idx_a in by_block.items():
            for nb in _neighbors(bid):
                if nb not in by_block:
                    continue
                if bid > nb:
                    continue
                idx_b = by_block[nb]

                # Spatial join between these two small sets
                ga = gdf.loc[idx_a, "geometry"]
                gb = gdf.loc[idx_b, "geometry"]
                gA = gpd.GeoDataFrame(geometry=ga)
                gB = gpd.GeoDataFrame(geometry=gb)

                # Use sjoin to find touching/intersecting pairs
                # For 8-connectivity: touch is enough
                # For 4-connectivity: intersect + boundary-length>0 filtering
                pred = "touches" if connectivity == 8 else "intersects"

                sj = gpd.sjoin(gA, gB, predicate=pred, how="inner")
                if len(sj) == 0:
                    continue

                if connectivity == 4:
                    # Filter to edge-sharing only (no corner-only)
                    left_geom = ga.geometry.loc[sj.index].values
                    right_geom = gb.geometry.loc[sj["index_right"]].values
                    keep = []
                    for g1, g2 in zip(left_geom, right_geom):
                        inter = g1.boundary.intersection(g2.boundary)
                        keep.append(inter.length > 0)
                    sj = sj.iloc[np.asarray(keep, dtype=bool).nonzero()[0]]
                    if len(sj) == 0:
                        continue

                # Union corresponding original indices
                for a, b in zip(sj.index.values, sj["index_right"].values):
                    uf.union(int(a), int(b))

    # Relabel and dissolve
    root_to_gid = uf.compact_root_ids()
    comp = np.array([root_to_gid[uf.find(i)] for i in range(len(gdf))], dtype=np.int64)
    out = gdf.assign(_comp_id=comp)
    out = out.dissolve(by="_comp_id", as_index=False, aggfunc="first")
    out = out.drop(columns=["_comp_id"], errors="ignore")

    return out.reset_index(drop=True)

# Common chunked strategy wrapper
##################################

def _chunked_polygonize_core(
    runner: _ChunkedRunner,
    reader: Any,
    tiling: ChunkedGeoGrid,
    block_geogrids: list[GeoGrid],
    block_ids: list[dict[str, int]],
    *,
    crs: Any,
    prepared: _PolygonizePrepared,
    shape: tuple[int, int],
) -> gpd.GeoDataFrame:
    """
    Chunked polygonize core (shared by Dask and Multiprocessing backends).

    The function supports three chunked strategies:

    1) ``label_union`` (exact, label-based)
       - Build connected-component labels per block.
       - Scan seams between neighboring blocks and build a union-find mapping that merges labels
         that touch across seams (and have identical raster value).
       - Polygonize each block from its label image, attach global ids via the seam mapping,
         then dissolve by global id.
       -> Goal: reproduce the result of a global connected-component labeling + polygonize.

    2) ``label_stitch`` (label-based + vector stitching)
       - Build connected-component labels per block.
       - Polygonize each block from its label image (no halo), producing polygons that are
         guaranteed connected *within* the block (and carry the raster value).
       - Stitch polygons across neighboring blocks using vector geometry adjacency within each
         value group (only polygons touching the block boundary are considered for cross-block merges).
       -> Goal: avoid seam union-find; merge splits across blocks in vector space.

    3) ``geometry_stitch`` (direct shapes + halo + vector stitching)
       - Do NOT compute labels.
       - For each block, polygonize directly with ``rasterio.features.shapes`` on a halo-expanded
         window (to avoid cutting polygons at block edges), then clip back to the block interior.
       - Stitch polygons across neighboring blocks using vector geometry adjacency within each
         value group.
       -> Goal: avoid labeling entirely; rely on shapes() + halo + vector stitching.

    Notes on step (2) “polygonize each block”
    -----------------------------------------
    - For ``label_union`` and ``label_stitch``: polygonization is performed on the *label raster*
      (one polygon per labeled component) and a representative raster value is attached to each polygon.
    - For ``geometry_stitch``: polygonization is performed on the *value raster* directly
      (shapes(values, mask=...)) on a halo window, then clipped to the block interior.
    """
    t = tiling.grid.transform

    # 1) Optional seam mapping (only required for label_union)
    seam_mapping: dict[int, int] | None = None
    if prepared.strategy == "label_union":
        seam_mapping = _build_seam_mapping(
            runner,
            reader,
            tiling=tiling,
            block_ids=block_ids,
            shape=shape,
            connectivity=prepared.connectivity,
            float_tol=prepared.float_tol,
        )

    # 2) Per-block polygonization (shared dispatch)
    if prepared.strategy in ("label_union", "label_stitch"):

        def _block_task(
            i: int,
            b: dict[str, int],
            block_transform: rio.Affine,
            block_bounds: rio.coords.BoundingBox,
        ) -> gpd.GeoDataFrame:
            g = _polygonize_block_from_labels(
                reader,
                block_id=i,
                b=b,
                tiling_transform=t,
                block_transform=block_transform,
                block_bounds=block_bounds,
                connectivity=prepared.connectivity,
                value_column=prepared.value_column,
                float_tol=prepared.float_tol,
            )
            if len(g) == 0:
                return g

            # Attach global ids only for label_union
            if prepared.strategy == "label_union":
                assert seam_mapping is not None
                g = _attach_label_union_ids(
                    g,
                    block_id=i,
                    id_column=prepared.id_column,
                    seam_mapping=seam_mapping,
                )

            # The local ids are internal to labeling only
            return g.drop(columns=["local_id"], errors="ignore")

        handles = [
            runner.submit(_block_task, i, block_ids[i], block_geogrids[i].transform, block_geogrids[i].bounds)
            for i in range(len(block_ids))
        ]

    elif prepared.strategy == "geometry_stitch":

        def _block_task(b: dict[str, int]) -> gpd.GeoDataFrame:
            return _polygonize_block_geometry_halo(
                reader,
                b=b,
                tiling_transform=t,
                crs=crs,
                data_column_name=prepared.data_column_name,
                value_column=prepared.value_column,
                halo=prepared.halo,
                connectivity=prepared.connectivity,
                float_tol=prepared.float_tol,
                shape=shape,
            )

        handles = [runner.submit(_block_task, block_ids[i]) for i in range(len(block_ids))]

    else:
        raise ValueError(
            f"Unsupported chunked strategy '{prepared.strategy}'. "
            "Should be one of 'label_union', 'label_stitch', or 'geometry_stitch'."
        )

    parts = runner.gather(handles)
    gdf = _concat_nonempty(parts)

    if len(gdf) == 0:
        return gpd.GeoDataFrame({prepared.id_column: [], prepared.value_column: []}, geometry=[], crs=crs)
    gdf = gdf.set_crs(crs)

    # 3) Stitch polygons
    if prepared.strategy == "label_union":

        # Global ids already attached; just dissolve
        out = gdf.dissolve(by=prepared.id_column, as_index=False, aggfunc="first")
        return out.reset_index(drop=True)

    # label_stitch / geometry_stitch: stitch across neighbor blocks then assign ids
    stitched = _chunked_stitch_by_value_neighbor_blocks(
        gdf,
        value_column=prepared.value_column,
        connectivity=prepared.connectivity,
        block_ids=block_ids,
        chunks=tiling.chunks,
    )
    stitched.insert(0, prepared.id_column, range(1, len(stitched) + 1))
    stitched = stitched.set_geometry("geometry").set_crs(crs)

    return stitched.reset_index(drop=True)


# Dask and Multiprocessing wrappers
###################################

def _dask_polygonize(
    array: da.Array,
    prepared: _PolygonizePrepared,
    *,
    transform: rio.Affine,
    crs: Any,
) -> gpd.GeoDataFrame:
    """
    Polygonize a Dask-backed raster without materializing arrays.
    """
    import_optional("dask")
    import dask.array as da

    if not hasattr(array, "chunks"):
        raise ValueError("_dask_polygonize expects dask-backed raster data.")

    # Keep numeric values; cast lazily for GeoPandas/rasterio compatibility
    values = array.astype(prepared.final_dtype)

    # Build selection mask lazily
    mask = _build_selection_mask(values, prepared)

    # Labels lazily if needed
    labels = None
    if prepared.strategy in ("label_union", "label_stitch"):

        def _label_wrap(v_blk: NDArrayNum, m_blk: NDArrayBool) -> NDArrayNum:
            v_np = np.asarray(v_blk)
            m_np = np.asarray(m_blk)
            if prepared.use_boolean_labeling:
                return _chunked_label_block_boolean(m_np, connectivity=prepared.connectivity)
            return _chunked_label_block_per_value(v_np, m_np, connectivity=prepared.connectivity,
                                                  float_tol=prepared.float_tol)

        labels = da.map_blocks(_label_wrap, values, mask, dtype=np.int32, chunks=values.chunks)
    else:
        # not used for geometry_stitch
        labels = da.zeros_like(values, dtype=np.int32)

    # Build tiling directly from Dask chunks
    tiling, block_geogrids, block_ids = _chunked_build_dst_geotiling(
        shape=(int(array.shape[0]), int(array.shape[1])),
        transform=transform,
        crs=crs,
        chunks=values.chunks,
    )

    # Create common runner/reader mirrored with Dask to use common logic
    runner = _DaskRunner()
    reader = _ChunkedDaskReader(values=values, mask=mask, labels=labels, prepared=prepared)

    # reader = _ChunkedDaskReader(values=values, prepared=prepared) #mask=mask, labels=labels)

    # Call core function
    return _chunked_polygonize_core(
        runner,
        reader,
        tiling,
        block_geogrids,
        block_ids,
        crs=crs,
        prepared=prepared,
        shape=(int(array.shape[0]), int(array.shape[1])),
    )


def _multiproc_polygonize(
    source_raster: Raster,
    prepared: _PolygonizePrepared,
    *,
    mp_config: MultiprocConfig,
) -> gpd.GeoDataFrame:
    """
    Multiprocessing polygonize wrapper.

    Uses windowed reads and reuses the same chunked core strategy functions as
    Dask, via runner/reader abstractions.
    """
    shape = (int(source_raster.shape[0]), int(source_raster.shape[1]))

    # Determine chunks from mp_config
    chunksizes = (mp_config.chunk_size, mp_config.chunk_size)
    chunks = _chunks2d_from_chunksizes_shape(chunksizes=chunksizes, shape=shape)

    # Build tiling from the chosen chunking scheme
    tiling, block_geogrids, block_ids = _chunked_build_dst_geotiling(
        shape=shape,
        transform=source_raster.transform,
        crs=source_raster.crs,
        chunks=chunks,
    )

    # Create common runner/reader mirrored with Dask to use common logic
    runner = _MultiprocRunner(mp_config)
    reader = _ChunkedRasterReader(
        raster=source_raster,
        prepared=prepared,
    )

    # Call chunked core function
    return _chunked_polygonize_core(
        runner,
        reader,
        tiling,
        block_geogrids,
        block_ids,
        crs=source_raster.crs,
        prepared=prepared,
        shape=shape,
    )


# Parent polygonize (dispatch only)
###################################

def _polygonize(
    source_raster: RasterType,
    target_values: Any,
    data_column_name: str,
    *,
    connectivity: Literal[4, 8] = 4,
    strategy: Literal["label_union", "label_stitch", "geometry_stitch"] = "label_union",
    mp_config: "MultiprocConfig | None" = None,
    float_tol: float = 0.1,
) -> Vector:
    """
    Polygonize a raster, either in-memory or lazily through Dask or Multiprocessing.

    This function only:
      - Validates backend choice,
      - Prepares shared inputs (values/mask/dtype),
      - Dispatches to base, Dask, or Multiprocessing wrapper.

    :param connectivity: Pixel connectivity for label-based strategies (4 or 8).
    :param strategy: Chunked strategy. Has no effect for base backend.
    :param mp_config: Multiprocessing configuration.
    :param float_tol: Absolute tolerance to distinguish two classes for floating input.
    """
    from geoutils.vector import Vector  # runtime import to avoid circularity issues

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    dask_backend = source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config. "
            "To use Multiprocessing, open the file without chunks."
        )

    # Get input arguments
    prepared = _polygonize_prepare(
        source_raster,
        target_values,
        data_column_name,
        connectivity=connectivity,
        strategy=strategy,
        value_column="raster_value",
        id_column="component_id",
        float_tol=float_tol,
    )

    # For Multiprocessing
    if mp_backend:
        gdf = _multiproc_polygonize(
            source_raster=source_raster,
            prepared=prepared,
            mp_config=mp_config,
        )
    # For Dask
    elif dask_backend:
        gdf = _dask_polygonize(
            source_raster.data,
            prepared=prepared,
            transform=source_raster.transform,
            crs=source_raster.crs,
        )
    # For base implementation
    else:
        # Eager: materialize values and build selection mask with SAME semantics as chunked backends
        if np.issubdtype(prepared.final_dtype, np.integer):
            fill_value = source_raster.nodata
        else:
            fill_value = np.nan
        values = np.asarray(source_raster.data.filled(fill_value)).astype(prepared.final_dtype, copy=False)
        mask = np.asarray(_build_selection_mask(values, prepared))

        gdf = _polygonize_base(
            values,
            mask,
            transform=source_raster.transform,
            crs=source_raster.crs,
            data_column_name=data_column_name,
            value_column=prepared.value_column,
            connectivity=prepared.connectivity,
            float_tol=prepared.float_tol,
        )

        # Very useful debugger: Extract neighbours array locations that were polygonized different,
        # and displays them in a practical way

        # if False:
        #     m = np.asarray(mask, dtype=bool)
        #     v = np.asarray(values)
        #     vv = v[m]
        #
        #     u_r = np.unique(vv)
        #     u_p = np.unique(np.asarray(gdf[prepared.value_column]))
        #
        #     # values present in raster selection but absent from polygon output
        #     missing = np.setdiff1d(u_r, u_p)
        #     print(f"[missing vals] count={missing.size} (showing up to 8)")
        #     # if missing.size == 0:
        #     #     return
        #
        #     missing = missing[:8]
        #
        #     for val in missing:
        #         # find one occurrence
        #         ys, xs = np.where(m & (v == val))
        #         if ys.size == 0:
        #             print(f"  val={val!r}: no pixel found? (unexpected)")
        #             continue
        #         y, x = int(ys[0]), int(xs[0])
        #
        #         # 3x3 neighborhood
        #         y0, y1 = max(0, y - 1), min(v.shape[0], y + 2)
        #         x0, x1 = max(0, x - 1), min(v.shape[1], x + 2)
        #         patch = v[y0:y1, x0:x1]
        #         pmask = m[y0:y1, x0:x1]
        #
        #         print(f"  val={val!r} at (y={y},x={x}) patch:\n{patch}")
        #         print(f"    patch_mask:\n{pmask.astype(np.uint8)}")
        #
        #         # compare to neighbors (4-neigh)
        #         neigh = []
        #         for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        #             yy, xx = y + dy, x + dx
        #             if 0 <= yy < v.shape[0] and 0 <= xx < v.shape[1] and m[yy, xx]:
        #                 neigh.append(float(v[yy, xx]))
        #         if neigh:
        #             diffs = [abs(float(val) - n) for n in neigh]
        #             print(f"    4-neigh vals={neigh} abs_diffs={diffs}")


        return Vector(gdf)

    # Finalize Vector output
    # For chunked methods, we only need to add the id column that is missing
    if len(gdf) == 0:
        gdf = gpd.GeoDataFrame({data_column_name: [], "raster_value": []}, geometry=[], crs=source_raster.crs)
    else:
        if data_column_name not in gdf.columns:
            gdf.insert(0, data_column_name, range(0, len(gdf)))
        gdf = gdf.set_geometry(col="geometry")
        gdf = gdf.set_crs(source_raster.crs)

    return Vector(gdf)


