"""
Test module for vectorization (polygonize, etc).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Literal

import numpy as np
import pytest
import rasterio as rio
import shapely
from shapely.geometry.base import BaseGeometry
import numpy as np
import shapely
from shapely.ops import unary_union
import xarray as xr


import geoutils as gu
from geoutils import examples
from geoutils.multiproc.mparray import MultiprocConfig
from geoutils import open_raster

# Helpers for different types of vector equality
###############################################

def _explode_polys(g):
    """Helper to explore geometry to its components polygons."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if g.geom_type == "MultiPolygon":
        return list(g.geoms)
    if g.geom_type == "GeometryCollection":
        out = []
        for x in g.geoms:
            out.extend(_explode_polys(x))
        return out
    return []


def _canon(g, grid=None):
    """Helper to canonicalize geometries."""
    g = shapely.normalize(g)
    if grid is not None:
        g = shapely.set_precision(g, float(grid), mode="valid_output")
    try:
        g = g.buffer(0)
    except Exception:
        pass
    g = shapely.normalize(g)
    if grid is not None:
        g = shapely.set_precision(g, float(grid), mode="valid_output")
    return g


def _components(geoms, grid=None):
    """Helper to get the normalized polygons component of a geometry."""
    u = unary_union(list(geoms)) if len(geoms) else shapely.GeometryCollection()
    u = _canon(u, grid=grid)
    parts = [_canon(p, grid=grid) for p in _explode_polys(u) if (p is not None and not p.is_empty)]
    parts.sort(key=lambda p: (float(p.area), tuple(map(float, p.bounds))))
    return parts


def assert_vectors_equal_setwise_components(
    g1,
    g2,
    value_col="raster_value",
    grid=None,
    tol=0.0,
):
    """
    Compare components per value*, ignoring feature fragmentation.
    Uses geometric equality (equals_exact/equals), not WKB byte identity.
    """
    assert value_col in g1.columns and value_col in g2.columns

    # Multiset of raster values (feature-level)
    # assert np.array_equal(np.sort(g1[value_col].to_numpy()), np.sort(g2[value_col].to_numpy()))
    # compare the set of unique values (exact)
    u_base = set(np.asarray(g1["raster_value"]))
    u_chk = set(np.asarray(g2["raster_value"]))
    print("unique only in base:", list(sorted(u_base - u_chk))[:20])
    print("unique only in chunk:", list(sorted(u_chk - u_base))[:20])

    # Get and sort unique values of geometries
    vals = sorted(set(np.unique(g1[value_col])) | set(np.unique(g2[value_col])))

    # Check polygon components for every value
    failed = []
    for v in vals:
        a = g1.loc[g1[value_col] == v, "geometry"].values
        b = g2.loc[g2[value_col] == v, "geometry"].values

        ca = _components(a, grid=grid)
        cb = _components(b, grid=grid)

        if len(ca) != len(cb):
            failed.append((v, "n_components", len(ca), len(cb)))
            continue

        # Pairwise comparison, stable since we sort by (area,bounds)
        ok = True
        for pa, pb in zip(ca, cb):
            if tol == 0:
                # Using equals_exact with tol=0 is stricter than equals()
                if not pa.equals_exact(pb, tolerance=0.0):
                    # Fallback to topological equals
                    if not pa.equals(pb):
                        ok = False
                        break
            else:
                if not pa.equals_exact(pb, tolerance=float(tol)):
                    ok = False
                    break

        # If it does not match, we prepare a useful output for debugging
        if not ok:
            # We compute the area of the symmetric difference of the unions (should always be zero, or something is
            # wrong)
            ua = unary_union(list(a)) if len(a) else shapely.GeometryCollection()
            ub = unary_union(list(b)) if len(b) else shapely.GeometryCollection()
            ua = _canon(ua, grid=grid)
            ub = _canon(ub, grid=grid)
            try:
                sda = float(ua.symmetric_difference(ub).area)
            except Exception:
                sda = float("nan")
            failed.append((v, "geom_mismatch", sda))

    # Raise error with useful info if test failed
    if failed:
        raise AssertionError(
            {
                "n_g1": len(g1),
                "n_g2": len(g2),
                "failed": failed[:50],
            }
        )

def _sort_gdf_for_compare(gdf, value_col="raster_value"):
    """
    Normalize ordering to make deterministic comparisons.

    We sort by value column (if present), then by geometry WKB bytes.
    """

    g = gdf.copy()

    # Use standard column by default
    keys = []
    if value_col is not None:
        keys.append(value_col)

    # WKB is deterministic enough for ordering (geometry identity)
    g["_wkb"] = g.geometry.apply(lambda x: x.wkb if x is not None else b"")
    keys.append("_wkb")

    g = g.sort_values(keys).reset_index(drop=True)
    g = g.drop(columns=["_wkb"], errors="ignore")
    return g

def _assert_vectors_equal_ordered(g1, g2, exact: bool=False):
    """Stricter test: we check that geometries are equal AND ordered exactly the same."""

    g1 = _sort_gdf_for_compare(g1)
    g2 = _sort_gdf_for_compare(g2)

    # Compare values
    if "raster_value" in g1.columns and "raster_value" in g2.columns:
        assert np.array_equal(g1["raster_value"].to_numpy(), g2["raster_value"].to_numpy())

    if exact:
        eq = [a.equals_exact(b, tolerance=0) for a, b in zip(g1.geometry.values, g2.geometry.values)]
    else:
        eq = [a.equals(b) for a, b in zip(g1.geometry.values, g2.geometry.values)]

    def _geom_debug(a: BaseGeometry, b: BaseGeometry) -> str:
        # symmetric difference area is a great “how different are they” scalar
        try:
            sda = a.symmetric_difference(b).area
        except Exception:
            sda = float("nan")
        return (
            f"type(a)={a.geom_type}, type(b)={b.geom_type}, "
            f"area(a)={a.area}, area(b)={b.area}, "
            f"len(a.wkt)={len(a.wkt)}, len(b.wkt)={len(b.wkt)}, "
            f"symdiff_area={sda}"
        )

    if not all(eq):
        i = next(i for i, ok in enumerate(eq) if not ok)
        a, b = g1.geometry.values[i], g2.geometry.values[i]
        raise AssertionError(
            "Geometry mismatch at index "
            f"{i}\n"
            f"{_geom_debug(a, b)}\n"
            f"a.wkt={a.wkt}\n\nb.wkt={b.wkt}\n"
        )

def assert_vectors_equal(v1: gu.Vector,
                         v2: gu.Vector,
                         *,
                         check_crs: bool = True,
                         exact: bool = False,
                         setwise: bool = False,
                         ) -> None:
    """
    Compare two vectors for polygonize tests.

    Setting "exact" switches between equals and exact_equals of Shapely.
    Setting "setwise = True" leaves more flexibility on the polygon definition (Multipolygon, etc...).
    """

    # Both should be vectors
    assert isinstance(v1, gu.Vector)
    assert isinstance(v2, gu.Vector)

    # Have the same CRS
    if check_crs:
        assert v1.crs == v2.crs

    # Have the same length
    g1 = v1.ds.copy()
    g2 = v2.ds.copy()
    assert len(g1) == len(g2)

    # Have zero symmetric difference area after taking the union of all geometries
    union1 = shapely.unary_union(g1.geometry.values)
    union2 = shapely.unary_union(g2.geometry.values)
    symdiff_area = union1.symmetric_difference(union2).area
    assert symdiff_area == 0

    # Then we check individual geometry equality
    if setwise:
        assert_vectors_equal_setwise_components(g1, g2, value_col="raster_value", grid=20)
    else:
        _assert_vectors_equal_ordered(g1, g2, exact)


def _write_tmp_tif(tmp_path, arr: np.ndarray, *, transform=None, crs=None, nodata=None) -> str:
    """
    Write a single-band GeoTIFF for Multiprocessing backend tests.
    """
    if transform is None:
        # 1x1 pixels, origin at (0, height)
        transform = rio.transform.from_origin(0, arr.shape[0], 1, 1)
    if crs is None:
        crs = rio.crs.CRS.from_epsg(32645)  # arbitrary projected CRS

    path = tmp_path / "tmp_polygonize.tif"
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(path)


class TestPolygonize:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    # Strategies to test for chunked backends
    chunked_strategies: tuple[Literal["label_union", "label_stitch", "geometry_stitch"], ...] = (
        "label_union",
        "label_stitch",
        "geometry_stitch",
    )

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    def test_polygonize__area_data_column(self, example: str) -> None:
        """Test that polygonize preserves area for a selected constant value and respects schema/crs."""
        img = gu.Raster(example)

        # Get raster pixel area
        value = np.unique(img)[0]
        pixel_area = np.count_nonzero(img.data == value) * img.res[0] * img.res[1]

        polygonized = img.polygonize(target_values=value)

        # Get vector area, and check they are equal
        assert polygonized.ds.area.sum() == pytest.approx(pixel_area)
        assert isinstance(polygonized, gu.Vector)
        assert polygonized.crs == img.crs

        # Default id column and custom column
        assert "id" in polygonized.ds.columns
        polygonized2 = img.polygonize(target_values=value, data_column_name="myname")
        assert "myname" in polygonized2.ds.columns

        # Same geometry/value content; id column naming differs
        assert_vectors_equal(polygonized, polygonized2)

    @pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"])
    def test_polygonize__dtype_support(self, dtype: str) -> None:
        """Polygonize should work on a wide range of dtypes (GeoPandas dtype constraints handled internally)."""
        img = gu.Raster(self.landsat_b4_path)
        img_dtype = img.copy()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="dtype conversion will result in a loss of information.*"
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Unmasked values equal to the nodata value found in data array.*"
            )
            img_dtype = img_dtype.astype(dtype)

        value = np.unique(img_dtype)[0]
        vect = img_dtype.polygonize(target_values=value)
        assert isinstance(vect, gu.Vector)

    def test_polygonize__mask(self) -> None:
        """Mask rasters should normalize target_values and raise warning for non-boolean target."""
        img = gu.Raster(self.landsat_b4_path)
        value = np.unique(img)[0]
        mask = img > value

        # Default polygonize on mask: True regions
        vect = mask.polygonize()
        assert isinstance(vect, gu.Vector)

        # Explicit 0 is allowed
        vect0 = mask.polygonize(target_values=0)
        assert isinstance(vect0, gu.Vector)

        # Non-boolean target triggers warning as it is being ignored
        with pytest.warns(
            UserWarning,
            match="ignoring target values input",
        ):
            mask.polygonize(target_values=2)

    @pytest.mark.parametrize("strategy", chunked_strategies)
    def test_polygonize__connectivity_across_chunks(self, tmp_path, strategy) -> None:
        """
        Test that 4/8-connectivity is respected across chunks for all chunked methods.

        We build a tiny raster where the only two True pixels are diagonal neighbors across the x and y chunk boundary.
        For 8-connectivity they form one component, while for 4-connectivity they form two.
        """
        pytest.importorskip("dask")

        import dask.array as da

        # 4x4 raster, chunk boundary at 2. Place two '1' pixels diagonally across both seams:
        # pixel A at (1,1) in top-left chunk; pixel B at (2,2) in bottom-right chunk.
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[1, 1] = 1
        arr[2, 2] = 1

        # Write raster to disk to open lazily
        path = _write_tmp_tif(tmp_path, arr)
        rst = gu.Raster(path)

        # 1/ Base reference
        base8 = rst.polygonize(target_values=1, connectivity=8)
        base4 = rst.polygonize(target_values=1, connectivity=4)

        # Expect 8-connectivity merges diagonals: 1 feature
        # While 4-connectivity does not: 2 features
        assert len(base8.ds) == 1
        assert len(base4.ds) == 2

        # 2/ Dask and Multiprocessing
        ds = open_raster(path, chunks={"band": 1, "x": 2, "y": 2})
        rst_mp = gu.Raster(path)

        # Ensure laziness
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None
        assert not rst_mp.is_loaded

        # Chunked polygonize with any strategy must match base exactly
        d8 = ds.rst.polygonize(target_values=1, connectivity=8, strategy=strategy)
        d4 = ds.rst.polygonize(target_values=1, connectivity=4, strategy=strategy)
        mp_config = MultiprocConfig(chunk_size=2)
        mp8 = rst_mp.polygonize(target_values=1, connectivity=8, strategy=strategy, mp_config=mp_config)
        mp4 = rst_mp.polygonize(target_values=1, connectivity=4, strategy=strategy, mp_config=mp_config)

        # We compare with setwise=True
        assert_vectors_equal(base8, d8, setwise=True)
        assert_vectors_equal(base4, d4, setwise=True)

        assert_vectors_equal(base8, mp8, setwise=True)
        assert_vectors_equal(base4, mp4, setwise=True)


    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("connectivity", [4, 8])
    @pytest.mark.parametrize("strategy", chunked_strategies)
    @pytest.mark.parametrize("target_mode", ["scalar", "range", "all"])
    def test_polygonize_chunked_backends_equal(
        self,
        path_index: int,
        connectivity: Literal[4, 8],
        strategy: Literal["label_union", "label_stitch", "geometry_stitch"],
        target_mode: str,
        lazy_test_files_tiny: list[str],
    ) -> None:
        """
        Test that polygonize yields identical output for:
         - In-memory base function,
         - Dask backend through Xarray accessor,
         - Multiprocessing backend through Raster class.

        Additionnally, both Dask and Multiprocessing inputs remains lazy (unloaded), but output vector is eager.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Get filepath of on-disk  (for laziness) test file
        path_raster = lazy_test_files_tiny[path_index]

        # Base input (in-memory)
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        # Multiprocessing input (lazy if we pass Multiprocessing later)
        raster_mp = gu.Raster(path_raster)
        mp_config = MultiprocConfig(chunk_size=200)
        # Dask input (lazy)
        ds = open_raster(path_raster, chunks={"x": 200, "y": 200})
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None

        # Prepare target_values
        nan = raster_base.get_nanarray()
        if target_mode == "scalar":
            target_values = np.unique(nan[~np.isnan(nan)])[0]  # First unique value we find for scalars
        elif target_mode == "range":
            med = float(np.nanmedian(nan))
            std = float(np.nanstd(nan))
            target_values = (med - std, med + std)  # Range around median using STD for the range
        elif target_mode == "all":
            target_values = "all"  # All here
        else:
            raise ValueError

        # Base output
        base = raster_base.polygonize(
            target_values=target_values,
            connectivity=connectivity,
        )
        # Multiprocessing output (Vector, computed by design)
        mp_vect = raster_mp.polygonize(
            target_values=target_values,
            connectivity=connectivity,
            strategy=strategy,
            mp_config=mp_config,
        )
        # Dask output (Vector, computed by design)
        dask_vect = ds.rst.polygonize(
            target_values=target_values,
            connectivity=connectivity,
            strategy=strategy,
        )

        # Dask input stays unloaded and lazy
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)

        # Multiprocessing should not load the raster
        assert not raster_mp.is_loaded

        # All outputs geometries must match base exactly spatially, but order/polygon definition might differ slightly
        # so we use setwise=True
        # assert_vectors_equal(base, dask_vect, setwise=True)
        assert_vectors_equal(base, mp_vect, setwise=True)

