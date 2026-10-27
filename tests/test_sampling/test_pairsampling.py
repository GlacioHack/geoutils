"""Tests for sampling pairs of values in rasters and point clouds without loading unnecessary data."""

from __future__ import annotations

import warnings
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.multiproc import ClusterGenerator, MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.sampling.pairsampling import _PointPairSampler, _RasterPairSampler


@pytest.fixture
def pointcloud_file(tmp_path: Any) -> Any:
    """Write point fixtures as GeoParquet row groups with their original labels and attribute dtypes."""

    filenames = []

    def write_points(frame: gpd.GeoDataFrame, partitions: int) -> tuple[str, int, str | None]:
        """Save one point fixture and return its filename, row chunk size and active value column."""

        pytest.importorskip("pyarrow")

        # Preserve the selected value column or infer one
        frame = frame.copy()
        column = frame.pc.data_name
        if "data_name" not in frame.attrs and column is None:
            column = next((name for name in ("height", "z", "intensity") if name in frame), None)
        if column is None and not frame.geometry.has_z.all():
            frame["z"] = np.zeros(len(frame))
            column = "z"

        # A final shorter row group checks that file readers preserve every row
        chunks = max(1, int(np.ceil(len(frame) / partitions)))
        filename = tmp_path / f"points-{len(filenames)}.parquet"
        filenames.append(filename)
        frame.to_parquet(
            filename, index=True, geometry_encoding="geoarrow", schema_version="1.1.0", row_group_size=chunks
        )
        return str(filename), chunks, column

    return write_points


@pytest.fixture
def raster() -> gu.Raster:
    """Return a finite raster with a small nodata region."""

    array = np.arange(900, dtype=float).reshape(30, 30)
    array[2:5, 4:8] = np.nan
    return gu.Raster.from_array(array, from_origin(0, 30, 2, 3), 32633, nodata=-99999)


class TestRasterPairSampling:
    """
    Test module for pairsample() on raster grids.

    Dask and multiprocessing behavior is covered in TestPairSampleChunked further below.

    This module is checking that:
    - All sampling "strategies" return the correct shape of outputs.
    - User masks, min/max sampling distance and duplicate pairs are all respected.
    """

    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    def test_pairsample__raster_strategies(self, raster: gu.Raster, strategy: str) -> None:
        """Checks that every strategy returns the requested pairs and labelled values."""

        # Draw raster pairs with one strategy and min/max
        pairs = raster.pairsample(
            n_pairs=200,
            min_distance=2,
            max_distance=40,
            strategy=strategy,
            random_state=42,
            anchors_per_round=100,
            distances_per_anchor=3,
            angles_per_distance=3,
        )

        # Check layout, finite values and distance limits
        assert isinstance(pairs, xr.Dataset)
        assert pairs.sizes == {"pair": 200, "endpoint": 2}
        assert set(pairs.data_vars) == {"index", "value", "distance", "row", "column", "x", "y"}
        assert np.all(np.isfinite(pairs.value))
        assert np.all((pairs.distance >= 2) & (pairs.distance <= 40))
        assert np.array_equal(pairs.value, raster.data.data.ravel()[pairs["index"]])
        expected_distance = np.hypot(np.diff(pairs.x, axis=1), np.diff(pairs.y, axis=1)).ravel()
        np.testing.assert_allclose(pairs.distance, expected_distance)

    def test_pairsample__raster_batch_limit(self, raster: gu.Raster, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that pairsample() never samples more pairs at once than batch_pairs allows."""

        # Record candidate batch sizes
        candidate_counts = []
        original_candidates = _RasterPairSampler._candidates

        def candidates(sampler: _RasterPairSampler, count: int) -> tuple[NDArrayNum, NDArrayNum]:
            """Record how many pairs were requested, then generate those pairs with the original method."""
            candidate_counts.append(count)
            return original_candidates(sampler, count)

        # Record calls to _candidates()
        monkeypatch.setattr(_RasterPairSampler, "_candidates", candidates)

        # Request 200 pairs while allowing only 37 at once: this forces several calls
        pairs = raster.pairsample(n_pairs=200, batch_pairs=37, strategy="independent", random_state=8)

        # Check pair count and batch sizes
        assert pairs.sizes["pair"] == 200
        assert len(candidate_counts) > 1
        assert max(candidate_counts) <= 37

    def test_pairsample__raster_reproducible(self, raster: gu.Raster) -> None:
        """Checks that a fixed seed returns the same unique raster pairs in the same order."""

        # Draw the same globally unique sample twice with one seed
        first = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        second = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        indexes = np.sort(first["index"].values, axis=1)

        # Check exact repeatability and treat reversed endpoint order as the same pair
        assert first.identical(second)
        assert len(np.unique(indexes, axis=0)) == first.sizes["pair"]

    @pytest.mark.parametrize("sampling", ["loglag", "random_xy"])
    @pytest.mark.parametrize("mask_form", ["array", "geodataframe"])
    def test_pairsample__raster_masks(self, raster: gu.Raster, sampling: str, mask_form: str) -> None:
        """Checks that array and polygon masks select the same raster pairs for both sampling methods."""

        # Mask the upper-left quarter of the raster
        eligible = np.zeros(raster.shape, dtype=bool)
        eligible[:15, :15] = True
        mask = eligible if mask_form == "array" else gpd.GeoDataFrame(geometry=[box(0, -15, 30, 30)], crs=raster.crs)
        options = {"n_pairs": 100, "sampling": sampling, "max_distance": 20, "random_state": 8}

        # Compare both masks with the same seed
        expected = raster.pairsample(mask=eligible, **options)
        pairs = raster.pairsample(mask=mask, **options)

        # Check selected cells and map coordinates
        xr.testing.assert_identical(pairs, expected)
        assert np.all(eligible[pairs.row, pairs.column])
        assert np.all(pairs.x < 30)
        assert np.all(pairs.y > -15)
        assert pairs.attrs["sampling"] == sampling

    @pytest.mark.parametrize("read_strategy", ["native", "reuse_values", "packed_validity"])
    @pytest.mark.parametrize("sampling", ["loglag", "random_xy"])
    @pytest.mark.parametrize("raster_mask", [False, True])
    def test_pairsample__raster_integers(self, raster_mask: bool, read_strategy: str, sampling: str) -> None:
        """Checks that masked integer values and masked boolean pixels never enter raster pairs."""

        # Mask separate cells through data, mask nodata and false values
        data = np.ma.array(np.arange(100, dtype=np.int32).reshape(10, 10), mask=False)
        data.mask[0, 0] = True
        mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
        mask.mask[1, 1] = True
        mask[2, 2] = False
        raster = gu.Raster.from_array(data, from_origin(0, 10, 1, 1), 32633, nodata=-9999)
        selected_mask = raster.from_array(mask, raster.transform, raster.crs) if raster_mask else mask

        # Draw pairs with either the boolean array or its Raster form
        pairs = raster.pairsample(
            n_pairs=200, mask=selected_mask, random_state=3, read_strategy=read_strategy, sampling=sampling
        )

        # Check each endpoint against the combined mask and original integer values
        eligible = ~data.mask & mask.filled(False)
        indexes = pairs["index"].values
        assert pairs.sizes["pair"] == 200
        assert np.all(eligible.ravel()[indexes])
        assert np.array_equal(pairs["value"].values, data.data.ravel()[indexes])


class TestRasterPairReads:
    """Test module for raster pair read strategies."""

    @pytest.mark.parametrize("read_strategy", ["reuse_values", "packed_validity"])
    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    @pytest.mark.parametrize("deduplicate", ["none", "per_anchor", "global"])
    def test_pairsample__read_strategies(
        self, raster: gu.Raster, read_strategy: str, strategy: str, deduplicate: str
    ) -> None:
        """Checks that read strategies return identical pairs."""

        # Create a mask and use small sampling batches
        eligible = np.ones(raster.shape, dtype=bool)
        eligible[:4] = False
        options = {
            "n_pairs": 150,
            "batch_pairs": 47,
            "random_state": 42,
            "strategy": strategy,
            "deduplicate": deduplicate,
            "mask": eligible,
            "anchors_per_round": 13,
        }

        # Run both read strategies with the same seed
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always")
            expected = raster.pairsample(**options)
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            result = raster.pairsample(**options, read_strategy=read_strategy)
        assert [str(item.message) for item in actual_warnings] == [str(item.message) for item in expected_warnings]
        assert result.attrs.pop("read_strategy") == read_strategy
        xr.testing.assert_identical(result, expected)
        assert not result.chunks


class TestPointPairSampling:
    """
    Test module for pairsample() on in-memory point clouds.

    It checks that:
    - Search strategies return pairs at the requested distances with their original row indexes.
    - Array and spatial masks select the same points.
    - Exact searches cover reused anchors and the upper distance limit.
    """

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector"])
    def test_pairsample__point_strategies(self, strategy: str) -> None:
        """Checks that every point strategy returns the requested distances and original rows."""

        # Create a point grid with varying values
        y, x = np.mgrid[:20, :20]
        values = np.sin(x.ravel() / 3) + np.cos(y.ravel() / 4)
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        # Sample with each search strategy
        pairs = points.pairsample(
            n_pairs=150,
            min_distance=1,
            max_distance=15,
            strategy=strategy,
            anchors_per_round=200,
            nn_tolerance=0.6,
            random_state=3,
        )

        # Check pair count, distances and source coordinates
        assert pairs.sizes["pair"] == 150
        assert np.all((pairs.distance >= 1) & (pairs.distance <= 15))
        assert np.array_equal(pairs["x"], x.ravel()[pairs["index"]])
        assert np.array_equal(pairs["y"], y.ravel()[pairs["index"]])

    @pytest.mark.parametrize(
        "mask_form", ["array", "masked", "xarray", "dask", "raster", "pointcloud", "vector", "geodataframe"]
    )
    def test_pairsample__point_masks(self, mask_form: str) -> None:
        """Checks that point masks select the same random pairs as a boolean array."""

        # Create points with one NaN and mask left half
        y, x = np.mgrid[:12, :12]
        values = (x + y).astype(float).ravel()
        values[5] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        eligible = x.ravel() < 6
        mask: Any = eligible.copy()

        if mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[13] = True
            eligible[13] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask.reshape(12, 12), dims=("row", "column"))
        elif mask_form == "dask":
            mask = pytest.importorskip("dask.array").from_array(mask, chunks=17)
        elif mask_form == "raster":
            mask = gu.Raster.from_array(eligible.reshape(12, 12)[::-1], from_origin(0, 11, 1, 1), points.crs)
        elif mask_form == "pointcloud":
            mask = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), eligible, crs=points.crs)
        elif mask_form in {"vector", "geodataframe"}:
            # Put the first excluded column on the polygon boundary
            mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 6, 11.5)], crs=points.crs)
            if mask_form == "vector":
                mask = gu.Vector(mask)

        # Compare masks using smaller output dtypes
        options = {
            "n_pairs": 100,
            "sampling": "random_xy",
            "min_distance": 1,
            "max_distance": 10,
            "random_state": 9,
            "index_dtype": np.int16,
            "distance_dtype": np.float32,
        }
        expected = points.pairsample(mask=eligible, **options)
        pairs = points.pairsample(mask=mask, **options)

        # Check dtypes, nodata and selected rows
        xr.testing.assert_identical(pairs, expected)
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32
        assert not np.any(pairs["index"] == 5)
        assert np.all(pairs.x < 6)
        assert np.all(eligible[pairs["index"]])

    def test_pairsample__point_reuses_anchors(self) -> None:
        """Checks that exact point searches can reuse first endpoints when one round requests more than exist."""

        # Create fewer points than the requested number of first endpoints per round
        y, x = np.mgrid[:5, :5]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        # Request enough pairs to require reuse during the single allowed round
        pairs = points.pairsample(
            n_pairs=50,
            min_distance=1,
            max_distance=5,
            strategy="kdtree",
            anchors_per_round=100,
            attempts_per_anchor=2,
            max_rounds=1,
            random_state=1,
        )

        # Check that reuse still fills the requested sample
        assert pairs.sizes["pair"] == 50

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid"])
    def test_pairsample__point_maxdist(self, strategy: str) -> None:
        """Checks that point searches include pairs at the maximum distance."""

        # Use two points whose only nonzero separation is exactly the upper boundary
        points = gu.PointCloud.from_xyz(np.array([0, 1]), np.array([0, 0]), np.array([3, 5]), crs=32633)
        pairs = points.pairsample(
            n_pairs=10, strategy=strategy, n_bins=1, min_distance=0.5, max_distance=1, random_state=2
        )

        # Check the only available pair, allowing reversed endpoints
        assert np.array_equal(pairs.distance, np.ones(10))
        assert np.array_equal(np.sort(pairs["index"], axis=1), np.tile([0, 1], (10, 1)))


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestPairSampleChunked:
    """
    Test module for pair sampling through Dask chunks and multiprocessing file partitions.

    Dask cases check that inputs remain lazy and source chunks are reused. Multiprocessing cases compare worker reads
    with in-memory results and check that file-backed inputs remain unloaded. Point cases also check that complete
    chunked tables are saved in disk-backed arrays. Every returned pair dataset is in-memory.
    """

    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    @pytest.mark.parametrize("shape,chunks", [((24, 24), (6, 8)), ((20, 20), ((4, 9, 7), (5, 3, 12)))])
    def test_pairsample__raster_uneven_chunks(self, strategy: str, shape: tuple[int, int], chunks: Any) -> None:
        """Checks that pairs respect regular/uneven chunks."""

        # Create regular or irregular chunks with different row and column boundaries
        da = pytest.importorskip("dask.array")

        eager = np.arange(np.prod(shape), dtype=float).reshape(shape)
        array = da.from_array(eager, chunks=chunks)
        transform = from_origin(0, shape[0], 1, 1)
        raster = gu.DataArrayRasterAccessor.from_array(array, transform, 32633)
        options = {
            "n_pairs": 100,
            "strategy": strategy,
            "hybrid_local_fraction": 1,
            "min_distance": 1,
            "max_distance": 6,
            "random_state": 4,
            "index_dtype": np.int16,
            "distance_dtype": np.float32,
        }

        # Draw only local pairs so both endpoints must belong to one original Dask chunk
        pairs = raster.rst.pairsample(**options, force_output_to_memory=True)

        # The Dask input stays lazy and pairs are returned as an in-memory dataset
        assert isinstance(raster.data, da.Array)
        assert not raster._in_memory
        assert not pairs.chunks

        # Locate endpoint chunks from cumulative boundaries
        rows = np.searchsorted(np.cumsum(array.chunks[0]), pairs.row, side="right")
        columns = np.searchsorted(np.cumsum(array.chunks[1]), pairs.column, side="right")
        assert np.array_equal(rows[:, 0], rows[:, 1])
        assert np.array_equal(columns[:, 0], columns[:, 1])
        assert np.array_equal(pairs.value, eager[pairs.row, pairs.column])
        assert pairs.sizes["pair"] == 100
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32

        # Compare regular chunks with in-memory sampling
        if isinstance(chunks[0], int):
            expected = gu.Raster.from_array(eager, transform, 32633).pairsample(
                **options, mp_config=MultiprocConfig(chunks=chunks)
            )
            xr.testing.assert_equal(pairs, expected)

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud", "dataframe"])
    @pytest.mark.parametrize("force_output_to_memory", [False, True])
    @pytest.mark.parametrize("sampling", ["loglag", "random_xy"])
    def test_pairsample__lazy_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source_kind: str,
        force_output_to_memory: bool,
        sampling: str,
    ) -> None:
        """Checks that pairs stay lazy by default, but can be forced into memory with user input."""

        # Create chunked raster or point data with a NaN
        da = pytest.importorskip("dask.array")

        rows, columns = np.indices((7, 9))
        values = np.arange(63.0).reshape(7, 9)
        values[1, 2] = np.nan
        eager: Any
        source: Any
        if source_kind == "raster":
            transform = from_origin(0, 7, 1, 1)
            eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
            source = gu.DataArrayRasterAccessor.from_array(da.from_array(values, chunks=(3, 4)), transform, 32633).rst
        else:
            points = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633)
            if source_kind == "pointcloud":
                eager = points.to_xarray().pc
                source = points.to_xarray().chunk({"point": 13}).pc
            else:
                from geoutils._misc import import_optional
                from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

                dgpd = import_optional("dask_geopandas")
                _register_dask_pointcloud_accessor()
                eager = points
                source = dgpd.from_geopandas(points.gdf, npartitions=4).pc
        options: dict[str, Any] = {"n_pairs": 37, "sampling": sampling, "random_state": 42, "output_chunks": 11}
        if source_kind == "raster":
            options["strategy"] = "independent"
        expected = eager.pairsample(**options)

        # Prevent temporary file creation
        from geoutils.sampling import pairsampling

        def unexpected_files(*args: Any, **kwargs: Any) -> None:
            pytest.fail("Dask pair sampling must not create temporary files.")

        monkeypatch.setattr(pairsampling, "TemporaryDirectory", unexpected_files)

        # Sample pairs and check loading
        result = source.pairsample(**options, force_output_to_memory=force_output_to_memory)
        expected_lazy = not force_output_to_memory
        assert not source.is_loaded
        if source_kind != "dataframe":
            assert isinstance(source.data, da.Array)
        assert all(isinstance(column.data, da.Array) == expected_lazy for column in result.data_vars.values())
        if expected_lazy:
            assert all(max(column.data.chunks[0]) <= 11 for column in result.data_vars.values())
            assert result.value.data.chunks[0][-1] < 11

        # Check equality
        computed = result.compute()
        computed.attrs.pop("lazy_output", None)
        computed.attrs.pop("output_chunks", None)
        xr.testing.assert_identical(computed, expected)
        assert not source.is_loaded
        assert not list(tmp_path.iterdir())

        # Check that raster sampling includes pairs crossing the 3 x 4 source chunks
        if source_kind == "raster":
            cross_chunk = (computed.row[:, 0] // 3 != computed.row[:, 1] // 3) | (
                computed.column[:, 0] // 4 != computed.column[:, 1] // 4
            )
            assert np.any(cross_chunk)

        # Save and check the lazy dataset
        if expected_lazy:
            filename = tmp_path / "pairs.nc"
            result.to_netcdf(filename)
            with xr.open_dataset(filename) as saved:
                xr.testing.assert_identical(saved.load(), result.compute())
            derived = result.value.data + 1
            del result
            np.testing.assert_array_equal(derived.compute(), expected.value.data + 1)
            assert not source.is_loaded

    @pytest.mark.parametrize("read_strategy", ["reuse_values", "packed_validity"])
    @pytest.mark.parametrize("deduplicate", ["none", "per_anchor", "global"])
    def test_pairsample__read_strategies_chunks(self, read_strategy: str, deduplicate: str) -> None:
        """Checks that read strategies agree across (potentially uneven) chunks."""

        # Create a raster with uneven chunks and nonfinite values
        da = pytest.importorskip("dask.array")

        values = np.arange(20 * 23, dtype=np.float32).reshape(20, 23)
        values[2:5, 3:8] = np.nan
        values[7:9, 10:13] = np.inf
        chunks = ((4, 9, 7), (5, 3, 15))
        transform = from_origin(0, 20, 1, 1)
        source = gu.DataArrayRasterAccessor.from_array(da.from_array(values, chunks=chunks), transform, 32633).rst
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options = {
            "n_pairs": 151,
            "batch_pairs": 61,
            "deduplicate": deduplicate,
            "strategy": "independent",
            "max_distance": 5,
            "random_state": 7,
            "anchors_per_round": 17,
        }

        # Compare pairs using the same seed
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always", UserWarning)
            expected = eager.pairsample(**options)
        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always", UserWarning)
            result = source.pairsample(**options, read_strategy=read_strategy, output_chunks=37)

        # Compare sampling warnings without deprecation notices from Xarray transform lookups
        expected_messages = [str(item.message) for item in expected_warnings if isinstance(item.message, UserWarning)]
        actual_messages = [str(item.message) for item in actual_warnings if isinstance(item.message, UserWarning)]
        assert actual_messages == expected_messages

        # Check lazy inputs and outputs, output chunk sizes and value dtype
        assert not source.is_loaded and isinstance(source.data, da.Array)
        assert isinstance(result.value.data, da.Array)
        assert max(result.value.data.chunks[0]) <= 37
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        assert result.value.dtype == np.float32

        # Compare computed pairs with the in-memory result
        computed = result.compute()
        for name in ("read_strategy", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)

    @pytest.mark.parametrize("read_strategy", ["native", "reuse_values", "packed_validity"])
    def test_pairsample__read_strategies_reads(self, read_strategy: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that read strategies avoid unnecessary reads."""

        # Record one native chunk or nine uneven cached chunks
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        reads: list[int] = []
        values = np.arange(20 * 23, dtype=np.float32).reshape(20, 23)
        chunk_rows, chunk_columns = values.shape if read_strategy == "native" else (7, 9)

        @dask.delayed
        def load_values(index: int, block: NDArrayNum) -> NDArrayNum:
            """Record a source read without returning a full raster array."""
            reads.append(index)
            return block

        rows = []
        for row in range(0, 20, chunk_rows):
            columns = []
            for column in range(0, 23, chunk_columns):
                index = row // chunk_rows * 3 + column // chunk_columns
                block = values[row : row + chunk_rows, column : column + chunk_columns]
                task = load_values(index, block)
                columns.append(da.from_delayed(task, shape=block.shape, dtype=np.float32))
            rows.append(columns)
        source = gu.DataArrayRasterAccessor.from_array(da.block(rows), from_origin(0, 20, 1, 1), 32633).rst
        rounds = []
        original = _RasterPairSampler._candidates

        # Record source reads immediately before/after candidate generation
        def candidates(sampler: Any, count: int) -> Any:
            rounds.append(len(reads))
            return original(sampler, count)

        monkeypatch.setattr(_RasterPairSampler, "_candidates", candidates)
        options: dict[str, Any] = {
            "n_pairs": 150,
            "batch_pairs": 47,
            "random_state": 42,
            "strategy": "independent",
            "read_strategy": read_strategy,
        }
        if read_strategy == "native":
            options["force_output_to_memory"] = True
        else:
            options["output_chunks"] = 37

        with dask.config.set(scheduler="synchronous"):
            result = source.pairsample(**options)
            sampling_reads = len(reads)
            assert isinstance(result.value.data, da.Array) == (read_strategy != "native")
            assert not source.is_loaded and isinstance(source.data, da.Array)
            computed = result.compute()

        # Check that each strategy avoids rereading values
        assert len(rounds) > 1
        if read_strategy == "native":
            # Count once, read once per round, then read both endpoints together
            assert len(reads) == len(rounds) + 2
            expected = gu.Raster.from_array(values, source.transform, source.crs).pairsample(**options)
            xr.testing.assert_equal(computed, expected)
        elif read_strategy == "packed_validity":
            assert rounds == [9] * len(rounds)
            assert sampling_reads == 9
            assert max(np.bincount(reads)) == 2
        else:
            assert sampling_reads > 9
            assert len(reads) == sampling_reads
        np.testing.assert_array_equal(computed.value, values.ravel()[computed["index"]])

    @pytest.mark.parametrize("execution", ["dask", "mp"])
    def test_pairsample__bounded_count(self, execution: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that pair sampling can count enough finite cells without scanning every source chunk."""

        # Create 56 uneven chunks (with 8 missing chunks in the first row)
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        from geoutils.sampling import pairsampling

        values = np.arange(25 * 37, dtype=np.float32).reshape(25, 37)
        values[:4] = np.nan
        transform = from_origin(0, 25, 1, 1)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)

        # Record finite counts for either backend
        counted: list[int] = []
        original_count = pairsampling._count_finite_reader_block

        def count_finite(block: Any) -> int:
            counted.append(1)
            return original_count(block)

        monkeypatch.setattr(pairsampling, "_count_finite_reader_block", count_finite)

        reads: list[int] = []
        source: Any
        backend_options: dict[str, Any]
        if execution == "dask":
            # Record source reads to check that computing lazy output reuses sampled values
            def load_values(block: NDArrayNum) -> NDArrayNum:
                reads.append(1)
                return block

            array = da.from_array(values, chunks=(4, 5)).map_blocks(
                load_values, meta=np.empty((0, 0), dtype=values.dtype)
            )
            source = gu.DataArrayRasterAccessor.from_array(array, transform, 32633, nodata=-99999).rst
            backend_options = {"output_chunks": 37}
        else:
            # Write source tiles for windowed file reads
            filename = tmp_path / "values.tif"
            eager.to_file(filename)
            source = gu.Raster(filename)
            backend_options = {"mp_config": MultiprocConfig(chunks=(4, 5))}

        # Compare chunked sampling with the same call on eager values
        options: dict[str, Any] = {
            "n_pairs": 127,
            "batch_pairs": 47,
            "strategy": "independent",
            "max_distance": 6,
            "random_state": 42,
            "read_strategy": "reuse_values",
        }
        assert not source.is_loaded
        with dask.config.set(scheduler="synchronous"):
            result = source.pairsample(**options, **backend_options)
            sampling_reads = len(reads)
            assert not source.is_loaded
            assert isinstance(result.value.data, da.Array) == (execution == "dask")
            computed = result.compute()
        expected = eager.pairsample(**options)

        # Check counting passes the 8 missing chunks, but stops before 56
        assert 8 < len(counted) < 56
        assert len(reads) == sampling_reads
        assert not source.is_loaded
        if execution == "dask":
            assert isinstance(source.data, da.Array)
            computed.attrs.pop("lazy_output")
            computed.attrs.pop("output_chunks")
        else:
            assert not result.chunks
        xr.testing.assert_identical(computed, expected)

    @pytest.mark.parametrize("scheduler", ["distributed", "synchronous"])
    def test_pairsample__packed_validity_scheduler_batches(
        self, scheduler: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that validity scans respect the scheduler and batch limit."""

        # We create 81 chunks on purpose to exceed the 64-result batch limit
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        pytest.importorskip("distributed")
        from distributed import Client, LocalCluster

        values = np.arange(27 * 27, dtype=np.float32).reshape(27, 27)
        source = gu.DataArrayRasterAccessor.from_array(
            da.from_array(values, chunks=(3, 3)), from_origin(0, 27, 1, 1), 32633
        ).rst
        submitted, gathered, packed_states = [], [], []

        def check_task(key: Any, value: Any, *args: Any) -> None:
            """Check that packed tasks transfer two bytes and a count for nine raster cells."""

            if str(key).startswith("_pack_raster_validity-"):
                packed, count = value
                assert packed.dtype == np.uint8 and packed.size == 2
                assert count == 9
                packed_states.append(count)

        with LocalCluster(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as cluster:
            with Client(cluster) as client, dask.config.set(scheduler=scheduler):
                original_compute, original_gather = client.compute, client.gather

                # Record the number of packed chunk tasks/results
                def compute(collections: Any, **kwargs: Any) -> Any:
                    submitted.append(len(collections))
                    return original_compute(collections, **kwargs)

                def gather(futures: Any, **kwargs: Any) -> Any:
                    gathered.append(len(futures))
                    return original_gather(futures, **kwargs)

                # Record scan batches before computing output
                with monkeypatch.context() as scan, Callback(posttask=check_task):
                    scan.setattr(client, "compute", compute)
                    scan.setattr(client, "gather", gather)
                    result = source.pairsample(
                        n_pairs=151,
                        read_strategy="packed_validity",
                        strategy="independent",
                        random_state=42,
                        output_chunks=37,
                    )
                if scheduler == "distributed":
                    assert submitted == [81]
                    assert gathered == [64, 17]
                else:
                    assert not submitted and not gathered
                    assert packed_states == [9] * 81
                assert not source.is_loaded and isinstance(source.data, da.Array)
                assert isinstance(result.value.data, da.Array)
                computed = result.compute()

        # Check equality
        expected = gu.Raster.from_array(values, source.transform, source.crs).pairsample(
            n_pairs=151, strategy="independent", random_state=42
        )
        xr.testing.assert_equal(computed, expected)

    @pytest.mark.parametrize(
        "execution,as_object_type",
        [("dask", "dataarray"), ("dask", "geodataframe"), ("mp", None)],
    )
    @pytest.mark.parametrize("mask_form", ["vector", "raster", "pointcloud", "chunked_pointcloud"])
    def test_pairsample__point_masks_chunked(
        self,
        execution: str,
        as_object_type: Literal["dataarray", "geodataframe"] | None,
        pointcloud_file: Any,
        mask_form: str,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Checks that Dask/MP masks match in-memory sampling without loading the point tables."""

        # Create points with one NAN
        y, x = np.mgrid[:12, :12]
        values = (x + y).astype(float).ravel()
        values[5] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633, data_name="height")
        reads: list[int] = []
        source: Any

        # Record Dask reads to detect repeated coordinate loads
        def read_partition(partition: gpd.GeoDataFrame | xr.DataArray) -> gpd.GeoDataFrame | xr.DataArray:
            """Record a source partition read."""
            reads.append(1)
            return partition

        with ExitStack() as stack:
            if execution == "dask":
                dask = pytest.importorskip("dask")
                stack.enter_context(dask.config.set(scheduler="synchronous"))
                filename, chunks, column = pointcloud_file(points.gdf, 3)
                lazy_points = gu.open_pointcloud(
                    filename, data_name=column, columns="all", chunks=chunks, as_object_type=as_object_type
                )
                if as_object_type == "dataarray":
                    # Drop row labels so map_blocks() does not load them
                    lazy_points = lazy_points.drop_vars("point")
                    lazy_points = lazy_points.map_blocks(read_partition, template=lazy_points)
                else:
                    lazy_points = lazy_points.map_partitions(read_partition, meta=points.gdf.iloc[:0])
                source = lazy_points.pc
                options: dict[str, Any] = {}
            else:
                source_filename = tmp_path / "points.gpkg"
                points.to_file(source_filename)
                source = gu.PointCloud(source_filename, data_name="height")
                cluster = request.getfixturevalue("mp_cluster")
                # Chunks of 37 on purpose to leave a shorter final partition
                options = {"mp_config": MultiprocConfig(chunks=37, cluster=cluster)}

            # Mask the left half of the point grid
            mask: Any
            if mask_form == "raster":
                mask = gu.Raster.from_array((x < 6)[::-1], from_origin(0, 11, 1, 1), points.crs)
            elif mask_form in {"pointcloud", "chunked_pointcloud"}:
                mask = gu.PointCloud.from_xyz(
                    x.ravel(), y.ravel(), (x < 6).ravel(), crs=points.crs, data_name="include"
                )
            else:
                mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 5.5, 11.5)], crs=points.crs)
            source_mask = mask

            # Read point masks in different partitions from the source
            if mask_form == "chunked_pointcloud":
                if execution == "dask":
                    filename, chunks, column = pointcloud_file(mask.gdf, 4)
                    source_mask = gu.open_pointcloud(
                        filename, data_name=column, columns="all", chunks=chunks, as_object_type=as_object_type
                    )
                    assert not source_mask.pc.is_loaded
                else:
                    mask_filename = tmp_path / "mask.gpkg"
                    mask.to_file(mask_filename)
                    source_mask = gu.PointCloud(mask_filename, data_name="include")
                    assert not source_mask.is_loaded

            # Compare pairs with the same in-memory call
            options.update(n_pairs=100, sampling="random_xy", random_state=9)
            expected = points.pairsample(mask=mask, n_pairs=100, sampling="random_xy", random_state=9)
            assert not source.is_loaded
            result = source.pairsample(mask=source_mask, **options, force_output_to_memory=True)

        # Check pairs, loading and source reads
        xr.testing.assert_equal(result, expected)
        assert result.sizes["pair"] == 100
        assert np.all(result.x < 6)
        assert not result.chunks
        assert not source.is_loaded
        if execution == "dask":
            assert len(reads) == 3
        if mask_form == "chunked_pointcloud":
            mask_loaded = source_mask.pc.is_loaded if execution == "dask" else source_mask.is_loaded
            assert not mask_loaded

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector"])
    @pytest.mark.parametrize(
        "execution,as_object_type",
        [("dask", "dataarray"), ("dask", "geodataframe"), ("mp", None)],
    )
    def test_pairsample__point_loglag_chunk_invariance(
        self,
        execution: str,
        as_object_type: Literal["dataarray", "geodataframe"] | None,
        pointcloud_file: Any,
        strategy: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        """Checks that Dask/MP point searches match in-memory pairs."""

        # 1/ Create points with one NaN
        y, x = np.mgrid[:16, :16]
        values = (np.sin(x / 3) + np.cos(y / 4)).ravel()
        values[9] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633, data_name="height")
        dataframe = points.gdf
        options: dict[str, Any] = {
            "n_pairs": 75,
            "min_distance": 1,
            "max_distance": 12,
            "strategy": strategy,
            "anchors_per_round": 200,
            "nn_tolerance": 0.6,
            "random_state": 3,
        }
        expected = dataframe.pc.pairsample(**options)

        # 2/ Record coordinate storage passed to the sampler
        disk_backed: list[bool] = []
        original_init = _PointPairSampler.__init__

        def record_coordinate_storage(sampler: _PointPairSampler, coordinates: NDArrayNum, **kwargs: Any) -> None:
            """Record whether sampler coordinates use a memory map."""
            disk_backed.append(isinstance(coordinates, np.memmap))
            original_init(sampler, coordinates, **kwargs)

        monkeypatch.setattr(_PointPairSampler, "__init__", record_coordinate_storage)

        # 3/ Sample Dask partitions or MP chunks
        with ExitStack() as stack:
            source: Any
            if execution == "dask":
                dask = pytest.importorskip("dask")
                stack.enter_context(dask.config.set(scheduler="synchronous"))
                filename, chunks, column = pointcloud_file(dataframe, 4)
                source = gu.open_pointcloud(
                    filename, data_name=column, columns="all", chunks=chunks, as_object_type=as_object_type
                ).pc
            else:
                source_filename = tmp_path / "loglag-points.gpkg"
                points.to_file(source_filename)
                source = gu.PointCloud(source_filename, data_name="height")
                cluster = request.getfixturevalue("mp_cluster")
                options["mp_config"] = MultiprocConfig(chunks=67, cluster=cluster)
            assert not source.is_loaded
            result = source.pairsample(**options)

        # 4/ Check pairs, coordinate storage and loading
        assert disk_backed == [execution == "mp"]
        assert not source.is_loaded
        if execution == "dask":
            assert all(isinstance(column.data, dask.array.Array) for column in result.data_vars.values())
        else:
            assert not result.chunks
        xr.testing.assert_equal(result.compute(), expected)
        assert result.sizes["pair"] == 75
        assert not source.is_loaded

    @pytest.mark.parametrize(
        "sampling,strategy",
        [
            ("loglag", "independent"),
            ("loglag", "anchors"),
            ("loglag", "chunk_anchors"),
            ("loglag", "anchor_batched"),
            ("random_xy", "chunk_anchors"),
        ],
    )
    @pytest.mark.parametrize("execution", ["dask", "mp"])
    def test_pairsample__raster_chunk_invariance(
        self, sampling: str, strategy: str, execution: str, tmp_path: Path, request: pytest.FixtureRequest
    ) -> None:
        """Checks that Dask and multiprocessing raster reads return the same pairs as in-memory sampling."""

        # 1/ Create a raster with NaNs
        values = np.arange(900, dtype=float).reshape(30, 30)
        values[2:5, 4:8] = np.nan
        transform = from_origin(0, 30, 2, 3)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=np.nan)
        source: Any
        if execution == "dask":
            da = pytest.importorskip("dask.array")
            source = gu.DataArrayRasterAccessor.from_array(
                da.from_array(values, chunks=values.shape), transform, 32633
            ).rst
        else:
            filename = tmp_path / "pairs.tif"
            eager.to_file(filename)
            source = gu.Raster(filename)

        # 2/ Set the seed and distance limits
        options = {
            "n_pairs": 120,
            "sampling": sampling,
            "strategy": strategy,
            "min_distance": 2,
            "max_distance": 40,
            "random_state": 42,
            "anchors_per_round": 100,
            "distances_per_anchor": 3,
            "angles_per_distance": 3,
        }

        # 3/ Compare one source tile with in-memory sampling
        if execution == "dask":
            result = source.pairsample(**options, force_output_to_memory=True)
            assert isinstance(source.data, da.Array)
        else:
            cluster = request.getfixturevalue("mp_cluster")
            config = MultiprocConfig(chunks=source.shape, cluster=cluster)
            result = source.pairsample(mp_config=config, **options)
        expected = eager.pairsample(**options)

        # 4/ Check pairs and loading
        xr.testing.assert_equal(result, expected)
        assert result.sizes["pair"] == 120
        assert not result.chunks
        assert not source.is_loaded

    @pytest.mark.parametrize("read_strategy", ["native", "reuse_values", "packed_validity"])
    def test_pairsample__raster_multiprocessing_tiles(
        self, tmp_path: Path, read_strategy: str, mp_cluster: MpCluster
    ) -> None:
        """Checks that multiprocessing tiles keep local pairs inside a file mask and match in-memory values."""

        # 1/ Create Dask and file rasters with a left-half mask
        da = pytest.importorskip("dask.array")

        values = np.arange(576, dtype=float).reshape(24, 24)
        keep = np.zeros(values.shape, dtype=bool)
        keep[:, :12] = True
        transform = from_origin(0, 24, 1, 1)
        eager_source = gu.Raster.from_array(values, transform, 32633)
        eager_mask = gu.Raster.from_array(keep, transform, 32633)
        dask_source = gu.DataArrayRasterAccessor.from_array(da.from_array(values, chunks=(13, 17)), transform, 32633)
        dask_mask = gu.DataArrayRasterAccessor.from_array(da.from_array(keep, chunks=(13, 17)), transform, 32633)
        source_filename = tmp_path / "local-values.tif"
        mask_filename = tmp_path / "local-mask.tif"
        gu.Raster.from_array(values, transform, 32633).to_file(source_filename)
        gu.Raster.from_array(keep, transform, 32633).to_file(mask_filename)
        file_source = gu.Raster(source_filename)
        file_mask = gu.Raster(mask_filename, is_mask=True)
        options = {
            "n_pairs": 150,
            "min_distance": 1,
            "max_distance": 5,
            "hybrid_local_fraction": 1,
            "random_state": 7,
            "index_dtype": np.int16,
            "distance_dtype": np.float32,
            "read_strategy": read_strategy,
        }

        # 2/ Sample 13 x 17 chunks with shorter final rows and columns
        dask_pairs = dask_source.rst.pairsample(mask=dask_mask, **options)
        config = MultiprocConfig(chunks=(13, 17), cluster=mp_cluster)
        mp_pairs = file_source.pairsample(mask=file_mask, mp_config=config, **options)
        eager_pairs = eager_source.pairsample(mask=eager_mask, mp_config=config, **options)

        # 3/ Compare pairs using the same chunk boundaries
        xr.testing.assert_equal(dask_pairs, mp_pairs)
        xr.testing.assert_equal(mp_pairs, eager_pairs)
        row_edges = np.array([0, 13, 24])
        column_edges = np.array([0, 17, 24])
        rows = np.searchsorted(row_edges, dask_pairs.row, side="right")
        columns = np.searchsorted(column_edges, dask_pairs.column, side="right")
        assert np.array_equal(rows[:, 0], rows[:, 1])
        assert np.array_equal(columns[:, 0], columns[:, 1])
        assert np.all(keep[dask_pairs.row, dask_pairs.column])
        assert np.array_equal(dask_pairs.value, values[dask_pairs.row, dask_pairs.column])

        # 4/ Check dtypes and loading
        assert dask_pairs["index"].dtype == np.int16
        assert dask_pairs["distance"].dtype == np.float32
        assert not dask_source._in_memory and not dask_mask._in_memory
        assert not file_source.is_loaded and not file_mask.is_loaded

    @pytest.mark.parametrize("n_pairs", [1, 30])
    @pytest.mark.parametrize("read_strategy", ["reuse_values", "packed_validity"])
    @pytest.mark.parametrize("sampling", ["loglag", "random_xy"])
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_pairsample__multiprocess_cached_integer_values(
        self, tmp_path: Path, n_pairs: int, read_strategy: str, sampling: str, dtype: Any, mp_cluster: MpCluster
    ) -> None:
        """Checks that MP caches preserve 64-bit integer endpoints after excluding masked cells."""

        # Write masked integers above float64's exact range
        values = np.ma.array(2**60 + np.arange(100, dtype=dtype).reshape(10, 10), mask=False)
        values.mask[0] = True
        eager = gu.Raster.from_array(values, from_origin(0, 10, 1, 1), 32633, nodata=0)
        filename = tmp_path / "integers.tif"
        eager.to_file(filename)
        source = gu.Raster(filename)
        options: dict[str, Any] = {
            "n_pairs": n_pairs,
            "sampling": sampling,
            "strategy": "independent",
            "random_state": 3,
            "read_strategy": read_strategy,
        }

        # Sample the same file through actual workers without loading the source object
        config = MultiprocConfig(chunks=(6, 7), cluster=mp_cluster)
        result = source.pairsample(**options, mp_config=config)
        expected = eager.pairsample(**options)

        # Check integer values directly, then compare pairs with eager sampling
        # Eager nodata handling converts values to float64
        assert not source.is_loaded
        assert not result.chunks
        assert result.value.dtype == np.dtype(dtype)
        indexes = result["index"].values
        np.testing.assert_array_equal(result.value.values, values.data.ravel()[indexes])
        comparison = result.assign(value=result.value.astype(expected.value.dtype))
        xr.testing.assert_identical(comparison, expected)


class TestPairOutputChunked:
    """Test module for disk pair output and temporary file cleanup."""

    @pytest.mark.parametrize(
        "strategy,deduplicate,read_strategy",
        [
            (strategy, deduplicate, "native")
            for strategy in ("independent", "anchors", "chunk_anchors", "anchor_batched")
            for deduplicate in ("none", "per_anchor", "global")
        ]
        + [("independent", "global", read_strategy) for read_strategy in ("reuse_values", "packed_validity")],
    )
    def test_pairsample__disk_raster_columns(
        self, tmp_path: Path, strategy: str, deduplicate: str, read_strategy: str
    ) -> None:
        """Checks that disk and in-memory raster pairs match."""

        # Create a masked raster with nonfinite values
        da = pytest.importorskip("dask.array")
        import pandas as pd

        values = np.arange(31 * 37, dtype=np.float32).reshape(31, 37)
        values[2:7, 4:11] = np.nan
        values[9, 13] = np.inf
        mask = np.ones(values.shape, dtype=bool)
        mask[15:18] = False
        transform = from_origin(0, 31, 2, 3)
        source = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options: dict[str, Any] = {
            "n_pairs": 347,
            "random_state": 8,
            "batch_pairs": 73,
            "anchors_per_round": 19,
            "distances_per_anchor": 3,
            "angles_per_distance": 3,
            "strategy": strategy,
            "deduplicate": deduplicate,
            "read_strategy": read_strategy,
            "mask": mask,
            "index_dtype": np.int16,
            "distance_dtype": np.float32,
        }

        # Sample in-memory and disk pairs with the same chunks
        with warnings.catch_warnings(record=True) as native_warnings:
            warnings.simplefilter("always", UserWarning)
            expected = eager.pairsample(**options, mp_config=MultiprocConfig(chunks=(7, 9)))
        with warnings.catch_warnings(record=True) as disk_warnings:
            warnings.simplefilter("always", UserWarning)
            result = source.pairsample(
                **options,
                mp_config=MultiprocConfig(chunks=(7, 9)),
                output_storage="disk",
                output_directory=tmp_path,
                output_chunks=31,
            )

        # Check lazy columns and files
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        assert all(max(column.data.chunks[0]) <= 31 for column in result.data_vars.values())
        assert isinstance(result.indexes["pair"], pd.RangeIndex)
        assert result.attrs["output_storage"] == "disk"
        assert list(tmp_path.glob("*/batch-*.npz"))
        assert not list(tmp_path.glob("*/unique.sqlite"))

        # Check equality, dtypes and warnings
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)
        assert computed["index"].dtype == np.int16
        assert computed.distance.dtype == np.float32
        assert [str(w.message) for w in disk_warnings] == [str(w.message) for w in native_warnings]
        assert source.is_loaded

    @pytest.mark.parametrize("read_strategy", ["native", "reuse_values", "packed_validity"])
    def test_pairsample__disk_random_raster_refill(self, tmp_path: Path, read_strategy: str) -> None:
        """Checks that disk output preserves random pair selection."""

        # Create a small raster to force duplicate draws
        da = pytest.importorskip("dask.array")

        values = np.arange(36.0).reshape(6, 6)
        values[1:3, 2:4] = np.nan
        transform = from_origin(0, 6, 1, 1)
        source = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options: dict[str, Any] = {
            "sampling": "random_xy",
            "read_strategy": read_strategy,
            "n_pairs": 301,
            "min_distance": 1,
            "max_distance": 6,
            "random_state": 8,
            "batch_pairs": 67,
        }
        expected = eager.pairsample(**options)
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=7),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=17,
        )

        # Check equality and loading
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)
        assert source.is_loaded

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector", "random_xy"])
    def test_pairsample__disk_point_columns(self, tmp_path: Path, strategy: str) -> None:
        """Checks that disk and in-memory point pairs match."""

        # Write points with NaNs to a file
        da = pytest.importorskip("dask.array")

        y, x = np.mgrid[:8, :11]
        z = np.sin(np.arange(x.size, dtype=np.float32))
        z[7:11] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), z, crs=32633)
        filename = tmp_path / "source.gpkg"
        points.to_file(filename)
        source = gu.PointCloud(filename, data_name=points.data_name)
        options: dict[str, Any] = {
            "n_pairs": 329,
            "random_state": 8,
            "nn_batch_size": 71,
            "nn_tolerance": 0.4,
            "min_distance": 1,
            "max_distance": 6,
            "anchors_per_round": 19,
            "attempts_per_anchor": 3,
            "n_bins": 3,
        }
        if strategy == "random_xy":
            options["sampling"] = strategy
        else:
            options["strategy"] = strategy
        expected = points.pairsample(**options)
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=7),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=29,
        )

        # Check lazy output after sampling closes its source files
        assert not source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        assert all(max(column.data.chunks[0]) <= 29 for column in result.data_vars.values())
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)
        assert not source.is_loaded

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_pairsample__disk_forced_to_memory(self, tmp_path: Path, source_kind: str) -> None:
        """Checks that forcing disk output into memory removes temporary files."""

        pytest.importorskip("dask.array")

        # Create raster or point data
        import gc

        rows, columns = np.indices((7, 9))
        values = np.arange(63.0).reshape(7, 9)
        source: Any
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 7, 1, 1), 32633)
        else:
            source = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633)
        eager = source
        options = {"n_pairs": 37, "sampling": "random_xy", "random_state": 42}
        expected = eager.pairsample(**options)

        # Force disk output into memory
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=7),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=11,
            force_output_to_memory=True,
        )
        assert source.is_loaded
        assert all(isinstance(column.data, np.ndarray) for column in result.data_vars.values())
        assert result.attrs.pop("output_storage") == "memory"
        xr.testing.assert_identical(result, expected)

        # Check temporary files are removed
        gc.collect()
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_pairsample__disk_incomplete_sample(self, tmp_path: Path, source_kind: str) -> None:
        """Checks that incomplete disk samples match in-memory samples."""

        # Create nine locations and limit sampling to one round
        da = pytest.importorskip("dask.array")

        rows, columns = np.indices((3, 3))
        values = np.arange(9.0).reshape(3, 3)
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 32633)
            batch_options = {"batch_pairs": 31}
        else:
            source = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633)
            batch_options = {"nn_batch_size": 31}
        options: dict[str, Any] = {
            "n_pairs": 1000,
            "sampling": "random_xy",
            "min_distance": 0.5,
            "max_distance": 3,
            "max_rounds": 1,
            "random_state": 42,
            **batch_options,
        }

        # Request more pairs than one round can return
        with pytest.warns(UserWarning) as native_warnings:
            expected = source.pairsample(**options, force_output_to_memory=True)
        with pytest.warns(UserWarning) as disk_warnings:
            result = source.pairsample(
                **options,
                mp_config=MultiprocConfig(chunks=7),
                output_storage="disk",
                output_directory=tmp_path,
                output_chunks=4,
            )

        # Check the partial sample, warnings and loading
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        assert 0 < result.sizes["pair"] < 1000
        assert result.attrs["accepted_pairs"] == result.sizes["pair"]
        assert [str(w.message) for w in disk_warnings] == [str(w.message) for w in native_warnings]
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)

    def test_pairsample__disk_collection_batch_limit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that disk pair collection stays within the batch limit."""

        # Create a raster and request more pairs than fit in one batch
        da = pytest.importorskip("dask.array")

        values = np.arange(32 * 35.0).reshape(32, 35)
        transform = from_origin(0, 32, 1, 1)
        source = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        eager = source
        options: dict[str, Any] = {
            "n_pairs": 701,
            "strategy": "independent",
            "random_state": 42,
            "batch_pairs": 53,
            "max_rounds": 100,
        }
        expected = eager.pairsample(**options)
        concatenate, arange = np.concatenate, np.arange

        def bounded_concatenate(arrays: Any, *args: Any, **kwargs: Any) -> Any:
            """Fail if pair collection joins more rows than a single candidate batch allows."""

            rows = sum(len(array) for array in arrays if isinstance(array, np.ndarray) and array.ndim)
            assert rows <= 53
            return concatenate(arrays, *args, **kwargs)

        def bounded_arange(*args: Any, **kwargs: Any) -> Any:
            """Fail if output construction materializes one label per requested pair."""

            result = arange(*args, **kwargs)
            assert result.size != 701
            return result

        # Check allocation limits during sampling
        with monkeypatch.context() as patch:
            patch.setattr(np, "concatenate", bounded_concatenate)
            patch.setattr(np, "arange", bounded_arange)
            result = source.pairsample(
                **options,
                mp_config=MultiprocConfig(chunks=7),
                output_storage="disk",
                output_directory=tmp_path,
                output_chunks=17,
            )
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)
        assert source.is_loaded

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_pairsample__disk_multiprocessing_files(self, tmp_path: Path, source_kind: str) -> None:
        """Checks that multiprocessing pair output leaves file inputs unloaded."""

        # Write and reopen raster or point data
        da = pytest.importorskip("dask.array")

        values = np.arange(30 * 30.0).reshape(30, 30)
        options: dict[str, Any] = {"n_pairs": 201, "sampling": "random_xy", "random_state": 42}
        if source_kind == "raster":
            eager = gu.Raster.from_array(values, from_origin(0, 30, 1, 1), 32633, nodata=-99999)
            filename = tmp_path / "source.tif"
            eager.to_file(filename)
            source = gu.Raster(filename, load_data=False)
            chunks: Any = (13, 17)
        else:
            rows, columns = np.indices(values.shape)
            eager = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633, data_name="height")
            filename = tmp_path / "source.gpkg"
            eager.to_file(filename)
            source = gu.PointCloud(filename, data_name="height")
            chunks = 137
        expected = eager.pairsample(**options)

        # Sample pairs, then close the cluster before reading output
        with ClusterGenerator("multiprocessing", nb_workers=2) as cluster:
            config = MultiprocConfig(cluster=cluster, chunks=chunks)
            result = source.pairsample(
                **options,
                mp_config=config,
                output_storage="disk",
                output_directory=tmp_path,
                output_chunks=37,
            )
        assert not source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        computed = result.compute()
        for name in ("output_storage", "lazy_output", "output_chunks"):
            computed.attrs.pop(name)
        xr.testing.assert_identical(computed, expected)
        assert not source.is_loaded

    def test_pairsample__disk_graph_storage_lifetime(self, tmp_path: Path) -> None:
        """Checks that derived arrays can still read temporary pair files."""

        # Create disk pair output with uneven batches
        import gc

        da = pytest.importorskip("dask.array")

        values = np.arange(23 * 27.0).reshape(23, 27)
        source = gu.Raster.from_array(values, from_origin(0, 23, 1, 1), 32633, nodata=-99999)
        eager = gu.Raster.from_array(values, from_origin(0, 23, 1, 1), 32633, nodata=-99999)
        options: dict[str, Any] = {"n_pairs": 101, "strategy": "independent", "random_state": 42}
        expected = eager.pairsample(**options)
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=7),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=17,
        )
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        directory = next(tmp_path.iterdir())
        derived = result.value.data + 1

        # Delete the dataset and check the derived array
        del result
        gc.collect()
        assert directory.exists()
        np.testing.assert_array_equal(derived.compute(), expected.value.data + 1)
        assert source.is_loaded
        del derived
        gc.collect()
        assert not directory.exists()

    @pytest.mark.parametrize("set_as_default", [True, False])
    @pytest.mark.parametrize("derived", [False, True])
    def test_pairsample__disk_pending_persist_storage(
        self, tmp_path: Path, set_as_default: bool, derived: bool
    ) -> None:
        """Checks that temporary pair files survive pending distributed reads."""

        # Create disk pair output with uneven batches
        import gc
        import time

        da = pytest.importorskip("dask.array")
        pytest.importorskip("distributed")
        from distributed import Client, Event, LocalCluster

        values = np.arange(23 * 27.0).reshape(23, 27)
        transform = from_origin(0, 23, 1, 1)
        source = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options: dict[str, Any] = {"n_pairs": 101, "strategy": "independent", "random_state": 42}
        expected = eager.pairsample(**options)
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=7),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=17,
        )
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())
        directory = next(tmp_path.iterdir())
        namespace = directory.name

        def hold_worker() -> None:
            """Block the worker until all original graph references have been released."""

            Event(f"{namespace}-started").set()
            Event(f"{namespace}-release").wait(timeout=10)

        # Queue reads and delete the original dataset
        with LocalCluster(n_workers=1, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
            with Client(cluster, set_as_default=set_as_default) as client:
                blocked = client.submit(hold_worker)
                assert Event(f"{namespace}-started", client=client).wait(timeout=10)
                selected = result[["value", "distance"]]
                if derived:
                    selected = selected + 1
                persisted = client.persist(selected)
                del result, selected
                gc.collect()
                assert directory.exists()
                Event(f"{namespace}-release", client=client).set()
                blocked.result()
                np.testing.assert_array_equal(persisted.value.compute(scheduler=client), expected.value + int(derived))
                np.testing.assert_array_equal(
                    persisted.distance.compute(scheduler=client), expected.distance + int(derived)
                )
                assert source.is_loaded

                # Check files are removed after pending reads finish
                deadline = time.monotonic() + 5
                while directory.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert not directory.exists()

    @pytest.mark.parametrize("persist", [False, True])
    def test_pairsample__disk_cleanup(self, tmp_path: Path, persist: bool) -> None:
        """Checks that unrelated Dask tasks do not delay removal of pair files."""

        import gc

        da = pytest.importorskip("dask.array")
        pytest.importorskip("distributed")
        from distributed import Client, Event, LocalCluster, wait

        # Create disk output with a shorter final batch
        values = np.arange(100.0).reshape(10, 10)
        source = gu.Raster.from_array(values, from_origin(0, 10, 1, 1), 32633)
        options: dict[str, Any] = {"n_pairs": 19, "strategy": "independent", "random_state": 42}
        expected = source.pairsample(**options)
        result = source.pairsample(
            **options,
            mp_config=MultiprocConfig(chunks=4),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=7,
        )
        directory = next(tmp_path.iterdir())
        namespace = directory.name
        assert source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in result.data_vars.values())

        def hold_worker() -> None:
            """Wait while the test checks removal of unrelated files."""

            Event(f"{namespace}-started").set()
            Event(f"{namespace}-release").wait(timeout=10)

        # Leave one thread busy while another can finish pair reads
        with LocalCluster(n_workers=1, threads_per_worker=2, processes=False, dashboard_address=None) as cluster:
            with Client(cluster) as client:
                blocked = client.submit(hold_worker)
                assert Event(f"{namespace}-started", client=client).wait(timeout=10)
                try:
                    if persist:
                        saved = client.persist(result.value.data + 1)
                        wait(saved)
                        np.testing.assert_array_equal(saved.compute(), expected.value.data + 1)

                    # Release the graph while the unrelated task is still running
                    del result
                    gc.collect()
                    assert not directory.exists()
                    assert blocked.status == "pending"
                    assert source.is_loaded
                finally:
                    Event(f"{namespace}-release", client=client).set()
                    blocked.result()


class TestPairSampleErrors:
    """Test module for errors/warnings raised by raster and point pair sampling."""

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize(
        "options,error_type,message",
        [
            ({"n_pairs": 0}, ValueError, "n_pairs"),
            ({"max_rounds": 0}, ValueError, "positive integers"),
            ({"sampling": "unknown"}, ValueError, "sampling.*loglag.*random_xy"),
            ({"strategy": "unknown"}, ValueError, "strategy"),
            ({"index_dtype": float}, TypeError, "index_dtype.*distance_dtype"),
            ({"distance_dtype": int}, TypeError, "index_dtype.*distance_dtype"),
            ({"min_distance": -1}, ValueError, "Require 0 <"),
            ({"min_distance": 0}, ValueError, "Require 0 <"),
            ({"min_distance": 3, "max_distance": 2}, ValueError, "Require 0 <"),
            ({"random_state": -1}, ValueError, "non-negative integer"),
            ({"force_output_to_memory": "yes"}, TypeError, "force_output_to_memory"),
        ],
    )
    def test_pairsample__error_options_before_reading(
        self,
        source_kind: str,
        options: dict[str, Any],
        error_type: type[Exception],
        message: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks an error is raised for invalid options before reading data."""

        # Create a source and prevent data reads
        from geoutils.sampling import pairsampling

        rows, columns = np.indices((7, 9))
        values = np.arange(63.0).reshape(7, 9)
        source: Any
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 7, 1, 1), 32633)
            reader_name = "_prepare_raster_pair_source"
        else:
            source = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633)
            reader_name = "_prepare_point_pair_data"

        def fail_read(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("Invalid pair sampling options must fail before source reads")

        monkeypatch.setattr(pairsampling, reader_name, fail_read)

        # Check the error occurs before reading
        arguments = {"n_pairs": 10, **options}
        with pytest.raises(error_type, match=message):
            source.pairsample(**arguments)

    def test_pairsample__error_unknown_read_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks an error is raised for an unknown read strategy."""

        # Create a lazy raster and prevent computation
        da = pytest.importorskip("dask.array")

        source = gu.DataArrayRasterAccessor.from_array(
            da.from_array(np.ones((7, 9)), chunks=(3, 4)), from_origin(0, 7, 1, 1), 32633
        ).rst

        def fail_compute(*args: Any, **kwargs: Any) -> None:
            pytest.fail("An unsupported read strategy must fail before reading values")

        monkeypatch.setattr(da.Array, "compute", fail_compute)
        with pytest.raises(ValueError, match="read_strategy.*native.*reuse_values.*packed_validity"):
            source.pairsample(n_pairs=10, read_strategy="unknown")
        assert not source.is_loaded and isinstance(source.data, da.Array)

    def test_pairsample__error_multiproc_with_dask(self) -> None:
        """Checks that pairsample() rejects simultaneous Dask and multiprocessing schedulers before reading values."""

        # Create a lazy raster and save its original array
        dask_array = pytest.importorskip("dask.array")
        values = dask_array.from_array(np.arange(100, dtype=float).reshape(10, 10), chunks=(4, 6))
        source = gu.DataArrayRasterAccessor.from_array(values, from_origin(0, 10, 1, 1), 32633)

        # Check the scheduler error and unchanged source array
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            source.rst.pairsample(n_pairs=20, mp_config=MultiprocConfig(chunks=4))
        assert source.data is values
        assert not source._in_memory

    def test_pairsample__error_point_multiproc_rectangular_chunks(self) -> None:
        """Checks that point pair multiprocessing requires one integer row partition size."""

        # Create points in memory to avoid file reads
        points = gu.PointCloud.from_xyz([0, 1, 2], [0, 0, 0], [3, 4, 5], crs=32633)

        # Check that point rows require a scalar chunk size
        with pytest.raises(ValueError, match="Point-cloud multiprocessing requires an integer chunk size"):
            points.pairsample(n_pairs=1, mp_config=MultiprocConfig(chunks=(2, 2)))

    @pytest.mark.parametrize("option", ["batch_pairs", "chunks_per_round", "angles_per_distance"])
    def test_pairsample__error_raster_invalid_batch_controls(self, raster: gu.Raster, option: str) -> None:
        """Checks that zero sampling controls fail before creating batches or indexing empty anchors."""

        with pytest.raises(ValueError, match="controls must be"):
            raster.pairsample(n_pairs=20, **{option: 0})

    @pytest.mark.parametrize("mask_form", ["wrong_count", "numeric", "numeric_pointcloud", "different_coordinates"])
    def test_pairsample__error_point_invalid_masks(self, mask_form: str) -> None:
        """Checks that point masks require boolean values at the same number of ordered source locations."""

        # Create finite points so only the mask is invalid
        y, x = np.mgrid[:5, :5]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        mask: Any = np.ones(25, dtype=bool)
        error = "Argument ``mask`` must be boolean and contain one value per input location"
        if mask_form == "wrong_count":
            mask = mask[:-1]
        elif mask_form == "numeric":
            mask = mask.astype(float)
        elif mask_form == "numeric_pointcloud":
            mask = points
            error = "point support mask must contain boolean values"
        else:
            mask = gu.PointCloud.from_xyz(x.ravel() + 1, y.ravel(), mask, crs=points.crs)
            error = "does not share the ordered support coordinates"

        # Report the mask problem before choosing any pair endpoints
        with pytest.raises(ValueError, match=error):
            points.pairsample(n_pairs=10, mask=mask, random_state=9)

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_pairsample__error_index_capacity(self, source_kind: str) -> None:
        """Checks an error is raised when the index dtype cannot represent every source location."""

        # Create 144 locations (exceeding the 128 indexes available in int8)
        rows, columns = np.indices((12, 12))
        values = np.arange(144.0).reshape(12, 12)
        source: Any
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 12, 1, 1), 32633)
            location = "cell"
        else:
            source = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), crs=32633)
            location = "point"

        # Check that the error identifies the source locations that cannot be indexed
        with pytest.raises(ValueError, match=f"cannot represent every {location}"):
            source.pairsample(n_pairs=10, index_dtype=np.int8)

    @pytest.mark.parametrize(
        "mask_change,message",
        [("crs", "does not share the support CRS"), ("order", "does not share the ordered support coordinates")],
    )
    @pytest.mark.parametrize(
        "execution,as_object_type",
        [("dask", "dataarray"), ("dask", "geodataframe"), ("mp", None)],
    )
    def test_pairsample__error_point_mask_support_chunked(
        self,
        execution: str,
        as_object_type: Literal["dataarray", "geodataframe"] | None,
        pointcloud_file: Any,
        mask_change: str,
        message: str,
        tmp_path: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Checks an error is raised for a chunked point mask with a different CRS or row order."""

        # Create points and a boolean mask with a different CRS or row order
        y, x = np.mgrid[:4, :4]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), np.arange(16), crs=32633, data_name="height")
        order = slice(None, None, -1) if mask_change == "order" else slice(None)
        mask_crs = 4326 if mask_change == "crs" else points.crs
        point_mask = gu.PointCloud.from_xyz(
            x.ravel()[order], y.ravel()[order], np.ones(16, dtype=bool), crs=mask_crs, data_name="include"
        )

        # Split source and mask rows into different partitions
        with ExitStack() as stack:
            source: Any
            mask: Any
            options: dict[str, Any] = {}
            if execution == "dask":
                dask = pytest.importorskip("dask")
                pytest.importorskip("dask_geopandas")
                stack.enter_context(dask.config.set(scheduler="synchronous"))
                source_filename, source_chunks, source_column = pointcloud_file(points.gdf, 2)
                source = gu.open_pointcloud(
                    source_filename,
                    data_name=source_column,
                    columns="all",
                    chunks=source_chunks,
                    as_object_type=as_object_type,
                ).pc
                mask_filename, mask_chunks, mask_column = pointcloud_file(point_mask.gdf, 3)
                mask = gu.open_pointcloud(
                    mask_filename,
                    data_name=mask_column,
                    columns="all",
                    chunks=mask_chunks,
                    as_object_type=as_object_type,
                ).pc
            else:
                source_path = tmp_path / "source.gpkg"
                points.to_file(source_path)
                source = gu.PointCloud(source_path, data_name="height")
                mask_path = tmp_path / "mask.gpkg"
                point_mask.to_file(mask_path)
                mask = gu.PointCloud(mask_path, data_name="include")
                cluster = request.getfixturevalue("mp_cluster")
                options["mp_config"] = MultiprocConfig(chunks=7, cluster=cluster)

            # Check the error without loading either point table
            assert not source.is_loaded and not mask.is_loaded
            with pytest.raises(ValueError, match=message):
                source.pairsample(n_pairs=5, mask=mask, random_state=9, **options)
        assert not source.is_loaded
        assert not mask.is_loaded

    @pytest.mark.parametrize("finite_count", [0, 1])
    @pytest.mark.parametrize("read_strategy", ["native", "reuse_values", "packed_validity"])
    def test_pairsample__error_bounded_finite_count(self, finite_count: int, read_strategy: str) -> None:
        """Checks an error is raised after scanning every chunk with fewer than two finite cells."""

        # Create 56 chunks with the only possible finite cell at the end of the source
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        values = np.full((25, 37), np.nan, dtype=np.float32)
        if finite_count:
            values[-1, -1] = 1
        reads: list[int] = []

        @dask.delayed
        def load_values(index: int, block: NDArrayNum) -> NDArrayNum:
            """Record one source chunk read and return its original values."""

            reads.append(index)
            return block

        rows = []
        for row in range(0, 25, 4):
            columns = []
            for column in range(0, 37, 5):
                index = row // 4 * 8 + column // 5
                block = values[row : row + 4, column : column + 5]
                columns.append(da.from_delayed(load_values(index, block), shape=block.shape, dtype=values.dtype))
            rows.append(columns)
        transform = from_origin(0, 25, 1, 1)
        source = gu.DataArrayRasterAccessor.from_array(da.block(rows), transform, 32633, nodata=-99999).rst
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options: dict[str, Any] = {"n_pairs": 127, "read_strategy": read_strategy, "random_state": 42}

        # Check the same error after counting every source cell
        with pytest.warns(UserWarning, match="possible finite pairs"):
            with pytest.raises(ValueError, match="At least two finite raster cells"):
                eager.pairsample(**options)
        with dask.config.set(scheduler="synchronous"):
            with pytest.warns(UserWarning, match="possible finite pairs"):
                with pytest.raises(ValueError, match="At least two finite raster cells"):
                    source.pairsample(**options)

        # Check all 56 chunks were read before reporting too few cells
        assert len(reads) == 56
        assert not source.is_loaded
        assert isinstance(source.data, da.Array)


class TestPairOutputErrors:
    """Test module for errors in disk pair output."""

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize(
        "case",
        [
            "unknown_storage",
            "directory_without_disk",
            "missing_directory",
            pytest.param(
                "missing_dask",
                marks=pytest.mark.skipif(find_spec("dask") is not None, reason="Only runs if dask is missing."),
            ),
            "missing_mp",
            "dask_disk",
        ],
    )
    def test_pairsample__error_disk_options(self, tmp_path: Path, source_kind: str, case: str) -> None:
        """Checks an error is raised for invalid disk output options."""

        # Create a source and invalid output options
        if source_kind == "raster":
            source = gu.Raster.from_array(np.arange(100.0).reshape(10, 10), from_origin(0, 10, 1, 1), 32633)
        else:
            source = gu.PointCloud.from_xyz(np.arange(10.0), np.zeros(10), np.arange(10.0), 32633)
        options: dict[str, Any] = {"output_storage": "disk", "mp_config": MultiprocConfig(chunks=7)}
        error: type[Exception] = ValueError
        message = "output_directory"
        if case == "unknown_storage":
            options["output_storage"] = "unknown"
            message = "output_storage"
        elif case == "directory_without_disk":
            options.update(output_storage="memory", output_directory=tmp_path)
        elif case == "missing_directory":
            options["output_directory"] = tmp_path / "missing"
        elif case in {"missing_mp", "dask_disk"}:
            options.pop("mp_config")
            message = "requires mp_config"
            if case == "dask_disk":
                pytest.importorskip("dask.array")
                chunked = source.to_xarray().chunk()
                source = chunked.rst if source_kind == "raster" else chunked.pc
        else:
            error, message = ImportError, "Optional dependency 'dask' required"

        # Check the error and file cleanup
        with pytest.raises(error, match=message):
            source.pairsample(n_pairs=10, **options)
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize("completed_batches", [0, 2])
    def test_pairsample__error_disk_write_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed_batches: int
    ) -> None:
        """Checks an error is raised for failed writes and temporary files are removed."""

        pytest.importorskip("dask.array")

        # Make a disk batch write fail
        from geoutils.sampling import pairsampling

        source = gu.Raster.from_array(np.arange(100.0).reshape(10, 10), from_origin(0, 10, 1, 1), 32633)
        original = pairsampling.np.savez
        writes = []

        def failed_write(*args: Any, **kwargs: Any) -> None:
            if len(writes) == completed_batches:
                raise OSError("Disk write failed.")
            original(*args, **kwargs)
            writes.append(args[0])

        monkeypatch.setattr(pairsampling.np, "savez", failed_write)
        with pytest.raises(OSError, match="Disk write failed"):
            source.pairsample(
                n_pairs=100,
                strategy="independent",
                mp_config=MultiprocConfig(chunks=7),
                output_storage="disk",
                output_directory=tmp_path,
                output_chunks=11,
                deduplicate="global",
                random_state=42,
            )
        assert len(writes) == completed_batches
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_pairsample__error_disk_no_matching_pairs(self, tmp_path: Path, source_kind: str) -> None:
        """Checks an error is raised when no pairs match and temporary files are removed."""

        pytest.importorskip("dask.array")

        # Create nine locations closer together than the requested distance
        rows, columns = np.indices((3, 3))
        values = np.arange(9.0).reshape(3, 3)
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 32633)
        else:
            source = gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values.ravel(), 32633)

        # Check the error and file cleanup
        with pytest.raises(ValueError, match="No .* pairs could be sampled"):
            source.pairsample(
                n_pairs=10,
                sampling="random_xy",
                min_distance=4,
                max_distance=5,
                max_rounds=1,
                random_state=42,
                mp_config=MultiprocConfig(chunks=7),
                output_storage="disk",
                output_directory=tmp_path,
            )
        assert not list(tmp_path.iterdir())
