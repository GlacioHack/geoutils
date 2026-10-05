"""Tests for raster-vector interfacing."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely import LineString, MultiLineString, MultiPolygon, Polygon

import geoutils as gu
from geoutils import examples
from geoutils.exceptions import InvalidGridError
from geoutils.interface import rasterization
from geoutils.multiproc import MultiprocConfig


class TestRasterVectorInterface:
    """Test raster and point-cloud outputs created from vector geometries."""

    # Create a synthetic vector file with a square of size 1, started at position (10, 10)
    poly1 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    gdf = gpd.GeoDataFrame({"geometry": [poly1]}, crs="EPSG:4326")
    vector = gu.Vector(gdf)

    # Same with a square started at position (5, 5)
    poly2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    gdf = gpd.GeoDataFrame({"geometry": [poly2]}, crs="EPSG:4326")
    vector2 = gu.Vector(gdf)

    # Create a multipolygon with both
    multipoly = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame({"geometry": [multipoly]}, crs="EPSG:4326")
    vector_multipoly = gu.Vector(gdf)

    # Create a synthetic vector file with a square of size 5, started at position (8, 8)
    poly3 = Polygon([(8, 8), (13, 8), (13, 13), (8, 13)])
    gdf = gpd.GeoDataFrame({"geometry": [poly3]}, crs="EPSG:4326")
    vector_5 = gu.Vector(gdf)

    # Create a synthetic LineString geometry
    lines = LineString([(10, 10), (11, 10), (11, 11)])
    gdf = gpd.GeoDataFrame({"geometry": [lines]}, crs="EPSG:4326")
    vector_lines = gu.Vector(gdf)

    # Create a synthetic MultiLineString geometry
    multilines = MultiLineString([[(10, 10), (11, 10), (11, 11)], [(5, 5), (6, 5), (6, 6)]])
    gdf = gpd.GeoDataFrame({"geometry": [multilines]}, crs="EPSG:4326")
    vector_multilines = gu.Vector(gdf)

    # Package examples
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = gu.examples.get_path_test("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")
    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    @pytest.mark.parametrize("method", ["rasterize", "create_mask"])
    @pytest.mark.parametrize("input_type", ["raster", "xarray", "dask"])
    def test_rasterize_create_mask__multi_band(
        self,
        method: Literal["rasterize", "create_mask"],
        input_type: Literal["raster", "xarray", "dask"],
    ) -> None:
        """Checks that vector rasterization returns one spatial layer from every multi-band reference type."""

        pytest.importorskip("dask")
        import dask.array as da

        # Create equivalent native, Xarray and chunked Xarray references with two bands
        raster = gu.Raster.from_array(np.zeros((2, 4, 4)), (1, 0, 8, 0, -1, 12), 4326)
        if input_type == "raster":
            reference = raster
        else:
            reference = raster.to_xarray()
            if input_type == "dask":
                reference = reference.chunk({"band": 1, "y": 3, "x": 2})
        # Rasterize the vector or create a mask from the selected reference
        output = getattr(self.vector, method)(ref=reference)

        # Check that the result has one spatial layer and keeps Dask chunk sizes when applicable
        assert output.shape == (4, 4)
        if input_type == "dask":
            assert isinstance(output, xr.DataArray)
            assert isinstance(output.data, da.Array)
            assert isinstance(reference, xr.DataArray)
            assert isinstance(reference.data, da.Array)
            assert output.data.chunks == reference.data.chunks[-2:]
        else:
            assert isinstance(output, gu.Raster)

        # The unit polygon occupies exactly the pixel centred on (10.5, 10.5)
        expected = np.zeros((4, 4))
        expected[1, 2] = 1
        actual = output.to_numpy() if isinstance(output, xr.DataArray) else output.data
        np.testing.assert_array_equal(actual, expected)

    def test_rasterize(self) -> None:
        """Test rasterizing an EPSG:3426 dataset into a projection."""

        vct = gu.Vector(self.everest_outlines_path)
        rst = gu.Raster(self.landsat_b4_crop_path)

        # Use Web Mercator at 30 m.
        # Capture the warning on resolution not matching exactly bounds
        vct.rasterize(res=30, crs=3857)

        # Typically, rasterize returns a raster
        burned_in2_out1 = vct.rasterize(rst, in_value=2, out_value=1)
        assert isinstance(burned_in2_out1, gu.Raster)

        # For an in_value of 1 and out_value of 0 (default)
        burned_mask = vct.rasterize(rst, in_value=1)
        assert isinstance(burned_mask, gu.Raster)
        # Convert to boolean
        burned_mask = burned_mask.astype(bool)

        # Check that rasterizing with in_value=1 is the same as creating a mask
        assert burned_mask.raster_equal(vct.create_mask(rst), warn_failure_reason=True)

        # The two rasterization should match
        assert np.all(burned_in2_out1[burned_mask] == 2)
        assert np.all(burned_in2_out1[~burned_mask] == 1)

        # Check that errors are raised
        with pytest.raises(InvalidGridError, match="Either 'ref' or 'crs' must be provided"):
            vct.rasterize(rst, crs=3857)

    def test_rasterize__fractional_feature_and_union_layers(self) -> None:
        """Checks that overlapping polygons produce separate coverage layers or one layer for their union."""

        # Cross four cells with one centered square and cover the complete right column with another polygon
        centered = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        right = Polygon([(1, 0), (2, 0), (2, 2), (1, 2)])
        vector = gu.Vector(gpd.GeoDataFrame({"name": ["centered", "right"]}, geometry=[centered, right], crs=32631))
        options = {"res": 1, "bounds": (0, 0, 2, 2), "crs": 32631, "out_dtype": np.float64}

        # Preserve both feature contributions even where they overlap the same right-column cells
        layered = vector.rasterize(**options, fractional=True, overlap_backend="shapely")
        expected_layers = np.array([np.full((2, 2), 0.25), [[0, 1], [0, 1]]])
        np.testing.assert_allclose(layered.data, expected_layers)
        assert layered.tags["long_name"] == ("0", "1")

        # Union coverage counts shared polygon area once and cannot exceed complete cell coverage
        union = vector.rasterize(**options, fractional="union", overlap_backend="shapely")
        np.testing.assert_allclose(union.data, [[0.25, 1], [0.25, 1]])
        assert union.tags["long_name"] == ("union",)

    def test_rasterize__fractional_group_layers_union_repeated_features(self) -> None:
        """Checks that fractional_by unions overlapping features within each named output layer."""

        # Repeat the centered polygon in one group and place a full-height polygon in another group
        centered = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        right = Polygon([(1, 0), (2, 0), (2, 2), (1, 2)])
        dataframe = gpd.GeoDataFrame(
            {"zone": ["shared", "shared", "right"]},
            geometry=[centered, centered, right],
            crs=32631,
        )

        # Equal labels form one union geometry, so the repeated centered polygon still covers one quarter per cell
        result = gu.Vector(dataframe).rasterize(
            res=1,
            bounds=(0, 0, 2, 2),
            crs=32631,
            fractional=True,
            fractional_by="zone",
            overlap_backend="shapely",
        )
        expected = np.array([np.full((2, 2), 0.25), [[0, 1], [0, 1]]], dtype=np.float32)
        np.testing.assert_allclose(result.data, expected)
        assert result.tags["long_name"] == ("shared", "right")

    def test_rasterize__fractional_chunked_backends_match_eager(self, tmp_path: Any) -> None:
        """Checks that fractional layers stay lazy with Dask and match eager and multiprocessing chunks."""

        pytest.importorskip("dask")
        import dask.array as da

        # Use two polygons crossing each 1 x 1 chunk so every backend must preserve global layer positions
        polygons = [
            Polygon([(0.25, 0.25), (2.25, 0.25), (2.25, 2.25), (0.25, 2.25)]),
            Polygon([(1.25, 1.25), (2.75, 1.25), (2.75, 2.75), (1.25, 2.75)]),
        ]
        vector = gu.Vector(gpd.GeoDataFrame(geometry=polygons, crs=32631))
        options = {"res": 1, "bounds": (0, 0, 3, 3), "crs": 32631, "fractional": True}

        # Calculate the eager reference and construct the same coverage as a lazy block graph
        expected = vector.rasterize(**options, overlap_backend="shapely")
        lazy = vector.rasterize(**options, dask=True, chunksizes=(1, 1), overlap_backend="shapely")
        assert isinstance(lazy, xr.DataArray)
        assert isinstance(lazy.data, da.Array)
        assert not lazy._in_memory
        np.testing.assert_array_equal(lazy.data.compute(), expected.data)
        assert not lazy._in_memory

        # Write the same 1 x 1 blocks in workers and compare the initially file-backed output
        output_path = tmp_path / "fractional.tif"
        config = MultiprocConfig(chunks=(1, 1), outfile=str(output_path))
        multiproc = vector.rasterize(**options, mp_config=config, overlap_backend="shapely")
        assert not multiproc.is_loaded
        np.testing.assert_array_equal(multiproc.data, expected.data)
        assert multiproc.tags["long_name"] == ("0", "1")

    def test_rasterize__fractional_exactextract_matches_shapely(self) -> None:
        """Checks that optional ExactExtract coverage agrees with Shapely for partial and complete cells."""

        pytest.importorskip("exactextract")

        # Cut several cells with a non-axis-aligned polygon to compare the two independent geometry engines
        polygon = Polygon([(0.1, 0.2), (2.8, 0.6), (2.2, 2.9), (0.4, 2.4)])
        vector = gu.Vector(gpd.GeoDataFrame(geometry=[polygon], crs=32631))
        options = {"res": 1, "bounds": (0, 0, 3, 3), "crs": 32631, "fractional": True}

        # ExactExtract stores coverage as float32, so compare within its approximately seven significant digits
        shapely_result = vector.rasterize(**options, overlap_backend="shapely")
        exactextract_result = vector.rasterize(**options, overlap_backend="exactextract")
        np.testing.assert_allclose(exactextract_result.data, shapely_result.data, rtol=5e-7, atol=5e-7)

    def test_rasterize__error_fractional_options(self) -> None:
        """Checks that fractional rasterization rejects burn semantics and an empty set of feature layers."""

        # A grouping column only has meaning when rasterization creates fractional layers
        with pytest.raises(ValueError, match="fractional_by requires fractional=True"):
            self.vector.rasterize(res=1, fractional_by="value")

        # Fractional coverage cannot also define overwrite values because each feature owns its own layer
        with pytest.raises(ValueError, match="in_value cannot be combined"):
            self.vector.rasterize(res=1, fractional=True, in_value=2)

        # An empty vector has no feature layers, while its union remains one valid zero-coverage layer
        empty = gu.Vector(gpd.GeoDataFrame(geometry=[], crs=4326))
        with pytest.raises(ValueError, match="at least one input feature"):
            empty.rasterize(res=1, bounds=(0, 0, 2, 2), fractional=True)
        union = empty.rasterize(res=1, bounds=(0, 0, 2, 2), fractional="union")
        np.testing.assert_array_equal(union.data, np.zeros((2, 2), dtype=np.float32))

    def test_rasterize__nodata_background(self) -> None:
        """Checks that rasterize() works properly with a NaN background."""

        # Rasterize polygon and request a NaN background value
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Ignore the Affine 3.1 transition warning emitted inside Rasterio
            warnings.filterwarnings(
                "ignore",
                message=r"Use `@` matmul instead of `\*` mul operator for matrix multiplication",
                category=PendingDeprecationWarning,
            )
            raster = self.vector.rasterize(res=1, bounds=(9, 9, 13, 13), crs=4326, in_value=1, out_value=np.nan)

        # Check that background is masked, nodata metadata is finite and polygon value are valid
        assert raster.nodata == -99999
        assert np.count_nonzero(raster.data.mask) > 0
        np.testing.assert_array_equal(raster.data.compressed(), np.ones(raster.data.count()))

        # Check with manual nodata
        custom_nodata = self.vector.rasterize(
            res=1, bounds=(9, 9, 13, 13), crs=4326, in_value=1, out_value=np.nan, nodata=-1
        )
        assert custom_nodata.nodata == -1
        assert np.count_nonzero(custom_nodata.data.mask) > 0
        np.testing.assert_array_equal(custom_nodata.data.compressed(), np.ones(custom_nodata.data.count()))

        # Check with a float32 array with NaNs
        dataarray = self.vector.ds.vct.rasterize(
            res=1, bounds=(9, 9, 13, 13), crs=4326, in_value=1, out_value=np.nan, out_dtype=np.float32
        )
        assert isinstance(dataarray, xr.DataArray)
        assert dataarray.dtype == np.float32
        assert dataarray.rst.nodata == -99999
        assert np.count_nonzero(np.isnan(dataarray.data)) > 0
        np.testing.assert_array_equal(dataarray.data[np.isfinite(dataarray.data)], np.ones(1, dtype=np.float32))

        # Raise error on non-finite nodata
        with pytest.raises(ValueError, match="nodata must be finite"):
            self.vector.rasterize(res=1, bounds=(9, 9, 13, 13), crs=4326, in_value=1, out_value=np.nan, nodata=np.nan)

    def test_rasterize__nodata_background_chunked(self, tmp_path: Any) -> None:
        """Checks that chunked rasterization keeps NaNs in memory and writes finite nodata values to a file."""

        pytest.importorskip("dask")
        import dask.array as da

        # Rasterize lazily into 2 x 2 chunks
        options = {
            "res": 1,
            "bounds": (9, 9, 13, 13),
            "crs": 4326,
            "in_value": 1,
            "out_value": np.nan,
            "out_dtype": np.float32,
        }
        dask_result = self.vector.rasterize(**options, dask=True, chunksizes=(2, 2))
        assert isinstance(dask_result, xr.DataArray)
        assert isinstance(dask_result.data, da.Array)
        assert not dask_result._in_memory

        # Compute lazy values and check that background uses NaN with finite nodata metadata
        dask_data = dask_result.data.compute()
        assert dask_result.rst.nodata == -99999
        assert np.count_nonzero(np.isnan(dask_data)) > 0
        np.testing.assert_array_equal(dask_data[np.isfinite(dask_data)], np.ones(1, dtype=np.float32))

        # Write chunks to file and check that background contains the finite nodata value
        outfile = tmp_path / "rasterized-nodata.tif"
        config = MultiprocConfig(chunks=(2, 2), outfile=str(outfile))
        file_result = self.vector.rasterize(**options, mp_config=config)
        assert not file_result.is_loaded
        assert file_result.nodata == -99999
        assert np.count_nonzero(file_result.data.mask) > 0
        assert np.all(file_result.data.data[file_result.data.mask] == -99999)
        np.testing.assert_array_equal(file_result.data.compressed(), np.ones(file_result.data.count()))

    def test_create_mask(self) -> None:
        """Checks for create_mask()."""

        # First with given res and bounds -> Should be a 21 x 21 array with 0 everywhere except center pixel
        vector = self.vector.copy()
        out_mask = vector.create_mask(res=1, bounds=(0, 0, 21, 21), as_array=True)
        ref_mask = np.zeros((21, 21), dtype="bool")
        ref_mask[10, 10] = True
        assert out_mask.shape == (21, 21)
        assert np.all(ref_mask == out_mask)

        # Check that vector has not been modified by accident
        assert vector.bounds == self.vector.bounds
        assert len(vector.ds) == len(self.vector.ds)
        assert vector.crs == self.vector.crs

        # Then with a gu.Raster as reference, single band
        rst = gu.Raster.from_array(np.zeros((21, 21)), transform=(1.0, 0.0, 0.0, 0.0, -1.0, 21.0), crs="EPSG:4326")
        out_mask = vector.create_mask(rst, as_array=True)
        assert out_mask.shape == (21, 21)

        # With gu.Raster, 2 bands -> fails...
        # rst = gu.Raster.from_array(np.zeros((2, 21, 21)), transform=(1., 0., 0., 0., -1., 21.), crs='EPSG:4326')
        # out_mask = vector.create_mask(rst)

        # Check that no warning is raised when creating a mask with a xres not multiple of vector bounds
        mask = vector.create_mask(res=1.01)

        # Check that by default, create_mask() returns a mask raster
        assert isinstance(mask, gu.Raster) and mask.is_mask

        # Check that an error is raised if no input is passed
        with pytest.raises(
            ValueError,
            match="Input arguments must define a valid raster or point cloud.",
        ):
            vector.create_mask()

        # If the raster has the wrong type
        with pytest.raises(ValueError, match="Input arguments must define a valid raster or point cloud."):
            vector.create_mask("lol")  # type: ignore

    def test_geometry_mask__alias(self) -> None:
        """Checks that geometry_mask() is a direct alias with the same result as create_mask()."""

        # Check that both names are actually the same implementation, on Vector and the vct accessor
        assert self.vector.geometry_mask.__func__ is self.vector.create_mask.__func__
        assert self.vector.ds.vct.geometry_mask.__func__ is self.vector.ds.vct.create_mask.__func__

        # They should return the same boolean array
        expected = self.vector.create_mask(res=1, bounds=(0, 0, 21, 21), as_array=True)
        actual = self.vector.geometry_mask(res=1, bounds=(0, 0, 21, 21), as_array=True)
        accessor_actual = self.vector.ds.vct.geometry_mask(res=1, bounds=(0, 0, 21, 21), as_array=True)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(accessor_actual, expected)

    @pytest.mark.parametrize("all_touched", [False, True])
    @pytest.mark.parametrize("in_value_mode", ["scalar", "iterable", "none"])
    def test_rasterize_create_mask__chunked_backends_equal(
        self,
        tmp_path: Any,
        all_touched: bool,
        in_value_mode: str,
    ) -> None:
        """
        Checks that rasterize() and create_mask() return the same eager, Dask and multiprocessing outputs.

        Dask outputs must remain lazy until computed and Multiprocessing outputs must initially remain file-backed.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Use one output grid without an input raster so every backend targets the same cells
        vect = self.vector.copy()
        bounds = (0.0, 0.0, 21.0, 21.0)
        res = 1.0
        crs = "EPSG:4326"
        chunksizes = (10, 7)

        # Define scalar, iterable and automatic burn values through the public interface
        if in_value_mode == "scalar":
            in_value = 7
            out_value = 0
        elif in_value_mode == "iterable":
            in_value = [7]  # type: ignore
            out_value = 0
        elif in_value_mode == "none":
            in_value = None
            out_value = 0
        else:
            raise ValueError("Unexpected in_value_mode")

        # Multiprocessing writes each output tile directly to one temporary raster
        mp_outfile = tmp_path / f"mp_rasterize_{all_touched}_{in_value_mode}.tif"
        mp_config = MultiprocConfig(chunks=chunksizes, outfile=str(mp_outfile), driver="GTiff")

        # 1/ Rasterize through each backend
        rst_base = vect.rasterize(
            res=res,
            bounds=bounds,
            crs=crs,
            in_value=in_value,
            out_value=out_value,
            all_touched=all_touched,
            out_dtype=np.uint8,
        )
        assert isinstance(rst_base, gu.Raster)
        base_arr = np.asarray(rst_base.data)

        # Dask builds the same raster as a lazy array
        rst_dask = vect.rasterize(
            res=res,
            bounds=bounds,
            crs=crs,
            in_value=in_value,
            out_value=out_value,
            all_touched=all_touched,
            out_dtype=np.uint8,
            dask=True,
            chunksizes=chunksizes,
        )
        assert isinstance(rst_dask, xr.DataArray)
        assert isinstance(rst_dask.data, da.Array)
        assert not rst_dask._in_memory
        dask_arr = np.asarray(rst_dask.data.compute())

        # Multiprocessing returns the written raster without loading its values
        rst_mp = vect.rasterize(
            res=res,
            bounds=bounds,
            crs=crs,
            in_value=in_value,
            out_value=out_value,
            all_touched=all_touched,
            out_dtype=np.uint8,
            mp_config=mp_config,
        )
        assert isinstance(rst_mp, gu.Raster)
        assert not rst_mp.is_loaded
        mp_arr = np.asarray(rst_mp.data)

        # All raster values must be exactly equal and computing Dask must not alter its lazy object
        assert base_arr.shape == dask_arr.shape == mp_arr.shape == (21, 21)
        assert np.array_equal(base_arr, dask_arr)
        assert np.array_equal(base_arr, mp_arr)
        assert not rst_dask._in_memory
        assert rst_mp.is_loaded

        # 2/ Create the corresponding mask through each backend
        m_base = vect.create_mask(
            res=res,
            bounds=bounds,
            crs=crs,
            all_touched=all_touched,
        )
        assert isinstance(m_base, gu.Raster)
        mask_base = np.asarray(m_base.data, dtype=bool)

        # Dask again keeps the output delayed until the explicit calculation
        m_dask = vect.create_mask(
            res=res,
            bounds=bounds,
            crs=crs,
            all_touched=all_touched,
            dask=True,
            chunksizes=chunksizes,
        )
        assert isinstance(m_dask, xr.DataArray)
        assert isinstance(m_dask.data, da.Array)
        assert not m_dask._in_memory
        mask_dask = np.asarray(m_dask.data.compute(), dtype=bool)

        # Multiprocessing returns another initially unloaded file-backed raster
        m_mp = vect.create_mask(
            res=res,
            bounds=bounds,
            crs=crs,
            all_touched=all_touched,
            mp_config=mp_config,
        )
        assert isinstance(m_mp, gu.Raster)
        assert not m_mp.is_loaded
        mask_mp = np.asarray(m_mp.data)

        assert m_base.shape == m_dask.shape == m_mp.shape == (21, 21)
        assert np.array_equal(mask_base, mask_dask)
        assert np.array_equal(mask_base, mask_mp)
        assert not m_dask._in_memory
        assert m_mp.is_loaded

        # The known center cell also checks that each result contains the expected geometry
        assert m_base[10, 10] is np.True_

    def test_rasterize__dask_builds_one_spatial_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test that Dask builds one spatial index before running its raster blocks.

        Selecting the features once avoids rebuilding the complete index in every worker.
        """

        pytest.importorskip("dask")

        # Wrap the index builder to count calls without changing its result
        index_count = 0
        build_spatial_index = rasterization._build_spatial_index

        def count_spatial_index(geometries: Any) -> Any:
            """Record each construction of the complete vector spatial index."""

            nonlocal index_count
            index_count += 1
            return build_spatial_index(geometries)

        monkeypatch.setattr(rasterization, "_build_spatial_index", count_spatial_index)

        # Building the lazy raster selects features once for all four output blocks
        rasterized = self.vector.rasterize(
            res=1,
            bounds=(0, 0, 21, 21),
            crs=4326,
            in_value=1,
            dask=True,
            chunksizes=(11, 11),
        )
        assert index_count == 1

        # Computing the blocks uses their selected features without rebuilding the index
        rasterized.data.compute(scheduler="synchronous")
        assert index_count == 1
