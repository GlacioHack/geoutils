"""Tests for raster-vector interfacing."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, MultiLineString, MultiPolygon, Polygon
import xarray as xr

import geoutils as gu
from geoutils import examples
from geoutils.exceptions import InvalidGridError
from geoutils.multiproc import MultiprocConfig

class TestRasterVectorInterface:

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

    def test_create_mask(self) -> None:
        """
        Test Vector.create_mask.
        """
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

        # Check that by default, create_mask returns a Mask
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

    @pytest.mark.parametrize("all_touched", [False, True])
    @pytest.mark.parametrize("in_value_mode", ["scalar", "iterable", "none"])
    def test_rasterize_create_mask__chunked_backends_equal(
        self,
        tmp_path,
        all_touched: bool,
        in_value_mode: str,
    ) -> None:
        """
        Test rasterize and create_mask for base versus chunked (Dask, Multiprocessing).

        Uses an output-only grid definition (no input raster dependence) so that
        all three backends are forced to target the exact same grid.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Output grid spec
        vect = self.vector.copy()
        bounds = (0.0, 0.0, 21.0, 21.0)
        res = 1.0
        crs = "EPSG:4326"

        # Chunking
        chunksizes = (10, 10)

        # Burn value modes for rasterize
        if in_value_mode == "scalar":
            in_value = 7
            out_value = 0
        elif in_value_mode == "iterable":
            # single-geom iterable (length must match geom count)
            in_value = [7]
            out_value = 0
        elif in_value_mode == "none":
            # None -> burn values become [1 .. N] internally; here N=1 so burn=1
            in_value = None
            out_value = 0
        else:
            raise ValueError("Unexpected in_value_mode")

        # Multiprocessing config (writes tiles to file)
        mp_outfile = tmp_path / f"mp_rasterize_{all_touched}_{in_value_mode}.tif"
        mp_config = MultiprocConfig(chunk_size=chunksizes[0], outfile=str(mp_outfile), driver="GTiff")

        # 1) RASTERIZE
        ##############

        # Base (eager)
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
        # Dask (lazy output)
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
        dask_arr = np.asarray(rst_dask.data.compute())

        # Multiprocessing (writes to disk then opens, output Raster is unloaded)
        rst_mp = vect.rasterize(
            res=res,
            bounds=bounds,
            crs=crs,
            in_value=in_value,
            out_value=out_value,
            all_touched=all_touched,
            out_dtype=np.uint8,
            mp_config=mp_config,
            chunksizes=chunksizes,
        )
        assert isinstance(rst_mp, gu.Raster)
        assert not rst_mp.is_loaded
        mp_arr = np.asarray(rst_mp.data)

        # Exact equality
        assert base_arr.shape == dask_arr.shape == mp_arr.shape == (21, 21)
        assert np.array_equal(base_arr, dask_arr)
        assert np.array_equal(base_arr, mp_arr)

        # 2) CREATE_MASK
        ################

        # Base (eager)
        m_base = vect.create_mask(
            res=res,
            bounds=bounds,
            crs=crs,
            all_touched=all_touched,
        )
        assert isinstance(m_base, gu.Raster)
        mask_base = np.asarray(m_base.data, dtype=bool)

        # Dask (lazy)
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
        mask_dask = np.asarray(m_dask.data.compute(), dtype=bool)

        # MP (writes to disk then opens, output is unloaded)
        m_mp = vect.create_mask(
            res=res,
            bounds=bounds,
            crs=crs,
            all_touched=all_touched,
            mp_config=mp_config,
            chunksizes=chunksizes,
        )
        assert isinstance(m_mp, gu.Raster)
        assert not m_mp.is_loaded
        mask_mp = np.asarray(m_mp.data)

        assert m_base.shape == m_dask.shape == m_mp.shape == (21, 21)
        assert np.array_equal(mask_base, mask_dask)
        assert np.array_equal(mask_base, mask_mp)

        # Sanity check that the known "center pixel" is True for this geometry/grid
        assert m_base[10, 10] is np.True_
