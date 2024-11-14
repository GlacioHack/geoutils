"""Functionalities for geotransformations of vectors."""

from __future__ import annotations

import os

import geopandas as gpd
import pyogrio
import rasterio as rio
from rasterio.crs import CRS

import geoutils as gu


def _reproject(
    gdf: gpd.GeoDataFrame,
    ref: gu.Raster | rio.io.DatasetReader | gu.Vector | gpd.GeoDataFrame | str | None = None,
    crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Reproject a vector. See Vector.reproject() for more details."""

    # Check that either ref or crs is provided
    if (ref is not None and crs is not None) or (ref is None and crs is None):
        raise ValueError("Either of `ref` or `crs` must be set. Not both.")

    # Case a raster or vector is provided as reference
    if ref is not None:
        # Check that ref type is either str, Raster or rasterio data set
        # Preferably use Raster instance to avoid rasterio data set to remain open. See PR #45
        if isinstance(ref, (gu.Raster, gu.Vector)):
            ds_ref = ref
        elif isinstance(ref, (rio.io.DatasetReader, gpd.GeoDataFrame)):
            ds_ref = ref
        elif isinstance(ref, str):
            if not os.path.exists(ref):
                raise ValueError("Reference raster or vector path does not exist.")
            try:
                ds_ref = gu.Raster(ref, load_data=False)
            except rio.errors.RasterioIOError:
                try:
                    ds_ref = gu.Vector(ref)
                except pyogrio.errors.DataSourceError:
                    raise ValueError("Could not open raster or vector with rasterio or pyogrio.")
        else:
            raise TypeError("Type of ref must be string path to file, Raster or Vector.")

        # Read reprojecting params from ref raster
        crs = ds_ref.crs
    else:
        # Determine user-input target CRS
        crs = CRS.from_user_input(crs)

    new_ds = gdf.to_crs(crs=crs)

    return new_ds
