"""Functionalities for geotransformations of vectors."""

from __future__ import annotations

import geopandas as gpd
import rasterio as rio
from rasterio.crs import CRS

import geoutils as gu


def _reproject(
    gdf: gpd.GeoDataFrame,
    ref: gu.Raster | rio.io.DatasetReader | gu.Vector | gpd.GeoDataFrame | None = None,
    crs: CRS | str | int | None = None,
) -> gpd.GeoDataFrame:
    """Reproject a vector. See Vector.reproject() for more details."""

    # Check that either ref or crs is provided
    if (ref is not None and crs is not None) or (ref is None and crs is None):
        raise ValueError("Either of `ref` or `crs` must be set. Not both.")

    # Case a raster or vector is provided as reference
    if ref is not None:
        # Check that ref type is either str, Raster or rasterio data set
        if isinstance(ref, (gu.Raster, gu.Vector, rio.io.DatasetReader, gpd.GeoDataFrame)):
            ds_ref = ref
        else:
            raise TypeError("Type of ref must be a raster or vector.")

        # Read reprojecting params from ref raster
        crs = ds_ref.crs
    else:
        # Determine user-input target CRS
        crs = CRS.from_user_input(crs)

    new_ds = gdf.to_crs(crs=crs)

    return new_ds
