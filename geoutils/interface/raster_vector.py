"""Functionalities at the interface of rasters and vectors."""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Literal

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.features import shapes

import geoutils as gu
from geoutils._typing import NDArrayBool, NDArrayNum, Number


def _polygonize(
    source_raster: gu.Raster,
    target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"],
    data_column_name: str,
) -> gu.Vector:
    """Polygonize a raster. See Raster.polygonize() for details."""

    # Mask a unique value set by a number
    if isinstance(target_values, (int, float, np.integer, np.floating)):
        if np.sum(source_raster.data == target_values) == 0:
            raise ValueError(f"no pixel with in_value {target_values}")

        bool_msk = np.array(source_raster.data == target_values).astype(np.uint8)

    # Mask values within boundaries set by a tuple
    elif isinstance(target_values, tuple):
        if np.sum((source_raster.data > target_values[0]) & (source_raster.data < target_values[1])) == 0:
            raise ValueError(f"no pixel with in_value between {target_values[0]} and {target_values[1]}")

        bool_msk = ((source_raster.data > target_values[0]) & (source_raster.data < target_values[1])).astype(np.uint8)

    # Mask specific values set by a sequence
    elif isinstance(target_values, list) or isinstance(target_values, np.ndarray):
        if np.sum(np.isin(source_raster.data, np.array(target_values))) == 0:
            raise ValueError("no pixel with in_value " + ", ".join(map("{}".format, target_values)))

        bool_msk = np.isin(source_raster.data, np.array(target_values)).astype("uint8")

    # Mask all valid values
    elif target_values == "all":
        # Using getmaskarray is necessary in case .data.mask is nomask (False)
        bool_msk = (~np.ma.getmaskarray(source_raster.data)).astype("uint8")

    else:
        raise ValueError("in_value must be a number, a tuple or a sequence")

    # GeoPandas.from_features() only supports certain dtypes, we find the best common dtype to optimize memory usage
    # TODO: this should be a function independent of polygonize, reused in several places
    gpd_dtypes = ["uint8", "uint16", "int16", "int32", "float32"]
    list_common_dtype_index = []
    for gpd_type in gpd_dtypes:
        polygonize_dtype = np.promote_types(gpd_type, source_raster.dtype)
        if str(polygonize_dtype) in gpd_dtypes:
            list_common_dtype_index.append(gpd_dtypes.index(gpd_type))
    if len(list_common_dtype_index) == 0:
        final_dtype = "float32"
    else:
        final_dtype_index = min(list_common_dtype_index)
        final_dtype = gpd_dtypes[final_dtype_index]

    results = (
        {"properties": {"raster_value": v}, "geometry": s}
        for i, (s, v) in enumerate(
            shapes(source_raster.data.astype(final_dtype), mask=bool_msk, transform=source_raster.transform)
        )
    )

    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf.insert(0, data_column_name, range(0, 0 + len(gdf)))
    gdf = gdf.set_geometry(col="geometry")
    gdf = gdf.set_crs(source_raster.crs)

    return gu.Vector(gdf)


def _rasterize(
    gdf: gpd.GeoDataFrame,
    raster: gu.Raster | None = None,
    crs: CRS | int | None = None,
    xres: float | None = None,
    yres: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    in_value: int | float | Iterable[int | float] | None = None,
    out_value: int | float = 0,
) -> gu.Raster:
    if (raster is not None) and (crs is not None):
        raise ValueError("Only one of raster or crs can be provided.")

        # Reproject vector into requested CRS or rst CRS first, if needed
        # This has to be done first so that width/height calculated below are correct!
    if crs is None:
        crs = gdf.crs

    if raster is not None:
        crs = raster.crs  # type: ignore

    vect = gdf.to_crs(crs)

    # If no raster given, now use provided dimensions
    if raster is None:
        # At minimum, xres must be set
        if xres is None:
            raise ValueError("At least raster or xres must be set.")
        if yres is None:
            yres = xres

        # By default, use self's bounds
        if bounds is None:
            bounds = vect.total_bounds

        # Calculate raster shape
        left, bottom, right, top = bounds
        width = abs((right - left) / xres)
        height = abs((top - bottom) / yres)

        if width % 1 != 0 or height % 1 != 0:
            warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds.")

        width = int(np.round(width))
        height = int(np.round(height))
        out_shape = (height, width)

        # Calculate raster transform
        transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

    # otherwise use directly raster's dimensions
    else:
        out_shape = raster.shape  # type: ignore
        transform = raster.transform  # type: ignore

    # Set default burn value, index from 1 to len(self.ds)
    if in_value is None:
        in_value = gdf.index + 1

    # Rasterize geometry
    if isinstance(in_value, Iterable):
        if len(in_value) != len(vect.geometry):  # type: ignore
            raise ValueError(
                "in_value must have same length as self.ds.geometry, currently {} != {}".format(
                    len(in_value), len(vect.geometry)  # type: ignore
                )
            )

        out_geom = ((geom, value) for geom, value in zip(vect.geometry, in_value))

        mask = features.rasterize(shapes=out_geom, fill=out_value, out_shape=out_shape, transform=transform)

    elif isinstance(in_value, int | float | np.floating | np.integer):
        mask = features.rasterize(
            shapes=vect.geometry, fill=out_value, out_shape=out_shape, transform=transform, default_value=in_value
        )
    else:
        raise ValueError("in_value must be a single number or an iterable with same length as self.ds.geometry")

    # We return a mask if there is a single value to burn and this value is 1
    if isinstance(in_value, (int, np.integer, float, np.floating)) and in_value == 1:
        output = gu.Mask.from_array(data=mask, transform=transform, crs=crs, nodata=None)

    # Otherwise we return a Raster if there are several values to burn
    else:
        output = gu.Raster.from_array(data=mask, transform=transform, crs=crs, nodata=None)

    return output


def _create_mask(
    gdf: gpd.GeoDataFrame,
    raster: gu.Raster | None = None,
    crs: CRS | None = None,
    xres: float | None = None,
    yres: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    buffer: int | float | np.integer[Any] | np.floating[Any] = 0,
    as_array: bool = False,
) -> tuple[NDArrayBool, affine.Affine, CRS]:

    # If no raster given, use provided dimensions
    if raster is None:
        # At minimum, xres must be set
        if xres is None:
            raise ValueError("At least raster or xres must be set.")
        if yres is None:
            yres = xres

        # By default, use self's CRS and bounds
        if crs is None:
            crs = gdf.crs
        if bounds is None:
            bounds_shp = True
            bounds = gdf.total_bounds
        else:
            bounds_shp = False

        # Calculate raster shape
        left, bottom, right, top = bounds
        height = abs((right - left) / xres)
        width = abs((top - bottom) / yres)

        if width % 1 != 0 or height % 1 != 0:
            # Only warn if the bounds were provided, and not derived from the vector
            if not bounds_shp:
                warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds.")

        width = int(np.round(width))
        height = int(np.round(height))
        out_shape = (height, width)

        # Calculate raster transform
        transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

    # otherwise use directly raster's dimensions
    elif isinstance(raster, gu.Raster):
        out_shape = raster.shape
        transform = raster.transform
        crs = raster.crs
        bounds = raster.bounds
    else:
        raise TypeError("Raster must be a geoutils.Raster or None.")

    # Copying GeoPandas dataframe before applying changes
    gdf = gdf.copy()

    # Crop vector geometries to avoid issues when reprojecting
    left, bottom, right, top = bounds  # type: ignore
    x1, y1, x2, y2 = warp.transform_bounds(crs, gdf.crs, left, bottom, right, top)
    gdf = gdf.cx[x1:x2, y1:y2]

    # Reproject vector into raster CRS
    gdf = gdf.to_crs(crs)

    # Create a buffer around the features
    if not isinstance(buffer, (int, float, np.number)):
        raise TypeError(f"Buffer must be a number, currently set to {type(buffer).__name__}.")
    if buffer != 0:
        gdf.geometry = [geom.buffer(buffer) for geom in gdf.geometry]
    elif buffer == 0:
        pass

    # Rasterize geometry
    mask = features.rasterize(
        shapes=gdf.geometry, fill=0, out_shape=out_shape, transform=transform, default_value=1, dtype="uint8"
    ).astype("bool")

    # Force output mask to be of same dimension as input raster
    if raster is not None:
        mask = mask.reshape((raster.count, raster.height, raster.width))  # type: ignore

    return mask, transform, crs
