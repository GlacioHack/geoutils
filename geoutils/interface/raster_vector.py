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

import warnings
from typing import TYPE_CHECKING, Iterable, Literal

import affine
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features, warp
from rasterio.crs import CRS
from rasterio.features import shapes

from geoutils._dispatch import get_geo_attr, has_geo_attr, _check_match_grid
from geoutils._misc import silence_rasterio_message
from geoutils._typing import NDArrayBool, NDArrayNum, Number
from geoutils.raster.georeferencing import _bounds

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloud, PointCloudLike
    from geoutils.raster.base import Raster, RasterLike, RasterType
    from geoutils.vector.vector import Vector


def _polygonize(
    source_raster: RasterType,
    target_values: Number | tuple[Number, Number] | list[Number] | NDArrayNum | Literal["all"],
    data_column_name: str,
) -> Vector:
    """Polygonize a raster. See Raster.polygonize() for details."""

    # If target values is passed but does not correspond to 0 or 1, raise a warning
    if source_raster.is_mask:
        if target_values != "all" and (
            not isinstance(target_values, (int, np.integer, float, np.floating)) or target_values not in [0, 1]
        ):
            warnings.warn("Raster mask (boolean type) passed, using target value of 1 (True).")
        target_values = True

    nanarray = source_raster.get_nanarray()

    # Mask a unique value set by a number
    if isinstance(target_values, (int, float, np.integer, np.floating)):
        if np.sum(nanarray == target_values) == 0:
            raise ValueError(f"no pixel with in_value {target_values}")

        bool_msk = np.array(nanarray == target_values).astype(np.uint8)

    # Mask values within boundaries set by a tuple
    elif isinstance(target_values, tuple):
        if np.sum((nanarray > target_values[0]) & (nanarray < target_values[1])) == 0:
            raise ValueError(f"no pixel with in_value between {target_values[0]} and {target_values[1]}")

        bool_msk = ((nanarray > target_values[0]) & (nanarray < target_values[1])).astype(np.uint8)

    # Mask specific values set by a sequence
    elif isinstance(target_values, list) or isinstance(target_values, np.ndarray):
        if np.sum(np.isin(nanarray, np.array(target_values))) == 0:
            raise ValueError("no pixel with in_value " + ", ".join(map("{}".format, target_values)))

        bool_msk = np.isin(nanarray, np.array(target_values)).astype("uint8")

    # Mask all valid values
    elif target_values == "all":
        # Using getmaskarray is necessary in case .data.mask is nomask (False)
        bool_msk = (~np.ma.getmaskarray(nanarray)).astype("uint8")

    else:
        raise ValueError("in_value must be a number, a tuple or a sequence")

    # GeoPandas.from_features() only supports certain dtypes, we find the best common dtype to optimize memory usage
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
            shapes(nanarray.astype(final_dtype), mask=bool_msk, transform=source_raster.transform)
        )
    )

    # Warning that comes from GeoPandas
    with silence_rasterio_message(param_name="MEM", warn_code="CPLE_AppDefined"):
        gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf.insert(0, data_column_name, range(0, 0 + len(gdf)))
    gdf = gdf.set_geometry(col="geometry")
    gdf = gdf.set_crs(source_raster.crs)
    from geoutils.vector import Vector  # Runtime import to avoid circularity issues

    return Vector(gdf)


def _rasterize(
    source_vector: Vector,
    ref: RasterType | None = None,
    in_value: int | float | Iterable[int | float] | None = None,
    out_value: int | float = 0,
    res: tuple[Number, Number] | Number | None = None,
    shape: tuple[int, int] | None = None,
    grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    crs: CRS | int | None = None,
) -> Raster:

    out_shape, out_transform, out_crs = _check_match_grid(src=source_vector, ref=ref, res=res, shape=shape,
                                                          bounds=bounds, crs=crs, coords=grid_coords)
    if out_crs is not None:
        source_vector = source_vector.to_crs(out_crs)
    vect = source_vector.ds

    # Set default burn value, index from 1 to len(self.ds)
    if in_value is None:
        in_value = vect.index + 1

    # Rasterize geometry
    if isinstance(in_value, Iterable):
        if len(in_value) != len(vect.geometry):  # type: ignore
            raise ValueError(
                "in_value must have same length as self.ds.geometry, currently {} != {}".format(
                    len(in_value), len(vect.geometry)  # type: ignore
                )
            )

        out_geom = ((geom, value) for geom, value in zip(vect.geometry, in_value))

        mask = features.rasterize(shapes=out_geom, fill=out_value, out_shape=out_shape, transform=out_transform)

    elif isinstance(in_value, int | float | np.floating | np.integer):
        mask = features.rasterize(
            shapes=vect.geometry, fill=out_value, out_shape=out_shape, transform=out_transform, default_value=in_value
        )
    else:
        raise ValueError("in_value must be a single number or an iterable with same length as self.ds.geometry")

    from geoutils.raster import Raster  # Runtime import to avoid circularity issues
    output = Raster.from_array(data=mask, transform=out_transform, crs=out_crs, nodata=None)

    return output


def _create_mask_pointcloud(gdf: gpd.GeoDataFrame, pts: gpd.GeoSeries) -> NDArrayBool:
    """Subfunction to create a point cloud mask using geopandas."""

    # Project to same CRS
    pts_reproj = pts.to_crs(crs=gdf.crs)

    # Check that points are contained no matter alignment
    contained = pts_reproj.within(gdf, align=False)

    # Extract resulting boolean array
    mask = contained.values

    return mask


def _create_mask_raster(
    gdf: gpd.GeoDataFrame, out_shape: tuple[int, int], transform: affine.Affine, crs: CRS
) -> NDArrayBool:
    """Subfunction to create a raster mask using rasterio.features.rasterize()."""

    # Copying GeoPandas dataframe before applying changes
    gdf = gdf.copy()

    # Crop vector geometries to avoid issues when reprojecting
    bounds = _bounds(transform=transform, shape=out_shape)
    left, bottom, right, top = bounds  # type: ignore
    x1, y1, x2, y2 = warp.transform_bounds(crs, gdf.crs, left, bottom, right, top)
    gdf = gdf.cx[x1:x2, y1:y2]

    # Reproject vector to raster CRS (almost always faster)
    gdf = gdf.to_crs(crs)

    # Rasterize geometry
    mask = features.rasterize(
        shapes=gdf.geometry, fill=0, out_shape=out_shape, transform=transform, default_value=1, dtype="uint8"
    ).astype("bool")

    return mask


def _create_mask(
    gdf: gpd.GeoDataFrame,
    ref: RasterLike | PointCloudLike | None = None,
    crs: CRS | None = None,
    res: float | tuple[float, float] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    points: tuple[NDArrayNum, NDArrayNum] | None = None,
    as_array: bool = False,
) -> Raster | PointCloud | NDArrayBool:
    """See Vector.create_mask for description."""

    # Raise errors for wrong inputs
    is_ref_raster = ref is not None and (
        has_geo_attr(ref, "transform") and has_geo_attr(ref, "shape") and has_geo_attr(ref, "crs")
    )
    is_ref_pointcloud = ref is not None and (has_geo_attr(ref, "ds") and has_geo_attr(ref, "data_column"))
    if ref is not None:
        if not (is_ref_raster or is_ref_pointcloud):
            raise ValueError(
                "Reference must be a raster implementing 'transform', 'crs', and 'shape' "
                "or a point cloud implementing 'ds' and 'data_column'."
            )
    else:
        if res is None and points is None:
            raise ValueError(
                "Without a reference for masking, specify at least the resolution a raster mask, "
                "or the points coordinates for a point cloud mask."
            )

    # If raster reference or user-input exists, we compute a raster mask
    if is_ref_raster or res is not None:
        # For a reference, extract transform and CRS
        if ref is not None:
            transform = get_geo_attr(ref, "transform")
            out_shape = get_geo_attr(ref, "shape")
            crs = get_geo_attr(ref, "crs")

        # For a user-input res
        else:

            # By default, use self's CRS
            if crs is None:
                crs = gdf.crs

            # Case of a raster mask
            if res is not None:
                # Get resolution
                if isinstance(res, tuple):
                    xres, yres = res
                else:
                    xres = res
                    yres = res

            # Get bounds
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
                    warnings.warn("Bounds not a multiple of resolution, using rounded bounds.")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # Compute raster mask
        mask = _create_mask_raster(gdf=gdf, transform=transform, out_shape=out_shape, crs=crs)
        pts = None

    # For a point cloud reference or user-input, compute a point cloud mask
    else:

        # For a reference, extract geometry
        if ref is not None:
            ds = get_geo_attr(ref, "ds")
            pts = ds.geometry

        else:

            if crs is None:
                crs = gdf.crs

            pts = gpd.points_from_xy(x=points[0], y=points[1], crs=crs)  # type: ignore

        mask = _create_mask_pointcloud(gdf=gdf, pts=pts)
        transform = None

    from geoutils.pointcloud import (
        PointCloud,  # Runtime import to avoid circularity issues
    )
    from geoutils.raster import Raster  # Runtime import to avoid circularity issues

    # Return output as mask or as array
    if as_array:
        return mask.squeeze()
    else:
        # If pts is None, the output is a point cloud mask
        if pts is not None:
            return PointCloud.from_xyz(x=pts.x.values, y=pts.y.values, z=mask, crs=crs)
        # Otherwise, the transform is not None
        else:
            assert transform is not None  # For mypy
            return Raster.from_array(data=mask, transform=transform, crs=crs, nodata=None)
