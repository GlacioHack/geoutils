"""
geoutils.accessor provides an Xarray accessor "rst" mirroring the API of the Raster class.
"""

import numpy as np
import rasterio as rio
import rioxarray as rioxr
import xarray as xr
from geocube.vector import vectorize
from pyproj import CRS

from geoutils._typing import NDArrayNum
from geoutils.projtools import _get_bounds_projected, _get_footprint_projected
from geoutils.raster import Raster, RasterType
from geoutils.raster.sampling import subsample_array
from geoutils.vector import Vector


def open_raster(filename: str, **kwargs):

    # Open with Rioxarray
    ds = rioxr.open_rasterio(filename, **kwargs)

    # Remove the band dimension if there is only one
    if ds.rio.count == 1:
        del ds.coords["band"]  # Delete band coordinate (only one dimension)

    # Store disk attributes? Didn't follow the logic in Rioxarray on this, need to get more into code details
    ds.rst._count_on_disk = 1

    return ds


@xr.register_dataarray_accessor("rst")
class RasterAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    # First, properties that need to be parsed differently than in Raster
    @property
    def crs(self):
        return self._obj.rio.crs

    @property
    def transform(self):
        return self._obj.rio.transform(recalc=True)

    @property
    def nodata(self):
        return self._obj.rio.nodata

    @property
    def height(self):
        return self._obj.rio.height

    @property
    def width(self):
        return self._obj.rio.width

    @property
    def shape(self):
        return self._obj.rio.shape

    @property
    def res(self):
        return self._obj.rio.resolution(recalc=True)

    @property
    def bounds(self):
        return self._obj.rio.bounds(recalc=True)

    @property
    def count(self):
        return self._obj.rio.count

    def reproject(self):

        # Copy logic of Raster.reproject()

        # Perform final reprojection operation with RioXarray
        # reproj_obj = self._obj.rio.reproject(dst_crs, resolution, shape, transform, resampling, nodata, **kwargs)
        reproj_obj = None

        return reproj_obj

    def crop(self):

        # Copy logic of Raster.crop()

        # Perform final cropping operation with RioXarray
        # clipped_obj = self._obj.rio.clip_box(minx, miny, maxx, maxy, crs, *kwargs)
        clipped_obj = None

        return clipped_obj

    def proximity(self):

        # Copy logic of Raster.proximity()

        proximity = None

        return proximity

    def polygonize(self):

        # Copy logic of Raster.polygonize()

        gdf_polygonize = vectorize(self._obj)

        return Vector(gdf_polygonize)

    def georeferenced_grid_equal(self: xr.DataArray, raster: RasterType | xr.DataArray) -> bool:

        return all([self.shape == raster.shape, self.transform == raster.transform, self.crs == raster.crs])

    def get_footprint_projected(self, out_crs: CRS, densify_pts: int = 5000) -> Vector:
        """
        Get raster footprint projected in a specified CRS.

        The polygon points of the vector are densified during reprojection to warp
        the rectangular square footprint of the original projection into the new one.

        :param out_crs: Output CRS.
        :param densify_pts: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        return Vector(
            _get_footprint_projected(bounds=self.bounds, in_crs=self.crs, out_crs=out_crs, densify_pts=densify_pts)
        )

    def get_bounds_projected(self, out_crs: CRS, densify_pts: int = 5000) -> rio.coords.BoundingBox:
        """
        Get raster bounds projected in a specified CRS.

        :param out_crs: Output CRS.
        :param densify_pts: Maximum points to be added between image corners to account for non linear edges.
         Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.

        """
        # Max points to be added between image corners to account for non linear edges
        # rasterio's default is a bit low for very large images
        # instead, use image dimensions, with a maximum of 50000
        densify_pts = min(max(self.width, self.height), densify_pts)

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_pts=densify_pts)

        return new_bounds

    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: np.random.RandomState | int | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly subsample the raster. Only valid values are considered.

        :param subsample: If <= 1, a fraction of the total pixels to extract. If > 1, the number of pixels.
        :param return_indices: Whether to return the extracted indices only.
        :param random_state: Random state or seed number.

        :return: Array of subsampled valid values, or array of subsampled indices.
        """

        return subsample_array(
            array=self.data, subsample=subsample, return_indices=return_indices, random_state=random_state
        )

    def to_raster(self):
        """
        Convert to geoutils.Raster object.

        :return:
        """

        return Raster.from_array(data=self._obj.data, crs=self.crs, transform=self.transform, nodata=self.nodata)

    def xy2ij(self):

        # Copy exact logic of raster
        return

    def ij2xy(self):

        # Copy exact logic of raster
        return

    def outside_image(self):

        # Copy exact logic of raster
        return

    def show(self):

        # Copy exact logic of raster
        return


@xr.register_dataarray_accessor("sat")
class SatelliteImageAccessor(RasterAccessor):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
