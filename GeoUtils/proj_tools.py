"""
GeoUtils.proj_tools provides a toolset for dealing with different coordinate reference systems (CRS)
"""
import rasterio as rio
from rasterio.crs import CRS
from shapely.geometry.polygon import Polygon
from shapely.ops import transform
import pyproj


def bounds2poly(boundsGeom, in_crs=None, out_crs=None):
    """
    Converts self's bounds into a shapely Polygon. Optionally, returns it into a different CRS.

    :param boundsGeom: A geometry with bounds. Can be either a list of coordinates (xmin, ymin, xmax, ymax), a rasterio/Raster object, a geoPandas/Vector object
    :type boundsGeom: list, tuple, object with attributes bounds or total_bounds
    :param in_crs: Input CRS
    :type in_crs: rasterio.crs.CRS
    :param out_crs: Output CRS
    :type out_crs: rasterio.crs.CRS

    :returns: Output polygon
    :rtype: shapely Polygon
    """
    # If boundsGeom is a rasterio or Raster object
    if hasattr(boundsGeom, 'bounds'):
        xmin, ymin, xmax, ymax = boundsGeom.bounds
        in_crs = boundsGeom.crs
    # If boundsGeom is a GeoPandas or Vector object
    elif hasattr(boundsGeom, 'total_bounds'):
        xmin, ymin, xmax, ymax = boundsGeom.total_bounds
        in_crs = boundsGeom.crs
    # if a list of coordinates
    elif isinstance(boundsGeom, (list, tuple)):
        xmin, ymin, xmax, ymax = boundsGeom
    else:
        raise ValueError(
            "boundsGeom must a list/tuple of coordinates or an object with attributes bounds or total_bounds.")

    bbox = Polygon([(xmin, ymin), (xmax, ymin),
                    (xmax, ymax), (xmin, ymax)])

    if out_crs is not None:
        raise NotImplementedError()

    return bbox


def reproject_shape(inshape, in_crs, out_crs):
    """
    Reproject a shapely geometry from one CRS into another CRS.

    :param inshape: Shapely geometry to be reprojected.
    :type inshape: shapely geometry
    :param in_crs: Input CRS
    :type in_crs: rasterio.crs.CRS
    :param out_crs: Output CRS
    :type out_crs: rasterio.crs.CRS

    :returns: Reprojected geometry
    :rtype: shapely geometry
    """
    reproj = pyproj.Transformer.from_crs(
        in_crs, out_crs, always_xy=True, skip_equivalent=True).transform
    return transform(reproj, inshape)
