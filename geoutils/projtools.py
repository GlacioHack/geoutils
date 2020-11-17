"""
GeoUtils.projtools provides a toolset for dealing with different coordinate reference systems (CRS).
"""
import rasterio as rio
import pyproj
from rasterio.crs import CRS
from shapely.geometry.polygon import Polygon
import shapely.ops.transform
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
    return shapely.ops.transform(reproj, inshape)

  
def compare_proj(proj1, proj2):
    """
    Compare two projections to see if they are the same, using pyproj.CRS.is_exact_same.

    :param proj1: The first projection to compare.
    :type proj1: pyproj.CRS, rasterio.crs.CRS
    :param proj2: The first projection to compare.
    :type proj2: pyproj.CRS, rasterio.crs.CRS

    :returns: True if the two projections are the same.
    """
    assert all([isinstance(proj1, (pyproj.CRS,CRS)), isinstance(proj2, (pyproj.CRS, CRS))]), \
        'proj1 and proj2 must be rasterio.crs.CRS objects.'
    proj1 = pyproj.CRS(proj1.to_string())
    proj2 = pyproj.CRS(proj2.to_string())

    return proj1.is_exact_same(proj2)
