"""
geoutils.vectortools provides a toolset for working with vector data.
"""
from __future__ import annotations

import warnings
from collections import abc
from numbers import Number
from typing import Literal, TypeVar, overload

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import shapely
from rasterio import features, warp
from rasterio.crs import CRS
from scipy.spatial import Voronoi
from shapely.geometry.polygon import Polygon

import geoutils as gu
from geoutils.projtools import _get_bounds_projected, bounds2poly

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
VectorType = TypeVar("VectorType", bound="Vector")


class Vector:
    """
    Create a Vector object from a fiona-supported vector dataset.
    """

    def __init__(self, filename: str | gpd.GeoDataFrame):
        """
        Load a fiona-supported dataset, given a filename.

        :param filename: The filename or GeoDataFrame of the dataset.

        :return: A Vector object
        """

        if isinstance(filename, str):
            with warnings.catch_warnings():
                # This warning shows up in numpy 1.21 (2021-07-09)
                warnings.filterwarnings("ignore", ".*attribute.*array_interface.*Polygon.*")
                ds = gpd.read_file(filename)
            self.ds = ds
            self.name: str | gpd.GeoDataFrame | None = filename
        elif isinstance(filename, gpd.GeoDataFrame):
            self.ds = filename
            self.name = None
        else:
            raise ValueError("filename argument not recognised.")

        self.crs = self.ds.crs

    def __repr__(self) -> str:
        return str(self.ds.__repr__())

    def __str__(self) -> str:
        """Provide string of information about Raster."""
        return self.info()

    def __getitem__(self, value: gu.Raster | Vector | list[float] | tuple[float, ...]) -> Vector:
        """Subset the Raster object: calls the crop method with default parameters"""
        return self.crop(cropGeom=value, inplace=False)

    def info(self) -> str:
        """
        Returns string of information about the vector (filename, coordinate system, number of layers, features, etc.).

        :returns: text information about Vector attributes.
        :rtype: str
        """
        as_str = [  # 'Driver:             {} \n'.format(self.driver),
            f"Filename:           {self.name} \n",
            f"Coordinate System:  EPSG:{self.ds.crs.to_epsg()}\n",
            f"Number of features: {len(self.ds)} \n",
            f"Extent:             {self.ds.total_bounds.tolist()} \n",
            f"Attributes:         {self.ds.columns.tolist()} \n",
            self.ds.__repr__(),
        ]

        return "".join(as_str)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        """Get a bounding box of the total bounds of the Vector."""
        return rio.coords.BoundingBox(*self.ds.total_bounds)

    def copy(self: VectorType) -> VectorType:
        """Return a copy of the Vector."""
        # Utilise the copy method of GeoPandas
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.copy())  # type: ignore
        return new_vector  # type: ignore

    @overload
    def crop(
        self: VectorType,
        cropGeom: gu.Raster | Vector | list[float] | tuple[float, ...],
        inplace: Literal[True] = True,
    ) -> None:
        ...

    @overload
    def crop(
        self: VectorType,
        cropGeom: gu.Raster | Vector | list[float] | tuple[float, ...],
        inplace: Literal[False],
    ) -> VectorType:
        ...

    def crop(
        self: VectorType,
        cropGeom: gu.Raster | Vector | list[float] | tuple[float, ...],
        inplace: bool = True,
    ) -> VectorType | None:
        """
        Crop the Vector to given extent, or bounds of a raster or vector.

        Reprojection is done on the fly if georeferenced objects have different projections.

        :param cropGeom: Geometry to crop vector to, as either a Raster object, a Vector object, or a list of
            coordinates. If cropGeom is a Raster, crop() will crop to the boundary of the raster as returned by
            Raster.ds.bounds. If cropGeom is a Vector, crop() will crop to the bounding geometry. If cropGeom is a
            list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
        :param inplace: Update the vector inplace or return copy.
        """
        if isinstance(cropGeom, (gu.Raster, Vector)):
            # For another Vector or Raster, we reproject the bounding box in the same CRS as self
            xmin, ymin, xmax, ymax = cropGeom.get_bounds_projected(out_crs=self.crs)
        elif isinstance(cropGeom, (list, tuple)):
            xmin, ymin, xmax, ymax = cropGeom
        else:
            raise ValueError("cropGeom must be a Raster, Vector, or list of coordinates.")

        # Need to separate the two options, inplace update
        if inplace:
            self.ds = self.ds.cx[xmin:xmax, ymin:ymax]
            return None
        # Or create a copy otherwise
        else:
            new_vector = self.copy()
            new_vector.ds = new_vector.ds.cx[xmin:xmax, ymin:ymax]
            return new_vector

    def create_mask(
        self,
        rst: str | gu.Raster | None = None,
        crs: CRS | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        buffer: int | float | np.number = 0,
    ) -> np.ndarray:
        """
        Rasterize the vector features into a boolean raster which has the extent/dimensions of \
        the provided raster file.

        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Vector features which fall outside the bounds of the raster file are not written to the new mask file.

        :param rst: A Raster object or string to filename
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :param xres: Output raster spatial resolution in x. Only is rst is None.
        :param yres: Output raster spatial resolution in y. Only if rst is None. (Default to xres)
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None (Default to self bounds)
        :param buffer: Size of buffer to be added around the features, in the raster's projection units.
        If a negative value is set, will erode the features.

        :returns: array containing the mask
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = gu.Raster(rst)  # type: ignore

        # If no rst given, use provided dimensions
        if rst is None:

            # At minimum, xres must be set
            if xres is None:
                raise ValueError("at least rst or xres must be set")
            if yres is None:
                yres = xres

            # By default, use self's CRS and bounds
            if crs is None:
                crs = self.ds.crs
            if bounds is None:
                bounds_shp = True
                bounds = self.ds.total_bounds
            else:
                bounds_shp = False

            # Calculate raster shape
            left, bottom, right, top = bounds
            height = abs((right - left) / xres)
            width = abs((top - bottom) / yres)

            if width % 1 != 0 or height % 1 != 0:
                # Only warn if the bounds were provided, and not derived from the vector
                if not bounds_shp:
                    warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # otherwise use directly rst's dimensions
        elif isinstance(rst, gu.Raster):
            out_shape = rst.shape
            transform = rst.transform
            crs = rst.crs
            bounds = rst.bounds
        else:
            raise ValueError("`rst` must be either a str, geoutils.Raster or None")

        # Copying GeoPandas dataframe before applying changes
        gdf = self.ds.copy()

        # Crop vector geometries to avoid issues when reprojecting
        left, bottom, right, top = bounds  # type: ignore
        x1, y1, x2, y2 = warp.transform_bounds(crs, gdf.crs, left, bottom, right, top)
        gdf = gdf.cx[x1:x2, y1:y2]

        # Reproject vector into rst CRS
        gdf = gdf.to_crs(crs)

        # Create a buffer around the features
        if not isinstance(buffer, (int, float, np.number)):
            raise ValueError(f"`buffer` must be a number, currently set to {type(buffer)}")
        if buffer != 0:
            gdf.geometry = [geom.buffer(buffer) for geom in gdf.geometry]
        elif buffer == 0:
            pass

        # Rasterize geometry
        mask = features.rasterize(
            shapes=gdf.geometry, fill=0, out_shape=out_shape, transform=transform, default_value=1, dtype="uint8"
        ).astype("bool")

        # Force output mask to be of same dimension as input rst
        if rst is not None:
            mask = mask.reshape((rst.count, rst.height, rst.width))  # type: ignore

        return mask

    def rasterize(
        self,
        rst: str | gu.Raster | None = None,
        crs: CRS | None = None,
        xres: float | None = None,
        yres: float | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        in_value: int | float | abc.Iterable[int | float] | None = None,
        out_value: int | float = 0,
    ) -> np.ndarray:
        """
        Return an array with input geometries burned in.

        By default, output raster has the extent/dimensions of the provided raster file.
        Alternatively, user can specify a grid to rasterize on using xres, yres, bounds and crs.
        Only xres is mandatory, by default yres=xres and bounds/crs are set to self's.

        Burn value is set by user and can be either a single number, or an iterable of same length as self.ds.
        Default is an index from 1 to len(self.ds).

        :param rst: A raster to be used as reference for the output grid
        :param crs: A pyproj or rasterio CRS object (Default to rst.crs if not None then self.crs)
        :param xres: Output raster spatial resolution in x. Only is rst is None.
            Must be in units of crs, if set.
        :param yres: Output raster spatial resolution in y. Only if rst is None.
        Must be in units of crs, if set. (Default to xres)
        :param bounds: Output raster bounds (left, bottom, right, top). Only if rst is None
            Must be in same system as crs, if set. (Default to self bounds).
        :param in_value: Value(s) to be burned inside the polygons (Default is self.ds.index + 1)
        :param out_value: Value to be burned outside the polygons (Default is 0)

        :returns: array containing the burned geometries
        """
        # If input rst is string, open as Raster
        if isinstance(rst, str):
            rst = gu.Raster(rst)  # type: ignore

        if (rst is not None) and (crs is not None):
            raise ValueError("Only one of rst or crs can be provided.")

        # Reproject vector into requested CRS or rst CRS first, if needed
        # This has to be done first so that width/height calculated below are correct!
        if crs is None:
            crs = self.ds.crs

        if rst is not None:
            crs = rst.crs  # type: ignore

        vect = self.ds.to_crs(crs)

        # If no rst given, now use provided dimensions
        if rst is None:

            # At minimum, xres must be set
            if xres is None:
                raise ValueError("at least rst or xres must be set")
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
                warnings.warn("Bounds not a multiple of xres/yres, use rounded bounds")

            width = int(np.round(width))
            height = int(np.round(height))
            out_shape = (height, width)

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # otherwise use directly rst's dimensions
        else:
            out_shape = rst.shape  # type: ignore
            transform = rst.transform  # type: ignore

        # Set default burn value, index from 1 to len(self.ds)
        if in_value is None:
            in_value = self.ds.index + 1

        # Rasterize geometry
        if isinstance(in_value, abc.Iterable):
            if len(in_value) != len(vect.geometry):  # type: ignore
                raise ValueError(
                    "in_value must have same length as self.ds.geometry, currently {} != {}".format(
                        len(in_value), len(vect.geometry)  # type: ignore
                    )
                )

            out_geom = ((geom, value) for geom, value in zip(vect.geometry, in_value))

            mask = features.rasterize(shapes=out_geom, fill=out_value, out_shape=out_shape, transform=transform)

        elif isinstance(in_value, Number):
            mask = features.rasterize(
                shapes=vect.geometry, fill=out_value, out_shape=out_shape, transform=transform, default_value=in_value
            )
        else:
            raise ValueError("in_value must be a single number or an iterable with same length as self.ds.geometry")

        return mask

    def query(self: VectorType, expression: str, inplace: bool = False) -> VectorType:
        """
        Query the Vector dataset with a valid Pandas expression.

        :param expression: A python-like expression to evaluate. Example: "col1 > col2".
        :param inplace: Whether the query should modify the data in place or return a modified copy.

        :returns: Vector resulting from the provided query expression or itself if inplace=True.
        """
        # Modify inplace if wanted and return the self instance.
        if inplace:
            self.ds.query(expression, inplace=True)
            return self

        # Otherwise, create a new Vector from the queried dataset.
        new_vector = self.__new__(type(self))
        new_vector.__init__(self.ds.query(expression))  # type: ignore
        return new_vector  # type: ignore

    def proximity(
        self,
        raster: gu.Raster | None = None,
        grid_size: tuple[int, int] = (1000, 1000),
        geometry_type: str = "boundary",
        in_or_out: Literal["in"] | Literal["out"] | Literal["both"] = "both",
        distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    ) -> gu.Raster:
        """
        Proximity to this Vector's geometry computed for each cell of a Raster grid, or for a grid size with
        this Vector's extent.

        If passed, the Raster georeferenced grid will be used to burn the proximity. If not, a grid size can be
        passed to create a georeferenced grid with the bounds and CRS of this Vector.

        By default, the boundary of the Vector's geometry will be used. The full geometry can be used by
        passing "geometry",
        or any lower dimensional geometry attribute such as "centroid", "envelope" or "convex_hull".
        See all geometry attributes in the Shapely documentation at https://shapely.readthedocs.io/.

        :param raster: Raster to burn the proximity grid on.
        :param grid_size: If no Raster is provided, grid size to use with this Vector's extent and CRS
            (defaults to 1000 x 1000).
        :param geometry_type: Type of geometry to use for the proximity, defaults to 'boundary'.
        :param in_or_out: Compute proximity only 'in' or 'out'-side the polygon, or 'both'.
        :param distance_unit: Distance unit, either 'georeferenced' or 'pixel'.

        :return: Proximity raster.
        """

        # 0/ If no Raster is passed, create one on the Vector bounds of size 1000 x 1000
        if raster is None:

            # TODO: this bit of code is common in several vector functions (rasterize, etc): move out as common code?
            # By default, use self's bounds
            if self.bounds is None:
                raise ValueError("To automatically rasterize on the vector, bounds need to be defined.")

            # Calculate raster shape
            left, bottom, right, top = self.bounds

            # Calculate raster transform
            transform = rio.transform.from_bounds(left, bottom, right, top, grid_size[0], grid_size[1])

            raster = gu.Raster.from_array(data=np.zeros((1000, 1000)), transform=transform, crs=self.crs)

        proximity = gu.georaster.raster.proximity_from_vector_or_raster(
            raster=raster, vector=self, geometry_type=geometry_type, in_or_out=in_or_out, distance_unit=distance_unit
        )

        return raster.copy(new_array=proximity)

    def buffer_metric(self, buffer_size: float) -> Vector:
        """
        Buffer the vector in a metric system (UTM only).

        The outlines are projected to a local UTM, then reverted to the original projection after buffering.

        :param buffer_size: Buffering distance in meters

        :return: Buffered shapefile
        """

        from geoutils.projtools import latlon_to_utm, utm_to_epsg

        # Get a rough centroid in geographic coordinates (ignore the warning that it is not the most precise):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            shp_wgs84 = self.ds.to_crs(epsg=4326)
            lat, lon = shp_wgs84.centroid.y.values[0], shp_wgs84.centroid.x.values[0]
            del shp_wgs84

        # Get the EPSG code of the local UTM
        utm = latlon_to_utm(lat, lon)
        epsg = utm_to_epsg(utm)

        # Reproject the shapefile in the local UTM
        ds_utm = self.ds.to_crs(epsg=epsg)

        # Buffer the shapefile
        ds_buffered = ds_utm.buffer(distance=buffer_size)
        del ds_utm

        # Revert-project the shapefile in the original CRS
        ds_buffered_origproj = ds_buffered.to_crs(crs=self.ds.crs)
        del ds_buffered

        # Return a Vector object of the buffered GeoDataFrame
        # TODO: Clarify what is conserved in the GeoSeries and what to pass the GeoDataFrame to not lose any attributes
        vector_buffered = Vector(gpd.GeoDataFrame(geometry=ds_buffered_origproj.geometry, crs=self.ds.crs))

        return vector_buffered

    def get_bounds_projected(self, out_crs: CRS, densify_pts: int = 5000) -> rio.coords.BoundingBox:
        """
        Return self's bounds in the given CRS.

        :param out_crs: Output CRS
        :param densify_pts: Maximum points to be added between image corners to account for nonlinear edges.
        Reduce if time computation is really critical (ms) or increase if extent is not accurate enough.
        """

        # Calculate new bounds
        new_bounds = _get_bounds_projected(self.bounds, in_crs=self.crs, out_crs=out_crs, densify_pts=densify_pts)

        return new_bounds

    def buffer_without_overlap(self, buffer_size: int | float, metric: bool = True, plot: bool = False) -> Vector:
        """
        Returns a Vector object containing self's geometries extended by a buffer, without overlapping each other.

        The algorithm is based upon this tutorial: https://statnmap.com/2020-07-31-buffer-area-for-nearest-neighbour/.
        The buffered polygons are created using Voronoi polygons in order to delineate the "area of influence"
        of each geometry.
        The buffer is slightly inaccurate where two geometries touch, due to the nature of the Voronoi polygons,
        hence one geometry "steps" slightly on the neighbor buffer in some cases.
        The algorithm may also yield unexpected results on very simple geometries.

        Note: A similar functionality is provided by momepy (http://docs.momepy.org) and is probably more robust.
        It could be implemented in GeoPandas in the future: https://github.com/geopandas/geopandas/issues/2015

        :examples:
        >>> outlines = gu.Vector(gu.examples.get_path('everest_rgi_outlines'))
        >>> outlines = gu.Vector(outlines.ds.to_crs('EPSG:32645'))
        >>> buffer = outlines.buffer_without_overlap(500)
        >>> ax = buffer.ds.plot()  # doctest: +SKIP
        >>> outlines.ds.plot(ax=ax, ec='k', fc='none')  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        :param buffer_size: Buffer size in self's coordinate system units.
        :param metric: Whether to perform the buffering in a local metric system (default: True).
        :param plot: Set to True to show intermediate plots, useful for understanding or debugging.

        :returns: A Vector containing the buffered geometries.
        """

        from geoutils.projtools import latlon_to_utm, utm_to_epsg

        # Project in local UTM if metric is True
        if metric:
            # Get a rough centroid in geographic coordinates (ignore the warning that it is not the most precise):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=UserWarning)
                shp_wgs84 = self.ds.to_crs(epsg=4326)
                lat, lon = shp_wgs84.centroid.y.values[0], shp_wgs84.centroid.x.values[0]
                del shp_wgs84

            # Get the EPSG code of the local UTM
            utm = latlon_to_utm(lat, lon)
            epsg = utm_to_epsg(utm)

            gdf = self.ds.to_crs(epsg=epsg)
        else:
            gdf = self.ds

        # Dissolve all geometries into one
        merged = gdf.dissolve()

        # Add buffer around geometries
        merged_buffer = merged.buffer(buffer_size)

        # Extract only the buffered area
        buffer = merged_buffer.difference(merged)

        # Crop Voronoi polygons to bound geometry and add missing polygons
        bound_poly = bounds2poly(gdf)
        bound_poly = bound_poly.buffer(buffer_size)
        voronoi_all = generate_voronoi_with_bounds(gdf, bound_poly)
        if plot:
            plt.figure(figsize=(16, 4))
            ax1 = plt.subplot(141)
            voronoi_all.plot(ax=ax1)
            gdf.plot(fc="none", ec="k", ax=ax1)
            ax1.set_title("Voronoi polygons, cropped")

        # Extract Voronoi polygons only within the buffer area
        voronoi_diff = voronoi_all.intersection(buffer.geometry[0])

        # Split all polygons, and join attributes of original geometries into the Voronoi polygons
        # Splitting, i.e. explode, is needed when Voronoi generate MultiPolygons that may extend over several features.
        voronoi_gdf = gpd.GeoDataFrame(geometry=voronoi_diff.explode(index_parts=True))  # requires geopandas>=0.10
        joined_voronoi = gpd.tools.sjoin(gdf, voronoi_gdf, how="right")

        # Plot results -> some polygons are duplicated
        if plot:
            ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
            joined_voronoi.plot(ax=ax2, column="index_left", alpha=0.5, ec="k")
            gdf.plot(ax=ax2, column=gdf.index.values)
            ax2.set_title("Buffer with duplicated polygons")

        # Find non unique Voronoi polygons, and retain only first one
        _, indexes = np.unique(joined_voronoi.index, return_index=True)
        unique_voronoi = joined_voronoi.iloc[indexes]

        # Plot results -> unique polygons only
        if plot:
            ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
            unique_voronoi.plot(ax=ax3, column="index_left", alpha=0.5, ec="k")
            gdf.plot(ax=ax3, column=gdf.index.values)
            ax3.set_title("Buffer with unique polygons")

        # Dissolve all polygons by original index
        merged_voronoi = unique_voronoi.dissolve(by="index_left")

        # Plot
        if plot:
            ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
            gdf.plot(ax=ax4, column=gdf.index.values)
            merged_voronoi.plot(column=merged_voronoi.index.values, ax=ax4, alpha=0.5)
            ax4.set_title("Final buffer")
            plt.show()

        # Reverse-project to the original CRS if metric is True
        if metric:
            merged_voronoi = merged_voronoi.to_crs(crs=self.crs)

        return Vector(merged_voronoi)


# -----------------------------------------
# Additional stand-alone utility functions
# -----------------------------------------


def extract_vertices(gdf: gpd.GeoDataFrame) -> list[list[tuple[float, float]]]:
    r"""
    Function to extract the exterior vertices of all shapes within a gpd.GeoDataFrame.

    :param gdf: The GeoDataFrame from which the vertices need to be extracted.

    :returns: A list containing a list of (x, y) positions of the vertices. The length of the primary list is equal \
    to the number of geometries inside gdf, and length of each sublist is the number of vertices in the geometry.
    """
    vertices = []
    # Loop on all geometries within gdf
    for geom in gdf.geometry:
        # Extract geometry exterior(s)
        if geom.geom_type == "MultiPolygon":
            exteriors = [p.exterior for p in geom.geoms]
        elif geom.geom_type == "Polygon":
            exteriors = [geom.exterior]
        elif geom.geom_type == "LineString":
            exteriors = [geom]
        elif geom.geom_type == "MultiLineString":
            exteriors = list(geom.geoms)
        else:
            raise NotImplementedError(f"Geometry type {geom.geom_type} not implemented.")

        vertices.extend([list(ext.coords) for ext in exteriors])

    return vertices


def generate_voronoi_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate Voronoi polygons (tessellation) from the vertices of all geometries in a GeoDataFrame.

    Uses scipy.spatial.voronoi.

    :param: The GeoDataFrame from whose vertices are used for the Voronoi polygons.

    :returns: A GeoDataFrame containing the Voronoi polygons.
    """
    # Extract the coordinates of the vertices of all geometries in gdf
    vertices = extract_vertices(gdf)
    coords = np.concatenate(vertices)

    # Create the Voronoi diagram and extract ridges
    vor = Voronoi(coords)
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    polys = list(shapely.ops.polygonize(lines))
    if len(polys) == 0:
        raise ValueError("Invalid geometry, cannot generate finite Voronoi polygons")

    # Convert into GeoDataFrame
    voronoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    voronoi.crs = gdf.crs

    return voronoi


def generate_voronoi_with_bounds(gdf: gpd.GeoDataFrame, bound_poly: Polygon) -> gpd.GeoDataFrame:
    """
    Generate Voronoi polygons that are bounded by the polygon bound_poly, to avoid Voronoi polygons that extend \
far beyond the original geometry.

    Voronoi polygons are created using generate_voronoi_polygons, cropped to the extent of bound_poly and gaps \
are filled with new polygons.

    :param: The GeoDataFrame from whose vertices are used for the Voronoi polygons.
    :param: A shapely Polygon to be used for bounding the Voronoi diagrams.

    :returns: A GeoDataFrame containing the Voronoi polygons.
    """
    # Create Voronoi polygons
    voronoi = generate_voronoi_polygons(gdf)

    # Crop Voronoi polygons to input bound_poly extent
    voronoi_crop = voronoi.intersection(bound_poly)
    voronoi_crop = gpd.GeoDataFrame(geometry=voronoi_crop)  # convert to DataFrame

    # Dissolve all Voronoi polygons and subtract from bounds to get gaps
    voronoi_merged = voronoi_crop.dissolve()
    bound_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bound_poly))
    bound_gdf.crs = gdf.crs
    gaps = bound_gdf.difference(voronoi_merged)

    # Merge cropped Voronoi with gaps, if not empty, otherwise return cropped Voronoi
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Geometry is in a geographic CRS. Results from 'area' are likely incorrect.")
        tot_area = np.sum(gaps.area.values)

    if not tot_area == 0:
        voronoi_all = gpd.GeoDataFrame(geometry=list(voronoi_crop.geometry) + list(gaps.geometry))
        voronoi_all.crs = gdf.crs
        return voronoi_all
    else:
        return voronoi_crop
