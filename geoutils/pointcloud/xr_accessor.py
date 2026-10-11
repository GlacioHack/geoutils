# Copyright (c) 2026 GeoUtils developers
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

"""Module for the Xarray accessor ``pc`` mirroring the PointCloud API."""

from __future__ import annotations

import inspect
import pathlib
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.coords import BoundingBox

from geoutils._dispatch import _clip_geometry, _get_reproject_crs, get_geo_attr, has_geo_attr, is_dask_array
from geoutils._misc import _deprecate_keyword, import_optional
from geoutils._xarray import _rebuild_dataset, _same_coordinate, _select_dataset_variables
from geoutils.pointcloud.base import PointCloudBase
from geoutils.pointcloud.referencing import _point_coordinates, _point_crs
from geoutils.sampling.subsampling import _subsample_numpy

if TYPE_CHECKING:
    from geoutils.stats.variography import Variogram


@xr.register_dataarray_accessor("pc")
class DataArrayPointCloudAccessor(PointCloudBase):
    """
    This class defines the Xarray accessor 'pc' for point clouds in a DataArray.

    Most operations are inherited from PointCloudBase.
    This accessor defines specific DataArray storage, coordinate operations and conversions.

    The DataArray is 1-D and contains the active point values ``data_name``, with one x/y coordinate per row.
    Other coordinates on the same dimension can hold other 1D auxiliary values.
    """

    _is_xr = True
    _is_pd = False
    _ACCESSOR_OUTPUT = True

    def __init__(self, xarray_obj: xr.DataArray) -> None:

        # We check point dimensions/coordinate types without loading arrays
        if xarray_obj.ndim != 1:
            raise AttributeError("The point accessor requires a one-dimensional DataArray.")
        _, x_name, y_name = _point_coordinates(xarray_obj)
        if xarray_obj.name is None:
            xarray_obj.name = "z"
        if xarray_obj.name in (*xarray_obj.coords, *xarray_obj.dims, "geometry"):
            raise AttributeError("Point value names must differ from coordinates, dimensions and geometry.")
        # Give shared point algorithms their established X/Y names without copying array values
        names = {name: axis for name, axis in ((x_name, "x"), (y_name, "y")) if name != axis}
        self._obj = xarray_obj.rename(names) if names else xarray_obj

    @property
    def _dataset(self) -> Any:
        return self.ds

    @_dataset.setter
    def _dataset(self, new_ds: Any) -> None:
        self.ds = new_ds

    @property
    def ds(self) -> xr.DataArray:
        """Xarray object containing the point cloud's coordinates and values."""
        return self._obj

    @ds.setter
    def ds(self, new_ds: xr.DataArray) -> None:
        if not isinstance(new_ds, xr.DataArray):
            raise TypeError("Point DataArray storage requires an Xarray DataArray.")
        self._obj = new_ds

    @property
    def data(self) -> Any:
        return self._dataset.data

    @data.setter
    def data(self, new_data: Any) -> None:
        self._dataset.data = new_data

    @property
    def data_name(self) -> str:
        return str(self._dataset.name)

    @property
    def columns(self) -> pd.Index:
        """
        Names of the point cloud's value and attribute columns.

        Includes the DataArray name and auxiliary point coordinates, excluding X/Y and the point dimension.
        """
        auxiliary = [name for name, coord in self._dataset.coords.items() if coord.dims == self._dataset.dims]
        return pd.Index([self.data_name, *[name for name in auxiliary if name not in (*self._dataset.dims, "x", "y")]])

    @property
    def _nongeo_columns(self) -> pd.Index:
        """Attribute names available to shared point calculations."""
        return self.columns

    @property
    def crs(self) -> CRS | None:
        return _point_crs(self._dataset)

    @property
    def name(self) -> str | None:
        """Source filename when the array was opened from a file."""
        return self._dataset.encoding.get("source")

    @property
    def point_count(self) -> int:
        """
        Number of points in the point cloud.

        Read from the array shape without computing point values.
        """
        return self._dataset.size

    @property
    def _is_dask(self) -> bool:
        """Whether values or per-point coordinates use Dask arrays."""
        return any(is_dask_array(value.data) for value in (self._dataset, *self._dataset.coords.values()))

    @property
    def is_loaded(self) -> bool:
        return not self._is_dask

    @property
    def bounds(self) -> BoundingBox:
        """Horizontal extent, computing only coordinate reductions for a lazy array."""
        if self.point_count == 0:
            return BoundingBox(*(np.nan,) * 4)
        x, y, _ = self.to_xyz()
        extrema = [np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)]
        if self._is_dask:
            extrema = import_optional("dask").compute(*extrema)
        return BoundingBox(*[float(value) for value in extrema])

    @property
    def bbox(self) -> BoundingBox:
        """Total bounding box of the point cloud."""
        return self.bounds

    def to_xyz(self) -> tuple[Any, Any, Any]:
        return self._dataset.coords["x"].data, self._dataset.coords["y"].data, self.data

    def to_array(self) -> Any:
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""
        return np.stack(self.to_xyz())

    def to_tuples(self) -> Iterable[tuple[float, float, float]]:
        """
        Convert point cloud to a list of 3-tuples.

        Computes lazy X/Y coordinates and Z values before building the tuples.
        """
        return list(zip(*self._dataset.compute().pc.to_xyz()))

    def copy(self, new_array: Any = None, *, deep: bool = True) -> xr.DataArray:
        """
        Copy the point cloud.

        Lazy arrays are copied without computing their values.

        :param new_array: Optional replacement values with one value per point.
        :param deep: Whether to copy eager arrays as well as their metadata.
        :returns: A DataArray with the same point locations and attributes.
        """
        return self._dataset.copy(data=new_array, deep=deep)

    @classmethod
    @_deprecate_keyword("data_column", "data_name")
    def from_xyz(
        cls,
        x: Any,
        y: Any,
        z: Any,
        crs: Any,
        data_name: str | None = None,
        use_z: bool = False,
        *,
        auxiliary: Mapping[str, Any] | None = None,
    ) -> xr.DataArray:
        """
        Create a point cloud from separate X, Y and Z arrays.

        :param x: One-dimensional X coordinates.
        :param y: One-dimensional Y coordinates.
        :param z: One-dimensional active point values.
        :param crs: Coordinate reference system, or None for unspecified coordinates.
        :param data_name: Name of the active values, defaulting to z.
        :param use_z: Also mark the active values as geometric elevations for explicit geometry conversion.
        :param auxiliary: Additional arrays with one value per point, each stored with its original dtype.
        :returns: A one-dimensional DataArray with a point dimension and x/y coordinates.
        """

        # We just pass input arrays (to avoid loading Dask or affect other array types)
        arrays = [value if hasattr(value, "ndim") else np.asarray(value) for value in (x, y, z)]
        x, y, z = (np.atleast_1d(value) for value in arrays)
        if any(value.ndim != 1 or value.shape != z.shape for value in (x, y, z)):
            raise ValueError("X, Y and point values must be one-dimensional arrays with the same length.")
        if not all(np.issubdtype(value.dtype, np.number) for value in (x, y)):
            raise TypeError("Point X/Y coordinates must be numeric.")

        # Store attributes independently so uint64 IDs are not rounded by a common floating dtype
        name = data_name or "z"
        attributes = {} if auxiliary is None else dict(auxiliary)
        reserved = {"x", "y", "point", "geometry", name}
        if name in {"x", "y", "point", "geometry"} or reserved.intersection(attributes):
            raise ValueError("Point attribute names must be distinct from X/Y, point, geometry and the active values.")
        coordinates: dict[str, Any] = {"x": ("point", x), "y": ("point", y)}
        for key, value in attributes.items():
            coordinates[key] = ("point", value)
        point_crs = None if crs is None else CRS.from_user_input(crs)
        if point_crs is not None:
            canonical = CRS.from_user_input(point_crs.to_string())
            point_crs = canonical if canonical.equals(point_crs) else point_crs
        attrs = {"crs": None if point_crs is None else point_crs.to_wkt(), "geometry_z": use_z}
        return xr.DataArray(z, dims="point", coords=coordinates, name=name, attrs=attrs)

    @_deprecate_keyword("new_data_column", "new_data_name")
    def set_data_name(self, new_data_name: str | None) -> xr.DataArray:
        """
        Select an auxiliary attribute as the active values in a new DataArray.

        :param new_data_name: Existing active or auxiliary attribute name.
        :returns: A DataArray using the selected attribute and preserving the former values as a coordinate.
        """
        if new_data_name not in self.columns:
            raise ValueError(f"Point data column {new_data_name!r} does not exist.")
        if new_data_name == self.data_name:
            return self.copy()
        result = self._dataset.coords[new_data_name].copy().rename(new_data_name)
        result = result.drop_vars(new_data_name).assign_coords({self.data_name: self._dataset.variable})
        result.attrs = self._dataset.attrs.copy()
        if self._dataset.attrs.get("geometry_z"):
            result = result.assign_coords(_geometry_z=self._dataset.variable)
        result.attrs["geometry_z"] = False
        return result

    def _cast_pointcloud_output(self, output: xr.DataArray) -> xr.DataArray:
        """Return an array point result with independent metadata."""
        output.attrs = self._dataset.attrs.copy()
        output.encoding = {}
        return output

    def _cast_raster_output(self, raster: Any) -> xr.DataArray:
        """Expose a gridded result through the Xarray raster accessor."""
        from geoutils.raster.xr_accessor import DataArrayRasterAccessor, open_raster

        if isinstance(raster, xr.DataArray):
            raster.name = None
            return raster
        if not raster.is_loaded and raster.name is not None:
            return open_raster(raster.name)
        result = DataArrayRasterAccessor.from_array(raster.data, raster.transform, raster.crs, nodata=raster.nodata)
        result.name = None
        return result

    def _select_rows(self, indices: Any) -> xr.DataArray:
        """Select point positions while preserving auxiliary coordinates and lazy values."""
        if is_dask_array(indices):
            indices = indices.compute()
        # Record source row positions before selecting an array with an implicit index
        points = self._dataset
        dimension = points.dims[0]
        if dimension not in points.coords:
            points = points.assign_coords({dimension: pd.RangeIndex(self.point_count)})
        return self._cast_pointcloud_output(points.isel({dimension: indices}))

    def to_xarray(self) -> xr.DataArray:
        """Return the point DataArray without copying its values."""
        return self._dataset

    def to_geoutils(self) -> Any:
        """Convert to an eager GeoUtils PointCloud object."""
        from geoutils.pointcloud.pointcloud import PointCloud

        points = self._dataset.compute().pc
        x, y, values = points.to_xyz()
        geometry_z = bool(self._dataset.attrs.get("geometry_z"))
        elevation = (
            values
            if geometry_z
            else (points._dataset.coords["_geometry_z"].data if "_geometry_z" in points._dataset.coords else None)
        )
        columns = {} if geometry_z else {points.data_name: values}
        columns.update(
            {
                name: points._dataset.coords[name].data
                for name in points.columns
                if name not in (points.data_name, "_geometry_z")
            }
        )
        geometry = gpd.points_from_xy(x, y, z=elevation, crs=self.crs)
        frame = gpd.GeoDataFrame(columns, geometry=geometry, crs=self.crs)
        original_columns = self._dataset.attrs.get("dataframe_columns")
        if original_columns is not None and set(original_columns) == set(frame.columns):
            frame = frame[original_columns]
        if self._dataset.dims[0] in self._dataset.coords:
            frame.index = pd.Index(
                points._dataset.coords[self._dataset.dims[0]].data, name=self._dataset.attrs.get("dataframe_index_name")
            )
        return PointCloud(frame, data_name=None if geometry_z else points.data_name)

    ###################################
    # 3/ COMPARISON AND SPATIAL SELECTION
    ###################################

    def pointcloud_equal(self, other: Any, **kwargs: Any) -> bool:
        """
        Check if two point clouds are equal.

        :param other: Point arrays, PointCloud, or a point GeoDataFrame to compare.
        :param kwargs: Set check_dtype=False to compare values without requiring matching attribute dtypes.
        :returns: Whether ordered point rows and their metadata match.
        """
        from geoutils.pointcloud.testing import _compare_array_points

        return _compare_array_points(self, other, check_dtype=kwargs.get("check_dtype", True))

    def pointcloud_allclose(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, **kwargs: Any) -> bool:
        """
        Check that two point clouds have equal metadata and numerically close coordinates and values.

        :param other: Point arrays, PointCloud, or a point GeoDataFrame to compare.
        :param rtol: Relative tolerance for numeric values and coordinates.
        :param atol: Absolute tolerance for numeric values and coordinates.
        :param kwargs: Set check_dtype=False to allow different attribute dtypes.
        :returns: Whether the point rows match within the specified tolerance.
        """
        from geoutils.pointcloud.testing import _compare_array_points

        return _compare_array_points(self, other, rtol=rtol, atol=atol, check_dtype=kwargs.get("check_dtype", True))

    def reproject(
        self, ref: Any = None, crs: Any = None, inplace: bool = False, mp_config: Any = None
    ) -> xr.DataArray | None:
        """
        Reproject point coordinates, preserving their order and value columns.

        :param ref: Geospatial reference supplying the destination CRS.
        :param crs: Destination CRS, mutually exclusive with ref.
        :param inplace: Whether to replace coordinates on the original DataArray.
        :param mp_config: Unsupported for array accessors; use Dask chunks for parallel execution.
        :returns: Projected DataArray, or None when updating coordinates in place.
        """
        if mp_config is not None:
            raise ValueError("Array point clouds use Dask chunks rather than mp_config for reprojection.")
        target = CRS.from_user_input(_get_reproject_crs(ref, crs))
        if self.crs is None:
            raise ValueError("Point reprojection requires a source CRS.")

        # Equivalent X/Y systems need no new coordinates, metadata or Dask transformation tasks
        if self.crs.equals(target, ignore_axis_order=True):
            return None if inplace else self.copy(deep=False)
        transformer = Transformer.from_crs(self.crs, target, always_xy=True)
        x, y = xr.apply_ufunc(
            transformer.transform,
            self._dataset.coords["x"],
            self._dataset.coords["y"],
            dask="parallelized",
            output_core_dims=[[], []],
            output_dtypes=[float, float],
        )
        result = self._cast_pointcloud_output(self._dataset.assign_coords(x=x, y=y))
        result.attrs["crs"] = target.to_wkt()

        # Preserve custom coordinate metadata and describe projected axes in their new units
        # Stored value ranges describe the original coordinates and cannot follow a CRS change
        ranges = {"valid_range", "actual_range", "valid_min", "valid_max"}
        for coordinate in target.cs_to_cf():
            name = coordinate.get("axis", "").lower()
            if name in ("x", "y"):
                attributes = {
                    key: value for key, value in self._dataset.coords[name].attrs.items() if key not in ranges
                }
                result.coords[name].attrs = {**attributes, **coordinate, "crs": target.to_wkt()}
        mapping = self._dataset.encoding.get("grid_mapping", self._dataset.attrs.get("grid_mapping"))
        if mapping in result.coords:
            result.coords[mapping].attrs = {**target.to_cf(), "spatial_ref": target.to_wkt()}
            result.encoding["grid_mapping"] = mapping
        if inplace:
            self._dataset.coords["x"], self._dataset.coords["y"] = result.coords["x"], result.coords["y"]
            if mapping in result.coords:
                self._dataset.coords[mapping] = result.coords[mapping]
            self._dataset.attrs = result.attrs
            return None
        return result

    def crop(
        self,
        bbox: Any = None,
        mode: Literal["intersects", "within"] = "intersects",
        *,
        crs: Any = None,
        inplace: bool = False,
        crop_geom: Any = None,
        clip: bool = False,
    ) -> xr.DataArray | None:
        """
        Crop the point cloud to a bounding box.

        :param bbox: Four bounds or a geospatial object defining the crop extent.
        :param mode: Include boundary points with intersects, or exclude them with within.
        :param crs: CRS of explicit bounds, defaulting to the point CRS.
        :param inplace: Unsupported for changing array lengths; use the returned DataArray.
        :param crop_geom: Deprecated alias of bbox.
        :param clip: Whether to apply exact geometry clipping instead of bounds.
        :returns: DataArray containing the selected point rows.
        """
        if crop_geom is not None:
            warnings.warn("Argument 'crop_geom' is deprecated, use 'bbox' instead.", DeprecationWarning, stacklevel=2)
            bbox = crop_geom
        if bbox is None:
            raise ValueError("Argument 'bbox' must be passed.")
        if mode not in ("intersects", "within"):
            raise ValueError("Argument 'mode' must be either 'intersects' or 'within'.")
        if inplace:
            raise ValueError("Array point crops change row counts; use the returned DataArray.")
        if clip:
            return self.clip(bbox)
        from geoutils._dispatch import _check_match_bbox

        # Match the reference extent to point coordinates before testing each row
        bounds = BoundingBox(*_check_match_bbox(self, bbox))
        if crs is not None and CRS.from_user_input(crs) != self.crs:
            from geoutils.projtools import _get_bounds_projected

            bounds = BoundingBox(*_get_bounds_projected(bounds, CRS.from_user_input(crs), self.crs))
        x, y, _ = self.to_xyz()
        if mode == "within":
            mask = (x > bounds.left) & (x < bounds.right) & (y > bounds.bottom) & (y < bounds.top)
        else:
            mask = (x >= bounds.left) & (x <= bounds.right) & (y >= bounds.bottom) & (y <= bounds.top)
        return self._select_rows(mask)

    def clip(self, mask: Any, keep_geom_type: bool = False, sort: bool = False, mp_config: Any = None) -> xr.DataArray:
        """
        Remove points outside an exact clipping geometry.

        :param mask: Geometry, geospatial object, or four bounds to clip against.
        :param keep_geom_type: Accepted for compatibility; all selected rows are points.
        :param sort: Whether to return rows in their original order, as array selections always do.
        :param mp_config: Unsupported for array accessors; use Dask chunks.
        :returns: DataArray containing the selected rows and their attributes.
        """
        import shapely

        if mp_config is not None:
            raise ValueError("Array point clouds use Dask chunks rather than mp_config for clipping.")
        geometry = _clip_geometry(mask, target_crs=self.crs)
        selected = xr.apply_ufunc(
            lambda x, y: shapely.intersects_xy(geometry, x, y),
            self._dataset.coords["x"],
            self._dataset.coords["y"],
            dask="parallelized",
            output_dtypes=[bool],
        )
        return self._select_rows(selected.data)

    #####################################
    # 4/ COMPUTING, DISPLAY AND FILE OUTPUT
    #####################################

    def load(self) -> xr.DataArray:
        """Compute a separate eager DataArray, leaving the source's Dask arrays lazy."""
        return self._dataset.compute()

    def plot(
        self,
        column: str | None = None,
        ref: Any = None,
        *,
        max_points: Literal["auto"] | int | None = "auto",
        random_state: Any = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Plot the point cloud.

        :param column: Attribute to color, defaulting to the active values.
        :param ref: Optional CRS or geospatial reference to match.
        :param max_points: Maximum displayed points, auto for an axes-based limit, or None for all points.
        :param random_state: Seed or generator for deterministic sampling.
        :param kwargs: Matplotlib scatter and shared colorbar/axes options.
        :returns: Plot and colorbar axes when return_axes=True; otherwise None.
        """
        from geoutils.pointcloud.plotting import _plot_pointcloud

        return _plot_pointcloud(
            self,
            column=column,
            ref=ref,
            max_points=max_points,
            random_state=random_state,
            **kwargs,
        )

    def to_file(self, filename: str, **kwargs: Any) -> None:
        """
        Write points to LAS/LAZ, GeoParquet, or a geometry format.

        :param filename: Destination filename; its suffix selects the format.
        :param kwargs: Options passed to to_las(), to_parquet(), or GeoPandas to_file().
        """
        suffix = pathlib.Path(filename).suffix.lower()
        if suffix in (".las", ".laz"):
            self.to_las(filename, **kwargs)
        elif suffix == ".parquet":
            self.to_parquet(filename, **kwargs)
        else:
            self.to_geoutils()._dataset.to_file(filename, **kwargs)

    def to_las(self, filename: str, **kwargs: Any) -> None:
        """
        Write LAS/LAZ in numeric row partitions without constructing point geometries.

        :param filename: Destination LAS or LAZ file.
        :param kwargs: Header and partition options passed to the array LAS writer.
        """
        from geoutils.pointcloud.las import _write_array_las

        _write_array_las(self._dataset, filename, **kwargs)

    def to_parquet(self, filename: str, **kwargs: Any) -> None:
        """
        Write native GeoParquet points in bounded row groups or separate partition files.

        :param filename: Destination file, or directory when partitioned=True.
        :param kwargs: Partition size and compression options passed to the GeoParquet writer.
        """
        from geoutils.pointcloud.parquet import _write_point_parquet

        _write_point_parquet(self._dataset, filename, **kwargs)


# Dataset accessor


def _point_variables(dataset: xr.Dataset) -> list[str]:
    """Identify value variables aligned on the Dataset's common point dimension."""

    try:
        dimension, x_name, y_name = _point_coordinates(dataset)
    except AttributeError:
        return []
    return [
        str(name)
        for name, value in dataset.data_vars.items()
        if value.dims == (dimension,) and name not in (x_name, y_name, dimension)
    ]


def _prepare_point_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Expose associated point X/Y variables as coordinates without loading arrays."""

    try:
        _, x_name, y_name = _point_coordinates(dataset)
    except AttributeError:
        return dataset
    return dataset.set_coords([x_name, y_name])


def _point_dataset_support(dataset: xr.Dataset, variables: Sequence[str]) -> xr.DataArray:
    """Attach aligned value variables to one point DataArray after validating their shared CRS."""

    # Promote associated X/Y variables to coordinates without modifying the input Dataset
    _, x_name, y_name = _point_coordinates(dataset)
    compatible = _point_variables(dataset)
    source = dataset.set_coords([x_name, y_name])
    first = source[variables[0]].copy(deep=False)
    reference = None
    for name in compatible:
        crs = _point_crs(source[name])
        if crs is not None:
            if reference is not None and not reference.equals(crs, ignore_axis_order=True):
                raise ValueError("Point variables on a common dimension must have the same CRS.")
            reference = crs
    if reference is None:
        value = dataset.attrs.get("point_crs", dataset.attrs.get("crs"))
        reference = None if value is None else CRS.from_user_input(value)

    # Shared point algorithms read one support and access auxiliary values by their variable names
    auxiliary = {name: source[name].variable for name in compatible if name != variables[0]}
    first = first.assign_coords(auxiliary)
    first.attrs = first.attrs.copy()
    first.attrs.update({key: value for key, value in dataset.attrs.items() if key.startswith("dataframe_")})
    first.attrs["crs"] = None if reference is None else reference.to_wkt()
    return first


def _subsample_point_rows(
    support: DataArrayPointCloudAccessor,
    subsample: int | float,
    *,
    mask: Any,
    random_state: Any,
    strategy: Literal["topk", "sequential"],
) -> np.ndarray[Any, Any]:
    """
    Select common point rows without reading value arrays.

    An eager mask is checked against the point coordinates, CRS and row labels before _subsample_numpy() draws
    positions from the eligible rows. Sampling placeholder values includes rows with missing point measurements.
    """

    # Resolve eligible rows from an eager mask without reading point values
    eligible = None
    if mask is not None:
        from geoutils.sampling.support import _mask_at_support, _normalize_mask_array

        if is_dask_array(getattr(mask, "data", mask)):
            raise ValueError("A lazy mask has an unknown selected point count; compute the mask before subsample().")
        if isinstance(mask, xr.DataArray) and mask.dims == support.ds.dims:
            # Aligned point masks need no spatial search or comparison of computed coordinates
            mask_coordinates: tuple[str, str, str] | None
            try:
                mask_coordinates = _point_coordinates(mask)
            except AttributeError:
                mask_coordinates = None
            if mask_coordinates is not None:
                _, mask_x, mask_y = mask_coordinates
                pairs = [(mask_x, "x"), (mask_y, "y")]
                if any(
                    not _same_coordinate(mask.coords[left].variable, support.ds.coords[right].variable)
                    for left, right in pairs
                ):
                    raise ValueError("Point mask coordinates differ from the shared point support.")
                mask_crs = _point_crs(mask)
                if mask_crs is not None and (
                    support.crs is None or not mask_crs.equals(support.crs, ignore_axis_order=True)
                ):
                    raise ValueError("Point mask and value variables must have the same CRS.")
            # Check stored row labels before interpreting mask entries as positions
            dimension = support.ds.dims[0]
            if dimension in mask.coords and dimension in support.ds.coords:
                if not _same_coordinate(mask.coords[dimension].variable, support.ds.coords[dimension].variable):
                    raise ValueError("Point mask labels differ from the shared point support.")
            eligible = _normalize_mask_array(mask.data, (support.point_count,))
        else:
            eligible = _mask_at_support(mask, support)
        if is_dask_array(eligible):
            raise ValueError("A lazy mask has an unknown selected point count; compute the mask before subsample().")
    # Draw positions from placeholder values so missing measurements do not change the sample
    rows = _subsample_numpy(
        np.empty(support.point_count, dtype=np.uint8),
        subsample,
        return_indices=True,
        random_state=random_state,
        strategy=strategy,
        skip_nodata=False,
        mask=eligible,
    )[0]
    return rows


@xr.register_dataset_accessor("pc")
class DatasetPointCloudAccessor:
    """
    This class defines the Xarray accessor 'pc' for point clouds in a Dataset.

    Dataset methods delegate calculations to the DataArray accessor for all recognized point variables (default) or
    for selected ones (by passing ``variables`` to an operation), then combine the results to return a Dataset.
    Other variables and independent coordinates are preserved. Operations raise an error if their new point
    coordinates or row selection would affect an unselected variable.

    A point variable is recognized as a 1D variable on the common point dimension with shared X/Y coordinates.
    The ``variables`` property of the accessor lists all variables recognized as point values.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Store the Dataset without loading or changing its variables."""
        self._obj = xarray_obj

    @property
    def variables(self) -> list[str]:
        """Names of aligned point value variables, in Dataset order."""
        return _point_variables(self._obj)

    def _wrap_output_point(
        self, operation: str, variables: Sequence[str] | None, /, *args: Any, **kwargs: Any
    ) -> xr.Dataset:
        """
        Function to wrap operations yielding a point output for all variables in the Dataset.

        We perform a point operation on the selected variables, then reconstruct the complete Dataset.

        Internal logic:
        - _select_dataset_variables() validates the selection,
        - _point_dataset_support() supplies shared coordinates and CRS metadata,
        - row selection and reprojection run once, while filtering shares one neighbor search,
        - _rebuild_dataset() checks shared coordinates before combining results with independent variables.
        """

        # 1/ Check inputs, in particular their coordinates and CRS
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        if operation == "subsample":
            mask = kwargs.pop("mask", None)
            random_state = kwargs.pop("random_state", None)
            strategy = kwargs.pop("strategy", "topk")
            defaults = {"as_array": False, "return_indices": False, "force_output_to_memory": False, "mp_config": None}
            if any(name not in defaults or value != defaults[name] for name, value in kwargs.items()):
                raise ValueError(
                    "Dataset subsampling returns a Dataset; use DataArray.pc for array or multiprocessing output."
                )
        elif operation == "reproject" and kwargs.pop("inplace", False):
            raise ValueError("Dataset point operations return a new Dataset; inplace=True is unsupported.")
        elif operation == "filter" and any(dataset[name].dtype.kind not in "biuf" for name in selected):
            raise TypeError("Point filtering requires numeric value variables.")

        # 2/ We prepare shared point inputs for the operation
        support = _point_dataset_support(dataset, selected)
        dimension, _, _ = _point_coordinates(support)

        # 3/ We perform the operation on selected point variables
        # Some per-operation behaviour is covered below

        if operation == "subsample":
            # Select one set of positions for all values, including rows with missing measurements
            positions = _subsample_point_rows(
                support.pc, *args, mask=mask, random_state=random_state, strategy=strategy
            )
            transformed = {name: dataset[name].isel({dimension: positions}) for name in selected}

        elif operation == "reproject":
            # Reproject the shared point coordinates once for every selected variable
            projected = getattr(support.pc, operation)(*args, **kwargs)
            # A matching CRS also leaves unselected values on the shared locations valid
            if support.pc.crs.equals(projected.pc.crs, ignore_axis_order=True):
                return dataset.copy(deep=False)

            # Restore the original coordinate names and update their CRS metadata
            _, x_name, y_name = _point_coordinates(dataset)
            coordinates = {name: projected.coords[axis].variable for name, axis in ((x_name, "x"), (y_name, "y"))}
            for coordinate in coordinates.values():
                coordinate.attrs = coordinate.attrs.copy()
                coordinate.attrs["crs"] = projected.attrs["crs"]

            # Update named point grid mappings referenced by the selected variables
            for name in selected:
                original = dataset[name]
                mapping = original.encoding.get("grid_mapping", original.attrs.get("grid_mapping"))
                if mapping in original.coords:
                    # Update explicit point mappings without borrowing independent raster georeferencing
                    coordinates[mapping] = original.coords[mapping].variable.copy(deep=False)
                    coordinates[mapping].attrs = {
                        **CRS.from_user_input(projected.attrs["crs"]).to_cf(),
                        "spatial_ref": projected.attrs["crs"],
                    }

            # Scalar point mappings accompany every value, even when only one explicitly references them
            transformed = {}
            for name in selected:
                value = dataset[name].assign_coords(coordinates)
                value.attrs = value.attrs.copy()
                value.attrs["crs"] = projected.attrs["crs"]
                transformed[name] = value

        elif operation == "filter":
            from geoutils.filters.irregular import _filter_pointcloud

            x, y, _ = support.pc.to_xyz()

            # Filter every selected column with the same neighbor pairs
            values = np.stack([dataset[name].data for name in selected], axis=1)
            filtered = _filter_pointcloud((x, y, values), *args, **kwargs)

            # Restore each field's coordinates and metadata before combining the Dataset
            transformed = {}
            for index, name in enumerate(selected):
                transformed[name] = dataset[name].copy(data=filtered[:, index])

        else:
            # Row numbers replace active values so only coordinates can enter the spatial calculation
            if any(is_dask_array(axis) for axis in support.pc.to_xyz()[:2]):
                raise ValueError(
                    "Lazy point coordinates have an unknown selected row count; compute coordinates before selection."
                )
            rows = support.copy(data=np.arange(support.pc.point_count), deep=False)
            selected_rows = getattr(rows.pc, operation)(*args, **kwargs)
            positions = selected_rows.data
            transformed = {name: dataset[name].isel({dimension: positions}) for name in selected}

        # 4/ We combine transformed point values with untouched Dataset variables

        # Check shared coordinates and update Dataset-level point CRS metadata
        result = _rebuild_dataset(dataset, transformed)
        if operation == "reproject" and "point_crs" in result.attrs:
            result.attrs["point_crs"] = projected.attrs["crs"]
        return result

    def subsample(
        self,
        subsample: int | float,
        *,
        variables: Sequence[str] | None = None,
        mask: Any = None,
        random_state: Any = None,
        strategy: Literal["topk", "sequential"] = "topk",
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Randomly sample point rows allowed by mask, without replacement.

        Sampling uses locations independently of missing values in individual value variables. It therefore needs
        no scan of Dask value arrays. A supplied boolean or spatial mask restricts the shared selection.

        See :meth:`DataArray.pc.subsample() <geoutils.DataArrayPointCloudAccessor.subsample>` for shared
        arguments.

        :param subsample: Fraction up to one or maximum point count above one.
        :param variables: Point variables to select; None selects all aligned values.
        :param mask: Optional mask of eligible point rows.
        :param random_state: Seed or random generator for repeatable sampling.
        :param strategy: Shared sampling strategy, topk or sequential.
        :param kwargs: Default DataArray output options are accepted; array or multiprocessing output is unsupported.
        :returns: Dataset containing the same selected locations in every point variable.
        """

        return self._wrap_output_point(
            "subsample", variables, subsample, mask=mask, random_state=random_state, strategy=strategy, **kwargs
        )

    def reproject(
        self,
        ref: Any = None,
        crs: Any = None,
        inplace: bool = False,
        mp_config: Any = None,
        *,
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reproject point coordinates, preserving their order and value columns.

        See :meth:`DataArray.pc.reproject() <geoutils.DataArrayPointCloudAccessor.reproject>` for shared
        arguments.

        :param ref: Native geospatial object supplying the destination CRS.
        :param crs: Destination CRS, mutually exclusive with ref.
        :param inplace: Must be False; Dataset operations return a new Dataset.
        :param mp_config: Unsupported for array accessors; use Dask chunks.
        :param variables: Point names to transform; None selects all aligned variables.
        :param kwargs: Keyword reprojection arguments.
        :returns: Dataset with projected point coordinates and unchanged point values.
        """

        return self._wrap_output_point(
            "reproject", variables, ref=ref, crs=crs, inplace=inplace, mp_config=mp_config, **kwargs
        )

    def crop(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Crop the point cloud to a bounding box.

        See :meth:`DataArray.pc.crop() <geoutils.DataArrayPointCloudAccessor.crop>` for shared arguments.

        :param args: Crop bounds.
        :param variables: Point names to crop; None selects all aligned variables.
        :param kwargs: Keyword crop options.
        :returns: Dataset containing the selected point rows and independent variables.
        """
        return self._wrap_output_point("crop", variables, *args, **kwargs)

    def clip(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Remove points outside an exact clipping geometry.

        See :meth:`DataArray.pc.clip() <geoutils.DataArrayPointCloudAccessor.clip>` for shared arguments.

        :param args: Geometry or mask to clip with.
        :param variables: Point names to clip; None selects all aligned variables.
        :param kwargs: Keyword clipping options.
        :returns: Dataset containing the selected point rows and independent variables.
        """
        return self._wrap_output_point("clip", variables, *args, **kwargs)

    def filter(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Filter the point cloud.

        One neighbor search is shared by all value variables. See
        :meth:`DataArray.pc.filter() <geoutils.DataArrayPointCloudAccessor.filter>` for shared arguments.

        :param args: Filter method and positional neighborhood options.
        :param variables: Point names to filter; None selects all aligned values. Values must be numeric.
        :param kwargs: Options passed to the shared point filter.
        :returns: Dataset with filtered values and unchanged point locations.
        """
        return self._wrap_output_point("filter", variables, *args, **kwargs)

    def _wrap_output_raster(
        self, operation: str, variables: Sequence[str] | None, /, *args: Any, **kwargs: Any
    ) -> xr.Dataset:
        """
        Function to wrap operations yielding a raster output for all variables in the Dataset.

        We delegate a point operation to each selected DataArray, then reconstruct the complete Dataset given that
        the output is now raster data.
        """

        # 1/ Checks to validate selection/arguments
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)

        # 2/ We prepare point inputs and the destination grid for DataArray methods

        # Validate the destination grid without reading lazy coordinates
        support = _point_dataset_support(dataset, selected)
        options = inspect.signature(getattr(PointCloudBase, operation)).bind_partial(self, *args, **kwargs).arguments
        reference = options.get("ref")
        raster_reference = (
            reference is not None
            and has_geo_attr(reference, "transform")
            and isinstance(get_geo_attr(reference, "transform"), Affine)
        )
        if any(is_dask_array(axis) for axis in support.pc.to_xyz()[:2]):
            # A known destination grid avoids scanning coordinates just to discover its output shape
            if not raster_reference and options.get("bounds") is None and options.get("grid_coords") is None:
                raise ValueError(
                    "Gridding lazy point coordinates requires a raster ref, grid_coords or explicit bounds."
                )

        # 3/ We perform the operation on selected point variables
        transformed = {}
        for name in selected:
            if dataset[name].dtype.kind not in "biuf":
                raise TypeError("Point gridding requires numeric value variables.")

            # Grid each active value variable on the same destination
            point = support.pc.set_data_name(name)
            transformed[name] = getattr(point.pc, operation)(*args, **kwargs).rename(name)

        # 4/ We combine gridded values with untouched Dataset variables
        return _rebuild_dataset(dataset, transformed)

    @_deprecate_keyword("data_column", "data_name")
    def grid(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> xr.Dataset:
        """
        Grid the point cloud into a raster.

        See :meth:`DataArray.pc.grid() <geoutils.DataArrayPointCloudAccessor.grid>` for shared arguments.

        :param args: Positional gridding arguments.
        :param variables: Point names to grid; None selects all aligned values. Values must be numeric.
        :param kwargs: Grid options, including a native raster DataArray through ref.
        :returns: Dataset with gridded value variables and independent data preserved when coordinates agree.
        """
        return self._wrap_output_raster("grid", variables, *args, **kwargs)

    def _wrap_output_dict(
        self, operation: str, variables: Sequence[str] | None, /, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Function to wrap operations yielding a dictionary output for all variables in the Dataset.

        We delegate a point operation to each selected DataArray, then collect the results in a dictionary keyed by
        variable name.
        """

        # 1/ Checks to validate selection/arguments

        # Validate every selected value before reading point arrays or sampling pairs
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        if operation == "variogram" and any(dataset[name].dtype.kind not in "biuf" for name in selected):
            raise TypeError("Point variograms require numeric value variables.")

        # 2/ We prepare shared point inputs and calculation options
        support = _point_dataset_support(dataset, selected)

        # Reuse explicit distance bins even when their edges can be iterated only once
        bins = kwargs.get("bins")
        if bins is not None and not isinstance(bins, str):
            kwargs["bins"] = tuple(bins)

        # 3/ We collect a result for every selected point variable
        results = {}
        for name in selected:
            point = support.pc.set_data_name(name)

            # Pair sampling reads only this variable and its spatial coordinates
            point = point.drop_vars([other for other in self.variables if other != name])
            results[name] = getattr(point.pc, operation)(*args, **kwargs)
        return results

    def variogram(self, *, variables: Sequence[str] | None = None, **kwargs: Any) -> dict[str, Variogram]:
        """
        Estimate a lightweight empirical variogram from point pairs.

        Each selected variable is sampled independently after excluding its missing values. Dask point arrays are
        read in chunks; the returned variograms contain eager NumPy arrays. An integer random_state is reused for
        each variable, while a supplied generator advances in the selected variable order.

        See :meth:`DataArray.pc.variogram() <geoutils.DataArrayPointCloudAccessor.variogram>` for shared arguments.

        :param variables: Point names to analyze; None selects all aligned values. Values must be numeric.
        :param kwargs: Pair sampling, estimation and fitting options passed to DataArray.pc.variogram().
        :returns: Dictionary mapping each selected variable name to its Variogram.
        """

        return self._wrap_output_dict("variogram", variables, **kwargs)

    def stats(self, *args: Any, variables: Sequence[str] | None = None, **kwargs: Any) -> Any:
        """
        Compute statistics, global or grouped by categories, bins, or vector zones.

        Grouping and sampling of point locations are shared by all variables. See
        :meth:`DataArray.pc.stats() <geoutils.DataArrayPointCloudAccessor.stats>` for shared arguments.

        :param args: Statistics to calculate.
        :param variables: Point names to summarize; None selects all aligned values.
        :param kwargs: Grouping and reduction options. Use variables instead of values.
        :returns: Named statistics, a grouped table, or a grouped table and masks.
        """
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        if "values" in kwargs:
            raise ValueError("Use 'variables' to select Dataset point values instead of 'values'.")
        support = _point_dataset_support(dataset, selected)
        return support.pc.stats(*args, values={name: name for name in selected}, **kwargs)

    def to_file(self, filename: str, *, variables: Sequence[str] | None = None, **kwargs: Any) -> None:
        """
        Write Dataset points using the format selected by the filename suffix.

        :param filename: Destination LAS/LAZ, GeoParquet or geometry filename.
        :param variables: Point fields to write; None selects all aligned values.
        :param kwargs: Options passed to the DataArray point writer.
        """
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        support = _point_dataset_support(dataset[selected], selected)
        support.pc.to_file(filename, **kwargs)

    def to_las(self, filename: str, *, variables: Sequence[str] | None = None, **kwargs: Any) -> None:
        """
        Write points in the dataset to a LAS or LAZ file.

        See :meth:`DataArray.pc.to_las() <geoutils.DataArrayPointCloudAccessor.to_las>` for shared arguments.

        :param filename: Destination LAS or LAZ file.
        :param variables: Point fields to write; None selects all aligned values. The first supplies LAS elevations.
        :param kwargs: Header and partition options.
        """
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        support = _point_dataset_support(dataset[selected], selected)
        support.pc.to_las(filename, **kwargs)

    def to_parquet(self, filename: str, *, variables: Sequence[str] | None = None, **kwargs: Any) -> None:
        """
        Write points in the dataset to GeoParquet.

        See :meth:`DataArray.pc.to_parquet() <geoutils.DataArrayPointCloudAccessor.to_parquet>` for shared
        arguments.

        :param filename: Destination file or partition directory.
        :param variables: Point value fields to write; None selects all aligned values.
        :param kwargs: Partition and compression options.
        """
        dataset = _prepare_point_dataset(self._obj)
        selected = _select_dataset_variables(dataset, self.variables, variables)
        support = _point_dataset_support(dataset[selected], selected)
        support.pc.to_parquet(filename, **kwargs)
