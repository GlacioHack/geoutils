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

"""Base class for the point cloud object and the ``pc`` Pandas accessor."""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    TypeVar,
    overload,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from geoutils import profiler
from geoutils._dispatch import get_geo_attr, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface._nodata import NodataPropagation
from geoutils.interface.gridding import (
    GriddingEngine,
    GriddingMethod,
    _grid_pointcloud_to_raster,
)
from geoutils.stats.sampling import _subsample_pointcloud
from geoutils.stats.stats import _statistics
from geoutils.vector.base import VectorBase

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterLike


PointCloudBaseType = TypeVar("PointCloudBaseType", bound="PointCloudBase")


def _get_dataframe_attrs(ds: Any) -> dict[str, Any]:
    """Get GeoUtils metadata from Pandas or Dask dataframes."""

    # Dask does not carry Pandas ``attrs`` reliably through graph operations
    if is_dask_dataframe(ds):
        try:
            return object.__getattribute__(ds, "_geoutils_attrs")
        except AttributeError:
            return {}
    return getattr(ds, "attrs", {})


def _set_dataframe_attrs(ds: Any, attrs: dict[str, Any]) -> None:
    """Set GeoUtils metadata on Pandas or Dask dataframes."""

    # Keep a private copy on Dask collections and use the public mapping for Pandas
    if is_dask_dataframe(ds):
        object.__setattr__(ds, "_geoutils_attrs", attrs.copy())
    elif hasattr(ds, "attrs"):
        ds.attrs.update(attrs)


class PointCloudBase(VectorBase):
    """
    Shared implementation for :class:`geoutils.PointCloud` and the ``pc`` Pandas accessor.
    """

    _ACCESSOR_OUTPUT = False

    @property
    def _is_dask(self) -> bool:
        """Whether the backing point-cloud dataframe is partitioned by Dask."""

        if not self._is_pd:
            return False
        return is_dask_dataframe(self.ds)

    @property
    def _has_z(self) -> bool:
        """Whether the point geometries all have a Z coordinate or not."""

        if self._is_dask:
            return False
        return all(p.has_z for p in self.ds.geometry) if len(self.ds.geometry) > 0 else False

    @property
    def data(self) -> Any:
        """
        Data of the point cloud.

        Points to either the Z axis of the point geometries, or the associated data column of the geodataframe.
        """

        if self.data_column is not None:
            data = self.ds[self.data_column]
            return data if self._is_pd or self._is_dask else data.values
        if self._is_dask:
            raise ValueError("Dask-backed point clouds require an explicit data column.")
        return self.geometry.z.values

    @data.setter
    def data(self, new_data: NDArrayNum | Any) -> None:
        """Set new data for the point cloud."""

        if self.data_column is not None:
            if self._is_dask:
                # ``assign`` adds a lazy column operation without mutating partitions
                self.ds = self.ds.assign(**{self.data_column: new_data})
            else:
                self.ds[self.data_column] = new_data
        else:
            if self._is_dask:
                # Dask point geometries are kept two-dimensional for reliable metadata
                raise ValueError("Dask-backed point clouds require an explicit data column.")
            self.ds.geometry = gpd.points_from_xy(x=self.geometry.x, y=self.geometry.y, z=new_data, crs=self.crs)

    @property
    def _nongeo_columns(self) -> pd.Index:
        """Columns of the point cloud excluding the column of 2D point geometries."""

        return pd.Index([c for c in self.columns if c != "geometry"])

    @property
    def data_column(self) -> str | None:
        """
        Name of data column of the point cloud.

        Can be None if point geometries are 3D.
        """

        return getattr(self, "_data_column", None)

    @data_column.setter
    def data_column(self, new_data_column: str | None) -> None:
        """Select the dataframe column used as point-cloud values."""

        self.set_data_column(new_data_column=new_data_column)

    def set_data_column(self, new_data_column: str | None) -> None:
        """
        Select the dataframe column used as point-cloud values.

        :param new_data_column: Column to use, or None to use Z coordinates stored in 3D point geometry.
        """

        if self.is_loaded and not self._is_dask and self._has_z:
            if new_data_column is None:
                self._data_column = None
                return
            warnings.warn(
                f"Overriding 3D points with with data column '{new_data_column}'. Set data_column "
                f"to None to use the 3D point geometries instead."
            )

        if new_data_column is None:
            raise ValueError("A data column name must be passed for a point cloud with 2D point geometries.")

        if new_data_column not in self._nongeo_columns:
            raise ValueError(
                f"Data column {new_data_column} not found among columns. Available columns "
                f"are: {', '.join(self._nongeo_columns)}."
            )

        self._data_column = new_data_column
        if self._is_pd or self.is_loaded:
            attrs = _get_dataframe_attrs(self.ds)
            attrs["data_column"] = new_data_column
            _set_dataframe_attrs(self.ds, attrs)

    @property
    def is_loaded(self) -> bool:
        """Whether the point cloud data is loaded in memory."""

        if self._is_pd:
            return not self._is_dask
        return getattr(self, "_ds", None) is not None

    @property
    def point_count(self) -> int:
        """Number of points in the point cloud."""

        if not self._is_pd and not self.is_loaded:
            count = getattr(self, "_nb_points", -1)
            if count >= 0:
                return int(count)
            self.load()
        if self._is_dask:
            # Use file or construction metadata before falling back to a Dask row count
            count = _get_dataframe_attrs(self.ds).get("point_count")
            if count is not None:
                return int(count)
        return len(self.ds)

    @property
    def is_mask(self) -> bool:
        """Whether the point cloud mask is a mask (boolean type)."""

        return np.dtype(self.data.dtype) == np.bool_

    def _cast_pointcloud_output(self, new_ds: Any) -> Any:
        """Cast a GeoDataFrame-like point cloud output to the proper public type."""

        attrs = _get_dataframe_attrs(self.ds)
        attrs["data_column"] = self.data_column
        _set_dataframe_attrs(new_ds, attrs)

        # Accessors expose dataframe-like outputs while PointCloud wraps eager outputs
        if self._is_pd or self._is_dask:
            return new_ds

        from geoutils.pointcloud.pointcloud import PointCloud

        return PointCloud(new_ds, data_column=self.data_column)

    def copy(self, new_array: NDArrayNum | NDArrayBool | Any | None = None) -> Any:
        """
        Copy the point cloud in-memory or as a lazy dataframe.

        :param new_array: New data array to use in the copied point cloud's data column.
        :returns: A copied PointCloud or dataframe matching the source interface.
        """

        if self._is_dask:
            # Copying a Dask collection duplicates the graph rather than computing data
            new_ds = self.ds.copy()
            if new_array is not None:
                if self.data_column is None:
                    raise ValueError("Dask-backed point clouds require an explicit data column.")
                new_ds = new_ds.assign(**{self.data_column: new_array})
            return self._cast_pointcloud_output(new_ds)

        if new_array is not None:
            if not isinstance(new_array, np.ndarray):
                new_array = np.asarray(new_array)
            new_array = new_array.squeeze()
            if not (new_array.ndim == 1 and new_array.shape[0] == self.point_count):
                raise ValueError(
                    "New data array must be 1-dimensional with the same number of points as the point "
                    "cloud being copied."
                )
            data = new_array
        else:
            data = np.asarray(self.data).copy()

        return self.from_xyz(
            x=self.geometry.x.values,
            y=self.geometry.y.values,
            z=data,
            crs=self.crs,
            data_column=self.data_column,
            use_z=self._has_z and self.data_column is None,
        )

    @classmethod
    def from_xyz(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        crs: CRS,
        data_column: str | None = None,
        use_z: bool = False,
    ) -> Any:
        """
        Create a point cloud from separate X, Y and Z arrays.

        :param x: X coordinates.
        :param y: Y coordinates.
        :param z: Point values or Z coordinates.
        :param crs: Coordinate reference system of the point cloud.
        :param data_column: Column name used to store ``z`` when ``use_z`` is False. Defaults to ``z``.
        :param use_z: Whether to store ``z`` in 3D point geometry instead of a dataframe column.
        :returns: A PointCloud or GeoDataFrame matching the class interface.
        """

        if not use_z:
            data_column = data_column if data_column is not None else "z"
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=np.atleast_1d(x), y=np.atleast_1d(y), crs=crs),
                data={data_column: np.atleast_1d(z)},
            )
        else:
            data_column = None
            gdf = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=np.atleast_1d(x), y=np.atleast_1d(y), z=np.atleast_1d(z), crs=crs),
            )

        if getattr(cls, "_ACCESSOR_OUTPUT", False):
            gdf.attrs["data_column"] = data_column
            return gdf

        return cls(filename_or_dataset=gdf, data_column=data_column)  # type: ignore[call-arg]

    @classmethod
    def from_array(cls, data: NDArrayNum, crs: CRS, data_column: str | None = None, use_z: bool = False) -> Any:
        """
        Create a point cloud from a 3 x N or N x 3 array of X, Y and Z values.

        :param data: Coordinates and values arranged as 3 x N or N x 3.
        :param crs: Coordinate reference system of the point cloud.
        :param data_column: Column name used to store Z values when ``use_z`` is False. Defaults to ``z``.
        :param use_z: Whether to store Z values in 3D point geometry instead of a dataframe column.
        :returns: A PointCloud or GeoDataFrame matching the class interface.
        """

        if data.ndim != 2 or (data.shape[0] != 3 and data.shape[1] != 3):
            raise ValueError("Array must be of shape 3xN or Nx3.")

        if data.shape[0] != 3:
            data = data.T

        return cls.from_xyz(x=data[0, :], y=data[1, :], z=data[2, :], crs=crs, data_column=data_column, use_z=use_z)

    @classmethod
    def from_tuples(
        cls,
        tuples_xyz: Iterable[tuple[Number, Number, Number]],
        crs: CRS,
        data_column: str | None = None,
        use_z: bool = False,
    ) -> Any:
        """
        Create a point cloud from an iterable of X, Y and Z tuples.

        :param tuples_xyz: Coordinates and values as ``(x, y, z)`` tuples.
        :param crs: Coordinate reference system of the point cloud.
        :param data_column: Column name used to store Z values when ``use_z`` is False. Defaults to ``z``.
        :param use_z: Whether to store Z values in 3D point geometry instead of a dataframe column.
        :returns: A PointCloud or GeoDataFrame matching the class interface.
        """

        return cls.from_array(np.array(tuples_xyz), crs=crs, data_column=data_column, use_z=use_z)

    def to_xyz(self) -> tuple[Any, Any, Any]:
        """Convert point cloud to three 1D arrays of coordinates for X/Y/Z."""

        if self._is_dask:
            # Extract X and Y independently within each point partition
            x = self.ds["geometry"].map_partitions(lambda s: s.apply(lambda geom: geom.x), meta=("x", "float64"))
            y = self.ds["geometry"].map_partitions(lambda s: s.apply(lambda geom: geom.y), meta=("y", "float64"))
            return x, y, self.data
        return self.geometry.x.values, self.geometry.y.values, self.data

    def to_array(self) -> Any:
        """Convert point cloud to a 3 x N array of X coordinates, Y coordinates and Z values."""

        x, y, z = self.to_xyz()
        if self._is_dask:
            # Stack lazy coordinate Series into a 3 x N Dask array
            import_optional("dask")
            import dask.array as da

            # Dask Series expose lazy array conversion while coordinate arrays can pass through unchanged
            arrays = [
                value.to_dask_array(lengths=True) if hasattr(value, "to_dask_array") else value for value in (x, y, z)
            ]
            return da.stack(arrays, axis=0)
        return np.stack((x, y, z), axis=0)

    def to_tuples(self) -> Iterable[tuple[Number, Number, Number]]:
        """Convert point cloud to a list of 3-tuples."""

        if self._is_dask:
            # Tuple output is eager, so compute all three coordinate collections here
            return list(zip(*[v.compute() for v in self.to_xyz()]))
        return list(zip(self.geometry.x.values, self.geometry.y.values, self.data))

    def pointcloud_equal(self, other: Any, **kwargs: Any) -> bool:
        """
        Check if two point clouds are equal.

        :param other: PointCloud, point-cloud accessor or GeoDataFrame to compare.
        :param kwargs: Keyword arguments passed to :meth:`geoutils.Vector.vector_equal`.
        :returns: True if geometry, values, metadata and the selected data column are equal.
        """

        vector_eq = self.vector_equal(other, **kwargs)
        try:
            data_column_eq = self.data_column == get_geo_attr(other, "data_column")
        except AttributeError:
            return False
        return vector_eq and data_column_eq

    def pointcloud_allclose(self, other: Any, rtol: float = 1e-5, atol: float = 1e-8, **kwargs: Any) -> bool:
        """
        Check that two point clouds have equal metadata and numerically close coordinates and values.

        :param other: PointCloud, point-cloud accessor or GeoDataFrame to compare.
        :param rtol: Relative tolerance for coordinates and numeric values.
        :param atol: Absolute tolerance for coordinates and numeric values.
        :param kwargs: Additional options passed to :meth:`geoutils.Vector.vector_allclose`.
        :returns: True if metadata are equal and numeric values are within tolerance.
        """

        vector_close = self.vector_allclose(other, rtol=rtol, atol=atol, **kwargs)
        try:
            data_column_close = self.data_column == get_geo_attr(other, "data_column")
        except AttributeError:
            return False
        return vector_close and data_column_close

    def georeferenced_coords_equal(self: PointCloudBaseType, pc: Any) -> bool:
        """
        Check that point-cloud X/Y coordinates and CRS are equal.

        :param pc: PointCloud, point-cloud accessor or GeoDataFrame to compare.
        :returns: True if the point coordinates and CRS are equal.
        """

        if self.crs != get_geo_attr(pc, "crs"):
            return False

        if self._is_dask or is_dask_dataframe(get_geo_attr(pc, "ds")):
            return self.point_count == get_geo_attr(pc, "point_count")

        return all(
            [
                np.array_equal(self.geometry.x.values, get_geo_attr(pc, "geometry").x.values),
                np.array_equal(self.geometry.y.values, get_geo_attr(pc, "geometry").y.values),
            ]
        )

    @overload
    def get_stats(
        self,
        stats_name: str | Callable[[NDArrayNum], np.floating[Any]],
    ) -> np.floating[Any]: ...

    @overload
    def get_stats(
        self,
        stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
    ) -> dict[str, np.floating[Any]]: ...

    @profiler.profile("geoutils.pointcloud.base.get_stats", memprof=True)
    def get_stats(
        self,
        stats_name: (
            str | Callable[[NDArrayNum], np.floating[Any]] | list[str | Callable[[NDArrayNum], np.floating[Any]]] | None
        ) = None,
    ) -> np.floating[Any] | dict[str, np.floating[Any]] | None:
        """
        Retrieve statistics for the point-cloud values.

        :param stats_name: Statistic name, custom callable, or list of either. None returns the main statistics and
            ``all`` returns every available statistic.
        :returns: One value for a single statistic, or a dictionary for multiple, default or all statistics.
        """

        # Statistics return small eager values, so reduce the lazy data column here
        data = self.data.compute().values if self._is_dask else np.asarray(self.data)

        if isinstance(stats_name, list) or stats_name is None or stats_name == "all":
            return _statistics(data, stats_name)  # type: ignore
        if isinstance(stats_name, str):
            return _statistics(data, [stats_name])[stats_name]  # type: ignore
        if callable(stats_name):
            return stats_name(data)  # type: ignore
        warnings.warn(f"Statistic name {stats_name} is a not recognized string", category=UserWarning)
        return None

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum: ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[NDArrayNum, ...]: ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]: ...

    @profiler.profile("geoutils.pointcloud.base.subsample", memprof=True)
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly sample finite point-cloud values.

        :param subsample: Fraction of values to sample when at most 1, otherwise the number of values.
        :param return_indices: Whether to return sampled indexes instead of values.
        :param random_state: Random generator or seed used to make sampling reproducible.
        :returns: Sampled values, or sampled indexes when ``return_indices`` is True.
        """

        return _subsample_pointcloud(
            source_pointcloud=self,
            subsample=subsample,
            return_indices=return_indices,
            random_state=random_state,
        )

    @profiler.profile("geoutils.pointcloud.base.grid", memprof=True)
    def grid(
        self,
        ref: RasterLike | None = None,
        grid_coords: tuple[NDArrayNum, NDArrayNum] | None = None,
        res: float | tuple[float, float] | None = None,
        shape: tuple[int, int] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        resampling: GriddingMethod = "linear",
        dist_nodata_pixel: float = 1.0,
        nodata: int | float = -9999,
        *,
        distance_power: float = 2.0,
        min_points: int = 1,
        engine: GriddingEngine = "scipy",
        chunksizes: tuple[int, int] | None = None,
        mp_config: MultiprocConfig | None = None,
        n_threads: int = 0,
        nodata_propagation: NodataPropagation = "gdal",
    ) -> Any:
        """
        Grid the point cloud into a raster.

        Define the output grid with a reference raster, regular X/Y coordinates, or a combination of resolution or
        shape and optional bounds.

        :param ref: Reference raster whose grid should be matched.
        :param grid_coords: Regular X and Y coordinates defining the output grid.
        :param res: Output resolution in X and Y, mutually exclusive with ``shape``.
        :param shape: Output shape as ``(height, width)``, mutually exclusive with ``res``.
        :param bounds: Output bounds as ``(left, bottom, right, top)``. Defaults to the point-cloud bounds.
        :param resampling: Interpolation, circular statistic or distance method. ``average``, ``min`` and ``max`` are
            aliases for ``mean``, ``minimum`` and ``maximum``.
        :param dist_nodata_pixel: Maximum point distance or circular neighborhood radius in output pixels.
        :param nodata: Nodata value of the output raster.
        :param distance_power: Distance exponent used for inverse-distance weighting.
        :param min_points: Minimum number of finite points required inside a circular neighborhood.
        :param engine: Calculation engine, either ``scipy`` or ``numba``.
        :param chunksizes: Output chunk size as ``(rows, columns)`` for Dask or multiprocessing execution.
        :param mp_config: Multiprocessing configuration for computing output chunks in workers.
        :param n_threads: Number of SciPy threads for eager nearest gridding. ``0`` uses all but one available CPU.
        :param nodata_propagation: Whether invalid point values follow GDAL behavior, are ignored, or propagate.
        :returns: A gridded raster matching the concrete PointCloud or dataframe accessor interface.
        """

        return self._cast_raster_output(
            _grid_pointcloud_to_raster(
                source_pointcloud=self,
                ref=ref,
                grid_coords=grid_coords,
                res=res,
                shape=shape,
                bounds=bounds,
                resampling=resampling,
                dist_nodata_pixel=dist_nodata_pixel,
                nodata=nodata,
                distance_power=distance_power,
                min_points=min_points,
                engine=engine,
                chunksizes=chunksizes,
                mp_config=mp_config,
                dask=self._is_dask,
                n_threads=n_threads,
                nodata_propagation=nodata_propagation,
            )
        )
