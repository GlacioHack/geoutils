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

"""Base class for point cloud objects and the ``pc`` Pandas accessor."""

from __future__ import annotations

import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    TypeVar,
    cast,
    overload,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from geoutils import profiler
from geoutils._dispatch import get_geo_attr, is_dask_array
from geoutils._dispatch import is_dask_dataframe as _is_dask_dataframe
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface._nodata import NodataPropagation
from geoutils.interface.gridding import (
    GriddingEngine,
    GriddingMethod,
    _grid_pointcloud_to_raster,
)
from geoutils.stats.sampling import _subsample_numpy
from geoutils.stats.stats import _statistics
from geoutils.vector.base import VectorBase

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterLike


PointCloudBaseType = TypeVar("PointCloudBaseType", bound="PointCloudBase")


def _as_dask_array(values: Any) -> Any:
    """Convert a Dask Series to a Dask Array when possible, otherwise return the input unchanged."""

    # Dask Series exposes this conversion without computing its partitions
    if hasattr(values, "to_dask_array"):
        return values.to_dask_array(lengths=True)
    return values


def _get_dataframe_attrs(ds: Any) -> dict[str, Any]:
    """Get GeoUtils metadata from Pandas or Dask dataframes."""

    # Dask does not carry Pandas ``attrs`` reliably through graph operations
    if _is_dask_dataframe(ds):
        try:
            return object.__getattribute__(ds, "_geoutils_attrs")
        except AttributeError:
            return {}
    return getattr(ds, "attrs", {})


def _set_dataframe_attrs(ds: Any, attrs: dict[str, Any]) -> None:
    """Set GeoUtils metadata on Pandas or Dask dataframes."""

    # Keep a private copy on Dask collections and use the public mapping for Pandas
    if _is_dask_dataframe(ds):
        object.__setattr__(ds, "_geoutils_attrs", attrs.copy())
    elif hasattr(ds, "attrs"):
        ds.attrs.update(attrs)


def _cast_numeric_array_pointcloud(
    pc: PointCloudBaseType, other: PointCloudBaseType | NDArrayNum | Number | Any, operation_name: str
) -> Any:
    """
    Cast point cloud and other numeric inputs to compatible arrays, or raise an explicit error.
    """

    if isinstance(other, PointCloudBase):
        if not pc.georeferenced_coords_equal(other):
            raise ValueError(
                "Both point clouds must have the same points X/Y coordinates and CRS for " + operation_name + "."
            )
        return other.data

    try:
        other_pc = cast(Any, other).pc
    except AttributeError:
        other_pc = None
    if isinstance(other_pc, PointCloudBase):
        if not pc.georeferenced_coords_equal(other_pc):
            raise ValueError(
                "Both point clouds must have the same points X/Y coordinates and CRS for " + operation_name + "."
            )
        return other_pc.data

    if isinstance(other, (np.ndarray, pd.Series)):
        other_data = np.asarray(other).squeeze()
        if other_data.ndim == 1 and other_data.shape[0] == pc.point_count:
            return other_data
        raise ValueError(
            "The array must be 1-dimensional with the same number of points as the point cloud for "
            + operation_name
            + "."
        )

    if isinstance(other, (float, int, np.floating, np.integer)):
        return other

    if is_dask_array(other) or _is_dask_dataframe(other):
        return other

    raise NotImplementedError(
        f"Operation between an object of type {type(other)} and a point cloud impossible. Must be a point cloud, "
        f"np.ndarray or single number."
    )


class PointCloudBase(VectorBase):
    """
    Shared implementation for :class:`geoutils.PointCloud` and the ``pc`` Pandas accessor.
    """

    _ACCESSOR_OUTPUT = False

    @property
    def _is_dask(self) -> bool:
        """Whether the backing point-cloud dataframe is partitioned by Dask."""

        return _is_dask_dataframe(self.ds)

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
            return self.ds[self.data_column]
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
        """Set new column as point cloud data column."""

        if not self._is_dask and self._has_z:
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
        attrs = _get_dataframe_attrs(self.ds)
        attrs["data_column"] = new_data_column
        _set_dataframe_attrs(self.ds, attrs)

    @property
    def is_loaded(self) -> bool:
        """Whether the point cloud data is loaded in memory."""

        return not self._is_dask

    @property
    def point_count(self) -> int:
        """Number of points in the point cloud."""

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
        """Create point cloud from three 1D array-like coordinates for X/Y/Z."""

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
        """Create point cloud from a 3 x N or N x 3 array of X coordinates, Y coordinates and Z values."""

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
        """Create point cloud from an iterable of X/Y/Z tuples."""

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

            return da.stack([_as_dask_array(x), _as_dask_array(y), _as_dask_array(z)], axis=0)
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
        """

        vector_eq = self.vector_equal(other, **kwargs)
        data_column_eq = self.data_column == get_geo_attr(other, "data_column")

        return all([vector_eq, data_column_eq])

    def georeferenced_coords_equal(self: PointCloudBaseType, pc: Any) -> bool:
        """Check that point cloud X/Y coordinates and CRS are equal."""

        if self.crs != get_geo_attr(pc, "crs"):
            return False

        if self._is_dask or _is_dask_dataframe(get_geo_attr(pc, "ds")):
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
    ) -> np.floating[Any] | dict[str, np.floating[Any]]:
        """Retrieve specified statistics or all available statistics for the point cloud data."""

        # Statistics return small eager values, so reduce the lazy data column here
        data = self.data.compute().values if self._is_dask else np.asarray(self.data)

        if isinstance(stats_name, list) or stats_name is None:
            return _statistics(data, stats_name)  # type: ignore
        if isinstance(stats_name, str):
            return _statistics(data, [stats_name])[stats_name]  # type: ignore
        if callable(stats_name):
            return stats_name(data)  # type: ignore
        logging.warning("Statistic name '%s' is a not recognized string", stats_name)
        return np.nan

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
        """Randomly sample the point cloud. Only valid values are considered."""

        # Subsampling returns a small eager selection, so materialize the data column once
        data = self.data.compute().values if self._is_dask else np.asarray(self.data)
        if return_indices:
            return _subsample_numpy(
                array=data,
                subsample=subsample,
                return_indices=True,
                random_state=random_state,
            )
        return _subsample_numpy(
            array=data,
            subsample=subsample,
            return_indices=False,
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
        """Grid point cloud into a raster."""

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

    def _binary_numeric_operation(self, other: Any, op_name: str) -> Any:
        """Apply one named numeric operation to point values and retain point geometry."""

        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = getattr(self.data, op_name)(other_data)
        return self.copy(new_array=out_data)

    def __add__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__add__")

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __neg__(self) -> Any:
        return self.copy(-self.data)

    def __sub__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__sub__")

    def __rsub__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(new_array=other_data - self.data)

    def __mul__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__mul__")

    def __rmul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__truediv__")

    def __rtruediv__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(new_array=other_data / self.data)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__floordiv__")

    def __rfloordiv__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(new_array=other_data // self.data)

    def __mod__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__mod__")

    def __pow__(self, power: int | float) -> Any:
        if not isinstance(power, (float, int, np.floating, np.integer)):
            raise ValueError("Power needs to be a number.")
        return self.copy(new_array=self.data**power)

    def __eq__(self, other: Any) -> Any:  # type: ignore
        return self._binary_numeric_operation(other, "__eq__")

    def __ne__(self, other: Any) -> Any:  # type: ignore
        return self._binary_numeric_operation(other, "__ne__")

    def __lt__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__lt__")

    def __le__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__le__")

    def __gt__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__gt__")

    def __ge__(self, other: Any) -> Any:
        return self._binary_numeric_operation(other, "__ge__")

    def __and__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(self.data & other_data)

    def __rand__(self, other: Any) -> Any:
        return self.__and__(other)

    def __or__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(self.data | other_data)

    def __ror__(self, other: Any) -> Any:
        return self.__or__(other)

    def __xor__(self, other: Any) -> Any:
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        return self.copy(self.data ^ other_data)

    def __rxor__(self, other: Any) -> Any:
        return self.__xor__(other)

    def __invert__(self) -> Any:
        return self.copy(~self.data)
