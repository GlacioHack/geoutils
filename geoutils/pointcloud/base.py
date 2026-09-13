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
from collections.abc import Mapping
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
from geoutils._dispatch import get_geo_attr, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.interface._nodata import NodataPropagation
from geoutils.interface.gridding import (
    GriddingEngine,
    GriddingMethod,
    _grid_pointcloud_to_raster,
)
from geoutils.pointcloud.dataframe import (
    _build_pointcloud_output,
    _get_dataframe_attrs,
    _set_dataframe_attrs,
)
from geoutils.pointcloud.testing import _georeferenced_coords_equal
from geoutils.sampling.subsampling import _subsample_pointcloud
from geoutils.stats.stats import stats as _stats
from geoutils.stats.stats import variogram as _variogram
from geoutils.vector.base import VectorBase
from geoutils.vector.transformation import _get_reproject_crs

if TYPE_CHECKING:
    import xarray as xr

    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterLike
    from geoutils.stats.variography import Variogram
    from geoutils.vector.base import VectorLike


PointCloudBaseType = TypeVar("PointCloudBaseType", bound="PointCloudBase")


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
        """Whether all point geometries have a Z coordinate."""

        if self._is_dask:
            return False
        if not self.is_loaded:
            return getattr(self, "_geometry_type", None) in ("Point Z", "3D Point")
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

        if self._is_pd:
            # Multiple accessors can share a dataframe, so its metadata owns the selected column
            attrs = _get_dataframe_attrs(self.ds)
            if "data_column" in attrs:
                return attrs["data_column"]
        return getattr(self, "_data_column", None)

    @data_column.setter
    def data_column(self, new_data_column: str | None) -> None:
        """Select the dataframe column used as point-cloud values."""

        self.set_data_column(new_data_column=new_data_column)

    def set_data_column(self, new_data_column: str | None) -> None:
        """
        Select the dataframe column used as point-cloud values.

        Selecting a named column for 3D points does not change the Z coordinates stored in their geometry.

        :param new_data_column: Column to use, or None to use Z coordinates stored in 3D point geometry.
        """

        if self._has_z and new_data_column is None:
            self._data_column = None
            if self._is_pd or self.is_loaded:
                attrs = _get_dataframe_attrs(self.ds)
                attrs["data_column"] = None
                _set_dataframe_attrs(self.ds, attrs)
            return

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

        # Copy source metadata before updating the result so its cached values remain independent
        attrs = _get_dataframe_attrs(self.ds).copy()
        return _build_pointcloud_output(
            new_ds,
            data_column=self.data_column,
            as_dataframe=self._is_pd or self._is_dask,
            attrs=attrs,
        )

    def _override_gdf_output(self, other: Any) -> Any:
        """Keep point-preserving GeoDataFrame outputs as point clouds."""

        if is_dask_dataframe(other):
            return self._cast_pointcloud_output(other)
        if isinstance(other, gpd.GeoDataFrame):
            geometry_types = set(other.geom_type)
            if len(geometry_types) == 0 or geometry_types == {"Point"}:
                return self._cast_pointcloud_output(other)
        return super()._override_gdf_output(other)

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
        else:
            new_ds = self.ds.copy()
            if new_array is not None:
                if not isinstance(new_array, np.ndarray):
                    new_array = np.asarray(new_array)
                new_array = new_array.squeeze()
                if not (new_array.ndim == 1 and new_array.shape[0] == self.point_count):
                    raise ValueError(
                        "New data array must be 1-dimensional with the same number of points as the point "
                        "cloud being copied."
                    )
                if self.data_column is not None:
                    new_ds[self.data_column] = new_array
                else:
                    new_ds.geometry = gpd.points_from_xy(
                        x=self.geometry.x.to_numpy(),
                        y=self.geometry.y.to_numpy(),
                        z=new_array,
                        crs=self.crs,
                    )

        output = self._cast_pointcloud_output(new_ds)
        if self._is_dask:
            # A lazy copy has the same point locations, so it can reuse the source's known count and bounds
            source_attrs = _get_dataframe_attrs(self.ds)
            output_attrs = _get_dataframe_attrs(output).copy()
            output_attrs.update(point_count=source_attrs.get("point_count"), bounds=source_attrs.get("bounds"))
            _set_dataframe_attrs(output, output_attrs)
        return output

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

    def georeferenced_coords_equal(self: PointCloudBaseType, pc: Any, warn_3d_crs: bool = True) -> bool:
        """
        Check that point-cloud X/Y coordinates and CRS are equal.

        :param pc: PointCloud, point-cloud accessor or GeoDataFrame to compare.
        :param warn_3d_crs: Whether to warn if the vertical CRS differs.
        :returns: True if the point coordinates and CRS are equal.
        """

        return _georeferenced_coords_equal(self, pc, warn_3d_crs=warn_3d_crs)

    def to_geoutils(self) -> Any:
        """Convert to an eager GeoUtils PointCloud object."""

        from geoutils.pointcloud.pointcloud import PointCloud

        ds = self.ds.compute() if self._is_dask else self.ds
        return PointCloud(ds, data_column=self.data_column)

    def stats(
        self,
        statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] | None = None,
        *,
        by: Mapping[str, Any] | None = None,
        values: str | Iterable[str] | Mapping[str, Any] | None = None,
        bins: Mapping[str, Any] | None = None,
        categories: Mapping[str, Iterable[Any]] | None = None,
        at: Literal["self"] | RasterLike | PointCloudLike | None = None,
        mask: RasterLike | VectorLike | ArrayLike | None = None,
        mask_mode: Literal["inside", "outside"] = "inside",
        subsample: int | float = 1,
        subsample_per_group: bool = False,
        random_state: int | np.random.Generator | None = None,
        strategy: Literal["auto", "dense", "sparse", "groupwise"] = "auto",
        backend: Literal["geoutils", "flox"] = "geoutils",
        subsampling_strategy: Literal["sequential", "topk"] = "sequential",
        interpolation: InterpolationMethod = "linear",
        align: Literal["raise", "reproject"] = "raise",
        observed: bool = True,
        return_masks: bool = False,
        mp_config: MultiprocConfig | None = None,
    ) -> Any:
        """Calculate summary statistics or statistics grouped by categories, bins, or vector zones.

        Omit ``by`` to summarize the active point values. Grouped inputs follow the same ``by``, ``categories``,
        ``bins``, and vector-zone interface as Raster.stats().

        :param statistics: Statistics to calculate (e.g. "mean", ["mean", "nmad"], or np.nanmedian). None returns
            "min", "max", "mean", "median", "std", "nmad", "validcount", "totalcount" and "percentagevalidpoints".
            "all" also includes "sum", "sumofsquares", "90thpercentile", "iqr", "le90" and "rmse", plus inlier counts
            for masked global statistics. Grouped defaults replace "validcount" with "count"; every grouped result
            includes "count".
        :param by: Named variables to group by (e.g. {"elevation": elevation}); use {"glacier": (outlines, "id")}
            for vector zones. Arrays must match the selected locations. Omit for global statistics.
        :param values: Point columns to summarize (e.g. "height" or ["height", "intensity"]); defaults to the main
            data column. Use a mapping for named inputs (e.g. {"elevation": (dem, 1)}).
        :param bins: Continuous bins keyed by grouping name (e.g. {"elevation": 10}). Each definition is a count of
            equal-width bins, increasing edges (e.g. [0, 2, 5]), or a Pandas IntervalIndex to choose open/closed sides.
        :param categories: Ordered categories keyed by grouping name (e.g. {"landcover": [100, 110, 120]}).
            Values outside these categories are excluded.
        :param at: Grid or ordered point locations on which to calculate statistics (e.g. at=reference or at="self").
            Defaults to this point cloud's locations.
        :param mask: Locations to include (True in a boolean mask, e.g. mask=points.data > 1000, or vector features).
            Global counts describe values before this mask; "all" adds counts for values kept by the mask.
        :param mask_mode: Keep locations "inside" or "outside" vector features; ignored for boolean masks.
        :param subsample: Fraction (e.g. 0.1 for 10%) or maximum count (e.g. 10000) of eligible locations to use.
            A value of 1 keeps all locations. Counts describe the sampled locations.
        :param subsample_per_group: Apply subsample within each combined group (True, stratified sampling) or once
            across all groups (False). Without by, both use one global sample.
        :param random_state: Seed to reproduce subsampling (e.g. 42), or an existing random generator.
        :param strategy: Combine chunk statistics for all groups ("dense"), only groups present in each chunk
            ("sparse"), or gather each complete group ("groupwise"). "auto" chooses from the statistics and group count;
            exact quantiles and custom functions require "auto" or "groupwise" for chunked data.
        :param backend: Use the GeoUtils reducer ("geoutils") or optional Flox reducer ("flox") for grouped statistics;
            see stats() for the Flox restrictions.
        :param subsampling_strategy: "topk" keeps the same sampled locations across chunk layouts for a fixed seed;
            "sequential" draws random locations using traversal order and can depend on the chunks.
        :param interpolation: Raster values at point locations use interp_points() with SciPy methods "nearest",
            "linear", "slinear", "cubic", "quintic", "pchip" or "splinef2d".
            Raster groupers listed in categories use "nearest".
        :param align: "raise" rejects different grids or coordinate systems; "reproject" aligns them to the output
            locations. Point inputs must still share the same ordered coordinates.
        :param observed: Omit declared group combinations with no eligible locations (True), or include them (False).
        :param return_masks: Also return masks keyed by group labels (e.g. table, masks = points.stats(...)).
            Masks cover complete groups before subsampling. Requires by.
        :param mp_config: Worker and tile settings for multiprocessing, e.g. MultiprocConfig(chunks=512).
            Cannot be combined with Dask inputs.
        :returns: A statistic, summary dictionary, grouped dataframe, or grouped dataframe and mask mapping.
        """

        return _stats(
            self,
            statistics,
            by=by,
            values=values,
            bins=bins,
            categories=categories,
            at=at,
            mask=mask,
            mask_mode=mask_mode,
            subsample=subsample,
            subsample_per_group=subsample_per_group,
            random_state=random_state,
            strategy=strategy,
            backend=backend,
            subsampling_strategy=subsampling_strategy,
            interpolation=interpolation,
            align=align,
            observed=observed,
            return_masks=return_masks,
            mp_config=mp_config,
        )

    @profiler.profile("geoutils.pointcloud.base.get_stats", memprof=True)
    def get_stats(
        self,
        stats_name: (
            str
            | Callable[[NDArrayNum], np.floating[Any]]
            | Iterable[str | Callable[[NDArrayNum], np.floating[Any]]]
            | None
        ) = None,
    ) -> Any:
        """Call stats() with the legacy argument names; deprecated in favor of stats()."""

        warnings.warn("get_stats() is deprecated; use stats() instead.", DeprecationWarning, stacklevel=2)
        return _stats(self, statistics=stats_name)

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[False] = False,
        *,
        random_state: int | np.random.Generator | None = None,
        mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
    ) -> NDArrayNum: ...

    @overload
    def subsample(
        self,
        subsample: int | float,
        return_indices: Literal[True],
        *,
        random_state: int | np.random.Generator | None = None,
        mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
    ) -> tuple[NDArrayNum, ...]: ...

    @overload
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
        *,
        mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]: ...

    @profiler.profile("geoutils.pointcloud.base.subsample", memprof=True)
    def subsample(
        self,
        subsample: float | int,
        return_indices: bool = False,
        random_state: int | np.random.Generator | None = None,
        *,
        mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
    ) -> NDArrayNum | tuple[NDArrayNum, ...]:
        """
        Randomly sample finite point cloud values allowed by mask, without replacement.

        :param subsample: Fraction of eligible finite values to sample when at most 1, otherwise the maximum number
            of values. The mask is applied before calculating this size.
        :param return_indices: Whether to return sampled row positions instead of values.
        :param random_state: Random generator or seed used to make sampling reproducible.
        :param mask: Eligible points: True in a boolean array or spatial mask, or inside vector geometries.
            Arrays must have one entry per point. Point masks must follow the same ordered coordinates;
            raster masks use nearest interpolation. Point and raster masks must share this point cloud's CRS.
            Missing mask entries are excluded (e.g. mask=points.data > 0).
        :returns: One-dimensional NumPy values with the source dtype, or a one-element tuple of indices into the
            original row order. These indices are positions, independent of any dataframe index labels.
        """

        return _subsample_pointcloud(
            source_pointcloud=self,
            subsample=subsample,
            return_indices=return_indices,
            random_state=random_state,
            mask=mask,
        )

    def cosample(
        self,
        other: RasterLike | PointCloudLike | ArrayLike,
        *,
        other_band: int = 1,
        auxiliary: Mapping[str, Any] | None = None,
        auxiliary_at: Literal["self", "other"] | Mapping[str, Literal["self", "other"]] | None = None,
        at: Literal["self", "other"] | RasterLike | PointCloudLike | None = None,
        mask: RasterLike | VectorLike | ArrayLike | None = None,
        mask_mode: Literal["inside", "outside"] = "inside",
        subsample: int | float = 1,
        random_state: int | np.random.Generator | None = None,
        strategy: Literal["sequential", "topk"] = "topk",
        raster_point_mode: Literal["grid_points", "resample_raster"] | None = None,
        grid_method: GriddingMethod = "linear",
        resample_method: InterpolationMethod | Literal["reduce"] = "linear",
        grid_kwargs: Mapping[str, Any] | None = None,
        resample_kwargs: Mapping[str, Any] | None = None,
        align: Literal["raise", "reproject"] = "raise",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterLike | PointCloudLike:
        """
        Sample this point cloud and another dataset at common finite locations.

        This point cloud provides the default spatial support. Use ``raster_point_mode="grid_points"`` to grid
        point values onto a raster input instead. An explicit ``at`` chooses the exact output locations and must
        agree with any explicit mode. Raw auxiliary arrays must identify their primary input's grid or point ordering.

        Spatial inputs, explicit output support and raster or point masks must use one family: Raster/PointCloud
        objects, or DataArray/GeoDataFrame objects. The latter may mix eager and Dask storage. Plain arrays and
        vector outlines are accepted with either family.

        :param other: Dataset to sample alongside this point cloud. A plain array follows this point cloud's order.
        :param other_band: Band selected from other if it is a raster, counting from one.
        :param auxiliary: Additional values by output name (e.g. {"slope": slope_raster}). Select a raster band with
            {"slope": (slope_raster, 2)} or a point column with {"intensity": (points, "intensity")}. Spatial inputs
            without a selector use the first raster band or active point values.
        :param auxiliary_at: Input locations followed by plain auxiliary arrays: "self", "other", or a choice per name
            (e.g. {"slope": "other"}). Spatial auxiliaries use their own coordinates.
        :param at: Output locations: "self", "other", or a reference raster/point cloud. Defaults to this point cloud.
            Point inputs must share the selected point order; "grid_points" selects a raster grid instead.
        :param mask: Locations eligible for sampling, defined by a boolean array, spatial mask, or vector outlines.
        :param mask_mode: Whether a vector mask keeps locations "inside" or "outside" its geometries.
        :param subsample: Fraction of common finite locations (e.g. 0.1), or maximum count (e.g. 1000); 1 keeps all.
        :param random_state: Seed or random generator for reproducible sampling (e.g. 42).
        :param strategy: Raster sampling with "topk" or "sequential"; "topk" keeps the same seeded sample across chunk
            sizes. Point output always uses "sequential".
        :param raster_point_mode: Conversion direction: "grid_points" places points on a raster, "resample_raster"
            reads rasters at points. Defaults to at's locations, or this point cloud's locations. Must agree with at.
        :param grid_method: Point gridding by SciPy interpolation ("nearest", "linear", "cubic"), or circular "idw",
            "mean", "minimum", "maximum", "range", "count", "stdev", "average_distance", "average_distance_pts".
            The aliases "average", "min" and "max" select "mean", "minimum" and "maximum".
        :param resample_method: Raster interpolation using the SciPy methods "nearest", "linear", "cubic", "quintic",
            "slinear", "pchip" or "splinef2d". Window reduction ("reduce") is not implemented.
        :param grid_kwargs: Options for PointCloud.grid(), e.g. {"dist_nodata_pixel": 2, "min_points": 3} sets a
            two-pixel radius and minimum of three finite points for circular methods. Other options include
            "distance_power" for IDW and "engine" ("scipy" or "numba").
            Set locations and method with at and grid_method.
        :param resample_kwargs: Options for Raster.interp_points(), e.g. {"nodata_propagation": "ignore"}. The nodata
            policies are "gdal", "ignore" and "propagate"; "dist_nodata_spread" controls extra spreading in pixels.
            Set locations, band and method with the corresponding cosample() arguments.
        :param align: Handling of mismatched grids or coordinate systems: "raise" an error, or "reproject" to match at.
            Point inputs must still share the same ordered coordinates when sampled at points.
        :param mp_config: Worker and tile settings for multiprocessing. Raster output uses its outfile; cannot be
            combined with Dask inputs.
        :returns: Raster or point cloud on the selected support; Xarray DataArray or eager/lazy GeoDataFrame for
            accessor calls. Bands or columns contain "self", "other", then auxiliaries in mapping order.
            Raster outputs retain the target grid with a common mask; point outputs retain selected geometries
            and index labels, with "self" as the active data column.
        """

        from geoutils.sampling.cosampling import _cosample

        return _cosample(
            self,
            other,
            band=1,
            other_band=other_band,
            auxiliary=auxiliary,
            auxiliary_at=auxiliary_at,
            at=at,
            mask=mask,
            mask_mode=mask_mode,
            subsample=subsample,
            random_state=random_state,
            strategy=strategy,
            raster_point_mode=raster_point_mode,
            grid_method=grid_method,
            resample_method=resample_method,
            grid_kwargs=grid_kwargs,
            resample_kwargs=resample_kwargs,
            align=align,
            mp_config=mp_config,
        )

    def pairsample(
        self,
        *,
        n_pairs: int = 1_000_000,
        sampling: Literal["loglag", "random_xy"] = "loglag",
        min_distance: float | None = None,
        max_distance: float | None = None,
        random_state: int | np.random.Generator | None = None,
        mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
        strategy: Literal["kdtree", "hashgrid", "nn_logvector"] = "nn_logvector",
        n_bins: int = 24,
        anchors_per_round: int = 50_000,
        attempts_per_anchor: int = 1,
        max_rounds: int = 50,
        cell_size: float | None = None,
        nn_tolerance: float = 0.1,
        nn_batch_size: int = 250_000,
        nn_oversample: float = 2.0,
        nn_max_batches: int = 200,
        index_dtype: DTypeLike = np.int32,
        distance_dtype: DTypeLike = np.float32,
        mp_config: MultiprocConfig | None = None,
    ) -> xr.Dataset:
        """Sample finite point pairs for statistics by distance.

        Exact ring strategies use a KD-tree or hash grid. ``"nn_logvector"`` proposes isotropic log-spaced vectors
        and accepts a nearby observed endpoint, which is generally faster for large point clouds.

        Strategy controls apply to ``"loglag"``. ``"random_xy"`` uses ``max_rounds`` and ``nn_batch_size``.
        Dask and Multiprocessing point tables are collected because the search requires all coordinates.

        :param n_pairs: Requested number of pairs with two finite values; fewer may be returned if sampling stops early.
        :param sampling: ``"loglag"`` balances short and long distances on a log scale; ``"random_xy"`` draws
            endpoints uniformly.
        :param min_distance: Smallest distance in CRS units (e.g. meters). Defaults to half the spacing estimated
            from the eligible point density.
        :param max_distance: Largest distance in CRS units. Defaults to the bounding box diagonal of eligible points.
        :param random_state: Seed for reproducible sampling (e.g. 42).
        :param mask: Eligible points: True in a boolean array or spatial mask, or inside vector geometries.
            Arrays must have one entry per point. Point masks must follow the same ordered coordinates;
            raster masks use nearest interpolation. Point and raster masks must share this point cloud's CRS.
            Missing mask entries are excluded.
        :param strategy: GeoUtils log-lag strategy: ``"kdtree"`` uses SciPy to search distance rings, ``"hashgrid"``
            searches rings using a spatial grid, and ``"nn_logvector"`` uses SciPy to match proposed endpoints
            to nearby points.
        :param n_bins: Log-spaced distance rings used by ``"kdtree"`` and ``"hashgrid"`` (e.g. 24).
        :param anchors_per_round: First endpoints tested per round by ``"kdtree"`` and ``"hashgrid"``.
        :param attempts_per_anchor: Distance rings tried per first endpoint by ``"kdtree"`` and ``"hashgrid"``.
        :param max_rounds: Maximum rounds to fill the sample with ``"kdtree"``, ``"hashgrid"``, or ``"random_xy"``.
        :param cell_size: Grid cell width in CRS units for ``"hashgrid"``. Defaults to one eighth of max_distance.
        :param nn_tolerance: Allowed endpoint snap distance for ``"nn_logvector"``, as a fraction of the proposed
            pair distance (e.g. 0.1 allows a 10% offset).
        :param nn_batch_size: Maximum candidate pairs per batch with ``"nn_logvector"`` or ``"random_xy"``;
            smaller batches use less temporary memory.
        :param nn_oversample: Candidate count as a multiple of the remaining pairs with ``"nn_logvector"`` (e.g. 2).
        :param nn_max_batches: Maximum batches to fill the sample with ``"nn_logvector"``.
        :param index_dtype: Integer NumPy dtype for returned row indexes (e.g. ``"int64"`` for very large point clouds).
        :param distance_dtype: Floating NumPy dtype for returned distances (e.g. ``"float64"`` for greater precision).
        :param mp_config: Worker and row partition settings for reading an unloaded point cloud. Cannot be combined
            with Dask inputs. The global point search and returned Xarray Dataset are eager.
        :returns: Xarray Dataset with pair and endpoint dimensions, containing original row indexes, values,
            coordinates, and distances.
        """

        from geoutils.sampling.pairsampling import _sample_point_pairs

        return _sample_point_pairs(
            self,
            n_pairs=n_pairs,
            sampling=sampling,
            min_distance=min_distance,
            max_distance=max_distance,
            random_state=random_state,
            mask=mask,
            strategy=strategy,
            n_bins=n_bins,
            anchors_per_round=anchors_per_round,
            attempts_per_anchor=attempts_per_anchor,
            max_rounds=max_rounds,
            cell_size=cell_size,
            nn_tolerance=nn_tolerance,
            nn_batch_size=nn_batch_size,
            nn_oversample=nn_oversample,
            nn_max_batches=nn_max_batches,
            index_dtype=index_dtype,
            distance_dtype=distance_dtype,
            mp_config=mp_config,
        )

    def variogram(
        self,
        *,
        n_pairs: int = 1_000_000,
        sampling: Literal["loglag", "random_xy"] = "loglag",
        estimator: str | Callable[[NDArrayNum], float] = "dowd",
        bins: Literal["log", "uniform"] | Iterable[float] = "log",
        n_lags: int = 24,
        min_lag: float | None = None,
        max_lag: float | None = None,
        n_runs: int = 1,
        model: str | Callable[..., Any] | list[str | Callable[..., Any]] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        random_state: int | np.random.Generator | None = None,
        mask: VectorLike | ArrayLike | None = None,
        **pair_sampling_kwargs: Any,
    ) -> Variogram:
        """Estimate a lightweight empirical variogram from point pairs.

        :param n_pairs: Number of finite pairs targeted in each run (e.g. 100_000).
        :param sampling: How to select pairs: ``"loglag"`` balances short and long distances, while ``"random_xy"``
            selects endpoints independently.
        :param estimator: Semivariance estimator from SciKit-GStat: ``"dowd"``, ``"matheron"``, ``"cressie"``,
            ``"genton"``, ``"minmax"``, ``"entropy"`` or ``"percentile"``. A function can instead map absolute pair
            differences to one value per distance bin.
        :param bins: Distance bins: ``"log"`` for logarithmic spacing, ``"uniform"`` for equal widths, or explicit
            edges (e.g. [1, 10, 100]).
        :param n_lags: Number of distance bins when bins is ``"log"`` or ``"uniform"``.
        :param min_lag: Minimum sampled distance in CRS units; defaults to half the spacing estimated from density.
        :param max_lag: Maximum sampled distance in CRS units; defaults to the extent diagonal of eligible points.
        :param n_runs: Independent samples to average; repeat sampling to estimate each distance bin's standard error.
        :param model: SciKit-GStat model to fit: ``"spherical"``, ``"exponential"``, ``"gaussian"``, ``"cubic"``,
            ``"stable"`` or ``"matern"``, or the corresponding model function. Sum a list of models ordered from short
            to long range (e.g. ["spherical", "exponential"]). ``None`` keeps only the empirical variogram.
        :param fit_kwargs: Options for Variogram.fit(): ``use_nugget``, ``bounds``, ``p0`` or ``maxfev``
            (e.g. {"use_nugget": True}); optimization uses SciPy curve_fit().
        :param random_state: Seed or NumPy generator for reproducible sampling across runs (e.g. 42).
        :param mask: Points to keep: True values in a boolean mask or points inside vector geometries.
        :param pair_sampling_kwargs: Extra pairsample() options (e.g. ``strategy`` or ``max_rounds``).
        :returns: Variogram with distance bins, pair counts and semivariance, plus sampling errors and a fitted model
            when requested.
        """

        return _variogram(
            self,
            n_pairs=n_pairs,
            sampling=sampling,
            n_runs=n_runs,
            estimator=estimator,
            bins=bins,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
            model=model,
            fit_kwargs=fit_kwargs,
            random_state=random_state,
            mask=mask,
            **pair_sampling_kwargs,
        )

    @overload
    def reproject(
        self: PointCloudBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[False] = False,
        mp_config: MultiprocConfig | None = None,
    ) -> PointCloudBaseType | gpd.GeoDataFrame: ...

    @overload
    def reproject(
        self: PointCloudBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: Literal[True],
        mp_config: MultiprocConfig | None = None,
    ) -> None: ...

    @overload
    def reproject(
        self: PointCloudBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        *,
        inplace: bool = False,
        mp_config: MultiprocConfig | None = None,
    ) -> PointCloudBaseType | gpd.GeoDataFrame | None: ...

    @profiler.profile("geoutils.pointcloud.base.reproject", memprof=True)
    def reproject(
        self: PointCloudBaseType,
        ref: RasterLike | VectorLike | None = None,
        crs: CRS | str | int | None = None,
        inplace: bool = False,
        *,
        mp_config: MultiprocConfig | None = None,
    ) -> PointCloudBaseType | gpd.GeoDataFrame | None:
        """
        Reproject point coordinates, preserving their order and value columns.

        Without multiprocessing, eager inputs return eager results and Dask inputs remain lazy. Multiprocessing
        reads and writes row partitions, keeping file-backed PointCloud inputs and results unloaded. LAS/LAZ
        output rounds coordinates to its stored precision; GeoPackage preserves floating-point coordinates.
        Reopened indices follow the file format. LAS attributes must fit their dimension types; GeoPackage
        requires millisecond timestamps and nullable integers that remain exact when read as float64.

        :param ref: Raster or vector whose CRS should be matched; mutually exclusive with ``crs``.
        :param crs: Target coordinate reference system; mutually exclusive with ``ref``.
        :param inplace: Update this object for eager execution. Unsupported with Dask or multiprocessing.
        :param mp_config: Worker configuration with an integer number of points per chunk. The output format is
            inferred from ``outfile`` or selected by ``driver`` (``GPKG``, ``LAS`` or ``LAZ``), defaulting to
            GeoPackage. Cannot be combined with Dask input.
        :returns: Reprojected PointCloud or GeoDataFrame matching the input interface, or None when in place.
            Multiprocessing PointCloud results are unloaded; dataframe accessor results are eager.
        """

        # Keep the shared vector implementation for eager and lazy dataframe transformations
        if mp_config is None:
            return super().reproject(ref=ref, crs=crs, inplace=inplace)
        if self._is_dask:
            raise ValueError("Argument ``mp_config`` cannot be combined with a Dask point cloud.")
        if inplace:
            raise ValueError("Argument ``inplace`` is not supported with ``mp_config``; use the returned point cloud.")

        # Resolve the target without reading point data, then let workers build the output file
        from geoutils.pointcloud.transformation import _reproject_pointcloud

        target_crs = _get_reproject_crs(ref=ref, crs=crs)
        projected = _reproject_pointcloud(self, crs=target_crs, mp_config=mp_config)
        if self._is_pd:
            # Read every output attribute and use native LAS Z when the file represents heights as a column
            projected.load(columns="all")
            return _build_pointcloud_output(
                projected.ds,
                data_column=projected.data_column,
                as_dataframe=True,
                attrs=_get_dataframe_attrs(self.ds),
            )
        return cast(PointCloudBaseType, projected)

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
        data_column: str | None = None,
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

        :param ref: Reference raster whose grid should be matched. A Dask reference also selects lazy output.
        :param grid_coords: Regular X and Y coordinates defining the output grid.
        :param res: Output resolution in X and Y, mutually exclusive with ``shape``.
        :param shape: Output shape as ``(height, width)``, mutually exclusive with ``res``.
        :param bounds: Output bounds as ``(left, bottom, right, top)``. Defaults to the point-cloud bounds.
        :param resampling: Interpolation, circular statistic or distance method. ``average``, ``min`` and ``max`` are
            aliases for ``mean``, ``minimum`` and ``maximum``.
        :param dist_nodata_pixel: Maximum point distance or circular neighborhood radius in output pixels.
        :param nodata: Nodata value of the output raster.
        :param data_column: Point value column to grid. None uses the active point values.
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
                data_column=data_column,
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
