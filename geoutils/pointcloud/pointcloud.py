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
"""Module for PointCloud class."""

from __future__ import annotations

import os.path
import pathlib
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
    cast,
    overload,
)

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from rasterio.coords import BoundingBox
from shapely.geometry.base import BaseGeometry

from geoutils import profiler
from geoutils._dispatch import (
    get_geo_attr,
    has_geo_attr,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.base import PointCloudBase
from geoutils.pointcloud.las import (
    _load_laspy_data_partitions,
    _point_partition_size,
    _write_laspy,
    load_laspy_data,
    load_laspy_metadata,
)
from geoutils.vector.vector import Vector, VectorLike

if TYPE_CHECKING:
    import matplotlib

    from geoutils.raster.base import RasterLike

# This is a generic Vector-type (if subclasses are made, this will change appropriately)
PointCloudType = TypeVar("PointCloudType", bound="PointCloud")
PointCloudLike = Union["PointCloud", gpd.GeoDataFrame]

# List of NumPy "array" functions that are handled.
# Note: all universal function are supported: https://numpy.org/doc/stable/reference/ufuncs.html
# Array functions include: NaN math and stats, classic math and stats, logical, sorting/counting:
_HANDLED_FUNCTIONS_1NIN = (
    # NaN math: https://numpy.org/doc/stable/reference/routines.math.html
    # and NaN stats: https://numpy.org/doc/stable/reference/routines.statistics.html
    [
        "nansum",
        "nanmax",
        "nanmin",
        "nanargmax",
        "nanargmin",
        "nanmean",
        "nanmedian",
        "nanpercentile",
        "nanvar",
        "nanstd",
        "nanprod",
        "nancumsum",
        "nancumprod",
        "nanquantile",
    ]
    # Classic math and stats (same links as above)
    + [
        "sum",
        "amax",
        "amin",
        "max",
        "min",
        "argmax",
        "argmin",
        "mean",
        "median",
        "percentile",
        "var",
        "std",
        "prod",
        "cumsum",
        "cumprod",
        "quantile",
        "abs",
        "absolute",
    ]
    # Sorting, searching and counting: https://numpy.org/doc/stable/reference/routines.sort.html
    + ["sort", "count_nonzero", "unique"]
    # Logic functions: https://numpy.org/doc/stable/reference/routines.logic.html
    + ["all", "any", "isfinite", "isinf", "isnan", "logical_not"]
)

_HANDLED_FUNCTIONS_2NIN = [
    "logical_and",
    "logical_or",
    "logical_xor",
    "allclose",
    "isclose",
    "array_equal",
    "array_equiv",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",
]
handled_array_funcs = _HANDLED_FUNCTIONS_1NIN + _HANDLED_FUNCTIONS_2NIN


def _cast_numeric_array_pointcloud(
    pc: PointCloudType, other: PointCloudType | NDArrayNum | Number | Any, operation_name: str
) -> Any:
    """Cast point-cloud arithmetic inputs to compatible arrays or scalar values."""

    if isinstance(other, PointCloud):
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
    if is_dask_array(other) or is_dask_dataframe(other):
        return other
    raise NotImplementedError(
        f"Operation between an object of type {type(other)} and a point cloud impossible. Must be a point cloud, "
        f"np.ndarray or single number."
    )


class PointCloud(PointCloudBase, Vector):  # type: ignore[misc]
    """
    The georeferenced point cloud.

    A point cloud is a vector of 2D point geometries associated to numeric values from a main data column, and can
    also contain auxiliary data columns.

     Main attributes:
        ds: :class:`geopandas.GeoDataFrame`
            Geodataframe of the point cloud.
        data_column: str
            Name of point cloud data column.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the point cloud.
        bounds: :class:`rio.coords.BoundingBox`
            Coordinate bounds of the point cloud.


    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    @profiler.profile("geoutils.pointcloud.pointcloud.__init__", collect=False)
    def __init__(
        self,
        filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
        data_column: str | None = None,
    ):
        """
        Instantiate a point cloud from either a data column name and a vector (filename, GeoPandas dataframe or series,
        or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to vector file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: Name of main data column defining the point cloud (not required for LAS/LAZ formats).
        """

        self._ds: gpd.GeoDataFrame | None = None
        self._name: str | None = None
        self._crs: CRS | None = None
        self._data_column: str | None = None
        self._bounds: BoundingBox
        self._columns: pd.Index | None = None
        self._feature_count: int | None = None
        self._geometry_type: str | None = None
        self._data: NDArrayNum
        self._nb_points: int
        self.__nongeo_columns: pd.Index
        self._is_las = False

        # If PointCloud is passed, simply point back to PointCloud
        if isinstance(filename_or_dataset, PointCloud):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # For filename, rely on parent Vector class or LAS file reader
        else:
            if isinstance(filename_or_dataset, (str, pathlib.Path)) and os.path.splitext(
                os.fspath(filename_or_dataset)
            )[-1] in [
                ".las",
                ".laz",
            ]:

                self._is_las = True
                # No need to pass a data column for LAS/LAZ file, as Z is the logical default
                if data_column is None:
                    data_column = "Z"
                # Load only metadata, and not the data
                fn = os.fspath(filename_or_dataset)
                metadata = load_laspy_metadata(fn)
                self._name = fn
                self._crs = metadata.crs
                self._nb_points = metadata.point_count
                self.__nongeo_columns = metadata.columns
                self._bounds = metadata.bounds
                self._columns = pd.Index(list(metadata.columns) + ["geometry"])
                self._feature_count = metadata.point_count
                self._geometry_type = "Point"
                self._ds = None
            # Check on filename are done with Vector.__init__
            else:
                super().__init__(filename_or_dataset)
                if not self.is_loaded:
                    if self._geometry_type is not None and "Point" not in self._geometry_type:
                        raise ValueError(
                            "This vector file contains non-point geometries, "
                            "cannot be instantiated as a point cloud."
                        )
                    self.__nongeo_columns = pd.Index([c for c in Vector.columns.fget(self) if c != "geometry"])
                    self._nb_points = self._feature_count if self._feature_count is not None else -1
                elif not all(p == "Point" for p in self.ds.geom_type):
                    raise ValueError(
                        "This vector file contains non-point geometries, " "cannot be instantiated as a point cloud."
                    )

        # Set data column name based on user input
        self.set_data_column(new_data_column=data_column)

    ##############################################
    # OVERRIDDEN VECTOR METHODS TO SUPPORT LOADING
    ##############################################

    @property
    def ds(self) -> gpd.GeoDataFrame:
        """Geodataframe of the point cloud."""
        # We need to override the Vector method to introduce the is_loaded dynamic for LAS files
        if not self.is_loaded:
            self.load()
        return self._ds  # type: ignore

    @ds.setter
    def ds(self, new_ds: gpd.GeoDataFrame | gpd.GeoSeries) -> None:
        """Set a new geodataframe for the point cloud."""
        # We need to override the setter Vector method because we have overridden the property method
        # (even if the code below is the same)
        if isinstance(new_ds, gpd.GeoDataFrame):
            self._ds = new_ds
        elif isinstance(new_ds, gpd.GeoSeries):
            self._ds = gpd.GeoDataFrame(geometry=new_ds)
        else:
            raise ValueError("The dataset of a vector must be set with a GeoSeries or a GeoDataFrame.")
        self._set_metadata_from_ds(self._ds)

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the vector."""

        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().crs
        return self._crs

    @property
    def bounds(self) -> BoundingBox:
        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().bounds
        return self._bounds

    @property
    def columns(self) -> pd.Index:
        # Overriding method in Vector in case dataset is not loaded
        if self.is_loaded:
            return super().columns
        if self._is_las:
            # Return columns on disk (adding a placeholder geometry to replace X/Y)
            return pd.Index(list(self._nongeo_columns) + ["geometry"])
        return Vector.columns.fget(self)

    #####################################
    # METHODS SPECIFIC TO POINT CLOUD
    #####################################

    @property
    def _nongeo_columns(self) -> pd.Index:
        """Columns of the point cloud excluding the column of 2D point geometries."""
        # Overriding method in Vector
        if self.is_loaded:
            nongeo_columns = super().columns
            nongeo_columns = nongeo_columns[nongeo_columns != "geometry"]
            return nongeo_columns
        return self.__nongeo_columns

    def load(
        self,
        columns: Literal["all", "main"] | list[str] = "main",
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Load point cloud from disk (only supported for LAS files).

        :param columns: Columns to load. Defaults to main data column only.
        :param mp_config: Optional multiprocessing configuration to load LAS/LAZ files by chunks.
        :param kwargs: Optional keyword arguments passed to :func:`geopandas.read_file` for non-LAS files.
        """

        if self.is_loaded:
            raise ValueError("Data are already loaded.")

        if self.name is None:
            raise AttributeError(
                "Cannot load as filename is not set anymore. Did you manually update the filename attribute?"
            )

        if not self._is_las:
            Vector.load(self, **kwargs)
            if not all(p == "Point" for p in self.ds.geom_type):
                raise ValueError(
                    "This vector file contains non-point geometries, cannot be instantiated as a point cloud."
                )
            self.set_data_column(new_data_column=self._data_column)
            return

        if columns == "all":
            columns_to_load = self._nongeo_columns
        elif columns == "main":
            columns_to_load = [self.data_column]
        else:
            columns_to_load = columns

        if mp_config is None:
            ds = load_laspy_data(filename=self.name, columns=columns_to_load, data_column=self.data_column)
        else:
            ds = _load_laspy_data_partitions(
                filename=self.name,
                columns=columns_to_load,
                point_count=self.point_count,
                partition_size=_point_partition_size(mp_config),
                mp_config=mp_config,
            )
        self._ds = ds

    @overload
    def astype(
        self: PointCloud,
        dtype: DTypeLike,
        convert_coords: bool = False,
        *,
        inplace: Literal[False] = False,
    ) -> PointCloud: ...

    @overload
    def astype(
        self: PointCloud,
        dtype: DTypeLike,
        convert_coords: bool = False,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def astype(
        self: PointCloud,
        dtype: DTypeLike,
        convert_coords: bool = False,
        *,
        inplace: bool = False,
    ) -> PointCloud | None: ...

    def astype(
        self: PointCloud,
        dtype: DTypeLike,
        convert_coords: bool = False,
        inplace: bool = False,
    ) -> PointCloud | None:
        """
        Convert data type of the point cloud data column.

        :param dtype: Any numpy dtype or string accepted by numpy.astype.
        :param convert_coords: Whether to convert the data type of coordinates values as well.
        :param inplace: Whether to modify the point cloud in-place.

        :returns: Point cloud with updated dtype (or None if inplace).
        """

        out_data = self.data.astype(dtype)

        if inplace:
            self._data = out_data  # type: ignore
            if convert_coords:
                self.ds.geometry.x = self.ds.geometry.x.values.astype(dtype)
                self.ds.geometry.y = self.ds.geometry.y.values.astype(dtype)
            return None
        else:
            if convert_coords:
                x = self.ds.geometry.x.values.astype(dtype)
                y = self.ds.geometry.y.values.astype(dtype)
                return self.from_xyz(x=x, y=y, z=out_data, crs=self.crs, data_column=self.data_column)
            else:
                return self.copy(new_array=out_data)

    def to_las(
        self,
        filename: str | pathlib.Path,
        version: Any = None,
        point_format: Any = None,
        offsets: tuple[float, float, float] | None = None,
        scales: tuple[float, float, float] | None = None,
        chunks: int | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Write the point cloud to LAS/LAZ/COPC file.

        :param filename: Name of output file.
        :param version: LAS/LAZ/COPC version.
        :param point_format: Point format.
        :param offsets: Offsets for X/Y/Z.
        :param scales: Scales for X/Y/Z.
        :param chunks: Optional number of points per write chunk.
        :param mp_config: Optional multiprocessing configuration for chunked writing.
        :param kwargs: Other keyword arguments to set the LAS file header (e.g., "offsets", "scales").
        """

        _write_laspy(
            filename=filename,
            pc=self.ds,
            data_column=self.data_column,
            version=version,
            point_format=point_format,
            offsets=offsets,
            scales=scales,
            chunks=chunks,
            mp_config=mp_config,
            **kwargs,
        )

    def __getitem__(self, index: PointCloud | NDArrayBool | Any) -> PointCloud | Any:
        """
        Index the point cloud.

        In addition to all index types supported by GeoPandas, also supports a point cloud mask of same georeferencing.
        """

        # If input is mask with the same shape and georeferencing
        if isinstance(index, PointCloud) or (isinstance(index, np.ndarray) and len(index) == self.point_count):
            _cast_numeric_array_pointcloud(self, index, operation_name="an indexing operation")  # type: ignore
            if isinstance(index, PointCloud):
                ind = index.data
            else:
                ind = index  # type: ignore
            ind = ind.astype(bool)  # In case the 3D Z column was used, it can only be stored as floating

            return PointCloud(self.ds.loc[ind], data_column=self.data_column)

        # Otherwise, use index and leave it to GeoPandas
        else:
            ind = index  # type: ignore
            return super().__getitem__(ind)

    def __setitem__(self, index: Any, assign: Any) -> None:
        """
        Perform index assignment on the point cloud.
        """

        # If input is mask with the same shape and georeferencing
        if isinstance(index, PointCloud) or (isinstance(index, np.ndarray) and len(index) == self.point_count):
            _cast_numeric_array_pointcloud(self, index, operation_name="an indexing operation")  # type: ignore
            # Get index
            if isinstance(index, PointCloud):
                ind = index.data
            else:
                ind = index  # type: ignore
            ind = ind.astype(bool)  # In case the 3D Z column was used, it can only be stored as floating
            # Assign
            if self._has_z:
                new_geo = gpd.points_from_xy(
                    x=self.geometry.x.values[ind],
                    y=self.geometry.y.values[ind],
                    z=assign,
                    crs=self.crs,
                )
                self.ds.loc[ind, "geometry"] = new_geo
            else:
                self.ds.loc[ind, [self.data_column]] = assign

        else:
            # Let the vector class do the job
            super().__setitem__(index, assign)

        return None

    def __array_ufunc__(
        self,
        ufunc: Callable[
            [NDArrayNum | tuple[NDArrayNum, NDArrayNum]],
            NDArrayNum | tuple[NDArrayNum, NDArrayNum],
        ],
        method: str,
        *inputs: tuple[PointCloud]
        | tuple[PointCloud, PointCloud]
        | tuple[NDArrayNum, PointCloud]
        | tuple[PointCloud, NDArrayNum],
        **kwargs: Any,
    ) -> PointCloud | tuple[PointCloud, PointCloud]:
        """
        Method to cast NumPy universal functions directly on PointCloud classes, by passing to the masked array.
        This function basically applies the ufunc (with its method and kwargs) to .data, and rebuilds the PointCloud
        from self.__class__. The cases separate the number of input nin and output nout, to properly feed .data and
        return PointCloud objects.
        See more details in NumPy doc, e.g., https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch.
        """

        # In addition to running ufuncs, this function takes over arithmetic operations (__add__, __multiply__, etc...)
        # when the first input provided is a NumPy array and second input a PointCloud.
        final_ufunc = getattr(ufunc, method)

        # If the universal function takes only one input
        if ufunc.nin == 1:
            # If the universal function has only one output
            if ufunc.nout == 1:
                return self.copy(new_array=final_ufunc(inputs[0].data, **kwargs))  # type: ignore

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(inputs[0].data, **kwargs)  # type: ignore
                return self.copy(new_array=output[0]), self.copy(new_array=output[1])

        # If the universal function takes two inputs (Note: no ufunc exists that has three inputs or more)
        else:

            # Check the casting between Point cloud and array inputs, and return error messages if not consistent

            # Raise errors if necessary
            if isinstance(inputs[0], PointCloud):
                pc = inputs[0]
                other = inputs[1]
            else:
                pc = inputs[1]  # type: ignore
                other = inputs[0]
            _ = _cast_numeric_array_pointcloud(pc, other, "an arithmetic operation")  # type: ignore

            # Get data depending on argument order
            if isinstance(inputs[0], PointCloud):
                first_arg = inputs[0].data
            else:
                first_arg = inputs[0]

            if isinstance(inputs[1], PointCloud):
                second_arg = inputs[1].data
            else:
                second_arg = inputs[1]

            # For one output
            if ufunc.nout == 1:
                return self.copy(new_array=final_ufunc(first_arg, second_arg, **kwargs))

            # If the universal function has two outputs (Note: no ufunc exists that has three outputs or more)
            else:
                output = final_ufunc(first_arg, second_arg, **kwargs)  # type: ignore
                return self.copy(new_array=output[0]), self.copy(new_array=output[1])

    def __array_function__(
        self,
        func: Callable[[NDArrayNum, Any], Any],
        types: tuple[type],
        args: Any,
        kwargs: Any,
    ) -> Any:
        """
        Method to cast NumPy array function directly on a Point cloud object by applying it to the masked array.
        A limited number of function is supported, listed in point cloud.handled_array_funcs.
        """

        # If function is not implemented
        if func.__name__ not in _HANDLED_FUNCTIONS_1NIN + _HANDLED_FUNCTIONS_2NIN:
            return NotImplemented

        # For subclassing
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        # Get first argument
        first_arg = args[0].data

        # Separate one and two input functions
        if func.__name__ in _HANDLED_FUNCTIONS_1NIN:
            outputs = func(first_arg, *args[1:], **kwargs)  # type: ignore
        # Two input functions require casting
        else:
            # Check the casting between point cloud and array inputs, and return error messages if not consistent
            if isinstance(args[0], PointCloud):
                pc = args[0]
                other = args[1]
            else:
                pc = args[1]
                other = args[0]
            _ = _cast_numeric_array_pointcloud(pc, other, operation_name="an arithmetic operation")
            second_arg = args[1].data
            outputs = func(first_arg, second_arg, *args[2:], **kwargs)  # type: ignore

        # Below, we recast to PointCloud if the shape was preserved, otherwise return an array
        # First, if there are several outputs in a tuple which are arrays
        if isinstance(outputs, tuple) and isinstance(outputs[0], np.ndarray):
            if all(output.shape == args[0].data.shape for output in outputs):
                return tuple(self.copy(new_array=output) for output in outputs)
            else:
                return outputs
        # Second, if there is a single output which is an array
        elif isinstance(outputs, np.ndarray):
            if outputs.shape == args[0].data.shape:
                return self.copy(new_array=outputs)
            else:
                return outputs
        # Else, return outputs directly
        else:
            return outputs

    def plot(  # type: ignore
        self,
        column: str | None = None,
        ref_crs: RasterLike | VectorLike | CRS | int | None = None,
        cmap: matplotlib.colors.Colormap | str | None = None,
        vmin: float | int | None = None,
        vmax: float | int | None = None,
        alpha: float | int | None = None,
        cbar_title: str | None = None,
        add_cbar: bool = True,
        ax: matplotlib.axes.Axes | Literal["new"] | None = None,
        return_axes: bool = False,
        savefig_fname: str | None = None,
        **kwargs: Any,
    ) -> None | tuple[matplotlib.axes.Axes, matplotlib.colors.Colormap]:
        """
        Plot the point cloud.

        This method is a wrapper to geopandas.GeoDataFrame.plot. Any kwargs which
        you give this method will be passed to it.

        :param column: Column to plot. Default is the data column of the point cloud.
        :param ref_crs: Coordinate reference system to match when plotting.
        :param cmap: Colormap to use. Default is plt.rcParams['image.cmap'].
        :param vmin: Colorbar minimum value. Default is data min.
        :param vmax: Colorbar maximum value. Default is data max.
        :param alpha: Transparency of raster and colorbar.
        :param cbar_title: Colorbar label. Default is None.
        :param add_cbar: Set to True to display a colorbar. Default is True if a "column" argument is passed.
        :param ax: A figure ax to be used for plotting. If None, will plot on current axes. If "new",
            will create a new axis.
        :param return_axes: Whether to return axes.
        :param savefig_fname: Path to quick save the output figure (previously created if an ax is give, new if not)
            with a default DPI, no transparency and no metadata. Use `plt.savefig()` to specify other save
            parameters or after other customizations. Warning: `plt.close()` or `plt.show()` still needs to be called
            to close the figure.

        :returns: None, or (ax, caxes) if return_axes is True
        """

        matplotlib = import_optional("matplotlib")
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Ensure that the vector is in the same crs as a reference
        if has_geo_attr(ref_crs, "crs"):
            crs = get_geo_attr(ref_crs, "crs")
            vect_reproj = self.reproject(ref=crs)
        elif isinstance(ref_crs, (CRS, int)):
            vect_reproj = self.reproject(crs=ref_crs)
        else:
            vect_reproj = self

        if column is None:
            column = self.data_column

        # Create axes, or get current ones by default (like in matplotlib)
        if ax is None:
            ax0 = plt.gca()
        elif isinstance(ax, str) and ax.lower() == "new":
            _, ax0 = plt.subplots()
        elif isinstance(ax, matplotlib.axes.Axes):
            ax0 = ax
        else:
            raise ValueError("ax must be a matplotlib.axes.Axes instance, 'new' or None.")

        # Set add_cbar depending on column argument
        if add_cbar:
            add_cbar = True
        else:
            add_cbar = False

        # Update with this function's arguments
        if add_cbar:
            legend = True
        else:
            legend = False

        if "legend" in list(kwargs.keys()):
            legend = kwargs.pop("legend")

        # Get colormap arguments that might have been passed in the keyword args
        if "legend_kwds" in list(kwargs.keys()) and legend:
            legend_kwds = kwargs.pop("legend_kwds")
            if cbar_title is not None:
                legend_kwds.update({"label": cbar_title})  # Pad updates depending on figsize during plot,
        else:
            if cbar_title is not None:
                legend_kwds = {"label": cbar_title}
            else:
                legend_kwds = None

        # Add colorbar
        if add_cbar or cbar_title:
            divider = make_axes_locatable(ax0)
            cax = divider.append_axes("right", size="5%", pad="2%")
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cbar = matplotlib.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=norm
            )  # , orientation="horizontal", ticklocation="top")
            cbar.solids.set_alpha(alpha)
        else:
            cax = None
            cbar = None

        # Plot
        vect_reproj.ds.plot(
            ax=ax0,
            cax=cax,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            legend=legend,
            legend_kwds=legend_kwds,
            **kwargs,
        )
        plt.sca(ax0)

        # if savefig_fname filled, save the plot
        if savefig_fname:
            plt.savefig(savefig_fname)

        # If returning axes
        if return_axes:
            return ax0, cax
        else:
            return None

    def __add__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Sum two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data + other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __radd__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Sum two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1 + rst).
        """
        return self.__add__(other)  # type: ignore

    def __neg__(self: PointCloud) -> PointCloud:
        """
        Take the point cloud negation.

        Returns a point cloud with -self.data.
        """
        return self.copy(-self.data)

    def __sub__(self, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Subtract two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data - other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rsub__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Subtract two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1 - rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data - self.data
        return self.copy(new_array=out_data)  # type: ignore

    def __mul__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data * other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rmul__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Multiply two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 2 * rst).
        """
        return self.__mul__(other)  # type: ignore

    def __truediv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data / other_data
        return self.copy(new_array=out_data)  # type: ignore

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rtruediv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        True division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data / self.data
        return self.copy(new_array=out_data)  # type: ignore

    def __floordiv__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data // other_data  # type: ignore
        return self.copy(new_array=out_data)

    # Skip Mypy not resolving forward operator typing with NumPy numbers: https://github.com/python/mypy/issues/11595
    def __rfloordiv__(self: PointCloud, other: NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Floor division of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        For when other is first item in the operation (e.g. 1/rst).
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = other_data // self.data  # type: ignore
        return self.copy(new_array=out_data)

    def __mod__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:
        """
        Modulo of two point clouds, or a point cloud and a numpy array, or a point cloud and single number.

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data % other_data  # type: ignore
        return self.copy(new_array=out_data)

    def __pow__(self: PointCloud, power: int | float) -> PointCloud:
        """
        Power of a point cloud to a number.
        """
        # Check that input is a number
        if not isinstance(power, (float, int, np.floating, np.integer)):
            raise ValueError("Power needs to be a number.")

        # Calculate the product of arrays and save to new point cloud
        out_data = self.data**power
        return self.copy(new_array=out_data)

    def __eq__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise equality of two point clouds, or a point cloud and a numpy array, or a point cloud and single
        number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data == other_data
        return self.copy(new_array=out_data)

    def __ne__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise negation of two point clouds, or a point cloud and a numpy array, or a point cloud and single
        number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data != other_data
        return self.copy(new_array=out_data)

    def __lt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise lower than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data < other_data
        return self.copy(new_array=out_data)

    def __le__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise lower or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data <= other_data
        return self.copy(new_array=out_data)

    def __gt__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise greater than comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data > other_data
        return self.copy(new_array=out_data)

    def __ge__(self: PointCloud, other: PointCloud | NDArrayNum | Number) -> PointCloud:  # type: ignore
        """
        Element-wise greater or equal comparison of two point clouds, or a point cloud and a numpy array,
        or a point cloud and single number.

        This operation casts the result into a mask (boolean Raster).

        If other is a point cloud, it must have the same shape, coordinates and crs as self.
        If other is a np.ndarray, it must have the same shape.
        Otherwise, other must be a single number.
        """
        other_data = _cast_numeric_array_pointcloud(self, other, operation_name="an arithmetic operation")
        out_data = self.data >= other_data
        return self.copy(new_array=out_data)

    def __and__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise and between masks, or a mask and an array."""
        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data & other_data)  # type: ignore

    def __rand__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise and between masks, or a mask and an array."""

        return self.__and__(other)

    def __or__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise or between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data | other_data)  # type: ignore

    def __ror__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise or between masks, or a mask and an array."""

        return self.__or__(other)

    def __xor__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise xor between masks, or a mask and an array."""

        other_data = _cast_numeric_array_pointcloud(
            self, other, operation_name="an arithmetic operation"  # type: ignore
        )

        return self.copy(self.data ^ other_data)  # type: ignore

    def __rxor__(self: PointCloud, other: PointCloud | NDArrayBool) -> PointCloud:
        """Bitwise xor between masks, or a mask and an array."""

        return self.__xor__(other)

    def __invert__(self: PointCloud) -> PointCloud:
        """Bitwise inversion of a mask."""

        return self.copy(~self.data)

    @overload
    def info(self, verbose: Literal[True] = ..., stats: bool = False) -> None: ...

    @overload
    def info(self, verbose: Literal[False], stats: bool = False) -> str: ...

    def info(self, verbose: bool = True, stats: bool = False) -> None | str:
        """
        Print summary information about the point cloud.

        :param stats: Add statistics for each band of the dataset (max, min, median, mean, std. dev.). Default is to
            not calculate statistics.
        :param verbose: If set to True (default) will directly print to screen and return None

        :returns: Summary string or None.
        """

        # Get vector.info()
        as_str_split = super().info(verbose=False).split("\n")  # type: ignore

        if stats:
            as_str_split.append("\nStatistics:")
            statistics = self.get_stats()

            # Determine the maximum length of the stat names for alignment
            max_len = max(len(name) for name in statistics.keys())

            # Format the stats with aligned names
            for name, value in statistics.items():
                as_str_split.append(f"{name.ljust(max_len)}: {value:.2f}")

        if verbose:
            print("\n".join(as_str_split))
            return None
        else:
            return "\n".join(as_str_split)
