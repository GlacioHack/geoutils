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

"""Read/write raster files, with optional downsampling, from local or remote files."""

from __future__ import annotations

import os
import pathlib
import posixpath
import warnings
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from io import BytesIO
from typing import IO, TYPE_CHECKING, Any, Literal, TypedDict, cast
from xml.etree import ElementTree

import affine
import numpy as np
import rasterio as rio
import rioxarray as rioxr
import xarray as xr
from affine import Affine
from rasterio.shutil import copy as rio_copy
from xarray.conventions import encode_cf_variable
from xarray.core.utils import dict_equiv

from geoutils import projtools
from geoutils._config import config
from geoutils._dispatch import _check_resolution
from geoutils._io import _filesystem_file_options, _FilesystemOpener, _is_remote_path, _resolve_filesystem
from geoutils._misc import _validate_downsample, import_optional, silence_rasterio_message
from geoutils._typing import DTypeLike, MArrayNum, NDArrayNum, Number
from geoutils.raster.referencing import _cast_nodata, _default_nodata

if TYPE_CHECKING:
    from geoutils.raster.raster import Raster

##############
# 1/ OPENING
##############


class _DownsampledRasterFilesystem(_FilesystemOpener):
    """
    A small helper class for reading a downsampled raster through a filesystem.

    We use a virtual raster (VRT) to describe the smaller grid without copying the pixels. We make it look like a file
    beside the source, so workers can open both with the same filesystem.
    """

    def __init__(self, filename: str, filesystem: Any, xml: str) -> None:
        """Set up a virtual file with relative paths to its source."""
        self.name = filename + ".geoutils.vrt"
        super().__init__(filesystem)
        source_name = posixpath.basename(filename)
        if _is_remote_path(filename):
            source_name, opener = _filesystem_file_options(filename, filesystem)
            self.filesystem = opener.filesystem
            self.name = source_name + ".geoutils.vrt"
        root = ElementTree.fromstring(xml)
        for source in root.iter("SourceFilename"):
            source.text = source_name
            source.set("relativeToVRT", "1")
        self.xml = ElementTree.tostring(root)

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """Open the virtual file from memory, or read a file from the filesystem."""
        if path == self.name:
            return BytesIO(self.xml)
        return super().open(path, mode, **kwargs)

    def isfile(self, path: str) -> bool:
        """Check whether a stored or virtual file exists."""
        return path == self.name or self.filesystem.isfile(path)

    def size(self, path: str) -> int:
        """Return the size of a stored or virtual file in bytes."""
        return len(self.xml) if path == self.name else self.filesystem.size(path)


class RasterDownsampleOptions(TypedDict, total=False):
    """Opening options for raster downsampling: specifying exactly one of "factor", "res" or "shape"."""

    factor: Number
    res: Number | tuple[Number, Number]
    shape: tuple[int, int]


def _downsample_grid(
    source: rio.io.DatasetReader, downsample: Number | RasterDownsampleOptions
) -> tuple[tuple[int, int], affine.Affine, tuple[float, float]]:
    """
    Calculate output shape, transform and column/row factors for a downsample read.

    Shared by Raster and open_raster() for the rst Xarray accessors.
    """

    # 1/ An integer is a downsampling factor
    options: RasterDownsampleOptions = downsample if isinstance(downsample, dict) else {"factor": downsample}
    targets = set(options) & {"factor", "res", "shape"}
    if len(targets) != 1 or set(options) - {"factor", "res", "shape"}:
        raise ValueError("downsample must specify exactly one of factor, res or shape.")

    # 2/ Otherwise can be "shape" or "res" through a dictionary

    # Shape performs nearest match over the full extent
    if "shape" in options:
        shape = options["shape"]
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 for size in shape)
        ):
            raise ValueError("downsample shape must contain two positive integers (height, width).")
        height, width = shape
        x_factor, y_factor = source.width / width, source.height / height
    # Resolution uses CRS units and column/row order to get exact spacing, trimming the end
    else:
        if "res" in options:
            xres, yres = _check_resolution(options["res"])
            x_factor, y_factor = xres / source.res[0], yres / source.res[1]
        else:
            x_factor = y_factor = _validate_downsample(options["factor"])
        if min(x_factor, y_factor) < 1:
            raise ValueError("downsample cannot increase the number of pixels.")
        width = max(1, int(np.floor(source.width / x_factor)))
        height = max(1, int(np.floor(source.height / y_factor)))

    # We have to scale the full transform, so rotation/shear are preserved
    if min(x_factor, y_factor) < 1:
        raise ValueError("downsample cannot increase the number of pixels.")
    transform = projtools._affine_matmul(source.transform, affine.Affine.scale(x_factor, y_factor))
    return (height, width), transform, (x_factor, y_factor)


def _raster_overview_metadata(source: rio.io.DatasetReader, overview_level: int | None = None) -> dict[str, Any]:
    """
    Read available overview metadata.

    Shared by Raster and open_raster() for the rst Xarray accessors.
    """

    factors = source.overviews(1)
    selected_level = -1 if overview_level is None else int(overview_level)
    selected_factor = factors[selected_level] if selected_level >= 0 else 1
    metadata: dict[str, Any] = {
        "overview_factors": factors,
        "overview_selected_level": selected_level,
        "overview_selected_factor": selected_factor,
    }
    # Include resampling method if it exists
    resampling = source.tags(ns="rio_overview").get("resampling")
    if resampling is not None:
        metadata["overview_resampling"] = resampling
    return metadata


def _overview_level_for_downsample(source: rio.io.DatasetReader, downsample: float) -> int | None:
    """
    Return closest overview level for downsampling.

    Shared by Raster and open_raster() for the rst Xarray accessors.
    """

    # Match GDAL nearest-neighbor overview selection by allowing up to 20% oversampling
    overview_level = None
    for level, overview_factor in enumerate(source.overviews(1)):
        if overview_factor > downsample * 1.2:
            break
        overview_level = level
    return overview_level


def _downsampled_raster_xml(source: rio.io.DatasetReader, downsample: Number | RasterDownsampleOptions) -> str:
    """
    Build the XML describing a smaller raster without copying its pixels.

    Shared by Raster and open_raster() for the rst Xarray accessors.
    """

    (height, width), transform, (x_factor, y_factor) = _downsample_grid(source, downsample)

    # Copy metadata and mask references without reading pixels
    with rio.MemoryFile(ext=".vrt") as memory:
        rio_copy(source, memory.name, driver="VRT")
        root = ElementTree.fromstring(memory.read())
    root.set("rasterXSize", str(width))
    root.set("rasterYSize", str(height))
    geotransform = root.find("GeoTransform")
    if geotransform is None:
        geotransform = ElementTree.SubElement(root, "GeoTransform")
    geotransform.text = ", ".join(str(value) for value in transform.to_gdal())

    # A smaller source rectangle trims incomplete intervals when preserving resolution
    for rectangle in root.iter("SrcRect"):
        rectangle.set("xSize", str(width * x_factor))
        rectangle.set("ySize", str(height * y_factor))
    for rectangle in root.iter("DstRect"):
        rectangle.set("xSize", str(width))
        rectangle.set("ySize", str(height))

    return ElementTree.tostring(root, encoding="unicode")


@contextmanager
def _open_downsampled_raster(
    source: rio.io.DatasetReader,
    downsample: Number | RasterDownsampleOptions,
    *,
    filename: str | None = None,
    filesystem: Any = None,
) -> Iterator[rio.io.DatasetReader]:
    """
    Open a lazy downsampled grid with source metadata and masks.

    First, _downsample_grid() defines the output grid, then GDAL copies the source description into a simple virtual
    raster.

    The virtual raster lets Rioxarray reopen the reduced grid lazily with its metadata and masks.

    Shared by Raster and open_raster() for the rst Xarray accessors.
    """

    xml = _downsampled_raster_xml(source, downsample)
    open_kwargs = {}
    if filesystem is not None:
        assert filename is not None
        reduced_filesystem = _DownsampledRasterFilesystem(filename, filesystem, xml)
        xml = reduced_filesystem.name
        open_kwargs["opener"] = reduced_filesystem

    # The XML can be reopened by Rioxarray and Dask after this context closes
    with rio.open(xml, **open_kwargs) as vrt:
        yield vrt


@contextmanager
def _open_raster_dataset(
    filename: str | pathlib.Path,
    *,
    filesystem: Any = None,
    overview_level: int | None = None,
    downsample: Number | RasterDownsampleOptions = 1,
) -> Iterator[rio.io.DatasetReader]:
    """
    Open a raster file, optionally using an overview or downsampling.

    We pass the filesystem to Rasterio and use _open_downsampled_raster() when a smaller grid is requested.
    The source and virtual raster close when we leave the context.

    Shared by Raster, the rst Xarray accessors and multiprocessing readers.
    """
    open_kwargs: dict[str, Any] = {}
    open_filename = filename
    if filesystem is not None:
        open_filename, open_kwargs["opener"] = _filesystem_file_options(filename, filesystem)
    if overview_level is not None:
        open_kwargs["overview_level"] = overview_level

    # Close the virtual raster before closing its source
    with ExitStack() as stack:
        dataset = stack.enter_context(rio.open(open_filename, **open_kwargs))
        if downsample != 1:
            dataset = stack.enter_context(
                _open_downsampled_raster(dataset, downsample, filename=os.fspath(filename), filesystem=filesystem)
            )
        yield dataset


def open_raster(
    filename: str,
    is_mask: bool = False,
    downsample: Number | RasterDownsampleOptions = 1,
    *,
    chunks: int | tuple[int, ...] | dict[str, int] | Literal["auto"] | None = None,
    squeeze: bool = True,
    masked: bool = True,
    as_object_type: Literal["dataarray", "dataset"] | None = None,
    data_name: str | None = None,
    parse_sensor_metadata: bool = False,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Open a raster as a Xarray DataArray or Dataset.

    This function relies largely on rioxarray.open_rasterio(), to which keyword arguments are passed.

    By default, nodata values become NaNs so that GeoUtils operations handle them correctly. Integer rasters are
    therefore converted to floating point (float32 for 8- and 16-bit integers, float64 for 32- and 64-bit
    integers). Use ``masked=False`` at your own risk to read values at their original integer dtype:
    later operations may treat numeric nodata values (e.g., -9999) as valid data.

    The band dimension is removed for single-band rasters by default. Set ``squeeze=False`` to preserve it.

    Passing ``chunks`` opens the raster data lazily using Dask arrays.

    Use ``downsample`` to open a smaller raster, by default interpreted as a downsampling factor along both axes. For
    example, ``downsample=2`` returns about one out of 2 pixel for each axis. Alternatively, select a pixel size in CRS
    units with ``downsample={"res": 120}``, or an exact shape (height, width) with ``downsample={"shape": (64, 128)}``.
    To optimize file reads, downsampling uses only nearest-neighbor selection from the closest suitable stored
    overview (reduced copy of the raster), or from the full raster if no suitable overview exists. For other
    resampling methods, use reproject(), for example ``reproject(res=120, resampling="bilinear")``.

    Use ``overview_level`` to select a stored overview explicitly. Level 0 selects the first stored overview,
    level 1 the second, and so on. The returned array uses that overview's shape, resolution and resampling. Thus,
    ``overview_level`` cannot be combined with ``downsample``.

    See :ref:`scalability-downsampling` for examples.

    To open a remote file, pass an HTTPS URL or S3 address, such as ``"https://example.org/data.tif"`` or
    ``"s3://bucket/data.tif"``.

    Use ``storage_options`` to pass a dictionary of connection settings, such as credentials or a custom
    server address. GeoUtils creates a Fsspec filesystem for you, which handles access to files on that service.
    For example, ``storage_options={"anon": True}`` reads a public S3 bucket without login credentials.

    Use ``filesystem`` if you already have a Fsspec filesystem object and want to reuse its connection
    settings across files. Both options are optional and cannot be combined.

    These settings are preserved for deferred and chunked reads.
    See :ref:`scalability-formats` for supported formats and authentication.

    :param filename: Path to the raster file to open.
    :param is_mask: Whether to return the raster values as booleans.
    :param chunks: Chunk sizes in pixels, as an integer, a tuple of sizes for each dimension, or a dictionary such as
        {"x": 512, "y": 512}. True or "auto" chooses chunk sizes automatically. None uses deferred file reads.
    :param downsample: A downsampling factor, or a dictionary containing exactly one of "factor", "res" or "shape".
        For example, passing 2 is equivalent to {"factor": 2} and selects one pixel out of two on both axes. Passing
        {"res": 120} sets the pixel spacing to 120 CRS units, use (xres, yres) for different spacing on each axis.
        Passing {"shape": (64, 128)} reads 64 rows and 128 columns. Only "nearest" selection can be supported for
        downsample reads, use ``reproject()`` instead of other resamplings. "factor" and "res" preserve exact
        pixel spacing and can thus trim extent, while "shape" adjusts pixel spacing and thus preserves extent.
        Default 1 reads the full grid.
    :param squeeze: Remove the band dimension when it contains one band. Spatial dimensions are never removed.
    :param masked: Replace missing values with NaNs, promoting integer values to floating point. If False,
        numeric nodata values remain in the data and operations expecting NaNs may return incorrect results.
        An explicit mask_and_scale=True still decodes values through Rioxarray.
    :param as_object_type: Returned representation: dataarray or dataset. None uses config["raster_return_type"],
        which defaults to dataarray. A Dataset contains one raster variable, with multiple bands on its band dimension.
    :param data_name: Name of the DataArray or Dataset variable. Defaults to the source array name, or raster for
        an unnamed array returned as a Dataset.
    :param parse_sensor_metadata: Parse sensor metadata from the filename and add it to the raster attributes.
    :param overview_level: Stored overview level, starting at 0 for the first overview. Use -1 for the full raster.
        Cannot be combined with downsample. Default None selects an overview automatically when downsampling.
    :param filesystem: An existing Fsspec filesystem object whose connection settings are used to open files.
        Cannot be combined with storage_options.
    :param storage_options: Dictionary of connection settings used to create a Fsspec filesystem from the file URL,
        such as credentials or a custom server address. Cannot be combined with filesystem.
    :param kwargs: Keyword arguments passed to :func:`rioxarray.open_rasterio`.
    :returns: The opened raster as a DataArray or Dataset with Rioxarray georeferencing.
    """

    # 1/ Check inputs
    if not isinstance(downsample, dict):
        downsample = _validate_downsample(downsample)
    if as_object_type is None:
        as_object_type = config["raster_return_type"]
    if as_object_type not in ("dataarray", "dataset"):
        raise ValueError("Argument 'as_object_type' must be 'dataarray' or 'dataset'.")
    if downsample != 1 and kwargs.get("overview_level") is not None:
        raise ValueError("downsample and overview_level cannot be used together.")
    if not masked:
        warnings.warn(
            "masked=False allows numeric nodata values in the data. GeoUtils operations that expect NaNs "
            "may return incorrect results.",
            UserWarning,
            stacklevel=2,
        )

    # 2/ Prepare filesystem access
    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)
    open_filename = filename
    source_options = {}
    if filesystem is not None:
        if "opener" in kwargs:
            raise ValueError("Pass either filesystem or opener, not both.")
        open_filename, opener = _filesystem_file_options(filename, filesystem)
        source_options["opener"] = opener
        kwargs["opener"] = opener

    # 3/ Open the raster
    overview_level = kwargs.get("overview_level")

    # 3.1/ Open without downsampling
    if downsample == 1:
        ds = rioxr.open_rasterio(open_filename, chunks=chunks, masked=masked, **kwargs)
    # 3.2/ Open with downsampling
    # We use a simple VRT to allow reduced RasterIO reads to Rioxarray, without a reprojection warp
    else:
        context = (
            nullcontext(filename)
            if isinstance(filename, rio.io.DatasetReaderBase)
            else rio.open(open_filename, **source_options)
        )
        with context as source:
            _, _, factors = _downsample_grid(source, downsample)
            if source.gcps[0] or source.rpcs:
                raise NotImplementedError("Reproject GCP/RPC rasters to an affine grid before opening with downsample.")
            overview_level = _overview_level_for_downsample(source, min(factors))
            if filesystem is None:
                with _open_downsampled_raster(source, downsample) as vrt:
                    ds = rioxr.open_rasterio(vrt, chunks=chunks, masked=masked, **kwargs)
            else:
                # Relative VRT paths let each worker register its own source opener
                xml = _downsampled_raster_xml(source, downsample)
                reduced_filesystem = _DownsampledRasterFilesystem(str(filename), filesystem, xml)
                reduced_options = {**kwargs, "opener": reduced_filesystem}
                ds = rioxr.open_rasterio(reduced_filesystem.name, chunks=chunks, masked=masked, **reduced_options)
            ds.encoding["source"] = str(filename) if filesystem is not None else source.name

    # Xarray conversions can drop the source close callback
    opened_array = ds

    # 4/ Read source metadata

    # Store RPCs consistently across Rioxarray versions
    source_context = (
        nullcontext(filename)
        if isinstance(filename, rio.io.DatasetReaderBase)
        else rio.open(open_filename, **source_options)
    )
    with source_context as source:
        ds.encoding["driver"] = source.driver
        # Store overview metadata
        overview_metadata = _raster_overview_metadata(source, overview_level)
        disk_metadata = {"block_shape": tuple(source.block_shapes[0])}
        ds.attrs.pop("overview_resampling", None)
        ds.attrs.update(overview_metadata)
        ds.attrs.update(disk_metadata)
        if source.rpcs is not None:
            ds.attrs["RPC"] = source.rpcs.to_gdal()
        if source.gcps[0] and source.rpcs is not None:
            # Rioxarray assigns WGS84 when reading RPCs; restore the CRS of the stored GCP coordinates
            ds.rio.write_gcps(*source.gcps, inplace=True)

    # 5/ Prepare the output array

    # Remove the band dimension if there is only one (and user did not pass ``squeeze=False``)
    if squeeze and ds.sizes.get("band") == 1:
        ds = ds.squeeze("band")

    # If input is a mask, convert to bool lazily
    if is_mask:
        encoding = ds.encoding.copy()
        ds = ds.astype(bool)
        ds.encoding = encoding

    # Preserve access settings for metadata reads through the GeoUtils accessor
    ds.encoding["_filesystem"] = filesystem
    ds.encoding["_downsample"] = downsample
    ds.encoding["_overview_level"] = kwargs.get("overview_level")
    if filesystem is not None:
        ds.encoding["source"] = str(filename)

    # Parse sensor filenames without reading pixel values
    if parse_sensor_metadata:
        from geoutils.raster.satimg import parse_and_convert_metadata_from_filename

        source_name = getattr(filename, "name", filename)
        ds.attrs.update(parse_and_convert_metadata_from_filename(str(source_name), silent=True))

    # 6/ Return the requested Xarray object

    # Name the raster and wrap its coordinates and lazy data in one Dataset variable when requested
    if data_name is not None:
        ds = ds.rename(data_name)
    if as_object_type == "dataset":
        dataset = ds.to_dataset(name=ds.name or "raster")
        dataset.encoding = ds.encoding.copy()
        dataset.attrs.update(overview_metadata)
        dataset.attrs.update(disk_metadata)
        if parse_sensor_metadata:
            dataset.attrs.update(ds.attrs)
        dataset.set_close(opened_array.close)
        return dataset
    if ds is not opened_array:
        ds.set_close(opened_array.close)
    return ds


##############
# 2/ LOADING
##############


def _load_rio(
    dataset: rio.io.DatasetReader,
    only_mask: bool = False,
    convert_to_mask: bool = False,
    indexes: int | list[int] | None = None,
    masked: bool = False,
    transform: Affine | None = None,
    shape: tuple[int, int] | None = None,
    out_count: int | None = None,
    **kwargs: Any,
) -> MArrayNum:
    """
    Load raster bands or validity masks, optionally cropped or downsampled.

    A transform and shape define the read window and must be supplied together unless out_shape is passed.
    Other keyword arguments go to the dataset's read() or read_masks() method. For multiple output bands,
    out_count adds the band dimension to out_shape.

    Band indexes start at 1. A single index returns a 2D array, while a list or None includes the band dimension.
    Band reads return a masked array when masked=True, and convert_to_mask=True converts the result to booleans.

    Used only by the Raster class.
    """

    # Use the complete dataset unless a window is passed or derived from a transform
    window = kwargs.pop("window", None)

    # If out_shape is passed, no need to account for transform and shape
    if kwargs.get("out_shape") is not None:
        # If multi-band raster, the out_shape needs to contain the count
        if out_count is not None and out_count > 1:
            kwargs["out_shape"] = (out_count, *kwargs["out_shape"])
    else:
        if transform is not None and shape is not None:
            if transform == dataset.transform:
                row_off, col_off = 0, 0
            else:
                row_off, col_off = (round(val) for val in dataset.index(transform[2], abs(transform[4])))

            window = rio.windows.Window(col_off, row_off, *shape[::-1])
        elif sum(param is None for param in [shape, transform]) == 1:
            raise ValueError("If 'shape' or 'transform' is provided, BOTH must be given.")

    if indexes is None:
        if only_mask:
            data = dataset.read_masks(window=window, **kwargs)
        else:
            data = dataset.read(masked=masked, window=window, **kwargs)
    else:
        if only_mask:
            data = dataset.read_masks(indexes=indexes, window=window, **kwargs)
        else:
            data = dataset.read(indexes=indexes, masked=masked, window=window, **kwargs)

    if convert_to_mask:
        return data.astype(bool)

    return data


##############
# 3/ WRITING
##############


def _copy_preferred_chunks(ref: Any, result: Any) -> None:
    """Copy spatial preferred chunks from a reference to Xarray raster results."""

    # Accept a DataArray or its raster accessor
    ref = getattr(ref, "_obj", ref)
    if not isinstance(ref, xr.DataArray) or not isinstance(result, (xr.DataArray, xr.Dataset)):
        return
    preferred_chunks = ref.encoding.get("preferred_chunks", {})
    if not preferred_chunks:
        return

    # Exclude band chunks because the result can have different layers
    spatial_chunks = {
        axis: preferred_chunks[dimension]
        for axis, dimension in (("y", ref.rio.y_dim), ("x", ref.rio.x_dim))
        if dimension in preferred_chunks
    }
    if spatial_chunks:
        variables = [result] if isinstance(result, xr.DataArray) else result.rst.compatible_variables.values()
        for variable in variables:
            variable.encoding["preferred_chunks"] = spatial_chunks.copy()


def _set_raster_encoding(
    array: xr.DataArray,
    encoding: Mapping[str, Any],
    nodata: int | float | None,
    cast_nodata: bool = True,
) -> None:
    """
    Copy storage encoding and adapt it to the output raster nodata and grid mapping.

    Integer storage needs a fill value for decoded NaNs. We use the array's dtype if that fill value is removed or
    cannot fit in the stored dtype. Encoded fill values are assigned directly to avoid casting them to decoded floats.

    Used only by the rst Xarray accessors.
    """

    # Copy settings before editing them, using output grid mapping
    mapping = array.rio.grid_mapping
    encoded = "_FillValue" in encoding or "dtype" in encoding
    output_encoding = dict(encoding)
    if mapping in array.coords:
        output_encoding["grid_mapping"] = mapping

    # Check the stored dtype (without reading array value)
    storage_dtype = np.dtype(output_encoding.get("dtype") or array.dtype)
    with np.errstate(invalid="ignore", over="ignore"):
        compatible = nodata is None or rio.dtypes.can_cast_dtype(nodata, storage_dtype)
    if not compatible and not cast_nodata:
        raise ValueError(f"Nodata value {nodata} incompatible with self.dtype {storage_dtype}.")
    missing_without_fill = nodata is None and array.dtype.kind == "f" and storage_dtype.kind in "biu"
    if not compatible or missing_without_fill:
        for key in ("dtype", "rasterio_dtype", "_Unsigned"):
            output_encoding.pop(key, None)

    # Store encoded nodata only in encoding, unencoded arrays continue to use their attributes
    if encoded:
        array.attrs.pop("_FillValue", None)
        if nodata is None:
            output_encoding.pop("_FillValue", None)
        else:
            output_encoding["_FillValue"] = nodata
    array.rio.set_encoding(output_encoding, inplace=True)
    if not encoded:
        array.rio.write_nodata(nodata, inplace=True)


class _RasterFilesystemPath(os.PathLike[str]):
    """A small class to keep filesystem access through Rioxarray's deferred window writes."""

    def __init__(self, filename: str, filesystem: Any) -> None:
        """Store path and opener recognized by Rasterio."""
        self.path = filename
        self.fs = _FilesystemOpener(filesystem)

    def __fspath__(self) -> str:
        """Return the destination filename for format detection."""
        return self.path


def _write_rio(
    source_raster: Raster,
    filename: str | pathlib.Path | IO[bytes],
    driver: str = "GTiff",
    dtype: DTypeLike | None = None,
    nodata: Number | None = None,
    blank_value: int | float | None = None,
    co_opts: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    gcps: list[tuple[float, ...]] | list[rio.control.GroundControlPoint] | None = None,
    gcps_crs: rio.crs.CRS | None = None,
    rpcs: rio.rpc.RPC | None = None,
    *,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """
    Write a Raster array and metadata to file with Rasterio.

    Used only by the Raster class.
    """

    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)
    open_options = {"opener": _FilesystemOpener(filesystem)} if filesystem is not None else {}
    if co_opts is None:
        co_opts = {}

    # Set COMPRESS default value to DEFLATE
    if "COMPRESS" not in map(str.upper, co_opts.keys()):
        co_opts["COMPRESS"] = "DEFLATE"

    # Set BIGTIFF default value to IF_SAFER, to save the output image as a BigTIFF if it might exceed 4GB.
    if "BIGTIFF" not in map(str.upper, co_opts.keys()):
        co_opts["BIGTIFF"] = "IF_SAFER"

    meta = dict(source_raster.tags) if source_raster.tags is not None else {}
    if metadata is not None:
        meta.update(metadata)

    # RPC dictionaries belong in GDAL's RPC namespace, rather than a string in the default tags
    rpc_metadata = meta.pop("RPC", None)
    rpcs = source_raster.rpcs if rpcs is None else rpcs
    if rpcs is None and rpc_metadata is not None:
        rpcs = rio.rpc.RPC.from_gdal(rpc_metadata)

    # Normalize explicit GCPs before opening the output file, preserving the legacy tuple input
    explicit_gcps = gcps is not None
    if gcps is None:
        gcps = source_raster.gcps[0]
    if not isinstance(gcps, list):
        raise ValueError("gcps must be a list")
    rio_gcps = [
        gcp if isinstance(gcp, rio.control.GroundControlPoint) else rio.control.GroundControlPoint(*gcp) for gcp in gcps
    ]
    gcps_crs = gcps_crs or source_raster.gcps[1] or source_raster.crs
    if rio_gcps and gcps_crs is None:
        raise ValueError("gcps_crs is required when saving GCPs without a raster CRS.")
    if explicit_gcps and rio_gcps and source_raster.transform != rio.Affine.identity():
        warnings.warn(
            "A geotransform previously set is going to be cleared due to the setting of GCPs.",
            category=UserWarning,
        )

    # Use nodata set by user, otherwise default to self's
    nodata = nodata if nodata is not None else source_raster.nodata
    output_dtype = np.dtype(dtype) if dtype is not None else np.dtype(source_raster.dtype)

    # If the output is a mask, convert to uint8 before saving and force nodata to 255
    if output_dtype == np.bool_:
        output_dtype = np.dtype("uint8")
        nodata = 255

    # Declare type of save_data to work in all occurrences
    save_data: NDArrayNum

    # Define save_data depending on blank_value argument
    if (source_raster.data is None) & (blank_value is None):
        raise AttributeError("No data loaded, and alternative blank_value not set.")
    elif blank_value is not None:
        if isinstance(blank_value, int) | isinstance(blank_value, float):
            save_data = np.zeros(source_raster.data.shape, dtype=output_dtype)
            save_data[:] = blank_value
        else:
            raise ValueError("blank_values must be one of int, float (or None).")
    else:
        save_data = source_raster.data

    # Make nodata compatible with the requested file type before filling masked values
    nodata = _cast_nodata(output_dtype, cast(int | float | None, nodata))

    # If masked array, save with masked values replaced by nodata
    if isinstance(save_data, np.ma.masked_array):
        # In this case, nodata=None is not compatible, so revert to default values, only if masked values exist
        if (nodata is None) & (np.count_nonzero(save_data.mask) > 0):
            nodata = _default_nodata(output_dtype)
            warnings.warn(f"No nodata set, will use default value of {nodata}", category=UserWarning)
        # Convert masked data before filling so nodata is represented in the requested output type
        save_data = save_data.astype(output_dtype, copy=False).filled(nodata)
    else:
        save_data = save_data.astype(output_dtype, copy=False)

    # Cast to 3D before saving if single band
    if source_raster.count == 1:
        save_data = save_data[np.newaxis, :, :]

    with rio.open(
        filename,
        "w",
        driver=driver,
        height=source_raster.height,
        width=source_raster.width,
        count=source_raster.count,
        dtype=save_data.dtype,
        crs=gcps_crs if rio_gcps else source_raster.crs,
        transform=None
        if rio_gcps or (rpcs and source_raster.transform == rio.Affine.identity())
        else source_raster.transform,
        gcps=rio_gcps or None,
        rpcs=rpcs,
        nodata=nodata,
        **open_options,
        **co_opts,
    ) as dst:
        dst.write(save_data)

        # Add metadata (tags in rio)
        dst.update_tags(**meta)


def _write_raster_array(array: xr.DataArray, *args: Any, **kwargs: Any) -> Any:
    """
    Prepare a DataArray georeferencing and write its raster values.

    Used only by the rst Xarray accessors.
    """

    output = array
    rpcs = kwargs.pop("rpcs", None)
    if rpcs is None:
        rpcs = array.rst.rpcs
    if rpcs is not None:
        # RPC metadata is written by Rasterio, so we omit it from the default tags
        output = output.copy(deep=False)
        output.attrs.pop("RPC", None)
        if kwargs.get("tags") is not None:
            kwargs["tags"] = {key: value for key, value in kwargs["tags"].items() if key != "RPC"}

        # Newer Rioxarray versions serialize RPCs themselves, older versions accept the Rasterio option
        if hasattr(output.rio, "write_rpcs"):
            output.rio.write_rpcs(rpcs, inplace=True)
        else:
            kwargs["rpcs"] = rpcs
        if array.rst.gcps[0]:
            output.rio.write_gcps(*array.rst.gcps, inplace=True)

    # The identity affine "placeholder" can be omitted safely with GCP/RPC, so we ignore its warning
    with (
        warnings.catch_warnings(),
        silence_rasterio_message(param_name="geotransform previously set", warn_code="CPLE_AppDefined"),
    ):
        if array.rst.gcps[0] or rpcs:
            warnings.filterwarnings(
                "ignore",
                message="The given matrix is equal to Affine.identity",
                category=rio.errors.NotGeoreferencedWarning,
            )
        return _write_raster(output, *args, **kwargs)


def _write_raster(
    xarray_obj: xr.DataArray | xr.Dataset,
    *args: Any,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Write Xarray rasters with a scheduler lock when their values are chunked.

    Used only by the rst Xarray accessors.
    """

    # 1/ Prepare the output path and filesystem
    filename = args[0] if args else kwargs.get("raster_path")
    filename, filesystem = _resolve_filesystem(filename, filesystem, storage_options, remote_default=False)
    if filesystem is not None:
        filename = _RasterFilesystemPath(filename, filesystem)
    if args:
        args = (filename, *args[1:])
    elif filename is not None:
        kwargs["raster_path"] = filename
    if filesystem is not None:
        if "opener" in kwargs:
            raise ValueError("Pass either filesystem or opener, not both.")
        kwargs["opener"] = _FilesystemOpener(filesystem)

    # 2/ Prevent simultaneous chunk writes

    # Serialize chunk writes across threads or worker processes
    # (Dataset variables can have different chunks, so inspect each array independently)
    variables = xarray_obj.data_vars.values() if isinstance(xarray_obj, xr.Dataset) else (xarray_obj,)
    if "lock" not in kwargs and any(variable.chunks for variable in variables):
        import_optional("dask")
        from dask.utils import get_scheduler_lock

        kwargs["lock"] = get_scheduler_lock(xarray_obj)

    # 3/ Prepare Dataset variables for writing

    # Rioxarray Dataset writer drops dtype when combining variables, so common packing must follow the values
    if isinstance(xarray_obj, xr.Dataset):
        # 3.1/ Check that variable encodings match
        fields = ("dtype", "scale_factor", "add_offset", "_FillValue")
        encodings = [{key: value.encoding[key] for key in fields if key in value.encoding} for value in variables]
        if any(not dict_equiv(encoding, encodings[0]) for encoding in encodings[1:]):
            raise ValueError(
                "Raster variables must share storage encoding; write differently encoded variables separately."
            )

        # 3.2/ Encode variable values
        if any(encodings):
            x_dimension, y_dimension = xarray_obj.rio.x_dim, xarray_obj.rio.y_dim
            encoded = {name: encode_cf_variable(value.variable) for name, value in xarray_obj.data_vars.items()}
            xarray_obj = xr.Dataset(encoded, coords=xarray_obj.coords, attrs=xarray_obj.attrs)
            xarray_obj.rio.set_spatial_dims(x_dimension, y_dimension, inplace=True)

        # 3.3/ Convert a single raster variable to a DataArray
        if len(xarray_obj.data_vars) == 1:
            name, variable = next(iter(xarray_obj.data_vars.items()))
            if variable.ndim == 3:
                xarray_obj = variable.assign_attrs({**variable.attrs, **xarray_obj.attrs})
                xarray_obj.attrs.setdefault("long_name", name)

    # 4/ Handle nodata values for DataArray output

    # Encode first when nodata value does not fit dtype
    if isinstance(xarray_obj, xr.DataArray) and xarray_obj.encoding.get("_FillValue") is not None:
        fill_value = np.asarray(xarray_obj.encoding["_FillValue"]).item()
        with np.errstate(invalid="ignore", over="ignore"):
            compatible = np.isnan(fill_value) or np.asarray(fill_value).astype(xarray_obj.dtype).item() == fill_value
        if not compatible:
            x_dimension, y_dimension = xarray_obj.rio.x_dim, xarray_obj.rio.y_dim
            xarray_obj = xr.DataArray(
                encode_cf_variable(xarray_obj.variable), coords=xarray_obj.coords, name=xarray_obj.name
            )
            xarray_obj.rio.set_spatial_dims(x_dimension, y_dimension, inplace=True)

    # 5/ Write to file
    return xarray_obj.rio.to_raster(*args, **kwargs)
