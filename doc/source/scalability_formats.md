---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: geoutils-env
  language: python
  name: geoutils
---
(scalability-formats)=
# Formats and cloud storage

GeoUtils supports several **geospatial file formats** for rasters, vectors and point clouds, including formats
suited to **large datasets** and **cloud storage**.

Note that these formats are inherently supported through packages such as
[Rasterio](https://rasterio.readthedocs.io/en/stable/), [Pyogrio](https://pyogrio.readthedocs.io/en/latest/),
[PyArrow](https://arrow.apache.org/docs/python/), [LasPy](https://laspy.readthedocs.io/en/latest/) and
[Zarr](https://zarr.readthedocs.io/en/stable/). Below, we simply examplify **how to use them through GeoUtils**.
See {ref}`dependencies` for GeoUtils optional dependencies
and [Xarray's installation guide](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html#for-netcdf-and-io)
for its optional file readers.

This page focuses on **how to choose a file format that fits your scalable analysis**, as a file format and its storage
layout sometimes determine how much data must be read for an operation.
These choices complement {ref}`chunked execution <scalability-usage>`, which limits memory during computation.

## Choosing a file format

For large data analysis, we recommend **COG** for rasters, **COPC** for LiDAR,
**GeoParquet** for vectors and point cloud attributes, or **Zarr** for multidimensional datasets.
Choose the format and its tile, row group or chunk layout to match the data selections used in your analysis.

The table summarizes format capabilities.
**Chunked I/O** means efficient block/partition reading for large files.
**Cloud reading** means efficient access and streaming of remote files.
Formats marked "No" for cloud reading can still be opened remotely, but may require reading much more data.

```{list-table}
:widths: 2 1 1 4
:header-rows: 1
:align: left
:class: tight-table

* - File format
  - Chunked I/O
  - Cloud reading
  - Notes

* - <span class="gu-table-section">Raster formats</span>
  -
  -
  -

* - **GeoTIFF (untiled)**
  - Yes
  - No
  -
* - **Tiled GeoTIFF**
  - Yes
  - Yes
  - Supports overviews
* - **COG**
  - Yes
  - Yes
  - Supports overviews, arranges metadata and tiles to reduce remote requests

* - <span class="gu-table-section">Point cloud formats</span>
  -
  -
  -

* - **LAS / LAZ**
  - Yes
  - No
  - LAZ compresses LAS data
* - **COPC**
  - Yes
  - Yes
  - Supports spatial index and levels of detail

* - <span class="gu-table-section">Vector formats</span>
  -
  -
  -

* - **ESRI Shapefile**
  - Yes
  - No
  - Relies on multiple companion files
* - **GeoJSON**
  - Yes
  - No
  -
* - **GeoParquet**
  - Yes
  - Yes
  - Supports column selection and compression

* - <span class="gu-table-section">Mixed formats</span>
  -
  -
  -

* - **GeoPackage**
  - Yes
  - Yes
  - Stores vector layers and raster tiles
* - **NetCDF4**
  - Yes
  - Yes
  - Supports compression
* - **Zarr**
  - Yes
  - Yes
  - Supports compression, multidimensional chunks and parallel I/O
```

**Zarr is especially useful for large time series and datasets with many variables.** Its compressed chunks can
be read and written independently on local disks or in cloud object storage, allowing Dask workers to process and
save different parts of a dataset in parallel. This supports analysis of datasets larger than memory and avoids
rewriting the full dataset when updating selected regions. See
[Xarray's Zarr guide](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr).

Ordinary GeoTIFFs support [windowed I/O](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html),
and Shapefiles can be processed in feature batches; neither requires loading all values at once. For remote
GeoTIFFs, [GDAL recommends tiles and overviews](https://gdal.org/en/stable/user/virtual_file_systems.html#network-cloud-friendliness-and-file-formats).

**COPC can speed up spatial operations** by letting GeoUtils
[read points from the area needed](https://laspy.readthedocs.io/en/latest/api/laspy.copc.html) instead of repeatedly
scanning the file. Use `level` to select {ref}`levels of detail <scalability-downsampling-copc>`, or `downsample`
to select a smaller point sample automatically. Both work with `chunks` for lazy reading.

Writing COPC requires building its spatial index with a writer such as
[PDAL](https://pdal.io/en/stable/stages/writers.copc.html); LasPy supports reading COPC, but does not write it.

Below, we use a synthetic LiDAR survey and elevation raster, created in a temporary directory `data_dir`.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show synthetic file preparation"
:  code_prompt_hide: "Hide synthetic file preparation"

from pathlib import Path
from tempfile import TemporaryDirectory

import laspy
import numpy as np
import rasterio as rio
from pyproj import CRS
from rasterio.transform import from_origin

# Create tiled terrain with enough variation to demonstrate selective HTTP reads
files = TemporaryDirectory()
data_dir = Path(files.name)
rows, columns = np.mgrid[:512, :512]
terrain = 100 + 10 * np.sin(columns / 60) + 6 * np.cos(rows / 90)
terrain += np.random.default_rng(42).normal(0, 0.1, terrain.shape)
terrain = terrain.astype("float32")
with rio.open(
    data_dir / "elevation.tif", "w", driver="GTiff", width=512, height=512, count=1,
    dtype="float32", crs=32631, transform=from_origin(500_000, 5_120, 10, 10),
    nodata=-9999, tiled=True, blockxsize=128, blockysize=128, compress="deflate",
) as source:
    source.write(terrain, 1)

# Write elevation, classification and intensity for 4,096 survey points
header = laspy.LasHeader(point_format=3, version="1.2")
header.add_crs(CRS.from_epsg(32631))
cloud = laspy.LasData(header)
point_rows, point_columns = np.mgrid[4:512:8, 4:512:8]
cloud.x = 500_000 + (point_columns.ravel() + 0.5) * 10
cloud.y = 5_120 - (point_rows.ravel() + 0.5) * 10
cloud.z = terrain[point_rows, point_columns].ravel()
cloud.classification = np.where(point_rows.ravel() < 256, 2, 5).astype("uint8")
cloud.intensity = np.full(point_rows.size, 100, dtype="uint16")
cloud.write(data_dir / "survey.laz")
```

(scalability-point-reading)=
## Reading selected point attributes

By default, {func}`~geoutils.open_pointcloud` returns an **Xarray DataArray**. For LAS and native GeoParquet point
encodings, values and coordinates are read directly into numerical arrays. Other vector formats are read through
GeoPandas.

The `columns` argument selects which attributes to read from LAS or GeoParquet files. For example, we open the
survey in chunks with classification alongside the main elevation values:

```{code-cell} ipython3
import geoutils as gu

point_path = data_dir / "survey.laz"
ds_lidar = gu.open_pointcloud(point_path, data_name="Z", columns=["classification"], chunks=1_000)
print("Point count:", ds_lidar.pc.point_count)
print("Dask chunks:", ds_lidar.chunks)
print("Attributes:", list(ds_lidar.pc.columns))
```

With `columns="main"`, only the main attribute is read. Passing `"all"` reads every attribute, while a list reads
the named attributes alongside the main values.

The returned data structure can also be selected with `as_object_type="dataset"` for separate Xarray variables,
or `"geodataframe"` for GeoPandas geometry operations. See {ref}`core-point-accessors` for a comparison.

## Saving and reopening point clouds

Point clouds can be saved to **GeoParquet** with {meth}`~PointCloudBase.to_parquet`. For lazy data, computation and
writing happen **one row group at a time**, storing coordinates, CRS and the main attribute along with the selected
values:

```{code-cell} ipython3
:tags: [remove-cell]

import xarray as xr

xr.set_options(display_expand_attrs=False)
```

```{code-cell} ipython3
parquet_path = data_dir / "survey.parquet"
ds_lidar.pc.to_parquet(parquet_path, chunks=1_000, compression="zstd")
ds_saved = gu.open_pointcloud(parquet_path, columns="all", chunks=500, as_object_type="dataset")
ds_saved
```

A **row group** is the part of a Parquet file read at once. In the example above, `chunks=500` splits the arrays for
computation, while the stored groups still contain up to 1,000 rows. Row groups should therefore be small enough to
fit within a worker's memory. For more details, see
[Arrow's Parquet documentation](https://arrow.apache.org/docs/python/parquet.html).

Passing `partitioned=True` writes the point cloud into **separate files** in a new directory, which can then be
reopened directly:

```{code-cell} ipython3
parts_path = data_dir / "survey_parts"
ds_lidar.pc.to_parquet(parts_path, chunks=1_000, partitioned=True)
ds_parts = gu.open_pointcloud(parts_path, chunks=1_000)
print("Partition files:", len(list(parts_path.glob("*.parquet"))))
print("Reopened points:", ds_parts.pc.point_count)
```

```{code-cell} ipython3
:tags: [remove-cell]

import pyarrow.parquet as pq

# Check physical groups, lazy variables and preservation of all selected point rows
metadata = pq.ParquetFile(parquet_path).metadata
assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [1000, 1000, 1000, 1000, 96]
assert ds_saved["Z"].chunks == ((500,) * 8 + (96,),)
assert ds_saved["Z"].pc.crs == ds_lidar.pc.crs == ds_parts.pc.crs
assert not ds_saved["Z"]._in_memory and not ds_parts._in_memory
np.testing.assert_array_equal(ds_saved["Z"].compute(), ds_lidar.compute())
np.testing.assert_array_equal(ds_saved["classification"].compute(), ds_lidar.classification.compute())
np.testing.assert_array_equal(ds_parts.compute(), ds_lidar.compute())
```

## Raster tiles and chunks

For tiled rasters, reading a window requires reading the **tiles it intersects**, even if the window is smaller
than one tile. We recommend using **Dask chunks that are multiples of the tile size**, and larger than
the neighborhoods used by filters or interpolation.

For example, we open a raster stored in tiles of 128 × 128 pixels with chunks of 256 × 256 pixels:

```{code-cell} ipython3
raster_path = data_dir / "elevation.tif"
with rio.open(raster_path) as source:
    print("Disk tile shape:", source.block_shapes[0])
ds = gu.open_raster(str(raster_path), chunks={"x": 256, "y": 256})
print("Dask chunks:", ds.chunks)
```

The {meth}`~RasterBase.to_file` method can write raster results in tiles. For efficient remote access, a
**Cloud Optimized GeoTIFF (COG)** also arranges metadata and overviews for HTTP reading. Below, we write a tiled
result, then convert it to a COG with the [GDAL COG driver](https://gdal.org/en/stable/drivers/raster/cog.html):

```{code-cell} ipython3
from rasterio.shutil import copy as copy_raster

output_path = data_dir / "processed.tif"
ds.rst.to_file(str(output_path), tiled=True, blockxsize=128, blockysize=128, compress="deflate", compute=True)
cog_path = data_dir / "elevation_cog.tif"
copy_raster(output_path, cog_path, driver="COG", BLOCKSIZE=128, COMPRESS="DEFLATE")
with rio.open(cog_path) as source:
    print("Layout:", source.tags(ns="IMAGE_STRUCTURE")["LAYOUT"])
    print("Overview factors:", source.overviews(1))
```

(scalability-array-formats)=
## Zarr and NetCDF datasets

Open **Zarr** and **NetCDF** datasets with Xarray, then use GeoUtils' `rst` accessor on their raster variables.
The examples below assume an `elevation` variable with `x`/`y` coordinates and a stored CRS. Use
`decode_coords="all"` to restore georeferencing stored in a `grid_mapping` variable; if the CRS is missing, set it
from the source information with `rio.write_crs()` before using spatial operations.

Xarray needs the [Zarr package](https://zarr.readthedocs.io/en/stable/) for Zarr and a backend such as
[netCDF4](https://unidata.github.io/netcdf4-python/) for NetCDF4. Install these backends separately if needed.

```python
import xarray as xr

# Open rasters in chunks and calculate statistics through the GeoUtils accessor
ds_zarr = xr.open_zarr("elevation.zarr", chunks={"x": 256, "y": 256}, decode_coords="all")
ds_netcdf = xr.open_dataset(
    "elevation.nc", engine="netcdf4", chunks={"x": 256, "y": 256}, decode_coords="all",
)
print(ds_zarr["elevation"].rst.stats(["mean", "std"]))

# Specify NetCDF storage chunks and preserve the CRS reference
netcdf_encoding = {
    "elevation": {
        "chunksizes": (256, 256), "zlib": True,
        "grid_mapping": ds_netcdf["elevation"].rio.grid_mapping,
    },
}

# Write Zarr v2 and NetCDF4 results in chunks using Xarray's methods
ds_zarr.to_zarr("result.zarr", mode="w", zarr_format=2)
ds_netcdf.to_netcdf("result.nc", engine="netcdf4", encoding=netcdf_encoding)
ds_zarr.close()
ds_netcdf.close()
```

The NetCDF example uses **NetCDF4**, which supports stored chunks and compression.
For details on storage options and writing, see
[Xarray's file format guide](https://docs.xarray.dev/en/stable/user-guide/io.html).

```{code-cell} ipython3
:tags: [remove-cell]

# Close the local examples before removing their temporary files
ds.close()
ds_lidar.close()
ds_saved.close()
ds_parts.close()
files.cleanup()
```

(scalability-cloud)=
## Reading from cloud storage

Pass an **HTTPS URL** or an **S3 address** directly to {func}`~geoutils.open_raster`,
{func}`~geoutils.open_pointcloud` or {func}`~geoutils.open_vector`, using the same options as for a local file.
Replace the example location below with your own dataset address:

```python
import geoutils as gu

location = "https://example.org/data"  # Or "s3://my-bucket/data"
ds = gu.open_raster(f"{location}/elevation.tif", chunks={"x": 256, "y": 256})
points = gu.open_pointcloud(f"{location}/survey.copc.laz", level=range(3), chunks=100_000)
features = gu.open_vector(f"{location}/regions.gpkg", chunks=10_000)
```

The point cloud call also accepts **LAS, LAZ and GeoParquet** files (or a directory of GeoParquet partitions);
omit `level` for formats other than COPC. The vector call accepts formats supported by GDAL, such as GeoPackage
and GeoJSON. These functions select the file reader automatically, so no separate download is needed.

With `chunks`, the data stays **lazy** until a computation needs it. For example, select a spatial window before
calculating raster statistics:

```python
window = ds.rst.crop((501_280, 3_200, 501_920, 3_840))
print(window.rst.stats(["mean", "std"]))
```

Efficient partial reads require **HTTP range requests** and a suitable file layout, as summarized in the table above.
Readers also fetch metadata and complete tiles, compressed point blocks or row groups, so the transferred data can
exceed the selected values. {ref}`Downsampling and overviews <scalability-downsampling>` can further reduce reads.

Raster and vector formats use GDAL through Rasterio and Pyogrio. Native LAS/COPC and point GeoParquet readers use
[Fsspec](https://filesystem-spec.readthedocs.io/en/latest/): install `fsspec[http]` for HTTPS or
[S3fs](https://s3fs.readthedocs.io/en/latest/) for S3. For multidimensional Zarr and NetCDF datasets, use Xarray
as shown {ref}`above <scalability-array-formats>`; see its
[cloud storage guide](https://docs.xarray.dev/en/stable/user-guide/io.html#cloud-storage) for remote access.

### Passwords and other credentials

Authentication follows the reader's configuration; the open functions do not share a `password` or `storage_options`
argument. For an **HTTPS server using a username and password**, configure GDAL for rasters and vectors, or Fsspec
for native point cloud files, before opening:

```python
import os
from base64 import b64encode
from getpass import getpass

import fsspec

username = input("Username: ")
password = getpass("Password: ")

# Encode HTTP Basic credentials for the Authorization header
authorization = "Basic " + b64encode(f"{username}:{password}".encode()).decode("ascii")

# Configure raster and vector reads through GDAL
os.environ["GDAL_HTTP_HEADERS"] = f"Authorization: {authorization}"

# Configure LAS/COPC and point GeoParquet reads through Fsspec
fsspec.config.conf["http"] = {"headers": {"Authorization": authorization}}

ds_private = gu.open_raster("https://example.org/private/elevation.tif", chunks={"x": 256, "y": 256})
points_private = gu.open_pointcloud("https://example.org/private/survey.parquet", chunks=100_000)
```

For password-protected vector URLs, use `gu.open_vector("/vsicurl/https://example.org/private/regions.gpkg")`
to send requests directly through GDAL, including authentication, instead of GeoPandas' initial URL check.
See [GDAL's HTTP authentication options](https://gdal.org/en/stable/user/configoptions.html#networking-options)
and [Fsspec's HTTP options](https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/http.html)
for other authentication methods. A **signed URL** can be passed directly, including its query string.

For **private S3 storage**, both readers can use the standard `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and,
for temporary credentials, `AWS_SESSION_TOKEN` environment variables. See
[GDAL's S3 credentials](https://gdal.org/en/stable/user/virtual_file_systems.html#vsis3-aws-s3-files-random-access)
and [S3fs credentials](https://s3fs.readthedocs.io/en/latest/#credentials) for profiles and IAM roles.
For a public bucket, enable anonymous reads before opening:

```python
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"  # GDAL
fsspec.config.conf["s3"] = {"anon": True}  # Fsspec/S3fs
points_public = gu.open_pointcloud("s3://my-public-bucket/survey.copc.laz", level=range(3), chunks=100_000)
```

Credentials must remain available until lazy reads finish. With distributed Dask, configure them on **each worker**;
changes to `fsspec.config.conf` in the client process are not automatically applied to workers.
For repeated reads of large datasets, run computations near the storage; see {ref}`scalability-usage` and
[Dask's remote data guidance](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html).
