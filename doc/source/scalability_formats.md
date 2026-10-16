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

Note that these formats are supported through other packages such as
[Rasterio](https://rasterio.readthedocs.io/en/stable/), [Pyogrio](https://pyogrio.readthedocs.io/en/latest/),
[PyArrow](https://arrow.apache.org/docs/python/), [LasPy](https://laspy.readthedocs.io/en/latest/) and
[Zarr](https://zarr.readthedocs.io/en/stable/). Below, we simply **examplify how to use them through GeoUtils**.

First, we help {ref}`<scalability-format-choice` for your data, then explain how to perform
{ref}`scalability-cloud`, and finally how to deal with format-dependent intricacies to 
optimally perform {ref}`scalability-format-chunks`.

(scalability-format-choice)=
## Choosing a file format

For large data analysis, we recommend **COG** for rasters, **COPC** for LiDAR,
**GeoParquet** for vectors and point cloud attributes, or **Zarr** for multidimensional datasets.

The table below summarizes format capabilities.
**Chunked I/O** means reading or writing in blocks/partitions, without loading the whole file.
**Cloud reading** means efficient access and streaming of remote files.

```{important}
At present, LasPy (and thus GeoUtils) **supports only reading COPC files**, but not writing them.
```

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
save different parts of a dataset in parallel. See [Xarray's Zarr guide](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr).

**COGs support efficient spatial reads at different resolutions.** They organize tiles, overviews and
metadata so readers can fetch the blocks needed for a spatial window or a coarser resolution. 
Remote reads avoid downloading the full raster. See our {ref}`scalability-downsampling-overviews` example.

**COPC can speed up spatial operations** by letting GeoUtils read points from the area needed [through LasPy](https://laspy.readthedocs.io/en/latest/api/laspy.copc.html) instead of repeatedly
scanning the file. They also store levels of detail, progressive subsamples of the point data equivalent to overviews. 
Both work with chunked, lazy reading. See our {ref}`scalability-downsampling-copc` example.

**GeoParquet is useful for vectors and point clouds with many attributes.** It lets readers
load only selected column attributes, while row groups allow data to be processed in chunks. Compression 
reduces storage and transfer sizes. See [PyArrow's Parquet guide](https://arrow.apache.org/docs/python/parquet).

(scalability-cloud)=
## Reading from cloud storage

To read a remote file, simply pass an **HTTPS URL** or an **S3 address** to {func}`~geoutils.open_raster`,
{func}`~geoutils.open_pointcloud` or {func}`~geoutils.open_vector`, using the same options as for a local file.

```{note}
These executable examples reuse the **mock HTTP/S3 server from our tests**.
Replace `server_url` in the HTTP example and `bucket_url` in the S3 example below with your own addresses.
```

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

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show mock server preparation"
:  code_prompt_hide: "Hide mock server preparation"

import os
import sys
from contextlib import ExitStack
from shutil import copyfile

import fsspec
import geoutils as gu
from pytest import MonkeyPatch

sys.path.insert(0, str(Path("../..").resolve()))
from tests.remote_helpers import mock_remote_files

# Start the shared test server and restore its settings when the examples finish
resources = ExitStack()
settings = resources.enter_context(MonkeyPatch.context())
remote = resources.enter_context(mock_remote_files(data_dir, settings, basic_auth=("demo", "geoutils")))
server_port = remote.endpoint.rsplit(":", 1)[1]
settings.setenv("GDAL_HTTP_HEADERS", "")
settings.setitem(fsspec.config.conf, "http", {})

# Prepare a COG and GeoParquet survey for network reads
with gu.open_raster(str(data_dir / "elevation.tif")) as source:
    source.rst.to_file(
        str(remote.directory / "elevation.tif"), driver="COG", BLOCKSIZE=128, compress="deflate",
    )
copyfile(Path("_static/survey.copc.laz"), remote.directory / "survey.copc.laz")
survey = gu.open_pointcloud(data_dir / "survey.laz", data_name="Z", columns=["classification"], chunks=1_000)
survey.pc.to_parquet(data_dir / "survey.parquet", chunks=1_000, compression="zstd")
survey.close()
copyfile(data_dir / "survey.parquet", remote.directory / "survey.parquet")

# Write a fresh GeoPackage so remote reads do not need SQLite journal files
regions = gu.open_vector(gu.examples.get_path("everest_rgi_outlines"))
regions.to_file(remote.directory / "regions.gpkg")

# Copy files into the password-protected directory
private_directory = remote.directory / "private"
private_directory.mkdir()
for name in ("elevation.tif", "survey.parquet"):
    copyfile(remote.directory / name, private_directory / name)
```

```{code-cell} ipython3
import geoutils as gu

# Server address, for example, "https://example.org/data"
server_url = f"http://127.0.0.1:{server_port}/bucket"  

# Open raster, point cloud or vector
ds_remote = gu.open_raster(f"{server_url}/elevation.tif", chunks={"x": 256, "y": 256})
points = gu.open_pointcloud(f"{server_url}/survey.copc.laz", chunks=10_000)
features = gu.open_vector(f"{server_url}/regions.gpkg", chunks=100)

# Print shape
print("Raster shape:", ds_remote.rst.shape)
print("COPC points:", points.pc.point_count)
print("Vector partitions:", features.npartitions)
```

With `chunks`, the data stays **lazy**. For example, to compute stats on a window only:

```{code-cell} ipython3
window = ds_remote.rst.crop((501_280, 3_200, 501_920, 3_840))
summary = window.rst.stats(["mean", "std"])
print(summary)
```

Raster and vector formats use GDAL through Rasterio and Pyogrio. LAS/LAZ/COPC and GeoParquet point clouds use
[Fsspec](https://filesystem-spec.readthedocs.io/en/latest/) for remote access and
[S3fs](https://s3fs.readthedocs.io/en/latest/) for S3, both {ref}`optional dependencies <dependencies>` of GeoUtils.
For multidimensional Zarr and NetCDF datasets, follow
[Xarray's cloud storage guide](https://docs.xarray.dev/en/stable/user-guide/io.html#cloud-storage).

### Passwords and other credentials

For **HTTPS with a username and password**, configure GDAL for rasters/vectors or Fsspec for LAS/LAZ/COPC and
GeoParquet point clouds.

```{code-cell} ipython3
import os

import fsspec
from aiohttp import encode_basic_auth

username, password = "demo", "geoutils"
private_url = f"{server_url}/private"
authorization = encode_basic_auth(username, password)

# Configure raster and vector reads through GDAL, and native point cloud reads through Fsspec
os.environ["GDAL_HTTP_HEADERS"] = f"Authorization: {authorization}"
fsspec.config.conf["http"] = {"headers": {"Authorization": authorization}}

ds_private = gu.open_raster(f"{private_url}/elevation.tif", chunks={"x": 256, "y": 256})
points_private = gu.open_pointcloud(f"{private_url}/survey.parquet", chunks=1_000)
print("Private raster mean:", ds_private.rst.stats(["mean"])["mean"])
print("Private point mean:", points_private.pc.stats(["mean"])["mean"])
```

See [GDAL's authentication options](https://gdal.org/en/stable/user/configoptions.html#networking-options) and
[Fsspec's HTTP options](https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/http.html)
for other methods.

For **private S3 buckets**, set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (plus `AWS_SESSION_TOKEN` for
temporary credentials).
[GDAL](https://gdal.org/en/stable/user/virtual_file_systems.html#vsis3-aws-s3-files-random-access) and
[S3fs](https://s3fs.readthedocs.io/en/latest/#credentials) also support AWS profiles and IAM roles.
For **public S3 buckets**, enable anonymous reads:

```{code-cell} ipython3
bucket_url = "s3://bucket"
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"  # GDAL
fsspec.config.conf.setdefault("s3", {})["anon"] = True  # Fsspec/S3fs
points_public = gu.open_pointcloud(f"{bucket_url}/survey.copc.laz", level=0, chunks=10_000)
print("Public S3 level 0 points:", points_public.pc.point_count)
```

Leave credentials configured until lazy reads finish. With distributed Dask, set them on **each worker**.

(scalability-format-chunks)=
## Reading and writing in chunks

**Disk chunks are often overlooked**, while they are essential to efficient analysis.

Indeed, **disk chunks need to align with compute chunks** we choose (through the `chunks=` arguments), and 
be adapted to the nature of the analysis (spatial, temporal).

Below, we show how to inspect files for their **disk chunk sizes**, then how to define **disk chunks** when writing, both for 
the different file formats.

(scalability-point-reading)=
### LAS, LAZ and COPC

We can inspect chunk sizes of LAZ/COPC when opening (LAS can read in batches but has no explicit disk chunks):

```{code-cell} ipython3
ds_laz = gu.open_pointcloud(data_dir / "survey.laz", data_name="Z", columns=["classification"], chunks=1_000)
print("LAZ maximum points per chunk:", ds_laz.attrs["laz_chunk_size"])

ds_copc = gu.open_pointcloud("_static/survey.copc.laz", chunks=10_000)
print("COPC points per chunk (first five):", ds_copc.attrs["copc_chunk_sizes"][:5])
```

To write LAZ disk chunks, pass `laz_chunk_size` in {meth}`~geoutils.DataArrayPointCloudAccessor.to_las`
(`chunks` controls processing batches):

```{code-cell} ipython3
laz_output = data_dir / "rechunked.laz"
ds_laz.pc.to_las(laz_output, chunks=1_000, laz_chunk_size=2_000)
with gu.open_pointcloud(laz_output, chunks=1_000) as written:
    print("Written LAZ points per chunk:", written.attrs["laz_chunk_size"])
```

```{caution}
Currently, LasPy (and thus GeoUtils) cannot write COPC files.
```

(scalability-format-parquet)=
### GeoParquet

Similarly, we can inspect GeoParquet disk chunk sizes (called **row groups**) on opening:

```{code-cell} ipython3
# Inspect row counts stored in the source file
parquet_path = data_dir / "survey.parquet"
with gu.open_pointcloud(parquet_path, chunks=1_000) as source_points:
    print("Input row group sizes:", source_points.attrs["row_group_sizes"])
```

To write disk chunks, we simply pass `chunks` sizes in {meth}`~geoutils.DataArrayPointCloudAccessor.to_parquet`:

```{code-cell} ipython3
# Write groups of 2,000 points and inspect the saved file
parquet_output = data_dir / "rechunked.parquet"
ds_laz.pc.to_parquet(parquet_output, chunks=2_000, compression="zstd")
ds_saved = gu.open_pointcloud(parquet_output, columns="all", chunks=500, as_object_type="dataset")
print("Written row group sizes:", ds_saved.attrs["row_group_sizes"])
```

And use `partitioned=True` to write **separate files** that reopen as one point cloud:

```{code-cell} ipython3
parts_path = data_dir / "survey_parts"
ds_laz.pc.to_parquet(parts_path, chunks=1_000, partitioned=True)
ds_parts = gu.open_pointcloud(parts_path, chunks=1_000)
print("Partition files:", len(list(parts_path.glob("*.parquet"))))
```

(scalability-format-rasters)=
### GeoTIFF and COG

For GeoTIFFs or COGs, chunks are called **blocks** in row/col pixels:

```{code-cell} ipython3
raster_path = data_dir / "elevation.tif"
ds = gu.open_raster(str(raster_path))
print("Input disk tiles (rows, columns):", ds.attrs["block_shape"])
```

To define write **chunks**, use `blockxsize` and `blockysize` for GeoTIFF, or `BLOCKSIZE` for COG:

```{code-cell} ipython3
output_path = data_dir / "processed.tif"
ds.rst.to_file(str(output_path), tiled=True, blockxsize=256, blockysize=256, compress="deflate", compute=True)
with gu.open_raster(str(output_path)) as written:
    print("Written GeoTIFF disk tiles (rows, columns):", written.attrs["block_shape"])

cog_path = data_dir / "elevation_cog.tif"
ds.rst.to_file(str(cog_path), driver="COG", BLOCKSIZE=256, compress="deflate", compute=True)
cog = gu.open_raster(str(cog_path))
print("Written COG disk tiles (rows, columns):", cog.attrs["block_shape"])
```

(scalability-array-formats)=
### Zarr and NetCDF

For Zarr/NetCDFs, we inspect **disk chunk sizes along a given axis** (such as time, rows and columns):

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show Zarr and NetCDF preparation"
:  code_prompt_hide: "Hide Zarr and NetCDF preparation"

# Save the synthetic elevation raster with its CRS and spatial coordinates
elevation = ds.rename("elevation").to_dataset()
zarr_path = data_dir / "elevation.zarr"
netcdf_path = data_dir / "elevation.nc"
grid_mapping = elevation["elevation"].rio.grid_mapping
elevation.to_zarr(zarr_path, mode="w", zarr_format=2,
                  encoding={"elevation": {"chunks": (128, 128), "grid_mapping": grid_mapping}})
elevation.to_netcdf(netcdf_path, engine="netcdf4",
                    encoding={"elevation": {"chunksizes": (128, 128), "grid_mapping": grid_mapping}})
```

```{code-cell} ipython3
import xarray as xr

# Inspect disk chunk sizes in both files
ds_zarr = xr.open_zarr(zarr_path, chunks={"x": 256, "y": 256}, decode_coords="all")
ds_netcdf = xr.open_dataset(
    netcdf_path, engine="netcdf4", chunks={"x": 256, "y": 256}, decode_coords="all",
)
print("Zarr disk chunks:", ds_zarr["elevation"].rst.tags["disk_chunks"])
print("NetCDF disk chunks:", ds_netcdf["elevation"].rst.tags["disk_chunks"])
```

To write to specific dis chunks, set `chunks` for Zarr or `chunksizes` for NetCDF4 in `encoding`:

```{code-cell} ipython3
# Specify storage chunks and preserve the CRS reference
zarr_encoding = {
    "elevation": {"chunks": (256, 256), "grid_mapping": ds_zarr["elevation"].rio.grid_mapping},
}

netcdf_encoding = {
    "elevation": {
        "chunksizes": (256, 256), "zlib": True,
        "grid_mapping": ds_netcdf["elevation"].rio.grid_mapping,
    },
}

# Write Zarr and NetCDF4 results in chunks using Xarray
ds_zarr.to_zarr(data_dir / "result.zarr", mode="w", zarr_format=2, encoding=zarr_encoding)
ds_netcdf.to_netcdf(data_dir / "result.nc", engine="netcdf4", encoding=netcdf_encoding)

# Reopen outputs to inspect their stored chunk sizes
with xr.open_zarr(data_dir / "result.zarr", chunks={}, decode_coords="all") as written:
    print("Written Zarr disk chunks:", written["elevation"].rst.tags["disk_chunks"])
with xr.open_dataset(data_dir / "result.nc", engine="netcdf4", chunks={}, decode_coords="all") as written:
    print("Written NetCDF disk chunks:", written["elevation"].rst.tags["disk_chunks"])
```

```{code-cell} ipython3
:tags: [remove-cell]

# Close arrays and stop the server before restoring configuration and deleting its files
for array in (ds, cog, ds_zarr, ds_netcdf, ds_laz, ds_copc, ds_saved, ds_parts, ds_remote, window, ds_private, points,
              points_private, points_public):
    array.close()
resources.close()
files.cleanup()
```
