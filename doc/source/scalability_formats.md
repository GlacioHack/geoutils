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
scanning the file. Opening with `chunks` still reads points in file order.

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

(scalability-cloud)=
## Reading from cloud storage

Cloud reads use the file readers shown below. The Zarr, NetCDF, COPC, GeoParquet and GeoPackage examples use
placeholder addresses: replace them with your own dataset locations. Authentication follows each reader's
configuration. Install [Fsspec](https://filesystem-spec.readthedocs.io/en/latest/) with its HTTP extra
(`fsspec[http]`) for the HTTP examples, or with [S3fs](https://s3fs.readthedocs.io/en/latest/) for `s3://`.

### GeoTIFF and COG

GeoUtils can open remote rasters by passing an **HTTPS URL** or a supported cloud address, such as `s3://`, to
{func}`~geoutils.open_raster`. With a server supporting **HTTP range requests**, Rasterio reads only the required
sections of the file. Authentication follows
[Rasterio's virtual filesystems](https://rasterio.readthedocs.io/en/stable/topics/vsi.html).

For example, we read the COG created above through a local HTTP server with range support, using `cog_url` as its
address. The same calls work with a COG in cloud storage. We crop to a 64 × 64 window before calculating statistics:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show local HTTP service preparation"
:  code_prompt_hide: "Hide local HTTP service preparation"

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

# Serve one file with byte ranges, recording response sizes to check selective reading
cog_bytes = cog_path.read_bytes()
transfers = []

class RangeHandler(BaseHTTPRequestHandler):
    """Serve the example COG and record the byte ranges read by GDAL."""

    def do_HEAD(self) -> None:
        """Return the file size without a response body."""
        self.respond(send_body=False)

    def do_GET(self) -> None:
        """Return a complete file or the requested byte range."""
        self.respond(send_body=True)

    def respond(self, send_body: bool) -> None:
        """Handle file existence and single byte ranges for this example service."""
        if self.path != "/elevation.tif":
            self.send_error(404)
            return

        # GDAL requests explicit, inclusive byte ranges from the COG
        requested_range = self.headers.get("Range")
        start, end = 0, len(cog_bytes) - 1
        if requested_range:
            first, last = requested_range.removeprefix("bytes=").split("-")
            start = int(first)
            end = min(int(last) if last else end, end)
        self.send_response(206 if requested_range else 200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested_range:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(cog_bytes)}")
        self.end_headers()
        if send_body:
            transfers.append((start, end + 1))
            self.wfile.write(cog_bytes[start:end + 1])

    def log_message(self, format: str, *args: object) -> None:
        """Keep request logging out of notebook output."""
        return

server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
server_thread = Thread(target=server.serve_forever, daemon=True)
server_thread.start()
cog_url = f"http://127.0.0.1:{server.server_port}/elevation.tif"
```

```{code-cell} ipython3
# Avoid checking the remote directory for auxiliary files
with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    ds_remote = gu.open_raster(cog_url, chunks={"x": 128, "y": 128})
    ds_window = ds_remote.rst.crop((501_280, 3_200, 501_920, 3_840))
    summary = ds_window.rst.stats(["mean", "std"])
print("Window shape:", ds_window.rst.shape)
print(summary)
```

```{code-cell} ipython3
print("Complete COG:", len(cog_bytes), "bytes")
print("HTTP response bodies:", sum(end - start for start, end in transfers), "bytes")
```

The source and cropped arrays remain **lazy**, and only the statistics are returned to memory. The number of bytes
read includes metadata and depends on the window, tile layout and GDAL's cache. When a coarser raster is sufficient,
{ref}`downsampling and overviews <scalability-downsampling>` can further reduce the amount of data read.

```{code-cell} ipython3
:tags: [remove-cell]

# Check the actual HTTP results, then stop the service and remove the temporary files
assert ds_window.rst.shape == (64, 64)
assert not ds_remote._in_memory and not ds_window._in_memory
np.testing.assert_allclose(summary["mean"], np.mean(terrain[128:192, 128:192], dtype="float64"), rtol=1e-12)
np.testing.assert_allclose(summary["std"], np.std(terrain[128:192, 128:192], dtype="float64"), rtol=1e-12)
assert transfers and sum(end - start for start, end in transfers) < len(cog_bytes)
ds_remote.close()
ds_window.close()
server.shutdown()
server.server_close()
server_thread.join()
ds.close()
files.cleanup()
```

### Zarr

Xarray can open a Zarr store directly in cloud storage. As with the COG above, select a spatial window before
computing statistics so that only the intersecting chunks are read:

```python
import xarray as xr

# Open a Zarr store, preserving its CRS and spatial coordinates
ds_cloud = xr.open_zarr(
    "s3://my-bucket/elevation.zarr", chunks={"x": 256, "y": 256}, decode_coords="all",
)

# Read the same window as the COG example when calculating statistics
window = ds_cloud["elevation"].rst.crop((501_280, 3_200, 501_920, 3_840))
print(window.rst.stats(["mean", "std"]))
ds_cloud.close()
```

See [Xarray's Zarr guide](https://docs.xarray.dev/en/stable/user-guide/io.html#zarr) for cloud authentication,
consolidated metadata and writing results back to object storage.

### NetCDF

For a **NetCDF4 file over HTTP**, Fsspec provides a file object with remote byte reads and
[h5netcdf](https://h5netcdf.org/) with [h5py](https://www.h5py.org/) opens its arrays through Xarray.
Compute the result while the remote file is open:

```python
import fsspec
import xarray as xr

# Open a NetCDF4 file without downloading the complete file first
with fsspec.open("https://example.org/elevation.nc", "rb") as remote_file:
    with xr.open_dataset(
        remote_file, engine="h5netcdf", chunks={"x": 256, "y": 256}, decode_coords="all",
    ) as ds_cloud:
        window = ds_cloud["elevation"].rst.crop((501_280, 3_200, 501_920, 3_840))
        print(window.rst.stats(["mean", "std"]))
```

This requires an HTTP server supporting byte ranges; Fsspec can also open an `s3://` file with S3fs.
For a dataset served through **OPeNDAP**, use `xr.open_dataset(url, engine="netcdf4", chunks="auto")` instead.
See [Xarray's OPeNDAP guide](https://docs.xarray.dev/en/stable/user-guide/io.html#opendap).

### COPC

Use LasPy's CopcReader to read a spatial selection from a remote COPC. The query loads the selected points into
memory, so choose bounds and a resolution appropriate for the analysis:

```python
import laspy
import numpy as np
import geoutils as gu

# Query the same horizontal bounds as the raster examples, at a coarser point spacing
bounds = laspy.copc.Bounds(mins=np.array([501_280, 3_200]), maxs=np.array([501_920, 3_840]))
with laspy.CopcReader.open("https://example.org/survey.copc.laz") as reader:
    points = reader.query(bounds=bounds, resolution=10)
    ds_points = gu.pointcloud.from_xyz(
        np.asarray(points.x), np.asarray(points.y), np.asarray(points.z),
        crs=reader.header.parse_crs(), data_name="Z",
    )

# Use GeoUtils on the selected points
print(ds_points.pc.stats(["mean", "std"]))
```

[CopcReader's documentation](https://laspy.readthedocs.io/en/latest/api/laspy.copc.html) describes bounds and levels
of detail. HTTP reading also requires [Requests](https://requests.readthedocs.io/en/latest/).

### GeoParquet

PyArrow can read selected columns from a remote GeoParquet file **one batch at a time**. Here, each printed mean
describes one batch of elevation values:

```python
import fsspec
import numpy as np
import pyarrow.parquet as pq

# Read elevation values in batches rather than loading every point and attribute
with fsspec.open("https://example.org/survey.parquet", "rb") as remote_file:
    parquet = pq.ParquetFile(remote_file)
    for batch in parquet.iter_batches(batch_size=1_000, columns=["Z"]):
        values = batch.column("Z").to_numpy()
        print(np.nanmean(values))
```

The HTTP server must support byte ranges. For a remote GeoDataFrame, use
[GeoPandas' read_parquet()](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_parquet.html), which
loads the selected rows and columns into memory.

### GeoPackage

Pyogrio can read a remote GeoPackage through
[GDAL's virtual filesystems](https://gdal.org/en/stable/user/virtual_file_systems.html). A bounding box limits the
features loaded into memory, although the database layout can require many HTTP requests:

```python
import pyogrio

# Read features intersecting the example bounds from a remote GeoPackage
features = pyogrio.read_dataframe(
    "/vsicurl/https://example.org/regions.gpkg",
    bbox=(501_280, 3_200, 501_920, 3_840), columns=["name"],
)
print(features.head())
```

For a file on S3, use a `/vsis3/my-bucket/regions.gpkg` path and configure GDAL's credentials.

Opening remote arrays with `chunks` uses **Dask's local scheduler** by default.
For repeated reads of large datasets, we recommend running computations near the storage. A Dask `Client` allows
using several machines when needed; see {ref}`scalability-usage` and
[Dask's remote data guidance](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html).

GeoUtils' native LAS and GeoParquet readers and writers currently use **local paths**, so those inputs need to be
copied to the processing machine and accessible to any remote workers. Large outputs can be written in chunks,
then uploaded as COG or Parquet files with your storage provider's tools.
