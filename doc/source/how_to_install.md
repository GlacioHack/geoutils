(how-to-install)=

# How to install

## Install for users

```{tip}
:class: margin

To speed-up installation, replace `conda` with `mamba` (see [Mamba docs](https://mamba.readthedocs.io/en/latest/)), or
 replace `pip` with `uv pip` (see [Uv docs](https://docs.astral.sh/uv/)).
```

With [Conda](https://docs.conda.io/en/latest/):

```bash
conda install -c conda-forge geoutils
```

Or with [Pip](https://pip.pypa.io/en/stable/):

```bash
pip install geoutils
```

Use `"geoutils[opt]"` with Pip to include all optional features, or select the extras listed below.
Extras can be combined, for example:

```bash
pip install "geoutils[io,paral]"
```

```{caution}
:class: margin

If a pip update breaks your installation, make a **fresh install** in a new environment with your other packages.
```

## Install for developers

```bash
git clone https://github.com/GlacioHack/geoutils.git
cd geoutils
conda env create -f dev-environment.yml
conda activate geoutils-dev
```

(dependencies)=
## Dependencies

GeoUtils depends on [Rasterio](https://rasterio.readthedocs.io/en/stable/),
[GeoPandas](https://geopandas.org/en/stable/), [SciPy](https://scipy.org/), [Xarray](https://xarray.dev/) and
[Rioxarray](https://vylor.github.io/rioxarray/stable/).

These main dependencies themselves depend notably on [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [PyProj](https://pyproj4.github.io/pyproj/stable/) and [Pyogrio](https://pyogrio.readthedocs.io/en/latest/).

Optional dependencies are grouped in the following pip extras (**`opt`** includes all of them):

- **`gstat`**: [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/) for variogram estimation and fitting,
  and [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) and [GPyTorch](https://docs.gpytorch.ai/en/stable/) (with [PyTorch](https://pytorch.org/)) for kriging and random fields.
- **`cloud`**: [Fsspec](https://filesystem-spec.readthedocs.io/) with HTTP and S3 support for remote file access.
- **`io`**: [LasPy](https://laspy.readthedocs.io/en/latest/) with [Lazrs](https://github.com/tmontaigu/laz-rs) for LAS/LAZ/COPC files, [PyArrow](https://arrow.apache.org/docs/python/) with GeoPandas 1.1 and above for GeoParquet,
  and [netCDF4](https://unidata.github.io/netcdf4-python/) and [Zarr](https://zarr.readthedocs.io/) for array storage.
- **`paral`**: [Dask](https://www.dask.org/), [Distributed](https://distributed.dask.org/en/stable/) and [Dask-GeoPandas](https://dask-geopandas.readthedocs.io/en/stable/) for out-of-memory and distributed parallel execution.
- **`num`**: [Numba](https://numba.pydata.org/) for faster filters and point cloud gridding,
  [Flox](https://flox.readthedocs.io/en/latest/) for grouped statistics, [Crick](https://github.com/dask/crick) for approximate quantiles,
  and [ExactExtract](https://isciences.github.io/exactextract/) for fractional polygon coverage.
- **`plot`**: [Matplotlib](https://matplotlib.org/) for raster, vector and statistical plots.
- **`prof`**: [Psutil](https://psutil.readthedocs.io/en/latest/) and [Plotly](https://plotly.com/) for profiling execution time and memory.
