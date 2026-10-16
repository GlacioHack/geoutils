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

Use `"geoutils[opt]"` with Pip to include all optional features.

```{caution}
:class: margin

If Pip updates break your installation, make a **fresh install** in a new environment with your other packages.
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

Their dependencies include [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/),
[PyProj](https://pyproj4.github.io/pyproj/stable/) and [Pyogrio](https://pyogrio.readthedocs.io/en/latest/).

Optional dependencies are:

- [Matplotlib](https://matplotlib.org/) for plotting,
- [LasPy](https://laspy.readthedocs.io/en/latest/) with [Lazrs](https://github.com/tmontaigu/laz-rs) for LAS/LAZ/COPC files,
- [PyArrow](https://arrow.apache.org/docs/python/) for GeoParquet point cloud files (with GeoPandas 1.1 and above),
- [Numba](https://numba.pydata.org/) for faster filters and point cloud gridding,
- [Dask](https://www.dask.org/), [Distributed](https://distributed.dask.org/en/stable/) and [Dask-GeoPandas](https://dask-geopandas.readthedocs.io/en/stable/) for out-of-memory and distributed operations,
- [Flox](https://flox.readthedocs.io/en/latest/) for grouped statistics,
- [ExactExtract](https://isciences.github.io/exactextract/) for fractional polygon coverage,
- [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/) for variogram estimation and fitting,
- [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) and [GPyTorch](https://docs.gpytorch.ai/en/stable/) (with [PyTorch](https://pytorch.org/)) for kriging and random fields,
- [Psutil](https://psutil.readthedocs.io/en/latest/) and [Plotly](https://plotly.com/) for profiling computing speed and memory.
