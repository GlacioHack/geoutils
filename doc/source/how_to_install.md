(how-to-install)=

# How to install

## Installing with ``mamba`` (recommended)

```bash
mamba install -c conda-forge geoutils
```

```{tip}
Solving dependencies can take a long time with `conda`, `mamba` significantly speeds up the process. Install it with:

    conda install mamba -n base -c conda-forge

Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available in the [mamba documentation](https://mamba.readthedocs.io/en/latest/).
```

## Installing with ``pip``

```bash
pip install geoutils
```

```{warning}
Updating packages with `pip` (and sometimes `mamba`) can break your installation. If this happens, re-create an environment from scratch pinning directly all your other dependencies during initial solve (e.g., `mamba create -n geoutils-env -c conda-forge geoutils myotherpackage==1.0.0`).
```

## Installing for contributors

```bash
git clone https://github.com/GlacioHack/geoutils.git
mamba env create -f geoutils/dev-environment.yml
```

After installing, you can check that everything is working by running the tests: `pytest -rA`.

## Dependencies

GeoUtils' required dependencies are:
- [Rasterio](https://rasterio.readthedocs.io/en/stable/) (version 1.3 and above),
- [GeoPandas](https://geopandas.org/en/stable/) (version 0.12 and above),
- [SciPy](https://scipy.org/),
- [Xarray](https://xarray.dev/),
- [Rioxarray](https://corteva.github.io/rioxarray/stable/).

which themselves depend notably on [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [pyproj](https://pyproj4.github.io/pyproj/stable/) and [pyogrio](https://pyogrio.readthedocs.io/en/latest/).

Optional dependencies are:
- [Matplotlib](https://matplotlib.org/) for plotting,
- [LasPy](https://laspy.readthedocs.io/en/latest/) for reading and writing LAS/LAZ/COPC point cloud files,
- [Numba](https://numba.pydata.org/) for faster filters,
- [Dask](https://www.dask.org/) for out-of-memory operations,
- [Psutil](https://psutil.readthedocs.io/en/latest/) and [Plotly](https://plotly.com/) for profiling computing speed and memory.
