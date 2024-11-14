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
