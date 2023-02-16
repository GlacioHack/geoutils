(how-to-install)=

# How to install

## Installing with `mamba` (recommended)

```bash
mamba install -c conda-forge geoutils
```

```{note}
Solving dependencies can take a long time with `conda`, `mamba` significantly speeds up the process. Install it with:

    conda install mamba -n base -c conda-forge

Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available in the [mamba documentation](https://mamba.readthedocs.io/en/latest/).
```

## Installing with `pip`

```bash
pip install geoutils
```

```{note}
Setting up GDAL and PROJ may require some extra steps, depending on your operating system and configuration.
```

## Installing for contributors

```bash
git clone https://github.com/GlacioHack/xdem.git
cd ./xdem
mamba env create -f dev-environment.yml
mamba activate xdem
pip install -e .
```

After installing, check that everything is working by running the tests: `pytest -rA`.
