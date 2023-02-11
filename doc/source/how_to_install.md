(how-to-install)=

# How to install

## Installing with conda (recommended)

```bash
conda install -c conda-forge --strict-channel-priority geoutils
```

**Notes**

- The `--strict-channel-priority` flag seems essential for Windows installs to function correctly, and is recommended for UNIX-based systems as well.

- Solving dependencies can take a long time with `conda`. To speed up this, consider installing `mamba`:

  ```bash
  conda install mamba -n base -c conda-forge
  ```

  Once installed, the same commands can be run by simply replacing `conda` by `mamba`. More details available through the [mamba project](https://github.com/mamba-org/mamba).

## Installing with pip

```bash
pip install geoutils
```

**NOTE**: Setting up GDAL and PROJ may need some extra steps, depending on your operating system and configuration.

## Installing for contributors

```shell
git clone https://github.com/GlacioHack/xdem.git
cd ./xdem
mamba env create -f dev-environment.yml
mamba activate xdem
pip install -e .
```

After installing, we recommend to check that everything is working by running the tests:

`pytest -rA`
