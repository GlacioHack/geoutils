% GeoUtils documentation master file, created by
% sphinx-quickstart on Fri Nov 13 17:43:16 2020.
% You can adapt this file completely to your liking, but it should at least
% contain the root 'toctree' directive.

# Welcome to GeoUtils' documentation!

GeoUtils aims to make handling of georeferenced data, either in raster or vector format, easier and available to anyone.

:::{important}
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are
working on for reproducibility!
We are working on making features fully consistent for the first long-term release `v0.1` (likely sometime in 2023).
:::

```{toctree}
:caption: Getting started
:maxdepth: 2

about_xdem
how_to_install
quick_start
```

```{literalinclude} code/index_example.py
```

```{eval-rst}
.. program-output:: $PYTHON code/index_example.py
        :shell:
```

```{toctree}
:maxdepth: 1

raster-basics
satimg-basics
vector-basics
auto_examples/index.rst
api
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
