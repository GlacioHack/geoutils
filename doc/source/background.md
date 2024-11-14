(background)=

# Background

More information on how the package was created, who are the people behind it, and its mission.

## Inspiration

GeoUtils was created during the [GlacioHack](https://github.com/GlacioHack) hackaton event, that took place online on November 8, 2020 and was initiated by
Amaury Dehecq<sup>2</sup>.

```{margin}
<sup>2</sup>More on our GlacioHack founder at [adehecq.github.io](https://adehecq.github.io/)!
```

GeoUtils is inspired by previous efforts that were built directly on top of GDAL and OGR, namely:

- the older, homonymous package [GeoUtils](https://github.com/adehecq/geoutils_old),
- the class [pybob.GeoImg](https://github.com/iamdonovan/pybob/blob/master/pybob/GeoImg.py),
- the package [pygeotools](https://github.com/dshean/pygeotools), and
- the package [salem](https://github.com/fmaussion/salem).

## The people behind GeoUtils

The initial core development of GeoUtils was mainly performed by members of the Glaciology group of the _Laboratory of Hydraulics, Hydrology and
Glaciology (VAW)_ at ETH Zürich<sup>3</sup> and of the _University of Fribourg_, both in Switzerland. The package also received contributions by members of
the _University of Oslo_, Norway, the _University of Washington_, US and _Université Grenobles Alpes_, France.

```{margin}
<sup>3</sup>Check-out [glaciology.ch](https://glaciology.ch) on our founding group of VAW glaciology!
```

We are not software developers but geoscientists, and we try our best to offer tools that can be useful to a larger group,
documented, reliable and maintained. All development and maintenance is made on a voluntary basis and we welcome
any new contributors. See some information on how to contribute in the dedicated page of our
[GitHub repository](https://github.com/GlacioHack/geoutils/blob/main/CONTRIBUTING.md).

## Mission

```{epigraph}
The core mission of GeoUtils is to be **easy-of-use**, **robust**, **reproducible** and **fully open**.

Additionally, GeoUtils aims to be **efficient**, **scalable** and **state-of-the-art**.
```

In details, those mean:

- **Ease-of-use:** all basic operations or methods only require a few lines of code to be performed;

- **Robustness:** all methods are tested within our continuous integration test-suite, to enforce that they always perform as expected;

- **Reproducibility:** all code is version-controlled and release-based, to ensure consistency of dependent packages and works;

- **Open-source:** all code is accessible and reusable to anyone in the community, for transparency and open governance.

```{note}
:class: margin
Additional mission points, in particular **scalability**, are partly developed but not a priority until our first long-term release ``v0.1`` is reached.
```

And, additionally:

- **Efficiency**: all methods should be optimized at the lower-level, to function with the highest performance offered by Python packages;

- **Scalability**: all methods should support both lazy processing and distributed parallelized processing, to work with high-resolution data on local machines as well as on HPCs;

- **State-of-the-art**: all methods should be at the cutting edge of remote sensing science, to provide users with the most reliable and up-to-date tools.
