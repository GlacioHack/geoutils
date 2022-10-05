from glob import glob
from os import path
from typing import Optional

from setuptools import setup

FULLVERSION = "0.0.9"
VERSION = FULLVERSION

write_version = True


def write_version_py(filename: Optional[str] = None) -> None:
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if filename is None:
        filename = path.join(path.dirname(__file__), "geoutils", "version.py")

    a = open(filename, "w")
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()


with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geoutils",
    version=FULLVERSION,
    description="Tools for working with geospatial data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.github.com/GlacioHack/geoutils/",
    author="The GlacioHack Team",
    license="BSD-3",
    packages=["geoutils", "geoutils.georaster"],
    python_requires=">=3.8",
    install_requires=[
        "rasterio",
        "geopandas >= 0.10.0",
        "pyproj",
        "scipy",
        "typing-extensions; python_version < '3.8'",
        "matplotlib",
        "tqdm",
    ],
    extras_require={"rioxarray": ["rioxarray"]},
    scripts=["geoutils/geoviewer.py"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
)
