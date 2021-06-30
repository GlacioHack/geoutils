from glob import glob
from os import path
from typing import Optional

from setuptools import setup

FULLVERSION = "0.0.3"
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

# get all data in the datasets module

data_files = []

for item in glob("geoutils/datasets/*"):
    bname = path.basename(item)
    if not bname.startswith("__"):
        data_files.append(path.join("datasets", bname))


setup(
    name="geoutils",
    version=FULLVERSION,
    description="Tools for working with geospatial data",
    url="https://www.github.com/GlacioHack/geoutils/",
    author="The GlacioHack Team",
    license="BSD-3",
    packages=["geoutils", "geoutils.datasets"],
    package_data={"geoutils": data_files},
    install_requires=["rasterio", "geopandas", "pyproj", "scipy"],
    extras_require={"rioxarray": ["rioxarray"]},
    scripts=["geoutils/geoviewer.py"],
    zip_safe=False,
)
