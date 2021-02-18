from setuptools import setup
from os import path

FULLVERSION = '0.0.1'
VERSION = FULLVERSION

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'geoutils',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()


setup(name='geoutils',
      version=FULLVERSION,
      description='Tools for working with geospatial data',
      url='https://www.github.com/GlacioHack/geoutils/',
      author='The GlacioHack Team',
      license='BSD-3',
      packages=['geoutils'],
      install_requires=['rasterio', 'geopandas', 'pyproj'],
      extras_require={'rioxarray': ['rioxarray']},
      scripts=[],
      zip_safe=False)
