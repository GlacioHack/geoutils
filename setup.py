from setuptools import setup


setup(name='GeoUtils',
      version='0.1',
      description='',
      url='',
      author='The GlacioHack Team',
      license='BSD-3',
      packages=['GeoUtils'],
      install_requires=['rasterio', 'geopandas'],
      extras_require={'rioxarray':['rioxarray']},
      scripts=[],
      zip_safe=False)
