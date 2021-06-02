.. GeoUtils documentation master file, created by
   sphinx-quickstart on Fri Nov 13 17:43:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GeoUtils's documentation!
================================
GeoUtils aims to make handling of georeferenced data, either in raster or vector format, easier and available to anyone.

Our functionalities are mostly based on `rasterio <https://rasterio.readthedocs.io>`_ and `GeoPandas <https://geopandas.org/>`_, but here are some of the additional benefits provided by GeoUtils:

* for raster objects, georeferences and data are stored into a single object of the class ``Raster``, making it easier to modify the data in-place: reprojection, cropping, additions/subtractions are all (or will be...) on-line operations!
* the interactions between raster and vectors, such as rasterizing, clipping or cropping are made easier thanks to the class ``Vector``.


.. literalinclude:: code/index_example.py

.. program-output:: $PYTHON code/index_example.py
        :shell:

.. toctree::
    :maxdepth: 1

    raster-basics
    satimg-basics
    vector-basics
    api




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
