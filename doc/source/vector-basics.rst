.. _vector-basics:

Vector basics
=============

The Vector class builds upon the great functionalities of `GeoPandas <https://geopandas.org/>`_, with the aim to bridge the gap between vector and raster files.
It uses ``geopandas.GeoDataFrame`` as a base driver, accessible through ``Vector.ds``.


Opening a Vector file
---------------------

.. literalinclude:: code/vector-basics_open_file.py
        :lines: 2-6


Printing the Vector shows the underlying ``GeoDataFrame`` and some extra statistics:

.. code:: python
        
        print(outlines)

.. program-output:: $PYTHON -c "exec(open('code/vector-basics_open_file.py').read()); print(outlines)"
        :shell:


Masks can easily be generated for use with Rasters:

.. literalinclude:: code/vector-basics_open_file.py
        :lines: 8-13

We can prove that glaciers are bright (who could have known!?) by masking the values outside and inside of the glaciers:

.. code:: python
        
        print(f"Inside: {image.data[mask].mean():.1f}, outside: {image.data[~mask].mean():.1f}")

.. program-output:: $PYTHON -c "exec(open('code/vector-basics_open_file.py').read()); print(f'Inside: {image.data[mask].mean():.1f}, outside: {image.data[~mask].mean():.1f}')"
        :shell:

TODO: Add rasterize text.
