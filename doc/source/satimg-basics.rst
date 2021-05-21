.. _satimg-basics:

SatImg basics
=============

Opening a raster file through SatImg
------------------------------------

Example with a Landsat image:

.. literalinclude:: code/satimg-basics_open_file.py
        :lines: 2-


What the SatImg class does for you
----------------------------------

When reading your file, SatImg will try to load metadata information from the filename.
For the above filename, this will be printed in the console:

.. program-output:: $PYTHON -c "exec(open('code/satimg-basics_open_file.py').read())"
        :shell:

Currently supporting the nomenclatures used for: Landsat, Sentinel-2, ArcticDEM, REMA, ASTER L1A, ASTER GDEM, NASADEM, TanDEM-X, SRTM and SPOT-5

More to come...
