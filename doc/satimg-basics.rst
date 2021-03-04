.. _satimg-basics:

SatImg basics
=============

Opening a raster file through SatImg
------------------------------------

Example with a Landsat image:

.. code-block::

    from geoutils import satimg
    image = satimg.SatelliteImage('LE71400412000304SGS00_B4_crop.TIF')


What the SatImg class does for you
----------------------------------

When reading your file, SatImg will try to load metadata information from the filename.
Example for the above filename:

::

    From filename: setting satellite as Landsat 7
    From filename: setting sensor as ETM+
    From filename: setting tile_name as (140, 41)
    From filename: setting datetime as 2000-10-30 00:00:00

Currently supporting the nomenclatures used for: Landsat, Sentinel-2, ArcticDEM, REMA, ASTER L1A, ASTER GDEM, NASADEM, TanDEM-X, SRTM and SPOT-5

More to come...