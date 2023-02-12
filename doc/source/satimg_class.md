(satimg-class)=

# The satellite image (`SatelliteImage`)

## Object definition

## Datetime parsing

Example with a Landsat image:

```{literalinclude} code/satimg-basics_open_file.py
:lines: 2-
```

When reading your file, SatImg will try to load metadata information from the filename.
For the above filename, this will be printed in the console:

```{eval-rst}
.. program-output:: $PYTHON -c "exec(open('code/satimg-basics_open_file.py').read())"
        :shell:
```

Currently supporting the nomenclatures used for: Landsat, Sentinel-2, ArcticDEM, REMA, ASTER L1A, ASTER GDEM, NASADEM, TanDEM-X, SRTM and SPOT-5

More to come...

```{eval-rst}
.. minigallery:: geoutils.SatelliteImage
        :add-heading:
        :heading-level: -
```

## Tile parsing

## Instrument metadata parsing
