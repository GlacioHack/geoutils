"""
GeoUtils is a Python package for the analysis of geospatial data.
"""

from geoutils.raster import Mask, Raster, SatelliteImage  # noqa isort:skip
from geoutils.vector import Vector  # noqa isort:skip
from geoutils import examples, projtools, raster, vector  # noqa isort:skip

try:
    from geoutils.version import version as __version__  # noqa
except ImportError:  # pragma: no cover
    raise ImportError(
        "geoutils is not properly installed. If you are "
        "running from the source directory, please instead "
        "create a new virtual environment (using conda or "
        "virtualenv) and then install it in-place by running: "
        "pip install -e ."
    )
