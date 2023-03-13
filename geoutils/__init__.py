"""
GeoUtils is a python package of raster and vector tools.
"""

from geoutils import examples, raster, vector, projtools  # noqa
from geoutils.raster import Mask, Raster, SatelliteImage  # noqa
from geoutils.vector import Vector  # noqa

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
