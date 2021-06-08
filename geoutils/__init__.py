"""
GeoUtils is a python package of raster and vector tools.
"""

from geoutils import datasets, georaster, geovector, satimg  # noqa
from geoutils.georaster import Raster  # noqa
from geoutils.geovector import Vector  # noqa
from geoutils.satimg import SatelliteImage  # noqa

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
