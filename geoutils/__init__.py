"""
GeoUtils is a python package of raster and vector tools.
"""

from . import georaster
from . import geovector
from . import datasets
from . import satimg

try:
    from geoutils.version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('geoutils is not properly installed. If you are '
                      'running from the source directory, please instead '
                      'create a new virtual environment (using conda or '
                      'virtualenv) and then install it in-place by running: '
                      'pip install -e .')
