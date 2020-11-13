"""
GeoUtils is a python package of raster and vector tools.
"""
from . import raster_tools
from . import vector_tools

try:
    from GeoUtils.version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('GeoUtils is not properly installed. If you are '
                      'running from the source directory, please instead '
                      'create a new virtual environment (using conda or '
                      'virtualenv) and then install it in-place by running: '
                      'pip install -e .')

