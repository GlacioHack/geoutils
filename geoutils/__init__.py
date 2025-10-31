# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GeoUtils is a Python package for the analysis of geospatial data.
"""

from geoutils import examples, projtools  # noqa
from geoutils._config import config  # noqa

from geoutils.raster import Raster  # noqa isort:skip
from geoutils.vector import Vector  # noqa isort:skip
from geoutils.pointcloud import PointCloud  # noqa isort:skip

# To-be-deprecated
from geoutils.raster import Mask  # noqa isort:skip

try:
    from geoutils._version import __version__ as __version__  # noqa
except ImportError:  # pragma: no cover
    raise ImportError(
        "geoutils is not properly installed. If you are "
        "running from the source directory, please instead "
        "create a new virtual environment (using conda or "
        "virtualenv) and then install it in-place by running: "
        "pip install -e ."
    )
