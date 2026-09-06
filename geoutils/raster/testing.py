# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing helpers for eager and Dask-backed raster arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional


def _array_equal_or_close(
    left: Any,
    right: Any,
    *,
    use_allclose: bool,
    rtol: float,
    atol: float,
) -> bool:
    """Reduce eager or Dask array equality without collecting a lazy array."""

    if np.shape(left) != np.shape(right):
        return False
    if is_dask_array(left) or is_dask_array(right):
        import_optional("dask")
        import dask.array as da

        left_array = da.asarray(left)
        right_array = da.asarray(right)
        if use_allclose:
            result = da.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True)
        else:
            result = da.all((left_array == right_array) | (da.isnan(left_array) & da.isnan(right_array)))
        return bool(result.compute())
    if use_allclose:
        return bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(left, right, equal_nan=True))
