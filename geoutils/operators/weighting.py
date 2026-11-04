# Copyright (c) 2026 GeoUtils developers
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

"""Weighting through use of the error structure."""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from geoutils.operators.base import LocalData

if TYPE_CHECKING:
    from geoutils.operators.interpolator import Interpolator
    from geoutils.operators.reducer import Reducer
    from geoutils.uncertainty.error_structure import ErrorStructure

OperatorType = TypeVar("OperatorType", bound="Interpolator | Reducer")


def _with_error_structure(operator: OperatorType, error_structure: ErrorStructure | None) -> OperatorType:
    """Attach an error model to a copy, leaving the reusable operator unchanged."""

    if error_structure is None:
        return operator
    from geoutils.uncertainty.error_structure import ErrorStructure

    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    configured = copy(operator)
    configured.error_structure = error_structure
    return configured


def _local_error_data(
    data: LocalData, error_structure: ErrorStructure | None, predictors: Mapping[str, Any] | None = None
) -> LocalData:
    """Attach observation covariance in the neighborhood's value order.

    Repeated source IDs describe the same observation. Calculate covariance for each distinct observation,
    then repeat its rows and columns wherever that observation occurs in the neighborhood.
    """

    if error_structure is None or data.error_covariance is not None or len(data.values) == 0:
        return data

    # Coordinates and IDs must describe the same distinct observations
    valid = data._valid_values()
    if not np.any(valid):
        return data
    labels = pd.Index(data.source_ids[valid])
    first = np.flatnonzero(~labels.duplicated())
    unique = labels[first]
    positions = unique.get_indexer(labels)
    coordinates = None if data.coordinates is None else data.coordinates[valid][first]
    aligned_predictors = {}
    for name, values in (predictors or {}).items():
        if isinstance(values, Mapping):
            aligned_predictors[name] = np.asarray([values[source_id] for source_id in unique])
        elif np.ndim(values) == 0:
            aligned_predictors[name] = values
        else:
            raise ValueError("Spatial magnitude predictors must be scalars or mappings keyed by source ID.")
    from geoutils.uncertainty.error_structure import _prepare_source_errors, _source_covariance_matrix

    locations, component_data = _prepare_source_errors(error_structure, len(unique), coordinates, aligned_predictors)
    covariance = np.zeros((len(data.values), len(data.values)))
    covariance[np.ix_(valid, valid)] = _source_covariance_matrix(
        unique.to_numpy(), locations, component_data, positions
    )
    return replace(data, error_covariance=covariance)
