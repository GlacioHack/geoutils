# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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

"""Functionalities for classifications of Raster objects"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd

from geoutils._typing import NDArrayBool
from geoutils.raster import Mask, Raster, RasterType


class ClassificationLayer(ABC):
    def __init__(
        self,
        raster: RasterType | str,
        name: str,
    ):
        """
        Initialize the ClassificationLayer object with common attributes.

        :param raster: The Raster on which classification is applied (geoutils.Raster).
        :param name: Name of the classification layer.
        """
        if isinstance(raster, str):
            self.raster = Raster(raster)
        else:
            self.raster = raster
        self.name = name
        self.class_names: dict[str, int | list[int]] = {}  # Will be set in subclasses
        self.classification_masks: Mask | None = None  # Will store classification result (as geoutils.Mask)
        self.stats_df: pd.DataFrame | None = None  # Will store computed statistics for required classes

    @abstractmethod
    def apply(self) -> None:
        """
        Abstract method to apply the classification logic. Subclasses should implement their specific classification
        logic. This method will create the classification as a :class:`geoutils.Mask` object.
        """
        pass

    def get_stats(
        self,
        raster: RasterType | None = None,
        stats: str | list[str] | None = None,
        classes: str | list[str] | None = None,
    ) -> None:
        """
        Compute the required statistics on the classified pixels.

        :param raster: Raster on which the classification masks should be applied for computing the stats.
        :param stats: List of required statistics to compute (optional, default is all statistics in
            geoutils.Raster.get_stats).
        :param classes: List of classes on which to compute statistics (optional, default is all classes).

        :raise ValueError: if req_stats_classes are not class names.
        """
        if self.classification_masks is None or self.class_names is None:
            raise ValueError("Classification has not been applied yet. Call apply_classification() first.")

        # Convert req_stats and req_stats_classes to list
        if isinstance(stats, str):
            stats = [stats]
        if isinstance(classes, str):
            classes = [classes]

        # Check if req_stats_classes are class names
        if classes:
            for req_class in classes:
                if req_class not in self.class_names.keys():
                    raise ValueError(
                        f"{req_class} is not a class name. Class names are: {list(self.class_names.keys())}"
                    )

        if raster is None:
            raster = self.raster  # type: ignore
        assert isinstance(raster, Raster)

        stats_list = []

        # Loop over each class in the classification mask
        for i, (class_name, class_idx) in enumerate(self.class_names.items()):
            if classes and class_name not in classes:
                continue

            # Compute statistics for the class
            class_stats = raster.get_stats(stats_name=stats, inlier_mask=self.classification_masks[i])  # type: ignore

            # Store class, bin info (if applicable), and stats as a dictionary
            class_info: dict[str, str | int | list[int] | np.floating[Any]] = {
                "class_name": class_name,
                "class_idx": class_idx,
            }
            class_info.update(class_stats)

            stats_list.append(class_info)

            # Convert the list of dictionaries into a pandas DataFrame
        self.stats_df = pd.DataFrame(stats_list)

    def save(self, output_dir: str, save_list: list[Literal["masks", "names", "stats"]] | None = None) -> None:
        """
        Save the classification result, class names, and computed statistics to files.

        :param output_dir: Directory where the output files will be saved.
        :param save_list: Objects to save.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if save_list is None:
            save_list = ["masks", "names", "stats"]

        # Save classification as a GeoTIFF file
        if self.classification_masks is not None and "masks" in save_list:
            classification_path = os.path.join(output_dir, f"{self.name}.tif")
            self.classification_masks.save(classification_path)
            logging.warning("Classification masks saved under %s", classification_path)

        # Save class names as a JSON file
        if self.class_names is not None and "names" in save_list:
            class_names_path = os.path.join(output_dir, f"{self.name}_classes.json")
            with open(class_names_path, "w") as f:
                json.dump(self.class_names, f, indent=4)
            logging.warning("Class names saved under %s", class_names_path)

        # Save statistics as CSV file
        if self.stats_df is not None and "stats" in save_list:
            stats_csv_path = os.path.join(output_dir, f"{self.name}_stats.csv")
            self.stats_df.to_csv(stats_csv_path, index=False)
            logging.warning("Stats saved under %s", stats_csv_path)


class Binning(ClassificationLayer):
    """Apply binning to a raster using input bins"""

    def __init__(
        self,
        raster: RasterType | str,
        name: str,
        bins: list[float],
    ) -> None:
        """
        Initialize the RasterBinning object.
        :param raster: The Raster object to classify (only 1-band).
        :param name: The name of the classification.
        :param bins: List of bin edges to classify the raster values (e.g., [0, 10, 20, 50, 100, inf]).
        """
        super().__init__(raster, name)
        # Classification works with 1-band raster
        if self.raster.count != 1:
            self.raster._data = self.raster.data[0]
        self.bins = bins
        self.class_names = {f"[{self.bins[i - 1]}, {self.bins[i]})": i for i in range(1, len(self.bins))}

    def apply(self) -> None:
        """
        Apply classification based on the bin ranges. This creates a :class:`geoutils.Mask` object where each
        band represent one class.
        """
        # Use np.digitize to classify the raster values into bins
        class_indices = np.digitize(self.raster.data, bins=self.bins)

        # Create a boolean mask for each class
        masks_array = class_indices != np.arange(1, len(self.bins))[:, None, None]

        # Store classification as a Mask object
        self.classification_masks = Mask.from_array(masks_array, transform=self.raster.transform, crs=self.raster.crs)


class Segmentation(ClassificationLayer):
    """
    Segmentation classification using pre-defined masks.

    This class handles classification of a raster using a set of boolean masks
    that define class membership for each pixel.
    """

    def __init__(
        self,
        raster: RasterType | str,
        name: str,
        classification_masks: Mask | NDArrayBool,
        class_names: dict[str, int | list[int]],
    ) -> None:
        """
        Initialize a Segmentation classification object.

        :param raster: Input raster or path to raster file.
        :param name: Name of the classification layer.
        :param classification_masks: Boolean masks indicating class membership. Can be a `Mask` or raw NumPy array.
        :param class_names: Dictionary mapping class names to class indices.
        """
        super().__init__(raster, name)
        if isinstance(classification_masks, Mask):
            self.classification_masks = classification_masks
        else:
            self.classification_masks = Mask.from_array(
                classification_masks, transform=self.raster.transform, crs=self.raster.crs
            )
        self.class_names = class_names

    def apply(self) -> None:
        """Masks already existing -> No application is needed"""
        pass

    def save(self, output_dir: str, save_list: list[Literal["masks", "names", "stats"]] | None = None) -> None:
        """Save only statistic in pd.dataframe as classification masks and class names are already known"""
        if save_list is None:
            save_list = ["stats"]
        return super().save(output_dir, save_list=save_list)


class Fusion(ClassificationLayer):
    """
    A fusion classification layer that combines multiple classification masks into fused classes.

    This class takes a list of binary classification masks and fuses them using logical operations.
    The final fused masks are stored in a multi-band Mask, where each band represents a unique
    combination of classes from the input layers. Empty fused classes are removed.
    """

    def __init__(
        self,
        raster: RasterType | str,
        name: str,
        layers_to_fuse: list[Mask | NDArrayBool],
        layers_class_names: list[dict[str, int | list[int]]],
    ) -> None:
        """
        Initialize a Fusion classification object.

        :param raster: Input raster or path to raster file.
        :param name: Name of the classification layer.
        :param layers_to_fuse: The list of classification masks to fuse.
        :param layers_class_names: List of class-name-to-index mappings for each classification layer.
        """
        super().__init__(raster=raster, name=name)
        self.list_classification_layers = layers_to_fuse
        self.layers_class_names = layers_class_names

    def apply(self) -> None:
        """
        Apply fusion classification by computing the intersection of masks from each input classification layer.
        The fused classification mask is stored as a 3D Mask object in `self.classification_masks`.
        The combined class names are stored in `self.class_names`.
        """

        from itertools import product

        def _get_combined_mask(mask_layer: NDArrayBool, indices: int | list[int]) -> NDArrayBool:
            """
            Return a combined mask from a single classification layer using AND logic.
            """
            if isinstance(indices, list):
                combined = mask_layer[indices[0] - 1].copy()
                for idx in indices[1:]:
                    combined &= mask_layer[idx - 1]
                return combined
            return mask_layer[indices - 1].copy()

        # Generate all possible combinations of class names across layers
        class_combinations = list(product(*[list(cn.keys()) for cn in self.layers_class_names]))

        fused_masks: list[NDArrayBool] = []
        fused_class_names: dict[str, int] = {}

        for i, combo in enumerate(class_combinations):
            # Start by getting the mask(s) for the first layer
            idx = self.layers_class_names[0][combo[0]]
            combined_mask = _get_combined_mask(self.list_classification_layers[0], idx)  # type: ignore

            # Fuse with remaining layers using OR
            for j in range(1, len(combo)):
                idx = self.layers_class_names[j][combo[j]]
                next_mask = _get_combined_mask(self.list_classification_layers[j], idx)  # type: ignore
                combined_mask |= next_mask

            # Only keep if at least one pixel is False
            if not combined_mask.all():
                fused_masks.append(combined_mask)
                fused_class_names["_".join(combo)] = i + 1
            else:
                logging.warning("Empty class %s, removed from the fusion", "_".join(combo))

        # Handle empty case
        if fused_masks:
            stacked_mask = np.stack(fused_masks)
        else:
            stacked_mask = np.zeros((1,) + self.raster.shape, dtype=bool)
            fused_class_names["empty_class"] = 1

        # Store results
        self.classification_masks = Mask.from_array(
            stacked_mask,
            transform=self.raster.transform,
            crs=self.raster.crs,
        )
        self.class_names = fused_class_names  # type: ignore
