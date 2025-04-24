---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: geoutils-env
  language: python
  name: geoutils
---
(classification)=

# Classification

The classification module provides tools for classifying {class}`~geoutils.Raster` object into discrete classes.
This can be used to categorize continuous data (e.g., elevation, slope) or to group raster pixels based on predefined
categories (e.g., land-use, vegetation types).

Classification is achieved through the {class}`~geoutils.raster.ClassificationLayer` abstract class, which provides the
foundation for implementing specific classification workflows.

## The {class}`~geoutils.raster.ClassificationLayer` class
{class}`~geoutils.raster.ClassificationLayer` is the base class for all classification layers.
It provides methods for setting up raster classification, computing statistics for each class, and saving classification results.

### Methods:
- {func}`~geoutils.raster.ClassificationLayer.get_stats`: Calculates statistics using {func}`~geoutils.Raster.get_stats` for the classified raster data.
- {func}`~geoutils.raster.ClassificationLayer.save`: Saves the classification result, class names, and statistics to the specified directory:
  - The classification mask is saved as a `.tif` file.
  - Class names are saved as a `.json` file.
  - Computed statistics are saved as a `.csv` file.

## Apply binning
{class}`~geoutils.raster.Binning` is a specific implementation of {class}`~geoutils.raster.ClassificationLayer`
designed for classifying continuous raster data using binning. This class allows you to define ranges (bins) for continuous raster values
and classify the pixels into discrete classes based on these ranges. The resulting classification is stored as a multi-band {class}`~geoutils.Mask` object.

### Example workflow:
1. **Load raster:** Load a raster file.
2. **Define bins:** Set the bin ranges to classify the raster data (e.g., elevation ranges).
3. **Apply classification:** Classify the raster data based on the bin ranges.
4. **Compute statistics:** Compute various statistics for each class.
5. **Save results:** Save the classification and statistics to files.

```{code-cell} ipython3
import numpy as np
import geoutils as gu
from geoutils.raster import Binning

# load raster (note, the raster parameter can also be a file path)
raster_file = gu.examples.get_path("exploradores_aster_dem")
raster = gu.Raster(raster_file)

# Define bins
bins = [0, 1000, 2000, 3000, np.inf]

# Create Binning object
binning= Binning(raster, "elevation", bins)

# Apply binning
binning.apply()

# Compute statistics
binning.get_stats(classes=["[0, 1000)", "[3000, inf)"])

# Save results
binning.save("elevation_binning")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("elevation_binning/elevation.tif")
os.remove("elevation_binning/elevation_classes.json")
os.remove("elevation_binning/elevation_stats.csv")
os.rmdir("elevation_binning")
```

```{code-cell} ipython3
# print stats
print(binning.stats_df)
```

## Apply segmentation
{class}`~geoutils.raster.Segmentation` is a subclass of {class}`~geoutils.raster.ClassificationLayer` designed for
categorical classification from pre-defined segmentation masks. It allows assigning class names to each mask and computing related statistics.

### Example workflow:
1. **Load Raster:** Load a raster.
2. **Provide masks:** Supply segmentation masks as a {class}`~geoutils.Mask` or boolean NumPy array.
3. **Assign class names:** Provide a dictionary mapping class IDs to names.
4. **Apply classification & save results:** Optionally compute stats or export masks.

```{code-cell} ipython3
from geoutils.raster import Segmentation
from geoutils import Mask

# Load raster
raster = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

# Create dummy mask (e.g., two classes)
mask_array = np.stack([raster.data > 1000, raster.data <= 1000])
class_names = {"Water": 1, "Land": 2}

# Initialize Segmentation
seg = Segmentation(raster, "water", mask_array, class_names)

# Compute stats
seg.get_stats(stats=["Mean", "Median"])

# Save stats
seg.save("segmentation_output")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("segmentation_output/water_stats.csv")
os.rmdir("segmentation_output")
```

```{code-cell} ipython3
# print stats
print(seg.stats_df)
```

## Apply fusion
{class}`~geoutils.raster.Fusion` is a subclass of {class}`~geoutils.raster.ClassificationLayer` designed to combine multiple classification masks into fused categories.
It computes all possible combinations across input class layers, generating new composite classes and statistics accordingly.

### Example Workflow:
1. **Load Raster:** Load a raster.
2. **Define classifications:** Use classification layers like {class}`~geoutils.raster.Binning` or {class}`~geoutils.raster.Segmentation`.
3. **Create fusion layer:** Combine classifications nto a single fused output.
4. **Compute stats & save results:** Analyze and export the fused classification.

```{code-cell} ipython3
from geoutils.raster import Fusion

# Load raster
raster = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

# Define binning thresholds
bins1 = [0, 1000, 2000, 3000, np.inf]
bins2 = [0, 1500, 2500, np.inf]

# Create classification layers
classifier1 = Binning(raster=raster, name="bin1", bins=bins1)
classifier2 = Binning(raster=raster, name="bin2", bins=bins2)
classifier1.apply()
classifier2.apply()

# Create Fusion classification
fusion = Fusion(
    raster=raster,
    name="fusion",
    layers_to_fuse=[
        classifier1.classification_masks.data,
        classifier2.classification_masks.data,
    ],
    layers_class_names=[
        classifier1.class_names,
        classifier2.class_names,
    ],
)

# Apply fusion
fusion.apply()

# Compute stats
fusion.get_stats()

# Save results
fusion.save("fusion_output")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("fusion_output/fusion.tif")
os.remove("fusion_output/fusion_classes.json")
os.remove("fusion_output/fusion_stats.csv")
os.rmdir("fusion_output")
```

```{code-cell} ipython3
# print stats
print(fusion.stats_df)
```
