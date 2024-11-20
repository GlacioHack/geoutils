"""
Parsing sensor metadata
=======================

This example demonstrates the instantiation of a raster while parsing image sensor metadata.
"""

import geoutils as gu

# %%
# We print the filename of our raster that, as often with satellite data, holds metadata information.
filename_geoimg = gu.examples.get_path("everest_landsat_b4")
import os

print(os.path.basename(filename_geoimg))

# %%
# We open it as a raster with the option to parse metadata, un-silencing the attribute retrieval to see it printed.
img = gu.Raster(filename_geoimg, parse_sensor_metadata=True, silent=False)

# %%
# We have now retrieved the metadata, stored in the :attr:`geoutils.Raster.tags` attribute.
img.tags
