"""
Parsing sensor metadata
=======================

This example demonstrates opening a raster while parsing image sensor metadata.
"""

import geoutils as gu

# %%
# We print the filename of our raster that, as often with satellite data, holds metadata information.
filename_geoimg = gu.examples.get_path("everest_landsat_b4")
import os

print(os.path.basename(filename_geoimg))

# %%
# We open it as a raster and parse its filename, un-silencing the attribute retrieval to see it printed.
img = gu.open_raster(filename_geoimg)
sensor_metadata = gu.raster.satimg.parse_and_convert_metadata_from_filename(filename_geoimg, silent=False)
img.attrs.update(sensor_metadata)

# %%
# We have now retrieved the metadata, stored in the :attr:`geoutils.Raster.tags` attribute.
img.rst.tags
