"""
Opening a raster from file
==========================

This example demonstrates the instantiation of a :class:`~geoutils.Raster` from file.
"""

import geoutils as gu

# %%
# We open an example raster. The data is, by default, unloaded.
img = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
img

# %%
# We can print more info on the raster.
print(img.info())

# %%
# The data will be loaded explicitly by any function requiring its :attr:`~geoutils.Raster.data`, such as :func:`~geoutils.Raster.show`.
img.show(cmap="Greys_r")

# %%
# Opening can be performed with several parameters, for instance choosing a single band with ``index`` and re-sampling with ``downsample``.
img = gu.Raster(gu.examples.get_path("everest_landsat_rgb"), indexes=2, downsample=4)
img

# %%
# The data is unloaded by default, even if when specifying a band or re-sampling.
# We can load it explicitly by calling :func:`~geoutils.Raster.load` (could have also passed `load_data=True` to # :class:`~geoutils.Raster`).
img.load()
img
