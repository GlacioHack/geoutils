"""
Loading and understanding Vectors
=================================

This is (right now) a dummy example for showing the functionality of :class:`geoutils.Vector`.
"""

import matplotlib.pyplot as plt

import geoutils as gu

# %%
# Example raster:
glaciers = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

# %%
# Info:
print(glaciers)


# %%
# A plot:
for _, glacier in glaciers.ds.iterrows():
    plt.plot(*glacier.geometry.exterior.xy)
plt.show()
