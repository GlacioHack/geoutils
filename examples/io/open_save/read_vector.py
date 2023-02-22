"""
Opening a vector
================

This example demonstrates the instantiation of a vector through :class:`geoutils.Vector`.
"""

import geoutils as gu

# %%
# We open an example vector.
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
vect

# %%
# We can print more info on the vector.
print(vect)

# %%
# Let's plot:
import matplotlib.pyplot as plt
for _, glacier in vect.ds.iterrows():
    plt.plot(*glacier.geometry.exterior.xy)
plt.show()
