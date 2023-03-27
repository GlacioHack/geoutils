"""
Metric buffer and without overlap
=================================

This example demonstrates the metric buffering of a vector using :func:`~geoutils.Vector.buffer_metric` and  :func:`~geoutils.Vector.buffer_without_overlap`.
"""
# %%
# We open an example vector

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

# %%
# We buffer in metric units directly using :func:`~geoutils.Vector.buffer_metric`. Under the hood, this functionality reprojects to a local projection,
# buffers, and converts back to the original CRS.
vect_buff = vect.buffer_metric(buffer_size=500)

# %%
# Let's plot the raster and vector
ax = vect.ds.plot()
vect_buff.ds.plot(ax=ax, ec="k", fc="none")

# %%
# Many buffers are overlapping. To compute a buffer without overlap, one can use :func:`~geoutils.Vector.buffer_without_overlap`.
#
vect_buff_nolap = vect.buffer_without_overlap(buffer_size=500)
ax = vect_buff_nolap.ds.plot()
vect.ds.plot(ax=ax, ec="k", fc="none")

# %%
# We plot with color to see that the attributes are retained for every feature.
import matplotlib.pyplot as plt

ax = vect_buff_nolap.ds.plot(column="RGIId")
vect.ds.plot(ax=ax, ec="k", column="RGIId", alpha=0.5)
plt.show()
