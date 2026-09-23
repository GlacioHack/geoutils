"""
Metric buffer and without overlap
=================================

This example demonstrates the metric buffering of a vector using :meth:`~geoutils.vector.base.VectorBase.buffer_metric` and  :meth:`~geoutils.vector.base.VectorBase.buffer_without_overlap`.
"""

# %%
# We open an example vector

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_vect = gu.examples.get_path("everest_rgi_outlines")
vect = gu.open_vector(filename_vect)

# %%
# We buffer in metric units directly using :meth:`~geoutils.vector.base.VectorBase.buffer_metric`. Under the hood, this functionality reprojects to a local projection,
# buffers, and converts back to the original CRS.
vect_buff = vect.vct.buffer_metric(buffer_size=500)

# %%
# Let's plot the raster and vector
ax = vect.vct.plot()
vect_buff.vct.plot(ec="k", fc="none")

# %%
# Many buffers are overlapping. To compute a buffer without overlap, one can use :meth:`~geoutils.vector.base.VectorBase.buffer_without_overlap`.
#
vect_buff_nolap = vect.vct.buffer_without_overlap(buffer_size=500)
vect.vct.plot(ax="new")
vect_buff_nolap.vct.plot(ec="k", fc="none")

# %%
# We plot with color to see that the attributes are retained for every feature.
vect_buff_nolap.vct.plot(ax="new", column="Area")
vect.vct.plot(ec="k", column="Area", alpha=0.5)
