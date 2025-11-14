####
# Part not shown in the doc, to get data files ready
import geoutils

landsat_b4_path = geoutils.examples.get_path("everest_landsat_b4")
everest_outlines_path = geoutils.examples.get_path("everest_rgi_outlines")
geoutils.Raster(landsat_b4_path).to_file("myraster.tif")
geoutils.Vector(everest_outlines_path).to_file("myvector.gpkg")
####

import geopandas as gpd
import rasterio as rio

# Opening a vector and a raster
df = gpd.read_file("myvector.gpkg")
rast2 = rio.io.DatasetReader("myraster.tif")

# Equivalent of a metric buffering
# (while keeping a frame object)
gs_m_crs = df.to_crs(df.estimate_utm_crs())
gs_m_crs_buff = gs_m_crs.buffer(distance=100)
gs_buff = gs_m_crs_buff.to_crs(df.crs)
df_buff = gpd.GeoDataFrame(geometry=gs_buff)

# Equivalent of creating a rasterized mask
# (ensuring CRS are similar)
df_buff = df_buff.to_crs(rast2.crs)
mask = rio.features.rasterize(
    shapes=df.geometry, fill=0, out_shape=rast2.shape, transform=rast2.transform, default_value=1, dtype="uint8"
)
mask = mask.astype("bool")

# Equivalent of indexing with mask
values = rast2.read(1, masked=True)[mask]

####
# Part not shown in the doc, to get data files ready
import os

for file in ["myraster.tif", "myvector.gpkg"]:
    os.remove(file)
####
