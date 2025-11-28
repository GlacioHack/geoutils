####
# Part not shown in the doc, to get data files ready
import geoutils

landsat_b4_path = geoutils.examples.get_path("everest_landsat_b4")
landsat_b4_crop_path = geoutils.examples.get_path("everest_landsat_b4_cropped")
geoutils.Raster(landsat_b4_path).to_file("myraster1.tif")
geoutils.Raster(landsat_b4_crop_path).to_file("myraster2.tif")
####

import numpy as np
import rasterio as rio

# Opening of two rasters
rast1 = rio.io.DatasetReader("myraster1.tif")
rast2 = rio.io.DatasetReader("myraster2.tif")

# Equivalent of a match-reference reprojection
# (returns an array, not a raster-type object)
arr1_reproj, _ = rio.warp.reproject(
    source=rast1.read(),
    destination=np.ones(rast2.shape),
    src_transform=rast1.transform,
    src_crs=rast1.crs,
    src_nodata=rast1.nodata,
    dst_transform=rast2.transform,
    dst_crs=rast2.crs,
    dst_nodata=rast2.nodata,
)

# Equivalent of array interfacing
# (ensuring nodata and dtypes are rightly
# propagated through masked arrays)
ma1_reproj = np.ma.MaskedArray(data=arr1_reproj, mask=(arr1_reproj == rast2.nodata))
ma2 = rast2.read(masked=True)
ma_result = (1 + ma2) / (ma1_reproj)


# Equivalent of saving
# (requires to define a logical
# nodata for the data type)
def custom_func(dtype, nodata1, nodata2):
    return -9999


out_nodata = custom_func(dtype=ma_result.dtype, nodata1=rast1.nodata, nodata2=rast2.nodata)
with rio.open(
    "myresult.tif",
    mode="w",
    height=rast2.height,
    width=rast2.width,
    count=rast1.count,
    dtype=ma_result.dtype,
    crs=rast2.crs,
    transform=rast2.transform,
    nodata=rast2.nodata,
) as dst:
    dst.write(ma_result.filled(out_nodata))

####
# Part not shown in the docs, to clean up
import os

for file in ["myraster1.tif", "myraster2.tif", "myresult.tif"]:
    os.remove(file)
####
