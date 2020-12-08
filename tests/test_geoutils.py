"""
Test functions for geoutils (not using unittest)
"""
import os
import geoutils.georaster as gr
import geoutils.geovector as gv
import matplotlib.pyplot as plt
import numpy as np
import pytest

fn_img = os.path.join('/home/atom/code/devel/libs/GeoUtils/tests','data','LE71400412000304SGS00_B4_crop.TIF')
fn_img2 = os.path.join('/home/atom/code/devel/libs/GeoUtils/tests','data','LE71400412000304SGS00_B4_crop2.TIF')

#to execute from console

# test_img = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','LE71400412000304SGS00_B4_crop.TIF')
# test_img2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','LE71400412000304SGS00_B4_crop2.TIF')

def test_load_img(fn_img):

    r = gr.Raster(fn_img)
    print('Printing raster info (RIO)')
    print(r)

    print('Printing raster info with stats (Raster class)')
    print(r.info(stats=True))

    print('Loading data')
    r.load()

def test_crop_img(fn_img,fn_img2):
    r = gr.Raster(fn_img)
    print('Raster 1:')
    print(r.info())

    r2 = gr.Raster(fn_img2)
    print('Raster 2:')
    print(r2.info())

    plt.figure()
    r.show(title='full image 1')

    plt.figure()
    r2.show(title='full image 2')

    print('Cropped raster')
    r.crop(r2)
    print(r.info())

    plt.figure()
    r.show(title='Crop')

def test_reproj_img(fn_img,fn_img2):
    r = gr.Raster(fn_img)
    print('Raster 1:')
    print(r.info())

    r2 = gr.Raster(fn_img2)
    print('Raster 2:')
    print(r2.info())

    print('Reprojected raster')
    r3 = r.reproject(r2)
    print(r.info())

    plt.figure()
    r3.show(title='Reprojection')

def test_inters_img(fn_img,fn_img2):
    r = gr.Raster(fn_img)
    print('Raster 1:')
    print(r.info())

    r2 = gr.Raster(fn_img2)
    print('Raster 2:')
    print(r2.info())

    print('Intersected raster')
    inters = r.intersection(r2)
    print(inters)

def test_interp(fn_img):

    r = gr.Raster(fn_img)

    xmin, ymin, xmax, ymax = r.ds.bounds

    print(xmin, ymin, xmax, ymax)

    # testing interp, find_value, and read when it falls right on the coordinates
    xrand = np.random.randint(low=0,high=r.ds.width,size=(10,))*list(r.ds.transform)[0] + xmin + list(r.ds.transform)[0]/2
    yrand = ymax + np.random.randint(low=0,high=r.ds.height,size=(10,))*list(r.ds.transform)[4] - list(r.ds.transform)[4]/2
    pts = list(zip(xrand,yrand))
    i, j = r.xy2ij(xrand,yrand)
    list_z = []
    list_z_ind = []
    r.load()
    img = r.data
    for k in range(len(xrand)):
        z_ind = img[0,i[k],j[k]]
        z = r.value_at_coords(xrand[k],yrand[k])
        list_z_ind.append(z_ind)
        list_z.append(z)

    rpts = r.interp_points(pts)
    print(list_z_ind)
    print(list_z)
    print(rpts)

    #individual tests
    x = 493135.0
    y = 3104015.0
    print(r.value_at_coords(x,y))
    i,j = r.xy2ij(x,y)
    print(img[0,i,j])
    print(r.interp_points([(x,y)]))

    # random float
    xrand = np.random.uniform(low=xmin, high=xmax, size=(1000,))
    yrand = np.random.uniform(low=ymin, high=ymax, size=(1000,))
    pts = list(zip(xrand,yrand))
    rpts = r.interp_points(pts)
    # print(rpts)

def test_set_ndv(fn_img):

    r = gr.Raster(fn_img)

    print(r.nodata)

    r.set_ndv(ndv=[255])

    print(r.nodata)

    ndv_index = r.data==r.nodata

    #change data in case
    r.data[r.data == 254]=0
    r.set_ndv(ndv=254,update_array=True)

    print(r.nodata)
    ndv_index_2 = r.data==r.nodata

    print(np.count_nonzero(~ndv_index_2==ndv_index))

def test_set_dtypes(fn_img):

    r = gr.Raster(fn_img)

    arr_1 = np.copy(r.data).astype(np.int8)

    r.set_dtypes(np.int8)

    print(r.dtypes)

    arr_2 = np.copy(r.data)

    r.set_dtypes([np.int8],update_array=True)

    print(r.dtypes)

    arr_3 = r.data

    print(np.count_nonzero(~arr_1 == arr_2))
    print(np.count_nonzero(~arr_2 == arr_3))


if __name__ == '__main__':
    test_load_img(fn_img)
    test_crop_img(fn_img,fn_img2)
    test_reproj_img(fn_img,fn_img2)
    test_inters_img(fn_img,fn_img2)
    test_interp(fn_img)
    test_set_ndv(fn_img)
    test_set_dtypes(fn_img)