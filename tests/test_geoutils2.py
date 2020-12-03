"""
Test functions for geoutils (not using unittest)
"""
import os
import geoutils.georaster as gr
import geoutils.geovector as gv
import matplotlib.pyplot as plt

fn_img = os.path.join('/home/atom/code/devel/libs/GeoUtils/tests','data','LE71400412000304SGS00_B4_crop.TIF')
fn_img2 = os.path.join('/home/atom/code/devel/libs/GeoUtils/tests','data','LE71400412000304SGS00_B4_crop2.TIF')

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


if __name__ == '__main__':
    test_load_img(fn_img)
    test_crop_img(fn_img,fn_img2)
    test_reproj_img(fn_img,fn_img2)
    test_inters_img(fn_img,fn_img2)