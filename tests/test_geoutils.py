"""
Test functions for geoutils
"""
import os
import inspect
import geoutils.georaster as gr
import geoutils.geovector as gv
import matplotlib.pyplot as plt
import numpy as np
import pytest

DO_PLOT = False

@pytest.fixture()
def path_data():
    path_module = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(gr))))
    print(path_module)
    fn_img = os.path.join(path_module, 'tests', 'data', 'LE71400412000304SGS00_B4_crop.TIF')
    fn_img2 = os.path.join(path_module, 'tests', 'data', 'LE71400412000304SGS00_B4_crop2.TIF')

    return fn_img, fn_img2

class TestRaster:

    def test_info(self,path_data):

        fn_img, fn_img2 = path_data

        r = gr.Raster(fn_img)

        #check all is good with passing attributes
        default_attrs = ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes', 'name',
                         'nodata',
                         'res', 'shape', 'transform', 'width']
        for attr in default_attrs:
            assert r.__getattribute__(attr) == r.ds.__getattribute__(attr)

        #check summary matches that of RIO
        assert print(r) == print(r.info())

    def test_copy(self,path_data):

        fn_img, _ = path_data

        r = gr.Raster(fn_img)
        r2 = r.copy()

        #should have no filename
        assert r2.filename is None
        #check a temporary memory file different than original disk file was created
        assert r2.name != r.name
        #check all attributes except name and dataset_mask array
        default_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes','nodata',
                         'res', 'shape', 'transform', 'width']
        for attr in default_attrs:
            print(attr)
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        #check data array
        assert np.count_nonzero(~r.data == r2.data) == 0
        #check dataset_mask array
        assert np.count_nonzero(~r.dataset_mask() == r2.dataset_mask()) == 0

    def test_crop(self, path_data):

        fn_img, fn_img2 = path_data
        r = gr.Raster(fn_img)
        r2 = gr.Raster(fn_img2)

        b = r.bounds
        b2 = r2.bounds

        b_minmax = (max(b[0],b2[0]),max(b[1],b2[1]),min(b[2],b2[2]),min(b[3],b2[3]))

        #TODO: add copy here (with copy() method) and plot all at the end?

        if DO_PLOT:
            plt.figure()
            r.show(title='Raster 1')
            plt.figure()
            r2.show(title='Raster 2')

        r.crop(r2)
        b_crop = tuple(r.bounds)

        if DO_PLOT:
            plt.figure()
            r.show(title='Raster 1 cropped to Raster 2')

        assert b_minmax == b_crop

    def test_reproj(self, path_data):

        fn_img, fn_img2 = path_data
        r = gr.Raster(fn_img)
        r2 = gr.Raster(fn_img2)
        r3 = r.reproject(r2)

        if DO_PLOT:
            plt.figure()
            r.show(title='Raster 1')
            plt.figure()
            r2.show(title='Raster 2')
            plt.figure()
            r3.show(title='Raster 1 reprojected to Raster 2')

        #TODO: not sure what to assert here

    def test_inters_img(self, path_data):

        fn_img, fn_img2 = path_data

        r = gr.Raster(fn_img)
        r2 = gr.Raster(fn_img2)

        inters = r.intersection(r2)
        print(inters)

    def test_interp(self,path_data):

        fn_img, fn_img2 = path_data

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

    def test_set_ndv(self,path_data):

        fn_img, fn_img2 = path_data

        r = gr.Raster(fn_img)
        r.set_ndv(ndv=[255])
        ndv_index = r.data==r.nodata

        #change data in case
        r.data[r.data == 254]=0
        r.set_ndv(ndv=254,update_array=True)
        ndv_index_2 = r.data==r.nodata

        assert np.count_nonzero(~ndv_index_2==ndv_index) == 0

    def test_set_dtypes(self,path_data):

        fn_img, fn_img2 = path_data

        r = gr.Raster(fn_img)
        arr_1 = np.copy(r.data).astype(np.int8)
        r.set_dtypes(np.int8)
        arr_2 = np.copy(r.data)
        r.set_dtypes([np.int8],update_array=True)

        arr_3 = r.data

        assert np.count_nonzero(~arr_1 == arr_2) == 0
        assert np.count_nonzero(~arr_2 == arr_3) == 0