from __future__ import annotations

import numpy as np
import pytest

import geoutils as gu
from geoutils import examples


class TestGeoreferencing:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    landsat_b4_crop_path = examples.get_path_test("everest_landsat_b4_cropped")

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_ij2xy(self, example: str) -> None:
        """Test the outputs of ij2xy and that the function is reversible with xy2ij."""

        # Open raster
        rst = gu.Raster(example)
        xmin, ymin, xmax, ymax = rst.bounds

        # Check ij2xy manually for the four corners and center
        # With "force_offset", no considerations of pixel interpolation

        # With offset="center", should be pixel center
        xmin_center = xmin + rst.res[0] / 2
        ymin_center = ymin + rst.res[1] / 2
        xmax_center = xmax - rst.res[0] / 2
        ymax_center = ymax - rst.res[1] / 2
        assert rst.ij2xy([0], [0], force_offset="center") == ([xmin_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [0], force_offset="center") == ([xmin_center], [ymin_center])
        assert rst.ij2xy([0], [rst.shape[1] - 1], force_offset="center") == ([xmax_center], [ymax_center])
        assert rst.ij2xy([rst.shape[0] - 1], [rst.shape[1] - 1], force_offset="center") == (
            [xmax_center],
            [ymin_center],
        )

        # Same checks for offset="ll", "ul", "ur", "lr
        lims_ll = [xmin, ymin, xmax - rst.res[0], ymax - rst.res[1]]
        lims_ul = [xmin, ymin + rst.res[1], xmax - rst.res[0], ymax]
        lims_lr = [xmin + rst.res[0], ymin, xmax, ymax - rst.res[1]]
        lims_ur = [xmin + rst.res[0], ymin + rst.res[1], xmax, ymax]
        offsets = ["ll", "ul", "lr", "ur"]
        list_lims = [lims_ll, lims_ul, lims_lr, lims_ur]
        for i in range(len(list_lims)):
            offset = offsets[i]
            lim = list_lims[i]
            assert rst.ij2xy([0], [0], force_offset=offset) == ([lim[0]], [lim[3]])
            assert rst.ij2xy([rst.shape[0] - 1], [0], force_offset=offset) == ([lim[0]], [lim[1]])
            assert rst.ij2xy([0], [rst.shape[1] - 1], force_offset=offset) == ([lim[2]], [lim[3]])
            assert rst.ij2xy([rst.shape[0] - 1], [rst.shape[1] - 1], force_offset=offset) == ([lim[2]], [lim[1]])

        # Check that default coordinate is upper-left

        # With shift from pixel interpretation, coordinates will be half a pixel back for "Point"
        if rst.area_or_point is not None and rst.area_or_point == "Point" and gu.config["shift_area_or_point"]:
            # Shift is backward in X, forward in Y
            lims_ul[0] = lims_ul[0] - 0.5 * rst.res[0]
            lims_ul[1] = lims_ul[1] + 0.5 * rst.res[1]
            lims_ul[2] = lims_ul[2] - 0.5 * rst.res[0]
            lims_ul[3] = lims_ul[3] + 0.5 * rst.res[1]

        assert rst.ij2xy([0], [0]) == ([lims_ul[0]], [lims_ul[3]])
        assert rst.ij2xy([rst.shape[0] - 1], [0]) == ([lims_ul[0]], [lims_ul[1]])
        assert rst.ij2xy([0], [rst.shape[1] - 1]) == ([lims_ul[2]], [lims_ul[3]])
        assert rst.ij2xy([rst.shape[0] - 1], [rst.shape[1] - 1]) == ([lims_ul[2]], [lims_ul[1]])

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_xy2ij_ij2xy_reversible(self, example: str):

        # Open raster
        rst = gu.Raster(example)
        xmin, ymin, xmax, ymax = rst.bounds

        # We generate random points within the boundaries of the image
        rng = np.random.default_rng(42)
        xrand = rng.integers(low=0, high=rst.width, size=(10,)) * list(rst.transform)[0] + xmin
        yrand = ymax + rng.integers(low=0, high=rst.height, size=(10,)) * list(rst.transform)[4]

        # Test reversibility for any point or area interpretation
        i, j = rst.xy2ij(xrand, yrand)
        xnew, ynew = rst.ij2xy(i, j)
        assert all(xnew == xrand)
        assert all(ynew == yrand)

        # TODO: clarify this weird behaviour of rasterio.index with floats?
        # r.ds.index(x, y)
        # Out[33]: (75, 301)
        # r.ds.index(x, y, op=np.float32)
        # Out[34]: (75.0, 302.0)

    def test_xy2ij(self) -> None:
        """Test xy2ij with shift_area_or_point argument, and compare to interp_points function for consistency."""

        # First, we try on a Raster with a Point interpretation: values interpolated at the center of pixel
        r = gu.Raster(self.landsat_b4_path)
        r.set_area_or_point("Point")
        assert r.area_or_point == "Point"
        xmin, ymin, xmax, ymax = r.bounds

        # We generate random points within the boundaries of the image
        rng = np.random.default_rng(42)
        xrand = rng.integers(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + rng.integers(low=0, high=r.height, size=(10,)) * list(r.transform)[4]

        # Get decimal indexes based on "Point", should refer to the corner still (shift False by default)
        i, j = r.xy2ij(xrand, yrand, shift_area_or_point=False)
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # Those should all be .5 because values refer to the center and are shifted
        i, j = r.xy2ij(xrand, yrand, shift_area_or_point=True)
        assert np.all(i % 1 == 0.5)
        assert np.all(j % 1 == 0.5)

        # Force "Area", should refer to corner
        r.set_area_or_point("Area", shift_area_or_point=False)
        i, j = r.xy2ij(xrand, yrand, shift_area_or_point=True)
        assert np.all(i % 1 == 0)
        assert np.all(j % 1 == 0)

        # Now, we calculate the mean of values in each 2x2 slices of the data, and compare with interpolation at order 1
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # 2x2 slices
            z_ind = np.mean(
                img[
                    slice(int(np.floor(i[k])), int(np.ceil(i[k])) + 1),
                    slice(int(np.floor(j[k])), int(np.ceil(j[k])) + 1),
                ]
            )
            list_z_ind.append(z_ind)

        # First order interpolation
        rpts = r.interp_points((xrand, yrand), method="linear", as_array=True)
        # The values interpolated should be equal
        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

        # Test there is no failure with random coordinates (edge effects, etc)
        xrand = rng.uniform(low=xmin, high=xmax, size=(1000,))
        yrand = rng.uniform(low=ymin, high=ymax, size=(1000,))
        r.interp_points((xrand, yrand), as_array=True)

        # Second, test after a crop: the Raster now has an Area interpretation, those should fall right on the integer
        # pixel indexes
        r2 = gu.Raster(self.landsat_b4_crop_path)
        r.crop(r2)
        assert r.area_or_point == "Area"

        xmin, ymin, xmax, ymax = r.bounds

        # We can test with several method for the exact indexes: interp, and simple read should
        # give back the same values that fall right on the coordinates
        xrand = rng.integers(low=0, high=r.width, size=(10,)) * list(r.transform)[0] + xmin
        yrand = ymax + rng.integers(low=0, high=r.height, size=(10,)) * list(r.transform)[4]
        # By default, i and j are returned as integers
        i, j = r.xy2ij(xrand, yrand, op=np.float32)
        list_z_ind = []
        img = r.data
        for k in range(len(xrand)):
            # We directly sample the values
            z_ind = img[int(i[k]), int(j[k])]
            list_z_ind.append(z_ind)

        rpts = r.interp_points((xrand, yrand), method="linear", as_array=True)

        assert np.array_equal(np.array(list_z_ind, dtype=np.float32), rpts, equal_nan=True)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_coords(self, example: str) -> None:

        img = gu.Raster(example)

        # With lower left argument
        xx0, yy0 = img.coords(grid=False, force_offset="ll")

        assert len(xx0) == img.width
        assert len(yy0) == img.height

        assert xx0[0] == pytest.approx(img.bounds.left)
        assert xx0[-1] == pytest.approx(img.bounds.right - img.res[0])
        if img.res[1] > 0:
            assert yy0[0] == pytest.approx(img.bounds.bottom)
            assert yy0[-1] == pytest.approx(img.bounds.top - img.res[1])
        else:
            # Currently not covered by test image
            assert yy0[0] == pytest.approx(img.bounds.top)
            assert yy0[-1] == pytest.approx(img.bounds.bottom + img.res[1])

        # With center argument
        xx, yy = img.coords(grid=False, force_offset="center")
        hx = img.res[0] / 2
        hy = img.res[1] / 2
        assert xx[0] == pytest.approx(img.bounds.left + hx)
        assert xx[-1] == pytest.approx(img.bounds.right - hx)
        if img.res[1] > 0:
            assert yy[0] == pytest.approx(img.bounds.bottom + hy)
            assert yy[-1] == pytest.approx(img.bounds.top - hy)
        else:
            # Currently not covered by test image
            assert yy[-1] == pytest.approx(img.bounds.top + hy)
            assert yy[0] == pytest.approx(img.bounds.bottom - hy)

        # With grid argument (default argument, repeated here for clarity)
        xxgrid, yygrid = img.coords(grid=True, force_offset="ll")
        assert np.array_equal(xxgrid, np.repeat(xx0[np.newaxis, :], img.height, axis=0))
        assert np.array_equal(yygrid, np.flipud(np.repeat(yy0[:, np.newaxis], img.width, axis=1)))
