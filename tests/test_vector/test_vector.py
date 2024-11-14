"""Test functions specific to the Vector class."""

from __future__ import annotations

import inspect
import os.path
import pathlib
import tempfile
import warnings

import geopandas as gpd
import geopandas.base
import pyproj
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_series_equal
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon

import geoutils as gu

GLACIER_OUTLINES_URL = "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"


class TestVector:
    landsat_b4_crop_path = gu.examples.get_path("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")
    aster_outlines_path = gu.examples.get_path("exploradores_rgi_outlines")
    glacier_outlines = gu.Vector(GLACIER_OUTLINES_URL)

    def test_init(self) -> None:
        """Test class initiation works as intended"""

        # First, with a URL filename
        v = gu.Vector(GLACIER_OUTLINES_URL)
        assert isinstance(v, gu.Vector)

        # Second, with a string filename
        v0 = gu.Vector(self.aster_outlines_path)
        assert isinstance(v0, gu.Vector)

        # Third, with a pathlib path
        path = pathlib.Path(self.aster_outlines_path)
        v1 = gu.Vector(path)
        assert isinstance(v1, gu.Vector)

        # Fourth, with a geopandas dataframe
        v2 = gu.Vector(gpd.read_file(self.aster_outlines_path))
        assert isinstance(v2, gu.Vector)

        # Fifth, passing a Vector itself (points back to Vector passed)
        v3 = gu.Vector(v2)
        assert isinstance(v3, gu.Vector)

        # Check errors are raised when filename has wrong type
        with pytest.raises(TypeError, match="Filename argument should be a string, Path or geopandas.GeoDataFrame."):
            gu.Vector(1)  # type: ignore

    def test_copy(self) -> None:
        vector2 = self.glacier_outlines.copy()

        assert vector2 is not self.glacier_outlines

        vector2.ds = vector2.ds.query("NAME == 'Ayerbreen'")

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_info(self) -> None:

        v = gu.Vector(GLACIER_OUTLINES_URL)

        # Check default runs without error (prints to screen)
        output = v.info()
        assert output is None

        # Otherwise returns info
        output2 = v.info(verbose=False)
        assert isinstance(output2, str)
        list_prints = ["Filename", "Coordinate system", "Extent", "Number of features", "Attributes"]
        assert all(p in output2 for p in list_prints)

    def test_query(self) -> None:
        vector2 = self.glacier_outlines.query("NAME == 'Ayerbreen'")

        assert vector2 is not self.glacier_outlines

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_save(self) -> None:
        """Test the save wrapper for GeoDataFrame.to_file()."""

        vector = gu.Vector(self.aster_outlines_path)

        # Create a temporary file in a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")

        # Save and check the file exists
        vector.save(temp_file)
        assert os.path.exists(temp_file)

        # Open and check the object is the same
        vector_save = gu.Vector(temp_file)
        vector_save.vector_equal(vector)

    def test_bounds(self) -> None:
        bounds = self.glacier_outlines.bounds

        assert bounds.left < bounds.right
        assert bounds.bottom < bounds.top

        assert bounds.left == self.glacier_outlines.ds.total_bounds[0]
        assert bounds.bottom == self.glacier_outlines.ds.total_bounds[1]
        assert bounds.right == self.glacier_outlines.ds.total_bounds[2]
        assert bounds.top == self.glacier_outlines.ds.total_bounds[3]

    def test_footprint(self) -> None:

        footprint = self.glacier_outlines.footprint

        assert isinstance(footprint, gu.Vector)
        assert footprint.vector_equal(self.glacier_outlines.get_footprint_projected(self.glacier_outlines.crs))


class NeedToImplementWarning(FutureWarning):
    """Warning to remember to implement new GeoPandas methods"""


class TestGeoPandasMethods:
    # Use two synthetic vectors
    poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    gdf1 = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    synthvec1 = gu.Vector(gdf1)

    # Create a synthetic LineString geometry
    lines = LineString([(10, 10), (10.5, 10.5), (11, 11)])
    gdf2 = gpd.GeoDataFrame({"geometry": [lines]}, crs="EPSG:4326")
    synthvec2 = gu.Vector(gdf2)

    # Use two real-life vectors
    realvec1 = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
    realvec2 = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

    # Properties and methods derived from Shapely or GeoPandas
    # List of properties and methods with non-geometric output that are implemented in GeoUtils
    main_properties = ["crs", "geometry", "total_bounds"]
    nongeo_properties = [
        "area",
        "length",
        "interiors",
        "geom_type",
        "is_empty",
        "is_ring",
        "is_simple",
        "is_valid",
        "has_z",
    ]
    nongeo_methods = [
        "contains",
        "geom_equals",
        "geom_almost_equals",
        "geom_equals_exact",
        "crosses",
        "disjoint",
        "intersects",
        "overlaps",
        "touches",
        "within",
        "covers",
        "covered_by",
        "distance",
    ]

    # List of properties and methods with geometric output that are implemented in GeoUtils
    geo_properties = ["boundary", "unary_union", "centroid", "convex_hull", "envelope", "exterior"]
    geo_methods = [
        "representative_point",
        "normalize",
        "make_valid",
        "difference",
        "symmetric_difference",
        "union",
        "intersection",
        "clip_by_rect",
        "buffer",
        "simplify",
        "affine_transform",
        "translate",
        "rotate",
        "scale",
        "skew",
        "dissolve",
        "explode",
        "sjoin",
        "sjoin_nearest",
        "overlay",
        "to_crs",
        "set_crs",
        "rename_geometry",
        "set_geometry",
        "clip",
    ]
    # List of class methods
    io_methods = [
        "from_file",
        "from_postgis",
        "from_dict",
        "from_features",
        "to_feather",
        "to_parquet",
        "to_file",
        "to_postgis",
        "to_json",
        "to_wkb",
        "to_wkt",
        "to_csv",
    ]

    # List of other properties and methods
    other = ["has_sindex", "sindex", "estimate_utm_crs", "cx", "iterfeatures"]
    all_declared = (
        main_properties + nongeo_methods + nongeo_properties + geo_methods + geo_properties + other + io_methods
    )

    # Exceptions for GeoPandasBase functions not implemented (or deprecrated) in GeoSeries/GeoDataFrame
    exceptions_unimplemented = [
        "plot",
        "explore",
        "cascaded_union",
        "bounds",
        "relate",
        "project",
        "interpolate",
        "equals",
        "type",
        "convert_dtypes",
        "merge",
        "apply",
        "astype",
        "minimum_bounding_circle",
        "minimum_bounding_radius",
        "get_coordinates",
        "hilbert_distance",
        "sample_points",
        "copy",
    ]
    # Exceptions for IO/conversion that can be done directly from .ds
    all_exceptions = exceptions_unimplemented

    # Get all GeoPandasBase public methods with some exceptions
    geobase_methods = gpd.base.GeoPandasBase.__dict__.copy()

    # Get all GeoDataFrame public methods with some exceptions
    gdf_methods = gpd.GeoDataFrame.__dict__.copy()

    def test_overridden_funcs_exist(self) -> None:
        """Check that all methods listed above exist in Vector."""

        # Check that all methods declared in the class above exist in Vector
        vector_methods = gu.Vector.__dict__

        list_missing = [method for method in self.all_declared if method not in vector_methods.keys()]

        assert len(list_missing) == 0, print(f"Test method listed that is not in GeoUtils: {list_missing}")

    def test_geopandas_coverage(self) -> None:
        """Check that all existing methods of GeoPandas are overridden, with a couple exceptions."""

        # Merge the two
        all_methods = self.geobase_methods.copy()
        all_methods.update(self.gdf_methods)

        # Remove exceptions we don't want to reuse from GeoPandas (mirrored in Vector)
        name_all_methods = list(all_methods.keys())
        public_methods = [method for method in name_all_methods if method[0] != "_"]

        covered_methods = [method for method in public_methods if method not in self.all_exceptions]

        # Check that all methods declared in the class above are covered in Vector
        list_missing = [method for method in covered_methods if method not in self.all_declared]

        if len(list_missing) != 0:
            warnings.warn(
                f"New GeoPandas methods are not implemented in GeoUtils: {list_missing}", NeedToImplementWarning
            )

    @pytest.mark.parametrize("method", nongeo_methods + geo_methods)  # type: ignore
    def test_overridden_funcs_args(self, method: str) -> None:
        """Check that all methods overridden have the same arguments as in GeoPandas."""

        # Get GeoPandas class where the methods live
        if method in self.geobase_methods.keys():
            upstream_class = gpd.base.GeoPandasBase
        elif method in self.gdf_methods.keys():
            upstream_class = gpd.GeoDataFrame
        else:
            raise ValueError("Method did not belong to GeoDataFrame or GeoPandasBase class.")

        # Get a full argument inspection object for each class
        argspec_upstream = inspect.getfullargspec(getattr(upstream_class, method))
        argspec_geoutils = inspect.getfullargspec(getattr(gu.Vector, method))

        # Check that all positional arguments are the same
        if argspec_upstream.args != argspec_geoutils.args:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        # Check that the *args and **kwargs argument are declared consistently
        if argspec_upstream.varargs != argspec_geoutils.varargs:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        if argspec_upstream.varkw != argspec_geoutils.varkw:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        # Check that default argument values are the same
        if argspec_upstream.defaults != argspec_geoutils.defaults:
            warnings.warn("Default argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

    @pytest.mark.parametrize("vector", [synthvec1, synthvec2, realvec1, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", nongeo_properties)  # type: ignore
    def test_nongeo_properties(self, vector: gu.Vector, method: str) -> None:
        """Check non-geometric properties are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        output_geoutils = getattr(vector, method)
        output_geopandas = getattr(vector.ds, method)

        # Assert equality
        assert_series_equal(output_geoutils, output_geopandas)

    @pytest.mark.parametrize("vector1", [synthvec1, realvec1])  # type: ignore
    @pytest.mark.parametrize("vector2", [synthvec2, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", nongeo_methods)  # type: ignore
    def test_nongeo_methods(self, vector1: gu.Vector, vector2: gu.Vector, method: str) -> None:
        """
        Check non-geometric methods are consistent with GeoPandas.
        All these methods require two inputs ("other", "df", or "right" argument), except one.
        """

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        if method != "geom_equals_exact":
            output_geoutils = getattr(vector1, method)(vector2)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds)
        else:
            output_geoutils = getattr(vector1, method)(vector2, tolerance=0.1)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds, tolerance=0.1)

        # Assert equality
        assert_series_equal(output_geoutils, output_geopandas)

    @pytest.mark.parametrize("vector", [synthvec1, synthvec2, realvec1, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", geo_properties)  # type: ignore
    def test_geo_properties(self, vector: gu.Vector, method: str) -> None:
        """Check geometric properties are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        output_geoutils = getattr(vector, method)
        output_geopandas = getattr(vector.ds, method)

        # Assert output types
        assert isinstance(output_geoutils, gu.Vector)
        assert isinstance(output_geopandas, (gpd.GeoSeries, gpd.GeoDataFrame, BaseGeometry))

        # Separate cases depending on GeoPandas' output
        if isinstance(output_geopandas, gpd.GeoSeries):
            # Assert geoseries equality
            assert_geoseries_equal(output_geoutils.ds.geometry, output_geopandas)
        elif isinstance(output_geopandas, BaseGeometry):
            assert_geodataframe_equal(
                output_geoutils.ds, gpd.GeoDataFrame({"geometry": [output_geopandas]}, crs=vector.crs)
            )
        else:
            assert_geodataframe_equal(output_geoutils.ds, output_geopandas)

    specific_method_args = {
        "buffer": {"distance": 1},
        "clip_by_rect": {"xmin": 10.5, "ymin": 10.5, "xmax": 11, "ymax": 11},
        "affine_transform": {"matrix": [1, 1, 1, 1, 1, 1]},
        "translate": {"xoff": 1, "yoff": 1, "zoff": 0},
        "rotate": {"angle": 90},
        "scale": {"xfact": 1.1, "yfact": 1.1, "zfact": 1.1, "origin": "center"},
        "skew": {"xs": 1.1, "ys": 1.1},
        "interpolate": {"distance": 1},
        "simplify": {"tolerance": 0.1},
        "to_crs": {"crs": pyproj.CRS.from_epsg(32610)},
        "set_crs": {"crs": pyproj.CRS.from_epsg(32610), "allow_override": True},
        "rename_geometry": {"col": "lol"},
        "set_geometry": {"col": synthvec1.geometry},
        "clip": {"mask": poly},
    }

    @pytest.mark.parametrize("vector1", [synthvec1, realvec1])  # type: ignore
    @pytest.mark.parametrize("vector2", [synthvec2, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", geo_methods)  # type: ignore
    def test_geo_methods(self, vector1: gu.Vector, vector2: gu.Vector, method: str) -> None:
        """Check geometric methods are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Methods that require two inputs
        if method in [
            "difference",
            "symmetric_difference",
            "union",
            "intersection",
            "sjoin",
            "sjoin_nearest",
            "overlay",
        ]:
            output_geoutils = getattr(vector1, method)(vector2)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds)
        # Methods that require zero input
        elif method in ["representative_point", "normalize", "make_valid", "dissolve", "explode"]:
            output_geoutils = getattr(vector1, method)()
            output_geopandas = getattr(vector1.ds, method)()
        elif method in self.specific_method_args.keys():
            output_geoutils = getattr(vector1, method)(**self.specific_method_args[method])
            output_geopandas = getattr(vector1.ds, method)(**self.specific_method_args[method])
        else:
            raise ValueError(f"The method '{method}' is not covered by this test.")

        # Assert output types
        assert isinstance(output_geoutils, gu.Vector)
        assert isinstance(output_geopandas, (gpd.GeoSeries, gpd.GeoDataFrame))

        # Separate cases depending on GeoPandas' output, and nature of the function
        # Simplify is a special case that can make geometries invalid, so adjust test
        if method == "simplify":
            # TODO: Unskip this random test failure (one index not matching) when this is fixed in GeoPandas/Shapely
            pass
            # assert_geoseries_equal(
            #     output_geopandas.make_valid(), output_geoutils.ds.geometry.make_valid(), check_less_precise=True
            # )
        # For geoseries output, check equality of it
        elif isinstance(output_geopandas, gpd.GeoSeries):
            assert_geoseries_equal(output_geoutils.ds.geometry, output_geopandas)
        # For geodataframe output, check equality
        else:
            assert_geodataframe_equal(output_geoutils.ds, output_geopandas)
