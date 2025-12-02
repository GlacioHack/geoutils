#
#
# Configuration file for the Sphinx documentation builder.
#
import glob
import os
import sys

# Allow conf.py to find the geoutils module
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../geoutils/"))
sys.path.append(os.path.abspath(".."))
sys.path.insert(0, os.path.dirname(__file__))

from sphinx_gallery.sorting import ExampleTitleSortKey, ExplicitOrder

project = "GeoUtils"
copyright = "2025, GeoUtils Developers"
author = "GeoUtils Developers"


# Set the python environment variable for programoutput to find it.
os.environ["PYTHON"] = sys.executable

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '3.3.1'
master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",  # Create the API documentation automatically
    "sphinx.ext.viewcode",  # Create the "[source]" button in the API to show the source code.
    "matplotlib.sphinxext.plot_directive",  # Render matplotlib figures from code.
    "sphinx.ext.autosummary",  # Create API doc summary texts from the docstrings.
    "sphinx.ext.inheritance_diagram",  # For class inheritance diagrams.
    "sphinx.ext.graphviz",  # To render graphviz diagrams.
    "sphinx_design",  # To render nice blocks
    "sphinx_autodoc_typehints",  # Include type hints in the API documentation.
    "sphinxcontrib.programoutput",
    "sphinx_gallery.gen_gallery",  # Examples gallery
    "sphinx.ext.intersphinx",
    # "myst_parser",  !! Not needed with myst_nb !! # Form of Markdown that works with sphinx, used a lot by the Sphinx Book Theme
    "myst_nb",  # MySt for rendering Jupyter notebook in documentation
    "sphinxarg.ext",  # To generate documentation for argparse tools
]

# For sphinx design to work properly
myst_enable_extensions = ["colon_fence"]

# For myst-nb to find the Jupyter kernel (=environment) to run from
nb_kernel_rgx_aliases = {".*geoutils.*": "python3"}
# To raise a Sphinx build error on notebook failure
nb_execution_raise_on_error = True  # To fail documentation build on notebook execution error
nb_execution_show_tb = True  # To show full traceback on notebook execution error
nb_output_stderr = "warn"  # To warn if an error is raised in a notebook cell (if intended, override to "show" in cell)

intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "xdem": ("https://xdem.readthedocs.io/en/stable", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

example_path = os.path.join("../", "../", "examples")

sphinx_gallery_conf = {
    "examples_dirs": [
        os.path.join(example_path, "io"),
        os.path.join(example_path, "handling"),
        os.path.join(example_path, "analysis"),
    ],  # path to your example scripts
    "gallery_dirs": [
        "io_examples",
        "handling_examples",
        "analysis_examples",
    ],  # path to where to save gallery generated output
    "subsection_order": ExplicitOrder(
        [
            os.path.join(example_path, "io", "open_save"),
            os.path.join(example_path, "io", "import_export"),
            os.path.join(example_path, "handling", "georeferencing"),
            os.path.join(example_path, "handling", "raster_vector"),
            os.path.join(example_path, "handling", "raster_point"),
            os.path.join(example_path, "analysis", "array_numerics"),
            os.path.join(example_path, "analysis", "distance_ops"),
        ]
    ),
    "within_subsection_order": ExampleTitleSortKey,
    "inspect_global_variables": True,  # Make links to the class/function definitions.
    "reference_url": {
        # The module you locally document uses None
        "geoutils": None,
    },
    "filename_pattern": r".*\.py",  # Run all python files in the gallery (by default, only files starting with "plot_" are run)
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("geoutils"),  # Which function/class levels are used to create galleries
    "remove_config_comments": True,  # To remove comments such as sphinx-gallery-thumbnail-number (only works in code, not in text)
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_mpl",
    ),
    # To reset matplotlib for each gallery (and run custom function that fixes the default DPI)
}

extlinks = {
    "issue": ("https://github.com/GlacioHack/geoutils/issues/%s", "GH"),
    "pull": ("https://github.com/GlacioHack/geoutils/pull/%s", "PR"),
}

# For matplotlib figures generate with sphinx plot: (suffix, dpi)
plot_formats = [(".png", 500)]

# To avoid long path names in inheritance diagrams
inheritance_alias = {
    "geoutils.raster.raster.Raster": "geoutils.Raster",
    "geoutils.pointcloud.pointcloud.PointCloud": "geoutils.PointCloud",
    "geoutils.vector.Vector": "geoutils.Vector",
    "xdem.dem.DEM": "xdem.DEM",
}

# To have an edge color that works in both dark and light mode
inheritance_edge_attrs = {"color": "dodgerblue1"}

# To avoid fuzzy PNGs
graphviz_output_format = "svg"

# Add any paths that contain templates here, relative to this directory.
templates_path = [os.path.join(os.path.dirname(__file__), "_templates")]

import geoutils

# The short X.Y version
version = geoutils.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = geoutils.__version__


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates"]

# autodoc_default_options = {
#        "special-members": "__init__",
# }


def clean_gallery_files(app, exception):
    fn_myraster = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/io/open_save/myraster.tif"))
    fn_myvector = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../examples/io/open_save/myvector.gpkg"))
    if os.path.exists(fn_myraster):
        os.remove(fn_myraster)
    if os.path.exists(fn_myvector):
        os.remove(fn_myvector)


# To ignore warnings due to having myst-nb reading the .ipynb created by sphinx-gallery
# Should eventually be fixed, see: https://github.com/executablebooks/MyST-NB/issues/363
def setup(app):
    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)
    app.connect("build-finished", clean_gallery_files)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_favicon = "_static/logo_only.png"
html_logo = "_static/logo.png"
html_title = "GeoUtils"

html_theme_options = {
    "path_to_docs": "doc/source",
    "use_sidenotes": True,
    "repository_url": "https://github.com/GlacioHack/geoutils",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org/",
        "notebook_interface": "jupyterlab",  # For launching Binder in Jupyterlab to open MD files as notebook (downloads them otherwise)
    },
    # "announcement": (
    #     "⚠️ Our 0.1 release refactored several early-development functions for long-term stability, "
    #     'to update your code see <a href="https://github.com/GlacioHack/geoutils/releases/tag/v0.1.0">here</a>. ⚠️'
    #     "<br>Future changes will come with deprecation warnings! 🙂"
    # ),
    "show_toc_level": 3,
    # "logo_only": True,
    # "icon_links": [
    #         {
    #             "name": "Conda",
    #             "url": "https://anaconda.org/conda-forge/geoutils",
    #             "icon": "https://img.shields.io/conda/vn/conda-forge/geoutils.svg",
    #             "type": "url",
    #         },
    #         {
    #             "name": "PyPI",
    #             "url": "https://pypi.org/project/geoutils/0.0.10/",
    #             "icon": "https://badge.fury.io/py/geoutils.svg",
    #             "type": "url",
    #         },
    #         {
    #             "name": "Testing",
    #             "url": "https://coveralls.io/github/GlacioHack/geoutils?branch=main",
    #             "icon": "https://coveralls.io/repos/github/GlacioHack/geoutils/badge.svg?branch=main",
    #             "type": "url",
    #         }],
}

# For dark mode
html_context = {
    # ...
    "default_mode": "auto"
}

# Add the search bar to be always displayed (not only on top)
# html_sidebars = {"**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]}


# html_logo = "path/to/myimage.png"

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
