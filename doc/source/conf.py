#
# Configuration file for the Sphinx documentation builder.
#
import os
import sys

# Allow conf.py to find the geoutils module
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../geoutils/"))
sys.path.append(os.path.abspath(".."))


project = "GeoUtils"
copyright = "2021, GeoUtils Developers"
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
    "sphinx.ext.inheritance_diagram",  # For class inheritance diagrams (see coregistration.rst).
    "sphinx_autodoc_typehints",  # Include type hints in the API documentation.
    "sphinxcontrib.programoutput",
    "sphinx_gallery.gen_gallery",  # Examples gallery
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": os.path.join(os.path.dirname(__file__), "../", "../", "examples"),  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "inspect_global_variables": True,  # Make links to the class/function definitions.
    "reference_url": {
        # The module you locally document uses None
        "geoutils": None,
    },
    "filename_pattern": r".*\.py",  # Run all python files in the gallery (by default, only files starting with "plot_" are run)
    # directory where function/class granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("geoutils"),  # which function/class levels are used to create galleries
}

extlinks = {
    "issue": ("https://github.com/GlacioHack/GeoUtils/issues/%s", "GH"),
    "pull": ("https://github.com/GlacioHack/GeoUtils/pull/%s", "PR"),
}

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

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
