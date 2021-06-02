#
# Configuration file for the Sphinx documentation builder.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../geoutils/"))


project = "GeoUtils"
copyright = "2020, GeoUtils Developers"
author = "GeoUtils Developers"


# Set the python environment variable for programoutput to find it.
os.environ["PYTHON"] = sys.executable

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '3.3.1'
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Create the API documentation automatically
    "sphinx.ext.viewcode",  # Create the "[source]" button in the API to show the source code.
    "matplotlib.sphinxext.plot_directive",  # Render matplotlib figures from code.
    "sphinx.ext.autosummary",  # Create API doc summary texts from the docstrings.
    "sphinx.ext.inheritance_diagram",  # For class inheritance diagrams (see coregistration.rst).
    "sphinx_autodoc_typehints",  # Include type hints in the API documentation.
    "sphinxcontrib.programoutput",  # Run scripts and show the output.
]

extlinks = {
    "issue": ("https://github.com/GlacioHack/GeoUtils/issues/%s", "GH"),
    "pull": ("https://github.com/GlacioHack/GeoUtils/pull/%s", "PR"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]


import geoutils

# The short X.Y version
version = geoutils.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = geoutils.__version__


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
