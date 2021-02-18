# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
import sys
import os

sys.path.insert(0, os.path.abspath('../geoutils/'))


project = 'GeoUtils'
copyright = '2020, GeoUtils Developers'
author = 'GeoUtils Developers'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '3.3.1'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme'
]

extlinks = {'issue': ('https://github.com/GlacioHack/GeoUtils/issues/%s',
                      'GH'),
            'pull': ('https://github.com/GlacioHack/GeoUtils/pull/%s', 'PR'),
            }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


import geoutils
# The short X.Y version
version = geoutils.__version__.split('+')[0]
# The full version, including alpha/beta/rc tags.
release = geoutils.__version__


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']