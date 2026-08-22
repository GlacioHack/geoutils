{% set name = objname.split('.')[-1] %}

Raster.{{ name }}{% if objtype == "method" %}(){% endif %} or ds.rst.{{ name }}{% if objtype == "method" %}(){% endif %}
=======================================================================

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
