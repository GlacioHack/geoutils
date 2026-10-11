{% set name = objname.split('.')[-1] %}

ds.rst.{{ name }}{% if objtype == "method" %}(){% endif %} or Raster.{{ name }}{% if objtype == "method" %}(){% endif %}
=======================================================================

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
{% if fullname.startswith("geoutils.raster.raster.Raster.") %}
    :no-index:
{% endif %}
