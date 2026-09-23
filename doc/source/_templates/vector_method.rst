{% set name = objname.split('.')[-1] %}

gdf.vct.{{ name }}{% if objtype == "method" %}(){% endif %} or Vector.{{ name }}{% if objtype == "method" %}(){% endif %}
=================================================================================

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
