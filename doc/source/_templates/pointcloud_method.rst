{% set name = objname.split('.')[-1] %}

gdf.pc.{{ name }}{% if objtype == "method" %}(){% endif %} or PointCloud.{{ name }}{% if objtype == "method" %}(){% endif %}
=====================================================================================

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
