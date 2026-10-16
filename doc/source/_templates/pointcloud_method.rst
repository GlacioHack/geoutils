{% set name = objname.split('.')[-1] %}

{% set title = 'gdf.pc.' ~ name ~ ('()' if objtype == 'method' else '') ~ ' or PointCloud.' ~ name ~ ('()' if objtype == 'method' else '') %}
{{ title }}
{{ '=' * (title|length) }}

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
