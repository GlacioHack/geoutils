{% set name = objname.split('.')[-1] %}

{% set title = 'gdf.vct.' ~ name ~ ('()' if objtype == 'method' else '') ~ ' or Vector.' ~ name ~ ('()' if objtype == 'method' else '') %}
{{ title }}
{{ '=' * (title|length) }}

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
