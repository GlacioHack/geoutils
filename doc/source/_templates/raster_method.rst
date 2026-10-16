{% set name = objname.split('.')[-1] %}

{% set title = 'ds.rst.' ~ name ~ ('()' if objtype == 'method' else '') ~ ' or Raster.' ~ name ~ ('()' if objtype == 'method' else '') %}
{{ title }}
{{ '=' * (title|length) }}

.. currentmodule:: geoutils

.. auto{{ objtype }}:: {{ fullname }}
{% if fullname.startswith("geoutils.raster.raster.Raster.") %}
    :no-index:
{% endif %}
