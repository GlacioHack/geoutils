{{ fullname }}
{{ underline }}

.. automodule:: {{ fullname }}

   .. contents:: Contents
      :local:

   {% block functions %}
   {% if functions %}

   Functions
   =========

   {% for item in functions %}

   {{item}}
   {{ "-" * (item | length) }}

   .. autofunction:: {{ item }}

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
       :add-heading:

   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   Classes
   =======

   {% for item in classes %}

   {% set special_classes = ["Raster", "SatelliteImage", "Vector"] %}

   {% if item not in special_classes %}

   {{item}}
   {{ "-" * (item | length) }}

   .. autoclass:: {{ item }}
      :show-inheritance:
      :special-members: __init__
      :members:

   .. _sphx_glr_backref_{{fullname}}.{{item}}:

   .. minigallery:: {{fullname}}.{{item}}
       :add-heading:

   {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}

   Exceptions
   ==========

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
