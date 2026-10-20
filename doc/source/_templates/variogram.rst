{% extends "autosummary/class.rst" %}

{% block methods %}
   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in methods if item != "__init__" %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endblock %}
