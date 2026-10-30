{% extends "autosummary/class.rst" %}

{% block attributes %}
   .. rubric:: Attributes

   {% for item in attributes %}
   .. autoattribute:: {{ name }}.{{ item }}

   {% endfor %}
{% endblock %}
