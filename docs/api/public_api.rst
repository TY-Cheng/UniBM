Public API
==========

The top-level :mod:`unibm` namespace is intentionally small. It keeps only the
two headline EVI entrypoints together with the grouped :mod:`unibm.evi` and
:mod:`unibm.ei` namespaces.

Use:

- :mod:`unibm` for ``estimate_evi_quantile`` and ``estimate_design_life_level``
- :mod:`unibm.evi` for the full EVI workflow, bootstrap backbones, result
  types, and external xi comparators
- :mod:`unibm.ei` for formal extremal-index preparation and estimators
- :mod:`unibm.diagnostics` and :mod:`unibm.plotting` as standalone public
  modules when you need them

.. automodule:: unibm
   :members:
   :undoc-members:
   :show-inheritance:
