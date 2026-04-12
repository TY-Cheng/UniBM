Public API
==========

The top-level :mod:`unibm` namespace is intentionally small. It keeps only the
two headline EVI entrypoints together with the grouped :mod:`unibm.evi` and
:mod:`unibm.ei` namespaces.

Use:

- :mod:`unibm` for ``estimate_evi_quantile`` and ``estimate_design_life_level``
- :mod:`unibm.evi` for the grouped EVI namespace, with canonical submodules
  such as :mod:`unibm.evi.blocks`, :mod:`unibm.evi.targets`,
  :mod:`unibm.evi.estimation`,
  :mod:`unibm.evi.design`, :mod:`unibm.evi.bootstrap`, and
  :mod:`unibm.evi.baselines`
- :mod:`unibm.ei` for the grouped EI namespace, with canonical submodules such
  as :mod:`unibm.ei.preparation`, :mod:`unibm.ei.paths`, :mod:`unibm.ei.bm`,
  and :mod:`unibm.ei.threshold`
- :mod:`unibm.cdf` as the standalone public empirical CDF helper
- :mod:`unibm.plotting` as the standalone plotting module

.. automodule:: unibm
   :members:
   :undoc-members:
   :show-inheritance:
