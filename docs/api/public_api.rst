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
  :mod:`unibm.evi.tail` / :mod:`unibm.evi.spectrum`
- :mod:`unibm.ei` for the grouped EI namespace, with canonical submodules such
  as :mod:`unibm.ei.preparation`, :mod:`unibm.ei.paths`,
  :mod:`unibm.ei.selection`, :mod:`unibm.ei.bm`,
  :mod:`unibm.ei.threshold`, and :mod:`unibm.ei.plotting`
- :mod:`unibm.cdf` as the standalone public empirical CDF helper
- :mod:`unibm.evi.plotting` for EVI fit plotting helpers

.. automodule:: unibm
   :members:
   :undoc-members:
   :show-inheritance:
