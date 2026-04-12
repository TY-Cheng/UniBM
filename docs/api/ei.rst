Formal EI API
=============

This submodule contains the formal extremal-index layer, distinct from the
lighter reciprocal diagnostics in :mod:`unibm.diagnostics`.

The recommended path is to call :func:`prepare_ei_bundle` once on the observed
series and then fit :func:`estimate_pooled_bm_ei` for the pooled BM headline
estimate. Use :func:`estimate_native_bm_ei` when you specifically want the
single-block-size native benchmark estimators.

Canonical public submodules:

- :mod:`unibm.ei.paths`
- :mod:`unibm.ei.native`
- :mod:`unibm.ei.threshold`
- :mod:`unibm.ei.bootstrap`
- :mod:`unibm.ei.models`

.. automodule:: unibm.ei
   :members:
   :undoc-members:
   :show-inheritance:
