Reading Returned Objects
========================

The main UniBM entrypoints return lightweight dataclasses rather than raw
tuples. In practice, you usually only need a few headline fields first.

EVI fits
--------

For a minimal EVI workflow:

.. code-block:: python

   import numpy as np
   from unibm import estimate_design_life_level, estimate_evi_quantile

   sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
   fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
   design_life = estimate_design_life_level(fit, years=np.array([10.0, 50.0]))

Read the result in this order:

- ``fit.slope`` is the headline UniBM ``xi`` estimate.
- ``fit.confidence_interval`` gives the uncertainty interval for ``xi``.
- ``fit.plateau_bounds`` shows which block-size window supported the fit.
- ``fit.bootstrap`` stores the bootstrap metadata and covariance inputs used by
  FGLS fitting.
- ``design_life`` contains the design-life-level estimates on the original data
  scale.

The remaining fields such as ``curve`` and ``plateau`` are mainly for plotting,
diagnostics, and workflow-side reuse.

EI fits
-------

For a minimal EI workflow:

.. code-block:: python

   import numpy as np
   from unibm.ei.preparation import prepare_ei_bundle
   from unibm.ei.bm import estimate_pooled_bm_ei

   sample = np.random.default_rng(21).pareto(2.0, 4096) + 1.0
   bundle = prepare_ei_bundle(sample)
   fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")

Read the result in this order:

- ``fit.theta_hat`` is the headline extremal-index estimate.
- ``fit.confidence_interval`` gives the uncertainty interval for ``theta``.
- ``fit.stable_window`` shows which block-size region was pooled.
- ``fit.base_path`` and ``fit.regression`` record which BM path and pooling rule
  produced the estimate.

The path-level fields are supporting diagnostics:

- ``fit.path_level`` records the observed block sizes retained on the finite
  path.
- ``fit.path_theta`` and ``fit.path_eir`` retain the observed path values for
  plotting and method audits.

FGLS versus OLS
---------------

Pooled EI fits always pool the observed stable-window path. If you switch from
``regression="OLS"`` to ``regression="FGLS"``, the observed path is still what
gets pooled. The bootstrap result only contributes the cross-block covariance
matrix used for FGLS weighting.
