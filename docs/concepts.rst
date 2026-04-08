EVI and EI Workflows
====================

EVI workflow
------------

The core EVI path is:

1. Choose a block extraction scheme and summary target.
2. Build a log block-summary curve across a candidate block-size grid.
3. Select a plateau window on the log-log scale.
4. Fit a regression slope on the selected window to estimate ``xi``.
5. Map the fitted scaling law to design-life levels.

In this package, the design-life-level step is not a separate annual-maxima
fit. UniBM reuses the same block-quantile scaling law and simply evaluates it
at larger block sizes that correspond to longer design-life spans. The fitting
view therefore lives on the ``block size`` axis, while the risk-interpretation
view lives on the ``design-life years`` axis.

The literature term closest to the current application output is a
``design-life level``: a quantile of the maximum over a design-life span, or
equivalently a ``T``-year block-maximum ``tau``-quantile. The main
manuscript/application default is ``tau = 0.5``, so the headline exported
curve is best read as a **median design-life level** rather than as a
classical return-period level. The application workflow also exports companion
design-life levels at ``tau = 0.90, 0.95, 0.99``. Those higher curves reuse
the same headline plateau and slope ``xi`` and only shift the intercept, so
they should be read as shared-``xi`` upper design-life quantiles rather than
as separate headline EVI fits.

In code, the main entrypoints are:

- :func:`unibm.core.generate_block_sizes`
- :func:`unibm.core.block_summary_curve`
- :func:`unibm.core.estimate_target_scaling`
- :func:`unibm.core.estimate_evi_quantile`
- :func:`unibm.core.estimate_design_life_level`

EI workflow
-----------

The formal EI path is distinct from the lighter reciprocal diagnostic plots:

1. Prepare block-size paths from the raw series.
2. Estimate a stable pooled block-maxima path.
3. Fit the preferred formal estimator, currently the manuscript headline
   ``BB-sliding-FGLS`` path.
4. Compare against reference estimators such as ``K-gaps``.

In code, the main entrypoints are:

- :func:`unibm.extremal_index.prepare_ei_bundle`
- :func:`unibm.extremal_index.estimate_pooled_bm_ei`
- :func:`unibm.extremal_index.estimate_k_gaps`
- :func:`unibm.extremal_index.estimate_ferro_segers`

Interpreting EVI vs EI outputs
------------------------------

The EVI plateau and the EI stable window answer different questions and need
not coincide.

- The EVI plateau is the block-size region where the log block-summary curve is
  sufficiently linear to support a stable ``xi`` estimate.
- The EI stable window is the block-size region where the extremal-index path
  is sufficiently stable to support a formal ``theta`` estimate.

Different block-maximum quantiles ``Q_tau(M_b)`` are expected to share the same
asymptotic slope ``xi`` and to differ mainly in intercept. In the direct
block-maxima framework used here, serial dependence is already internalized in
the fitted block-maximum law, so the design-life-level curve should be read
directly from the dependent-series fit rather than through a second BM-side
``theta`` adjustment. In the current application workflow, this is implemented
explicitly by treating ``tau = 0.50`` as the headline fit and deriving the
``0.90 / 0.95 / 0.99`` curves by holding the same plateau and slope fixed while
re-estimating only tau-specific intercepts.

In practice:

- use the EVI/design-life-level outputs for severity on the original physical scale;
- use the EI outputs for persistence, clustering, and recovery burden.

Diagnostic EI vs formal EI
--------------------------

``unibm.diagnostics`` is meant for exploratory stability views, such as
``1 / theta`` plots. ``unibm.extremal_index`` contains the formal inference
objects used in benchmark and application pipelines.

Module responsibilities
-----------------------

- ``core``: block-maxima summaries, regression, and design-life-level mapping.
- ``bootstrap``: shared bootstrap resampling and covariance backbones.
- ``extremal_index``: formal EI preparation, path construction, and estimators.
- ``diagnostics``: lighter exploratory summaries and reciprocal plots.
- ``external``: published xi estimators used as comparators.
- ``plotting``: figures for fitted objects.
