EVI and EI Workflows
====================

EVI workflow
------------

The core EVI path is:

1. Choose a block extraction scheme and summary target.
2. Build a log block-summary curve across a candidate block-size grid.
3. Select a plateau window on the log-log scale.
4. Fit a regression slope on the selected window to estimate ``xi``.
5. Map the fitted scaling law to return levels.

In code, the main entrypoints are:

- :func:`unibm.core.generate_block_sizes`
- :func:`unibm.core.block_summary_curve`
- :func:`unibm.core.estimate_target_scaling`
- :func:`unibm.core.estimate_evi_quantile`
- :func:`unibm.core.estimate_return_level`

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

Diagnostic EI vs formal EI
--------------------------

``unibm.diagnostics`` is meant for exploratory stability views, such as
``1 / theta`` plots. ``unibm.extremal_index`` contains the formal inference
objects used in benchmark and application pipelines.

Module responsibilities
-----------------------

- ``core``: block-maxima summaries, regression, and return-level mapping.
- ``bootstrap``: shared bootstrap resampling and covariance backbones.
- ``extremal_index``: formal EI preparation, path construction, and estimators.
- ``diagnostics``: lighter exploratory summaries and reciprocal plots.
- ``external``: published xi estimators used as comparators.
- ``plotting``: figures for fitted objects.
