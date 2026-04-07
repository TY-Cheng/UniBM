Getting Started
===============

The reusable statistical package lives in ``scripts/unibm`` and is importable
as ``unibm`` once the repository environment has been synced.

Install the development environment
-----------------------------------

.. code-block:: bash

   uv sync --dev

Typical imports
---------------

.. code-block:: python

   import numpy as np
   from unibm import estimate_evi_quantile, estimate_return_level

   sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
   fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
   rl = estimate_return_level(fit, years=np.array([10.0]), observations_per_year=365.25)

Package boundaries
------------------

- ``unibm.core`` contains the main block-maxima EVI machinery.
- ``unibm.extremal_index`` contains formal EI estimation.
- ``unibm.bootstrap`` contains reusable bootstrap backbones.
- ``unibm.external`` contains published EVI comparator estimators.
- ``unibm.plotting`` contains plotting helpers for notebook and manuscript use.
