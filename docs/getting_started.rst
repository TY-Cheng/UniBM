Getting Started
===============

The reusable statistical package lives in ``src/unibm`` and is importable
as ``unibm`` once the repository environment has been synced. This site focuses
on the reusable package layer, not on the full repo orchestration under
``scripts/benchmark`` and ``scripts/application``.

Repo workflow
-------------

.. code-block:: bash

   cp .env.example .env
   just verify

Top-level ``just`` tasks load ``.env`` automatically and sync the development
environment before they run. If you prefer raw ``uv`` commands instead, load
``.env`` into your shell first and then sync:

.. code-block:: bash

   set -a; source .env; set +a
   uv sync --dev

The repo-level workflow details still live in the repository ``README.md`` and
``justfile``. Use this documentation site when you want the reusable
``unibm`` package API itself.

Package usage
-------------

.. code-block:: python

   import numpy as np
   from unibm import estimate_evi_quantile, estimate_design_life_level

   sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
   fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
   design_life = estimate_design_life_level(
       fit,
       years=np.array([10.0]),
       observations_per_year=365.25,
   )

The shortest formal-EI package workflow is:

.. code-block:: python

   from unibm.extremal_index import prepare_ei_bundle, estimate_pooled_bm_ei

   bundle = prepare_ei_bundle(sample)
   ei_fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")

For a quick guide to which returned fields matter most, see
``Reading Returned Objects`` in the Guide navigation.

Package boundaries
------------------

- ``unibm.core`` contains the main block-maxima EVI machinery.
- ``unibm.extremal_index`` contains formal EI estimation.
- ``unibm.bootstrap`` contains reusable bootstrap backbones.
- ``unibm.external`` contains published EVI comparator estimators.
- ``unibm.plotting`` contains plotting helpers for notebook and manuscript use.
