Worked Examples
===============

Example 1: Median sliding-block EVI fit
---------------------------------------

.. code-block:: python

   import numpy as np
   from unibm import estimate_evi_quantile, estimate_return_level

   sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
   fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
   rl = estimate_return_level(fit, years=np.array([10.0, 50.0]), observations_per_year=365.25)

Example 2: Reusable bootstrap backbone
--------------------------------------

.. code-block:: python

   import numpy as np
   from unibm import (
       build_block_summary_bootstrap_backbone,
       evaluate_block_summary_bootstrap_backbone,
       generate_block_sizes,
   )

   sample = np.random.default_rng(13).pareto(2.0, 4096) + 1.0
   block_sizes = generate_block_sizes(sample.size)
   backbone = build_block_summary_bootstrap_backbone(sample, block_sizes, sliding=True, reps=64)
   quantile_boot = evaluate_block_summary_bootstrap_backbone(backbone, target="quantile")

Example 3: Formal extremal-index fit
------------------------------------

.. code-block:: python

   import numpy as np
   from unibm.extremal_index import prepare_ei_bundle, estimate_pooled_bm_ei

   sample = np.random.default_rng(21).pareto(2.0, 4096) + 1.0
   bundle = prepare_ei_bundle(sample)
   fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="FGLS")
