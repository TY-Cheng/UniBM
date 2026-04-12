Canonical EVI API
=================

This grouped package contains the full block-maxima EVI layer:

- headline quantile-scaling estimation and design-life mapping;
- reusable bootstrap backbones for FGLS;
- published xi comparators used in benchmark studies;
- EVI result types such as :class:`ScalingFit`.

Use :mod:`unibm.evi` when you want more than the two headline root functions
kept under :mod:`unibm`.

Raw circular bootstrap resampling remains internal infrastructure under the
top-level private shared layer; :mod:`unibm.evi` exposes only the block-summary
bootstrap/backbone API built on top of that sampling step.

.. automodule:: unibm.evi
   :members:
   :undoc-members:
   :show-inheritance:
