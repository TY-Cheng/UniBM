# CDF Helper

The standalone `unibm.cdf` module exposes the public empirical CDF helper
reused by EI path preparation. Its retained behavior is a scaled-rank helper,
not the ordinary normalized empirical distribution in every case:

- It flattens the sample and removes non-finite sample values.
- For more than one retained value, it returns `count(X <= q) / (n + 1)`.
  It therefore reaches `n / (n + 1)`, not 1, above the sample maximum.
- A singleton instead uses the exact one-point step distribution.
- An empty retained sample returns NaN; a NaN query also returns NaN.

EI preparation validates the observation series before calling this helper.
Do not use the helper's permissive input cleanup to justify deleting missing
time positions from an EI series.

::: unibm.cdf
    options:
      members: true
      show_root_heading: true
      show_source: false
      members_order: source
      filters:
        - "!^_"
