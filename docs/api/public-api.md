# Public API

The top-level `unibm` namespace is intentionally small. It keeps only the two
headline EVI entrypoints together with the grouped `unibm.evi` and
`unibm.ei` namespaces.

Use:

- `unibm` for `estimate_evi_quantile` and `estimate_design_life_level`
- `unibm.evi` for the grouped severity-side namespace
- `unibm.ei` for the grouped persistence-side namespace
- `unibm.cdf` for the standalone public empirical CDF helper

::: unibm
    options:
      members: true
      show_root_heading: true
      show_source: false
      members_order: source
      filters:
        - "!^_"
