# EI Namespace

The exported `unibm.ei` namespace groups the persistence-side workflow:
sample preparation, BM-path construction, stable-window selection, pooled
BM estimators, threshold estimators, plotting helpers, and bootstrap-based
covariance support.

`prepare_ei_bundle` owns the observed paths and candidate threshold quantiles.
It requires an explicit `allow_zeros` choice and preserves the caller's observation
clock. A threshold estimator's omitted candidate list uses the bundle's list;
an explicit subset must be increasing and present in that bundle.

For pooled FGLS, first call `bootstrap_bm_ei_path` with the same data, `base_path`,
sliding/disjoint scheme, and block-size grid. Then pass its result to
`estimate_pooled_bm_ei`. OLS does not accept a bootstrap result. See the
[complete FGLS example](../worked-examples.md#example-3-pooled-extremal-index-fit)
and [precision diagnostics](../reading-returned-objects.md#reading-adaptive-precision).

::: unibm.ei
    options:
      members: true
      show_root_heading: true
      show_source: false
      show_root_full_path: false
      show_root_toc_entry: false
      show_symbol_type_heading: false
      members_order: source
      filters:
        - "!^_"
