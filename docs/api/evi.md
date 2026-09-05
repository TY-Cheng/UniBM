# EVI Namespace

The exported `unibm.evi` namespace groups the severity-side workflow:
block extraction and summaries, plateau selection, covariance-aware fitting,
design-life-level helpers, plotting helpers, and the public comparator families
that remain part of the public package surface.

Start with `estimate_evi_quantile` for the median/quantile path or
`estimate_target_scaling` for mean and mode summaries. A custom `block_sizes`
grid must contain unique, strictly increasing integers between 2 and the sample
size; it is not silently sorted or truncated. Quantile targets require a finite,
non-boolean value strictly between 0 and 1.

FGLS covariance reuse must match the summary target, quantile, and sliding/disjoint
scheme; a supplied plateau must match its curve slice. See
[Worked Examples](../worked-examples.md) for a complete reuse call and
[Reading Returned Objects](../reading-returned-objects.md) for actual-regression
and adaptive-R metadata.

::: unibm.evi
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
