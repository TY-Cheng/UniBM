# Case studies

<p class="unibm-case-intro">
Six frozen case studies show where UniBM's severity and persistence branches apply—and where
they do not. All figures are generated locally from the repository's fixed-cutoff inputs; this
site performs no data download or statistical computation.
</p>

<div class="unibm-domain-grid">
  <a class="unibm-domain-card unibm-domain-climate" href="climate-extremes/">
    <span class="unibm-domain-index">GHCN · 2 cases</span>
    <h3>Climate extremes</h3>
    <p>Wet-season precipitation in Houston and compound hot–dry severity in Phoenix.</p>
    <span class="unibm-domain-meta">Severity branch · no formal EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-streamflow" href="streamflow/">
    <span class="unibm-domain-index">USGS · 2 cases</span>
    <h3>Streamflow</h3>
    <p>Long daily-discharge records at selected Texas and Florida streamgages.</p>
    <span class="unibm-domain-meta">Severity + persistence</span>
  </a>
  <a class="unibm-domain-card unibm-domain-nfip" href="nfip-claims/">
    <span class="unibm-domain-index">OpenFEMA · 2 cases</span>
    <h3>NFIP claims</h3>
    <p>Daily building payouts in Texas and Florida on distinct severity and persistence clocks.</p>
    <span class="unibm-domain-meta">Severity + persistence</span>
  </a>
</div>

## Reading the evidence

| Quantity | What it describes | What it does not establish |
|---|---|---|
| EVI `ξ` | Growth of the fitted upper tail across block sizes | A causal mechanism or event forecast |
| EI `θ` | Short-range extremal clustering on the declared observation clock | Tail severity or annual loss |
| Design-life level | A fitted horizon-maximum quantile on that clock | A waiting time, return-period label, or operational threshold |

All intervals shown here are conditional on the selected block-size window and fixed workflow
defaults. They do not include post-selection uncertainty or nonstationary future change.

The shared analysis cutoff is **2025-12-31**. See [Validation](../validation.md) for the synthetic
benchmark evidence that precedes these real-data illustrations.
