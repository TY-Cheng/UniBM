<section class="unibm-hero">
  <div class="unibm-hero-grid">
    <div class="unibm-panel unibm-panel-primary">
      <p class="unibm-eyebrow">Dependence-aware block-maxima inference</p>
      <h1>Severity, persistence, and design-life levels under serial dependence.</h1>
      <p class="unibm-lead">
        UniBM is a Python package for dependence-aware block-maxima inference in
        environmental extremes. It keeps severity inference, persistence
        inference, and design-life levels in one coherent workflow while
        exposing a small public API under <code>unibm</code>,
        <code>unibm.evi</code>, <code>unibm.ei</code>, and <code>unibm.cdf</code>.
      </p>
      <div class="unibm-actions">
        <a class="unibm-button unibm-button-primary" href="getting-started/">
          Start with the package
        </a>
        <a class="unibm-button unibm-button-secondary" href="worked-examples/">
          See runnable examples
        </a>
      </div>
    </div>
    <aside class="unibm-panel unibm-panel-secondary">
      <p class="unibm-kicker">At a glance</p>
      <div class="unibm-metrics">
        <div class="unibm-metric">
          <strong>Two branches</strong>
          <span>Severity via EVI and design-life levels; persistence via EI.</span>
        </div>
        <div class="unibm-metric">
          <strong>Public namespaces</strong>
          <span><code>unibm</code>, <code>unibm.evi</code>, <code>unibm.ei</code>, and <code>unibm.cdf</code>.</span>
        </div>
        <div class="unibm-metric">
          <strong>Repo workflow</strong>
          <span>Benchmarks, applications, and notebooks stay behind the root <a href="https://github.com/TY-Cheng/UniBM/blob/main/justfile">justfile</a>.</span>
        </div>
      </div>
    </aside>
  </div>
</section>

<p class="unibm-section-label">Guide</p>

<div class="unibm-card-grid">
  <a class="unibm-card" href="getting-started/">
    <h3>Getting Started</h3>
    <p>Install the package, sync the local environment, and make the first severity or persistence call.</p>
  </a>
  <a class="unibm-card" href="concepts/">
    <h3>Concepts</h3>
    <p>Read the conceptual split between the severity branch, the persistence branch, and design-life levels.</p>
  </a>
  <a class="unibm-card" href="worked-examples/">
    <h3>Worked Examples</h3>
    <p>Use short, runnable examples for the public package surface without stepping into the full repo orchestration.</p>
  </a>
  <a class="unibm-card" href="reading-returned-objects/">
    <h3>Reading Returned Objects</h3>
    <p>Interpret the most useful result fields and understand what the fitted objects actually contain.</p>
  </a>
</div>

<p class="unibm-section-label">API Surface</p>

<div class="unibm-card-grid">
  <a class="unibm-card" href="api/public-api/">
    <h3>Public API</h3>
    <p>Start from the top-level package namespace and the recommended entrypoints.</p>
  </a>
  <a class="unibm-card" href="api/evi/">
    <h3>EVI Namespace</h3>
    <p>Severity-side estimation, block-quantile scaling, and design-life helpers.</p>
  </a>
  <a class="unibm-card" href="api/ei/">
    <h3>EI Namespace</h3>
    <p>Persistence-side procedures for extremal-index estimation under dependence.</p>
  </a>
  <a class="unibm-card" href="api/cdf/">
    <h3>CDF Helper</h3>
    <p>Supporting helper functionality exposed as part of the public package surface.</p>
  </a>
</div>

<p class="unibm-note">
  This site is intentionally package-focused. Repository-level orchestration for
  benchmarks, applications, and notebook rebuilds remains in the root
  <a href="https://github.com/TY-Cheng/UniBM/blob/main/README.md">README</a>
  and
  <a href="https://github.com/TY-Cheng/UniBM/blob/main/justfile">justfile</a>.
</p>
