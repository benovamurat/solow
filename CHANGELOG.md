# Changelog

All notable changes to Solow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3] — 2026-09-05

Documentation refresh. Product-first descriptions across every crate on
crates.io. No API changes. Every crate lives under `#![forbid(unsafe_code)]`.

The workspace ships 57 focused crates covering regression, generalized
linear models, discrete choice, robust regression, time series and
state-space models, survival analysis, mixed effects, Bayesian inference,
clustering, tree ensembles, support vector machines, neural networks,
kernel methods, dimensionality reduction, and change-point detection.

### Beyond the classical stack

- Change-point detection: CUSUM, PELT (Killick, Fearnhead, Eckley 2012),
  Binary Segmentation.
- `GARCH(1, 1)` with iterated multi-step variance forecast.
- Extreme value analysis: `Gev` and `Gpd` distributions with
  `return_level(T)` and peaks-over-threshold fit.
- Effect sizes: Cohen's d, Hedges' g, Glass's delta, eta squared, omega
  squared, Cliff's delta, Cramer's V.
- Meta-analysis: fixed-effect and DerSimonian-Laird random-effects with
  Cochran's Q, I squared, tau squared.
- Two-sided CUSUM and EWMA (Roberts 1959) control charts.
- Moving, circular, and stationary block bootstrap for time series.

### Correctness

Every deterministic estimator is cross-verified against committed golden
reference fixtures on every CI run. Closed-form solvers agree bit-wise
to `1e-10`. Iterative solvers agree on parameters to `1e-6` or on
predictions to `5e-2` where reference solvers themselves disagree at
that scale. NIST StRD certified cases are re-run on every CI, with
worst-case certified relative error across the suite of `2.5e-10`.

Run the full CI locally with `cargo test --workspace` (1000+ tests).
