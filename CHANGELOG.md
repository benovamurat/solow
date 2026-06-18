# Changelog

All notable changes to Solow are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- User guide (mdBook under `docs/book/`) with Introduction, Installation,
  Quickstart, and how-to chapters for regression, GLM, discrete choice, time
  series, the formula interface, summary tables, and the Python bindings.
- Continuous-integration workflow running `cargo fmt --check`,
  `cargo clippy -D warnings`, `cargo test --workspace`, and an mdBook build on a
  stable-Rust ubuntu + macOS matrix.
- Contributor guide (`CONTRIBUTING.md`) documenting the model-plus-fixture
  workflow, the verification discipline, and the reference-naming rule.

## [0.1.0]

Initial release: a pure-Rust statistical-modeling, econometrics, and
data-visualization toolkit, every model cross-validated against an authoritative
reference implementation (most quantities to `1e-8`, MLE/VB estimates to `1e-6`,
the formula engine to `1e-12`, and the Python bindings to about `1e-12`).

### Foundation

- **solow-core** — error types, numeric aliases, and shared data-handling tools
  (including `add_constant`).
- **solow-linalg** — pure-Rust linear algebra: SVD, symmetric eigendecomposition,
  Cholesky, QR, LU, and the Moore-Penrose pseudo-inverse, with no system
  LAPACK/BLAS.
- **solow-distributions** — special functions (incomplete beta/gamma, `erf`), a
  broad library of continuous and discrete distributions, and an ECDF.
- **solow-optimize** — Newton and BFGS optimizers and numerical differentiation.

### Regression and generalized models

- **solow-regression** — OLS, WLS, and GLS via a whitening formulation, plus
  quantile regression, rolling and recursive least squares, GLSAR, sliced
  inverse regression, and robust (HC0-HC3, HAC/Newey-West, one-way cluster)
  sandwich covariances.
- **solow-glm** — generalized linear models by IRLS across the Gaussian,
  Binomial, Poisson, Gamma, Inverse-Gaussian, and Negative-Binomial families
  with selectable links, plus the Tweedie family.
- **solow-discrete** — Logit, Probit, and Poisson by exact Newton steps, plus
  multinomial logit, negative-binomial, ordered, generalized-Poisson,
  zero-inflated, conditional/fixed-effects, and truncated/hurdle count models.
- **solow-robust** — robust linear models by M-estimation.
- **solow-othermod** — beta regression.
- **solow-mixed** — linear mixed-effects models (REML).
- **solow-gee** — generalized estimating equations, including nominal and
  ordinal responses.
- **solow-gam** — generalized additive models (penalized splines).
- **solow-bayes** — Bayesian mixed GLM by variational Bayes.
- **solow-emplike** — empirical-likelihood inference.

### Time series and state space

- **solow-tsa** — autocovariance/acf/pacf/ccf, Ljung-Box, lag/trend design
  helpers, the augmented Dickey-Fuller and KPSS unit-root tests, AutoReg with
  automatic order selection, STL and classical seasonal decomposition,
  exponential smoothing (Holt-Winters), Granger causality, cointegration, ARMA
  processes and order selection, and the Hodrick-Prescott / Baxter-King /
  Christiano-Fitzgerald filters.
- **solow-statespace** — a time-invariant Kalman filter and fixed-interval
  smoother, SARIMAX by maximum likelihood, unobserved-components models, and
  dynamic-factor models.
- **solow-var** — vector autoregression (VAR), structural VAR (SVAR), and
  VECM / Johansen cointegration.
- **solow-regime** — regime-switching (Markov-switching) models.

### Other models and tests

- **solow-nonparametric** — lowess, kernel density estimation, and kernel
  regression.
- **solow-multivariate** — PCA, factor analysis with rotation, MANOVA, and
  canonical correlation.
- **solow-duration** — survival analysis: Kaplan-Meier, Cox proportional
  hazards (with Efron ties), and the log-rank test.
- **solow-impute** — multiple imputation with Rubin's rules.
- **solow-copula** — Archimedean and elliptical copulas.
- **solow-stats** — statistical tests and diagnostics: ANOVA, Tukey HSD,
  proportions, power, contingency, inter-rater agreement, mediation,
  Breusch-Godfrey, Ramsey RESET, ARCH, robust (HC/HAC/cluster) sandwich
  covariances, variance-inflation factors, Lilliefors, nearest-correlation
  projection, the Blinder-Oaxaca decomposition, distance correlation, and
  Zivot-Andrews / structural-break tests.

### Formulas, presentation, and bindings

- **solow-formula** — an R/patsy-style formula interface producing design
  matrices, validated against `patsy` to `1e-12`.
- **solow-summary** — labeled results and fixed-width summary tables.
- **solow-graphics** — statistical graphics (qqplot, plot_acf, influence) on the
  `solow-viz` backend.
- **solow-viz** — a general-purpose SVG data-visualization backend.
- **solow** — umbrella crate re-exporting the full public API behind a `prelude`.
- **solow-py** — PyO3 bindings exposing OLS, GLM, Logit, and Poisson to Python,
  reproducing reference outputs to about `1e-12`.
- **solow-bench** — a `criterion` benchmark harness.

[Unreleased]: https://github.com/solow-rs/solow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/solow-rs/solow/releases/tag/v0.1.0
