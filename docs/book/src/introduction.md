# Introduction

<p align="center">
  <img alt="Solow" src="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo.svg" width="280">
</p>

<p align="center">
  <a href="https://github.com/benovamurat/solow">GitHub&nbsp;repository</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/benovamurat/solow/tree/main/crates/solow-gallery">Runnable examples</a>
  &nbsp;·&nbsp;
  <a href="https://stochasticminds.com">Stochastic&nbsp;Minds</a>
</p>

> **Source code:** [github.com/benovamurat/solow](https://github.com/benovamurat/solow) — stars, issues, and contributions welcome.

**Solow** is a complete statistical-modeling, econometrics, and
data-visualization toolkit for Rust. It provides estimators for a wide range of
statistical models, along with statistical tests, diagnostics, and data
exploration — entirely in safe, dependency-light Rust.

If you have used the canonical Python statistics stack (NumPy / SciPy together
with the standard econometrics-modeling package), Solow will feel familiar: the
model names, the result quantities, and the formula syntax all mirror that
ecosystem. The difference is that Solow is a *from-scratch* re-implementation in
Rust, with no Python runtime and no system LAPACK/BLAS dependency.

## Design principles

- **Pure Rust.** The numerical core — linear algebra (SVD, eigendecomposition,
  Cholesky, QR, LU, pseudo-inverse), special functions (incomplete
  beta/gamma, `erf`), and the statistical distributions — is implemented in
  pure Rust. The only foundational dependency is [`ndarray`]. There is no
  system LAPACK/BLAS to install and no C toolchain to wrangle.

- **Verified against a reference.** Every numerical routine is cross-checked
  against an authoritative reference implementation of the same models. Golden
  fixtures are generated from that reference and committed to the repository;
  the test suite asserts agreement to a tight tolerance — most quantities to
  `1e-8`, maximum-likelihood and variational-Bayes estimates to `1e-6`, and the
  formula engine to `1e-12`. This means a model you fit with Solow returns the
  same numbers you would get from the established tool, not merely numbers that
  "look right."

- **Layered.** Solow is a Cargo workspace of focused crates, each mirroring a
  module of a mature statistics library. You depend only on what you use, or
  pull in the umbrella [`solow`] crate to get everything behind one import.

- **Faithful output.** Results carry the same battery of inference statistics
  you expect: standard errors, t/z statistics, p-values, confidence intervals,
  information criteria, and (where applicable) robust covariances. A
  human-readable summary table renders the way you are used to seeing.

## What is in the box

Solow covers, among other things:

| Area | Crate | Models / tools |
| --- | --- | --- |
| Linear regression | [`solow-regression`] | OLS, WLS, GLS, quantile, rolling/recursive LS, robust covariances |
| Generalized linear models | [`solow-glm`] | Gaussian, Binomial, Poisson, Gamma, Inverse-Gaussian, Negative-Binomial, Tweedie |
| Discrete choice | [`solow-discrete`] | Logit, Probit, Poisson, MNLogit, Negative-Binomial, ordered, zero-inflated |
| Robust regression | [`solow-robust`] | M-estimation (Huber, Tukey biweight, Andrew wave) |
| Time series | [`solow-tsa`] | acf/pacf, ADF, KPSS, AutoReg, STL, Holt-Winters, Granger, cointegration |
| State space | [`solow-statespace`] | Kalman filter/smoother, SARIMAX, unobserved components, dynamic factor |
| Vector autoregression | [`solow-var`] | VAR, SVAR, VECM / Johansen |
| Survival / duration | [`solow-duration`] | Kaplan-Meier, Cox PH, log-rank |
| Multivariate | [`solow-multivariate`] | PCA, factor analysis, MANOVA, canonical correlation |
| Nonparametric | [`solow-nonparametric`] | LOWESS, kernel density, kernel regression |
| Statistical tests | [`solow-stats`] | normality, heteroskedasticity, autocorrelation, ANOVA, multiple testing |
| Distributions | [`solow-distributions`] | distributions and special functions (`scipy`-compatible) |
| Formula interface | [`solow-formula`] | patsy-style design matrices from a formula string |
| Summary tables | [`solow-summary`] | labeled results and fixed-width summary rendering |
| Python bindings | `solow-py` | import `solow` from Python (PyO3) |

See the [Crate reference](./crates.md) for the full workspace.

## A taste

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];

let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design).unwrap().fit().unwrap();

println!("R-squared = {:.3}", res.rsquared);
println!("coefficients = {:?}", res.params);
```

Read on to [install](./installation.md) Solow and work through a fuller
[quickstart](./quickstart.md).

[`ndarray`]: https://docs.rs/ndarray
[`solow`]: https://github.com/solow-rs/solow
[`solow-regression`]: ./regression.md
[`solow-glm`]: ./glm.md
[`solow-discrete`]: ./discrete.md
[`solow-robust`]: ./robust.md
[`solow-tsa`]: ./time-series.md
[`solow-statespace`]: ./statespace.md
[`solow-var`]: ./var.md
[`solow-duration`]: ./duration.md
[`solow-multivariate`]: ./multivariate.md
[`solow-nonparametric`]: ./nonparametric.md
[`solow-stats`]: ./stats-tests.md
[`solow-distributions`]: ./distributions.md
[`solow-formula`]: ./formula.md
[`solow-summary`]: ./summary-tables.md
