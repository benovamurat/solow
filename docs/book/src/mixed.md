# Linear mixed effects models

The `solow-mixed` crate fits a **linear mixed-effects model** with a single
grouping factor, where each group carries its own random intercept on top of
shared fixed effects. The entry point is the `MixedLm` model; `.fit()` maximizes
the profile (restricted) likelihood and returns a `MixedLmResults` with the
fixed-effects estimates, the variance components, and the usual Wald inference.
Estimation uses restricted maximum likelihood (REML) by default, or ordinary
maximum likelihood (ML) via `RemlMethod`.

> **Intercepts are explicit.** As elsewhere in solow, you supply the intercept
> column of the fixed-effects design yourself. The *random* intercept is implied
> by the grouping factor and is not a column you provide.

## Background

The data are partitioned into \\( m \\) independent groups. Writing
\\( y_g \\in \mathbb{R}^{n_g} \\) for the responses in group \\( g \\) and
\\( X_g \\in \mathbb{R}^{n_g \times p} \\) for its fixed-effects design, the
random-intercept model is

\\[
y_g = X_g\,\beta + b_g\,\mathbf{1}_{n_g} + \varepsilon_g,
\qquad
b_g \sim \mathcal{N}(0,\ \psi\sigma^2),
\qquad
\varepsilon_g \sim \mathcal{N}(0,\ \sigma^2 I_{n_g}),
\\]

with \\( b_g \\) and \\( \varepsilon_g \\) mutually independent. Here
\\( \beta \in \mathbb{R}^p \\) are the **fixed effects**, the scalar \\( b_g \\)
is the group's **random intercept**, \\( \sigma^2 \\) is the residual variance,
and \\( \psi = \Psi/\sigma^2 \\) is the random-intercept variance expressed as a
ratio to \\( \sigma^2 \\). This is the simplest variance-components model: one
grouping factor, one random effect per group (the random-slope and
general-covariance cases are not implemented; see [Module reference](#module-reference)).

Marginalizing over \\( b_g \\) gives a group-block covariance with the
compound-symmetry form

\\[
V_g \;=\; \operatorname{Cov}(y_g) / \sigma^2
\;=\; I_{n_g} + \psi\,\mathbf{1}_{n_g}\mathbf{1}_{n_g}^{\top},
\\]

whose inverse and log-determinant are available in closed form,

\\[
V_g^{-1} = I_{n_g} - \frac{\psi}{1 + n_g\psi}\,\mathbf{1}_{n_g}\mathbf{1}_{n_g}^{\top},
\qquad
\log\det V_g = \log\!\left(1 + n_g\psi\right).
\\]

Given \\( \psi \\), the fixed effects are the generalized-least-squares (GLS)
solution and the residual scale has a closed form:

\\[
\hat\beta(\psi) = \Big(\textstyle\sum_g X_g^{\top} V_g^{-1} X_g\Big)^{-1}
\sum_g X_g^{\top} V_g^{-1} y_g,
\qquad
\hat\sigma^2(\psi) = \frac{1}{f}\sum_g r_g^{\top} V_g^{-1} r_g,
\quad r_g = y_g - X_g\hat\beta,
\\]

where the denominator is \\( f = N \\) for ML and \\( f = N - p \\) for REML
(\\( N = \sum_g n_g \\)). Substituting both profiled quantities leaves a
one-dimensional **profile objective** in \\( \psi \\),

\\[
\ell_p(\psi) = -\tfrac{1}{2}\sum_g \log\!\left(1 + n_g\psi\right)
- \tfrac{f}{2}\log\!\Big(\textstyle\sum_g r_g^{\top} V_g^{-1} r_g\Big)
+ \tfrac{f}{2}\log f - \tfrac{f}{2}\log(2\pi) - \tfrac{f}{2}
\;-\; \underbrace{\tfrac{1}{2}\log\det\!\Big(\textstyle\sum_g X_g^{\top} V_g^{-1} X_g\Big)}_{\text{REML only}},
\\]

which `fit` maximizes over \\( \theta = \log\psi \\) (an unconstrained
reparameterization that keeps \\( \psi > 0 \\)) by BFGS, followed by a few Newton
steps on the 1-D profile. Fixed-effects standard errors come from the negative
Hessian of the joint \\( [\beta, \psi] \\) log-likelihood. The reported
random-effects variance is \\( \widehat\Psi = \hat\psi\,\hat\sigma^2 \\)
(field `cov_re`), and \\( \hat\psi \\) itself is the field `psi`.

## Example

The data below have four groups of three observations each, a single fixed-effect
predictor (plus an intercept), and a clear group-to-group shift in level — exactly
what a random intercept absorbs.

```rust
use ndarray::{array, Array1, Array2};
use solow_mixed::{MixedLm, MixedLmResults, RemlMethod};

// Response, one row per observation.
let y: Array1<f64> = array![
    2.0, 2.4, 1.8,   // group 0
    5.1, 4.8, 5.4,   // group 1
   -1.0, -0.6, -1.3, // group 2
    3.2, 3.6, 2.9,   // group 3
];

// Fixed-effects design: column 0 is the intercept, column 1 a covariate.
let x: Array2<f64> = array![
    [1.0,  0.1], [1.0,  0.5], [1.0, -0.2],
    [1.0,  0.3], [1.0, -0.4], [1.0,  0.6],
    [1.0,  0.0], [1.0,  0.2], [1.0, -0.1],
    [1.0,  0.4], [1.0, -0.3], [1.0,  0.5],
];

// One integer group label per observation.
let groups: [i64; 12] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];

// REML is the default; call `.method(RemlMethod::Ml)` to switch.
let res: MixedLmResults = MixedLm::new(y, x, &groups)
    .unwrap()
    .fit()
    .unwrap();

println!("fixed effects β = {:?}", res.fe_params);
println!("random-intercept variance Ψ = {:.4}", res.cov_re);
println!("residual variance σ²        = {:.4}", res.scale);
println!("variance ratio ψ            = {:.4}", res.psi);
println!("profile log-likelihood      = {:.4}", res.llf);
println!("std errors  = {:?}", res.bse_fe);
println!("z-values    = {:?}", res.tvalues());
println!("p-values    = {:?}", res.pvalues());

// 95% Wald confidence intervals: one [lower, upper] row per coefficient.
let ci = res.conf_int(0.05);
println!("conf_int shape = {:?}", ci.dim());
```

What this prints (values are *illustrative* — run it for exact figures): the two
entries of `fe_params` are the GLS intercept and slope; `cov_re` is large
relative to `scale`, reflecting that almost all of the spread across the four
groups is between-group rather than within-group, so the fitted `psi` is well
above 1. `bse_fe`, `tvalues()`, and `pvalues()` give the Wald inference battery
for the fixed effects (normal approximation), and `conf_int(0.05)` returns a
`(2, 2)` array of symmetric 95% intervals.

To fit by ordinary maximum likelihood instead, chain `.method`:

```rust
use solow_mixed::{MixedLm, RemlMethod};
# use ndarray::{array, Array1, Array2};
# let y: Array1<f64> = array![1.0, 1.5, 3.0, 2.6, -0.5, 0.1];
# let x: Array2<f64> = Array2::<f64>::ones((6, 1));
# let groups: [i64; 6] = [0, 0, 1, 1, 2, 2];
let res = MixedLm::new(y, x, &groups)
    .unwrap()
    .method(RemlMethod::Ml)
    .fit()
    .unwrap();

assert!(!res.reml); // confirms ML was used
```

Under ML the residual scale divides by \\( N \\) rather than \\( N - p \\), so
the variance components differ slightly from the REML fit (REML corrects for the
degrees of freedom consumed by estimating \\( \beta \\)).

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `MixedLm` | Random-intercept mixed model. Build with `MixedLm::new(endog, exog, group_labels)`; chain `.method(..)` to choose the criterion, then `.fit()`. |

**Results**

| Name | Description |
| --- | --- |
| `MixedLmResults` | Fitted quantities returned by `MixedLm::fit`. |
| `MixedLmResults::tvalues` | Wald \\( z \\)-statistics for the fixed effects (`fe_params / bse_fe`). |
| `MixedLmResults::pvalues` | Two-sided normal-approximation p-values for the fixed effects. |
| `MixedLmResults::conf_int` | Two-sided Wald confidence intervals at level `alpha` (rows `[lower, upper]`). |

Public fields of `MixedLmResults`: `fe_params` (fixed effects \\( \hat\beta \\)),
`cov_re` (random-intercept variance \\( \widehat\Psi = \hat\psi\hat\sigma^2 \\)),
`scale` (residual variance \\( \hat\sigma^2 \\)), `psi` (variance ratio
\\( \hat\psi \\)), `bse_fe` (fixed-effects standard errors), `llf` (maximized
profile log-likelihood), and `reml` (whether REML was used).

**Enums**

| Name | Description |
| --- | --- |
| `RemlMethod` | Estimation criterion: `RemlMethod::Reml` (default) or `RemlMethod::Ml`. |

This crate is deliberately narrower than the reference's mixed-models module: it
covers the single-grouping-factor **random-intercept** model only. Random slopes,
multiple or crossed grouping factors, general (unstructured) random-effects
covariances, variance-weights, and explicit BLUP extraction for the random
intercepts are not implemented.

Full API: see the generated rustdoc for `solow-mixed`.

## References

- Laird, N. M., and Ware, J. H. (1982). *Random-Effects Models for Longitudinal
  Data.* Biometrics, 38(4), 963–974.
- Harville, D. A. (1977). *Maximum Likelihood Approaches to Variance Component
  Estimation and to Related Problems.* Journal of the American Statistical
  Association, 72(358), 320–338.
- Pinheiro, J. C., and Bates, D. M. (2000). *Mixed-Effects Models in S and
  S-PLUS.* Springer.
- McCulloch, C. E., Searle, S. R., and Neuhaus, J. M. (2008). *Generalized,
  Linear, and Mixed Models*, 2nd ed. Wiley.
