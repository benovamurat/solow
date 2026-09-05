# Generalized additive models (GAM)

The [`solow-gam`] crate fits generalized additive models with a single
penalized smooth term plus an intercept. A smooth covariate is expanded into a
spline basis — either a quantile-knot B-spline ([`BSplines`]) or a cyclic cubic
regression spline ([`CyclicCubicSplines`]) — and the model is estimated by
penalized iteratively reweighted least squares (P-IRLS) at a *fixed* smoothing
parameter `alpha`. The main entry points are [`GlmGam`] (canonical-link
families) and [`GlmGamExt`] (arbitrary, possibly non-canonical links, with
effective degrees of freedom taken from the observed information).

> This crate is deliberately narrower than the reference's GAM module: it fits
> **one** smooth term with an explicit intercept at a user-supplied `alpha`. It
> does not select `alpha` automatically, combine several smooths, or expose a
> formula interface. The numerics are validated term-for-term against an
> authoritative reference.

## Background

An additive model replaces the linear predictor of a GLM with a sum of smooth
functions of the covariates. With one smooth term `f` and an intercept `β₀`,
the mean response `μ` is related to the covariate `x` through a link `g`:

\\[
g(\mu_i) = \beta_0 + f(x_i), \qquad \mathbb{E}[y_i] = \mu_i,
\\]

where `y` follows an exponential-dispersion [`Family`] with variance function
`V(μ)`. The smooth `f` is represented in a spline basis with columns
`B₁(x), …, B_k(x)`, so that

\\[
f(x_i) = \sum_{j=1}^{k} \beta_j \, B_j(x_i).
\\]

For [`BSplines`] the columns are degree-`d` B-splines on a knot vector with
`degree + 1` multiplicity at the boundaries and interior knots placed at the
empirical quantiles of `x`; the constant column is dropped so the basis does not
collide with the explicit intercept, leaving `k = df − 1` columns. For
[`CyclicCubicSplines`] the basis is Wood's cyclic cubic regression spline, which
ties the value and the first two derivatives at the boundary knots so the curve
joins into a loop.

Wiggliness is controlled by a quadratic roughness penalty. The penalty matrix
`S` is the integrated cross-product of the basis' second derivative,

\\[
S_{ab} = \int B_a''(x)\, B_b''(x)\, dx,
\\]

computed by Simpson quadrature for the B-spline basis (the field `cov_der2`),
and by the closed-form `S = D' B⁻¹ D` construction for the cyclic basis. Fitting
maximises the penalized log-likelihood, equivalently minimising the penalized
deviance

\\[
D_p(\beta) = D(\beta) + 2\alpha\, \beta^{\top} S\, \beta,
\\]

where the penalty acts only on the spline coefficients (the intercept is
unpenalized) and enters with the factor of two of the reference's augmented
P-IRLS convention — so the penalty matrix is \\( 2\alpha S \\) throughout, which
is exactly the `penalized_deviance` field the crate reports. Each P-IRLS step
forms the working response `z` and weights `W` and solves the penalized normal
equations

\\[
\bigl(X^{\top} W X + 2\alpha S\bigr)\,\hat\beta = X^{\top} W z,
\qquad X = [\,\mathbf{1}\;\; B(x)\,],
\\]

the same penalty block \\( 2\alpha S \\) appearing here as in the penalized
deviance above. The smoothing parameter `alpha` trades fit against smoothness:
`alpha → 0` recovers an unpenalized spline fit, while large `alpha` drives `f`
toward the unpenalized null space of `S`.

Because the spline basis is rich, the nominal column count overstates the model
complexity. The **effective degrees of freedom** summarise the actual
flexibility used. With the penalized hat matrix
`H = X (X' W X + 2αS)⁻¹ X' W`, the per-column edf are the contributions of each
design column to `tr(H)`, and `edf_total = tr(H)`. For a canonical link the
observed and expected information coincide and [`GlmGam`] reads the edf off the
expected-information hat matrix; for a non-canonical link [`GlmGamExt`] uses the
observed information instead, so the two agree exactly when the link is
canonical.

## Example

A Gaussian smooth fit to a noiseless sine. We build the covariate and response
inline, fit a cubic B-spline GAM with `df = 10` and a moderate smoothing
parameter, and inspect the real result fields.

```rust
use ndarray::Array1;
use solow_gam::GlmGam;
use solow_glm::Family;

// One covariate on [0, 1]; response is a smooth sine of x.
let x = Array1::linspace(0.0, 1.0, 60);
let y = x.mapv(|xi| (2.0 * std::f64::consts::PI * xi).sin());

// df = 10 basis degrees of freedom, cubic (degree 3), alpha = 0.6,
// Gaussian family with its canonical identity link.
let res = GlmGam::new(y, &x, 10, 3, 0.6, Family::Gaussian)
    .unwrap()
    .fit()
    .unwrap();

assert!(res.converged);
println!("intercept     = {:.4}", res.intercept());
println!("spline coefs  = {:?}", res.spline_params());
println!("edf (total)   = {:.3}", res.edf_total);
println!("deviance      = {:.4}", res.deviance);
println!("penalized dev = {:.4}", res.penalized_deviance);
println!("scale         = {:.4}", res.scale);
```

The fitted [`GamResults`] carries `params` (the intercept followed by the
`df − 1` spline coefficients), `fittedvalues` (the fitted mean `μ` at each
observation), the per-column `edf` vector together with `edf_total`, the
estimated `scale`, the unpenalized `deviance` and the `penalized_deviance`,
`df_resid = nobs − edf_total`, the `converged` flag, `n_iter`, and `dim_basis`.
The convenience methods `intercept()` and `spline_params()` slice `params`.
*(Illustrative behaviour, not exact numbers:)* with `alpha = 0.6` the smooth
tracks the sine closely and `edf_total` sits a little below the nominal ten
columns; raising `alpha` lowers `edf_total` toward the unpenalized null space,
as the crate's own tests confirm.

For a count response, swap in the Poisson family — its canonical log link is
selected automatically and `scale` is fixed at 1:

```rust
use ndarray::Array1;
use solow_gam::GlmGam;
use solow_glm::Family;

let x = Array1::linspace(0.0, 1.0, 50);
let y = x.mapv(|xi| (1.0 + (2.0 * std::f64::consts::PI * xi).sin()).exp().round());

let res = GlmGam::new(y, &x, 8, 3, 1.0, Family::Poisson)
    .unwrap()
    .fit()
    .unwrap();
assert!(res.converged);
assert_eq!(res.scale, 1.0);
```

### Non-canonical links and cyclic smooths

To fit a non-canonical link, or to use the cyclic basis, build the smooth
explicitly and pass its basis and penalty to [`GlmGamExt`]. The fitter then
computes the effective degrees of freedom from the observed information, which
is what makes a non-canonical link correct:

```rust
use ndarray::Array1;
use solow_gam::{BSplines, GlmGamExt};
use solow_glm::{Family, Link};

let x = Array1::linspace(0.0, 1.0, 70);
let y = x.mapv(|xi| (0.5 + 0.7 * (2.0 * std::f64::consts::PI * xi).sin()).exp());

// Build a cubic B-spline basis and its curvature penalty.
let bs = BSplines::new(&x, 8, 3).unwrap();
let basis = bs.basis().clone();
let penalty = bs.cov_der2().clone();

// Gaussian family with a (non-canonical) log link.
let res = GlmGamExt::new(y, basis, penalty, 0.5, Family::Gaussian, Link::Log)
    .unwrap()
    .fit()
    .unwrap();
assert!(res.converged);
println!("intercept = {:.4}", res.intercept());
println!("edf total = {:.3}", res.edf_total);
```

[`GamExtResults`] mirrors [`GamResults`] field-for-field (with `smooth_params()`
in place of `spline_params()`). Because [`GlmGamExt`] accepts any precomputed
`(basis, penalty)` pair, the same fitter drives a [`CyclicCubicSplines`] smooth;
use [`CyclicCubicSplines::with_centering`] so the centered design `[1, basis]`
is full rank and the fit is identifiable.

## Module reference

### Models

| Name | Description |
| --- | --- |
| `GlmGam` | Penalized additive model: intercept plus one B-spline smooth, P-IRLS at a fixed `alpha`, canonical link (`new`, `with_link`, `fit`, `smoother`). |
| `GlmGamExt` | Penalized additive model accepting a precomputed `(basis, penalty)` and an arbitrary link; edf from the observed information (`new`, `fit`). |

### Results

| Name | Description |
| --- | --- |
| `GamResults` | Fitted output of `GlmGam`: `params`, `fittedvalues`, `edf`, `edf_total`, `scale`, `deviance`, `penalized_deviance`, `df_resid`, `converged`, `n_iter`, `dim_basis`; helpers `intercept()`, `spline_params()`. |
| `GamExtResults` | Fitted output of `GlmGamExt`: same fields as `GamResults`; helpers `intercept()`, `smooth_params()`. |

### Smoothers (bases)

| Name | Description |
| --- | --- |
| `BSplines` | Quantile-knot B-spline basis for one covariate with a curvature penalty (`new`, `basis`, `cov_der2`, `knots`, `dim_basis`, `degree`); yields `df − 1` columns. |
| `CyclicCubicSplines` | Cyclic cubic regression spline basis with the `D' B⁻¹ D` wiggliness penalty (`new`, `with_centering`, `basis`, `cov_der2`, `knots`, `dim_basis`); yields `df` columns. |

### Enums (from `solow-glm`, used in signatures)

| Name | Description |
| --- | --- |
| `Family` | Response distribution and variance function (`Gaussian`, `Poisson`, `Binomial`, `Gamma`, `InverseGaussian`, `NegativeBinomial`). |
| `Link` | Link function relating the linear predictor to the mean (`Identity`, `Log`, `Logit`, `Probit`, `CLogLog`, `InversePower`, `InverseSquared`, `Sqrt`). |

Full API: see the generated rustdoc for `solow-gam`.

## References

- T. J. Hastie and R. J. Tibshirani, *Generalized Additive Models*, Chapman &
  Hall/CRC, Monographs on Statistics and Applied Probability 43, 1990.
- S. N. Wood, *Generalized Additive Models: An Introduction with R*, 2nd ed.,
  Chapman & Hall/CRC, 2017.
- P. H. C. Eilers and B. D. Marx, "Flexible Smoothing with B-splines and
  Penalties," *Statistical Science*, vol. 11, no. 2, pp. 89–121, 1996.
- S. N. Wood, "Thin Plate Regression Splines," *Journal of the Royal
  Statistical Society, Series B*, vol. 65, no. 1, pp. 95–114, 2003.

[`solow-gam`]: https://github.com/solow-rs/solow
[`GlmGam`]: ./gam.md
[`GlmGamExt`]: ./gam.md
[`GamResults`]: ./gam.md
[`GamExtResults`]: ./gam.md
[`BSplines`]: ./gam.md
[`CyclicCubicSplines`]: ./gam.md
[`CyclicCubicSplines::with_centering`]: ./gam.md
[`Family`]: ./glm.md
[`Link`]: ./glm.md
