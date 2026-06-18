# Other models

The [`solow-othermod`] crate collects likelihood models that fall outside the
core linear-regression and GLM families. At present it provides a single entry
point, [`BetaModel`], a maximum-likelihood **beta regression** in which both the
conditional mean and the precision of a response confined to the open interval
\\( (0, 1) \\) get their own linear predictors and link functions. This is a
deliberately small corner of the reference's "other models" area: the reference
also ships ordinal / proportional-odds style models, which are **not yet**
implemented here — beta regression is the only model in the crate today.

## Background

Beta regression models a response \\( y_i \in (0, 1) \\) — a rate, proportion,
or fraction — as Beta-distributed, reparameterized in terms of a mean
\\( \mu_i \in (0,1) \\) and a precision \\( \phi_i > 0 \\). Writing the Beta in
its shape form with \\( a = \mu\phi \\) and \\( b = (1-\mu)\phi \\) gives

\\[
\mathbb{E}[y_i] = \mu_i, \qquad
\operatorname{Var}[y_i] = \frac{\mu_i(1-\mu_i)}{1 + \phi_i},
\\]

so larger \\( \phi \\) means less dispersion around the mean. Each of the two
parameters has its own linear predictor and monotone link:

\\[
g_\mu(\mu_i) = x_i^{\top}\beta, \qquad g_\phi(\phi_i) = z_i^{\top}\gamma .
\\]

The default mean link is the **logit**, \\( g_\mu(\mu) = \ln\frac{\mu}{1-\mu} \\),
and the default precision link is the **log**, \\( g_\phi(\phi) = \ln\phi \\);
both are exposed as the `Link` enum (`Link::Logit`, `Link::Log`). The
per-observation log-likelihood is

\\[
\ell_i = \ln\Gamma(\phi_i) - \ln\Gamma(\mu_i\phi_i) - \ln\Gamma\!\big((1-\mu_i)\phi_i\big)
       + (\mu_i\phi_i - 1)\ln y_i + \big((1-\mu_i)\phi_i - 1\big)\ln(1 - y_i),
\\]

and the model is fit by maximizing \\( \ell = \sum_i \ell_i \\). The crate
supplies the analytic score (gradient) in closed form using the digamma
function, chained through the link derivatives and the two design matrices. The
parameter vector stacks the mean coefficients \\( \beta \\) first and the
precision coefficients \\( \gamma \\) second.

Estimation proceeds in two phases from weighted-least-squares starting values:
BFGS gets into the basin of the optimum, then a short Newton phase drives the
analytic score to zero. The coefficient covariance is the inverse of the
observed information \\( -H^{-1} \\), where \\( H \\) is the Hessian of the
log-likelihood; standard errors are \\( \sqrt{\operatorname{diag}(-H^{-1})} \\),
with z-statistics and two-sided normal p-values derived from them.

## Example

The response is a proportion in \\( (0, 1) \\). Below, the mean submodel has an
intercept and one covariate; the precision submodel is intercept-only (a
constant \\( \phi \\)). You supply the intercept columns yourself, as elsewhere
in Solow.

```rust
use ndarray::{array, Array2};
use solow_othermod::BetaModel;

// Six proportions, all strictly inside (0, 1).
let y = array![0.30, 0.55, 0.62, 0.48, 0.71, 0.40];

// Mean design: intercept + one covariate.
let x = Array2::from_shape_vec(
    (6, 2),
    vec![
        1.0, -1.0,
        1.0, -0.3,
        1.0,  0.2,
        1.0,  0.0,
        1.0,  0.8,
        1.0, -0.5,
    ],
)
.unwrap();

// Precision design: intercept only (constant phi).
let z = Array2::from_shape_vec((6, 1), vec![1.0; 6]).unwrap();

// Default links: logit for the mean, log for the precision.
let res = BetaModel::new(y, x, z).unwrap().fit().unwrap();

assert!(res.converged);

// params stacks [mean beta..., precision gamma...].
println!("all params       = {:?}", res.params);
println!("mean beta        = {:?}", res.params_mean());
println!("precision gamma  = {:?}", res.params_precision());
println!("std errors (bse) = {:?}", res.bse);
println!("z-values         = {:?}", res.tvalues);
println!("p-values         = {:?}", res.pvalues);
println!("log-likelihood   = {:.4}", res.llf);
println!("fitted means     = {:?}", res.fittedvalues);

// 95% normal confidence intervals: one [lower, upper] row per coefficient.
let ci = res.conf_int(0.05);
println!("conf_int shape   = {:?}", ci.dim());
```

When the fit converges, `res.params` holds three numbers here — the two mean
coefficients \\( (\beta_0, \beta_1) \\) followed by the single precision
coefficient \\( \gamma_0 \\) (on the log scale, so \\( \hat\phi = e^{\gamma_0} \\)).
`res.fittedvalues` returns the fitted conditional means \\( \hat\mu_i \\), each
guaranteed to lie in \\( (0, 1) \\), and `res.llf` is the maximized
log-likelihood. The exact numbers depend on the data and are not reproduced
here to avoid quoting figures that have not been recomputed.

To choose non-default links, use `BetaModel::with_links`, passing the mean link
and the precision link explicitly:

```rust
use solow_othermod::{BetaModel, Link};

// Same data, but state the links explicitly (here, the defaults).
let model = BetaModel::with_links(y, x, z, Link::Logit, Link::Log).unwrap();
let res = model.fit().unwrap();
```

`BetaModel` also exposes the likelihood machinery directly: `loglike(&params)`
and `score(&params)` evaluate the total log-likelihood and its analytic gradient
at an arbitrary parameter vector, and `nobs()` returns the observation count.

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `BetaModel` | Beta regression awaiting estimation; separate mean and precision linear predictors. |

**Results**

| Name | Description |
| --- | --- |
| `BetaResults` | Fitted output of `BetaModel::fit`. |

**Enums**

| Name | Description |
| --- | --- |
| `Link` | Monotone link for a submodel: `Logit` (mean default) or `Log` (precision default). |

**Key methods**

| Name | Description |
| --- | --- |
| `BetaModel::new` | Construct with the default links (logit mean, log precision). |
| `BetaModel::with_links` | Construct with explicit mean and precision links. |
| `BetaModel::fit` | Estimate by maximum likelihood; returns `BetaResults`. |
| `BetaModel::loglike` | Total log-likelihood at a parameter vector. |
| `BetaModel::score` | Analytic score (gradient) at a parameter vector. |
| `BetaModel::nobs` | Number of observations. |
| `BetaResults::params_mean` | Mean-submodel coefficients \\( \beta \\). |
| `BetaResults::params_precision` | Precision-submodel coefficients \\( \gamma \\). |
| `BetaResults::conf_int` | Normal confidence intervals at level \\( 1 - \alpha \\). |

`BetaResults` fields: `params`, `bse`, `tvalues`, `pvalues`, `cov_params`,
`llf`, `fittedvalues`, `k_mean`, `k_prec`, `nobs`, `converged`.

Full API: see the generated rustdoc for `solow-othermod`.

## References

- Ferrari, S. L. P., and Cribari-Neto, F. (2004). "Beta Regression for Modelling
  Rates and Proportions." *Journal of Applied Statistics*, 31(7), 799–815.
- Smithson, M., and Verkuilen, J. (2006). "A Better Lemon Squeezer?
  Maximum-Likelihood Regression with Beta-Distributed Dependent Variables."
  *Psychological Methods*, 11(1), 54–71.
- Cribari-Neto, F., and Zeileis, A. (2010). "Beta Regression in R."
  *Journal of Statistical Software*, 34(2), 1–24.

[`solow-othermod`]: https://github.com/solow-rs/solow
[`BetaModel`]: ./othermod.md
