# Multiple imputation (MICE)

The [`solow-impute`] crate provides the *deterministic core* of multiple
imputation by chained equations (MICE): a single regression imputation step and
Rubin's rules for pooling. Its two entry points are the free functions
`conditional_mean_impute` — which fills one variable's missing entries with the
conditional mean from a least-squares fit on the observed rows — and `combine`,
which applies Rubin's combining rules to the per-imputation estimates and returns
a pooled [`CombinedEstimate`].

> **Scope.** This crate is deliberately smaller than the reference's full MICE
> driver: it implements only the exactly reproducible pieces. The stochastic
> posterior draws and predictive-mean-matching lookups that turn a single
> conditional mean into a proper random imputation depend on a random number
> generator and are out of scope here. You drive the chained-equations loop and
> the imputation draws yourself, then use `combine` to pool the analyses.

## Background

MICE imputes a data matrix one column at a time. Writing the columns as
\\( X_1, \dots, X_p \\), a single update of column \\( X_j \\) regresses it on
the remaining columns \\( X_{-j} \\) using only the rows where \\( X_j \\) is
observed, then uses the fit to fill the rows where \\( X_j \\) is missing.
`conditional_mean_impute` performs the deterministic heart of that step. With
the observed-row design matrix \\( X_o \\) (intercept column included by the
caller) and response \\( y_o \\), it solves ordinary least squares

\\[
\hat\beta = (X_o^\top X_o)^{-1} X_o^\top y_o,
\\]

and predicts the conditional mean \\( \hat{y}_m = X_m \hat\beta \\) at the
missing rows \\( X_m \\). The residual scale \\( \hat\sigma^2 \\) of the
observed-row fit is returned so that a caller can add the stochastic noise term
\\( \hat\sigma\, z \\) that a full Bayesian imputation requires.

After running the chained equations to completion \\( m \\) times — once per
imputed data set — each completed data set is analysed with the model of
interest, yielding parameter estimates \\( \hat\theta_k \\) and covariance
matrices \\( U_k \\) for \\( k = 1, \dots, m \\). **Rubin's rules** combine these.
The pooled estimate is the simple average

\\[
\bar\theta = \frac{1}{m} \sum_{k=1}^{m} \hat\theta_k .
\\]

The total covariance decomposes into a *within-imputation* part \\( \bar U \\)
and a *between-imputation* part \\( B \\),

\\[
\bar U = \frac{1}{m} \sum_{k=1}^{m} U_k,
\qquad
B = \frac{1}{m-1} \sum_{k=1}^{m} (\hat\theta_k - \bar\theta)(\hat\theta_k - \bar\theta)^\top,
\\]

which combine into the total covariance

\\[
T = \bar U + \Bigl(1 + \tfrac{1}{m}\Bigr) B .
\\]

Per parameter, write \\( \bar u \\), \\( b \\) and \\( t \\) for the
corresponding diagonal entries. The relative increase in variance due to
nonresponse and the fraction of missing information are

\\[
r = \frac{(1 + 1/m)\, b}{\bar u},
\qquad
\lambda = \frac{(1 + 1/m)\, b}{t} = \frac{r}{1 + r},
\\]

and the standard error is \\( \sqrt{t} \\). Inference uses a Student-\\( t \\)
reference with the Barnard–Rubin (1999) degrees of freedom, which refine the
original Rubin (1987) value \\( \nu_{\text{old}} = (m-1)/\lambda^2 \\) toward the
complete-data residual degrees of freedom \\( \nu_{\text{com}} \\):

\\[
\nu_{\text{obs}} = \frac{\nu_{\text{com}} + 1}{\nu_{\text{com}} + 3}\,
\nu_{\text{com}}\,(1 - \lambda),
\qquad
\nu = \frac{\nu_{\text{old}}\, \nu_{\text{obs}}}{\nu_{\text{old}} + \nu_{\text{obs}}} .
\\]

Passing \\( \nu_{\text{com}} = \infty \\) (`f64::INFINITY`) drops the
observed-data adjustment and recovers the large-sample Rubin (1987) degrees of
freedom.

## Example

### A single regression imputation step

Suppose a target variable follows \\( y \approx 1 + 2x \\) on the rows where it
is observed, and we want to fill two rows where it is missing. The caller splits
the data into observed and missing blocks and supplies the design matrices
directly, intercept column included.

```rust
use ndarray::array;
use solow_impute::conditional_mean_impute;

// Observed rows: y = 1 + 2*x exactly. First column is the intercept.
let endog_obs = array![3.0, 5.0, 7.0, 9.0];
let exog_obs = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];

// Two rows where the target is missing (x = 10 and x = 0).
let exog_miss = array![[1.0, 10.0], [1.0, 0.0]];

let res = conditional_mean_impute(endog_obs, exog_obs, &exog_miss).unwrap();

println!("coefficients   = {:?}", res.params);          // ~ [1.0, 2.0]
println!("imputed values = {:?}", res.imputed_missing); // ~ [21.0, 1.0]
println!("residual scale = {:e}", res.scale);           // ~ 0 (perfect fit)
```

Because the observed rows lie exactly on the line \\( y = 1 + 2x \\), the fitted
coefficients are \\( [1, 2] \\), the conditional means at \\( x = 10 \\) and
\\( x = 0 \\) are \\( 21 \\) and \\( 1 \\), and the residual scale is numerically
zero. The fields exposed on `ConditionalMeanImputation` are `params`,
`fitted_observed` (conditional means at the observed rows), `imputed_missing`
(the deterministic imputations), and `scale`.

### Pooling analyses with Rubin's rules

After producing \\( m \\) completed data sets and fitting your analysis model on
each, collect the per-imputation parameter vectors and covariance matrices and
pool them. Here \\( m = 3 \\) imputations of a single scalar parameter are
combined with a complete-data degrees of freedom of `50.0`.

```rust
use ndarray::array;
use solow_impute::combine;

// Per-imputation point estimates of one parameter ...
let p1 = array![1.0];
let p2 = array![1.4];
let p3 = array![0.7];

// ... and their (1x1) covariance matrices.
let c1 = array![[0.02]];
let c2 = array![[0.02]];
let c3 = array![[0.02]];

let res = combine(&[p1, p2, p3], &[c1, c2, c3], 50.0).unwrap();

println!("pooled estimate = {:?}", res.params);        // mean = 1.0333...
println!("std error       = {:?}", res.bse);
println!("fmi             = {:?}", res.fmi);
println!("df (Barnard-Rubin) = {:?}", res.df);

// Wald inference uses the per-parameter Barnard-Rubin df.
println!("t-values  = {:?}", res.tvalues());
println!("p-values  = {:?}", res.pvalues());
println!("95% CI    = {:?}", res.conf_int(0.05));
```

The pooled point estimate is the mean of the three inputs,
\\( (1.0 + 1.4 + 0.7)/3 \approx 1.033 \\). The remaining numbers (`bse`, `fmi`,
`df`, and the derived `tvalues`, `pvalues`, `conf_int`) follow from the
within/between covariance decomposition above; their exact values depend on the
spread of the per-imputation estimates relative to their average covariance, so
they are described here rather than transcribed. The `CombinedEstimate` struct
also carries `cov_within`, `cov_between`, `cov_total`, `relative_increase`, and
`m` (the number of imputations combined).

`combine` requires at least two imputations — the between-imputation covariance
is otherwise undefined — and returns an error on a single imputation or on
mismatched parameter/covariance shapes.

## Module reference

**Functions**

| Name | Description |
| --- | --- |
| `combine` | Pool `m` per-imputation parameter vectors and covariance matrices with Rubin's rules; takes `params_list`, `cov_list`, and the complete-data df `dfcom`. |
| `conditional_mean_impute` | Fit `endog_obs ~ exog_obs` by OLS on the observed rows and predict the conditional mean at the missing rows `exog_miss`. |

**Results**

| Name | Description |
| --- | --- |
| `CombinedEstimate` | Pooled estimate from Rubin's rules: `params`, `cov_within`, `cov_between`, `cov_total`, `bse`, `relative_increase`, `fmi`, `df`, `m`, plus methods `tvalues`, `pvalues`, and `conf_int(alpha)`. |
| `ConditionalMeanImputation` | One deterministic imputation step: `params`, `fitted_observed`, `imputed_missing`, and `scale` (the observed-row residual \\( \sigma^2 \\)). |

Full API: see the generated rustdoc for `solow-impute`.

## References

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley,
  New York.
- Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom with
  multiple imputation. *Biometrika*, 86(4), 948–955.
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate
  Imputation by Chained Equations in R. *Journal of Statistical Software*,
  45(3), 1–67.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.).
  Chapman & Hall/CRC, Boca Raton.

[`solow-impute`]: https://github.com/solow-rs/solow
[`CombinedEstimate`]: ./imputation.md
