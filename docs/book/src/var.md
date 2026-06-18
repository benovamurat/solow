# Vector autoregressions

The [`solow-var`] crate fits multivariate vector autoregressions (VAR) and the
two structural extensions that build on them. The reduced-form entry point is
[`Var`], estimated by equation-by-equation ordinary least squares; on top of it
sit [`Svar`] (a recursively identified structural VAR) and [`Vecm`] (a vector
error-correction model fitted by Johansen's reduced-rank maximum likelihood),
together with the standalone [`coint_johansen`] cointegration test. Every fit is
cross-validated to machine precision against an authoritative reference.

> This crate covers reduced-form estimation, recursive (Cholesky) structural
> identification, and cointegration analysis. Impulse-response functions,
> forecast-error variance decomposition, and Granger-causality tests are *not*
> yet implemented; lag selection is done manually by comparing the AIC / BIC /
> HQIC / FPE criteria that each fit reports (see the example).

## Background

For a `K`-dimensional series \\( y_t \\) and lag order \\( p \\), the
reduced-form VAR(\\( p \\)) with a constant is

\\[
y_t = \nu + A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + u_t,
\qquad u_t \sim (0, \Sigma_u),
\\]

where \\( \nu \\) is a \\( K \\)-vector intercept, each \\( A_i \\) is a
\\( K \times K \\) coefficient matrix, and \\( u_t \\) is mean-zero white noise.
Stacking the regressors of observation \\( t \\) into the row
\\( Z_t = [\,1,\; y_{t-1}^\top,\; y_{t-2}^\top,\; \dots,\; y_{t-p}^\top\,] \\)
(most recent lag first, following Lütkepohl's convention), the model becomes a
multivariate linear regression \\( Y = Z B + U \\) and each equation is
estimated independently by OLS:

\\[
\hat{B} = (Z^\top Z)^{-1} Z^\top Y .
\\]

With \\( T \\) usable observations the residual covariance is reported in both
the degrees-of-freedom-adjusted and maximum-likelihood scalings,

\\[
\hat{\Sigma}_u = \frac{\hat{U}^\top \hat{U}}{T - Kp - k_{\text{trend}}},
\qquad
\tilde{\Sigma}_u = \frac{\hat{U}^\top \hat{U}}{T},
\\]

and the Gaussian log-likelihood evaluated at the estimates is

\\[
\ell = -\frac{TK}{2}\ln(2\pi) - \frac{T}{2}\bigl(\ln\lvert\tilde{\Sigma}_u\rvert + K\bigr).
\\]

Lag selection compares information criteria built from
\\( \ln\lvert\tilde{\Sigma}_u\rvert \\) and the free-parameter count
\\( m = pK^2 + K\,k_{\text{trend}} \\):

\\[
\text{AIC} = \ln\lvert\tilde{\Sigma}_u\rvert + \frac{2}{T} m,
\qquad
\text{BIC} = \ln\lvert\tilde{\Sigma}_u\rvert + \frac{\ln T}{T} m,
\qquad
\text{HQIC} = \ln\lvert\tilde{\Sigma}_u\rvert + \frac{2\ln\ln T}{T} m,
\\]

alongside the final prediction error
\\( \text{FPE} = \bigl((T + \text{df}_{\text{model}})/\text{df}_{\text{resid}}\bigr)^{K}\,\lvert\tilde{\Sigma}_u\rvert \\).
You pick the order that minimizes your chosen criterion.

**Structural identification.** A recursive SVAR writes the reduced-form shocks
as \\( u_t = B\,\varepsilon_t \\) with orthonormal structural shocks
\\( \varepsilon_t \\) and a lower-triangular impact matrix \\( B \\). The
maximum-likelihood \\( B \\) is the Cholesky factor of the ML residual
covariance, \\( B B^\top = \tilde{\Sigma}_u \\), so the ordering of the series
determines the contemporaneous causal chain.

**Cointegration.** When the series are individually integrated but share
long-run equilibria, the VAR is reparametrized as a VEC model,

\\[
\Delta y_t = \Pi\, y_{t-1} + \Gamma_1 \Delta y_{t-1} + \cdots
           + \Gamma_{p-1} \Delta y_{t-p+1} + u_t,
\qquad \Pi = \alpha \beta^\top,
\\]

where \\( \Pi \\) has reduced rank \\( r \\): the \\( K \times r \\) matrix
\\( \beta \\) spans the cointegrating space and \\( \alpha \\) holds the
adjustment loadings. Johansen's procedure concentrates out the short-run terms
\\( \Gamma_i \\) and solves a generalized symmetric eigenproblem; the ordered
eigenvalues \\( \lambda_1 \ge \cdots \ge \lambda_K \\) drive the trace and
maximum-eigenvalue rank tests,

\\[
\text{LR}_{\text{trace}}(r) = -T \sum_{j>r} \ln(1-\lambda_j),
\qquad
\text{LR}_{\max}(r) = -T \ln(1-\lambda_{r+1}).
\\]

## Example

Fit a reduced-form VAR to a short bivariate series, inspect the coefficient
matrices and information criteria, then layer a recursive SVAR on top.

```rust
use ndarray::array;
use solow_var::{Svar, Var};

// A small, well-conditioned bivariate series (rows are time, columns are the
// two variables y1, y2). With trend = C (the default) a constant is included.
let y = array![
    [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
    [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
    [0.9, 0.8], [1.3, 0.5],
];

// Reduced-form VAR(1) with an intercept.
let res = Var::new(y.clone()).unwrap().fit(1).unwrap();

assert_eq!(res.neqs, 2);     // K = 2 equations
assert_eq!(res.k_ar, 1);     // lag order p = 1
assert_eq!(res.coefs.len(), 1);
assert_eq!(res.coefs[0].dim(), (2, 2)); // A_1 is K x K

println!("intercept nu = {:?}", res.intercept);
println!("A_1          = {:?}", res.coefs[0]);
println!("Sigma_u      = {:?}", res.sigma_u);
println!("llf = {:.4}  AIC = {:.4}  BIC = {:.4}  HQIC = {:.4}  FPE = {:.4}",
    res.llf, res.aic, res.bic, res.hqic, res.fpe);
```

`Var::new` includes a constant (`Trend::C`); use `Var::with_trend(y, Trend::N)`
to drop it. The fitted [`VarResults`] exposes `params` (shape
`(df_model, K)`, row blocks ordered `[intercept; A_1; ...; A_p]`), the split-out
`intercept` and `coefs`, `resid`, `fittedvalues`, both `sigma_u` and
`sigma_u_mle`, the likelihood `llf`, the criteria `aic` / `bic` / `hqic` /
`fpe`, and per-coefficient `bse`, `tvalues`, and `pvalues` (two-sided, standard
normal). The coefficient entry `coefs[i][[r, c]]` is the effect of variable
`c` at lag `i + 1` on equation `r`.

**Manual lag selection.** There is no `select_order` helper; fit each candidate
order and compare the criterion you care about:

```rust
use ndarray::array;
use solow_var::Var;
# let y = array![
#     [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
#     [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
#     [0.9, 0.8], [1.3, 0.5],
# ];
let model = Var::new(y).unwrap();
let best = (1..=3)
    .map(|p| (p, model.fit(p).unwrap().aic))
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .unwrap();
println!("AIC-minimizing lag order = {}", best.0);
```

**Recursive SVAR.** `Svar` reuses the reduced-form fit and returns the
lower-triangular impact matrix `b` (the Cholesky factor of `sigma_u_mle`) with
`a` fixed to the identity:

```rust
# use ndarray::array;
# use solow_var::Svar;
# let y = array![
#     [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
#     [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
#     [0.9, 0.8], [1.3, 0.5],
# ];
let svar = Svar::new(y).unwrap().fit(1).unwrap();
println!("structural impact B = {:?}", svar.b); // lower triangular, B Bᵀ = Σ_u
```

The exact printed numbers depend on the data above; the structural assertions
hold by construction — `b` is lower triangular and `b.dot(&b.t())` reproduces
`sigma_u_mle`.

**Cointegration.** For integrated series, run the Johansen test and, at a chosen
rank, fit a VECM:

```rust
use ndarray::array;
use solow_var::{coint_johansen, Vecm};

// Two series sharing a common stochastic trend (y1 - y2 stationary).
let data = array![
    [0.0, 0.1], [0.5, 0.4], [0.3, 0.5], [0.9, 0.7], [1.2, 1.3],
    [1.0, 1.1], [1.6, 1.5], [1.9, 2.0], [2.1, 1.9], [2.5, 2.6],
    [2.3, 2.4], [2.9, 3.0], [3.2, 3.1], [3.0, 3.2], [3.6, 3.5], [3.9, 4.0],
];

// det_order = 0 (constant), k_ar_diff = 1 lagged difference.
let jh = coint_johansen(&data, 0, 1).unwrap();
println!("eigenvalues       = {:?}", jh.eig);
println!("trace statistics  = {:?}", jh.lr1);   // compare against jh.cvt
println!("max-eig statistics= {:?}", jh.lr2);   // compare against jh.cvm

// Fit a rank-1 VECM with 1 lagged difference.
let vecm = Vecm::new(data, 1, 1).unwrap().fit().unwrap();
println!("alpha (loadings)      = {:?}", vecm.alpha); // (K, r)
println!("beta  (cointegration) = {:?}", vecm.beta);  // (K, r), first r rows = I
println!("gamma (short run)     = {:?}", vecm.gamma);
```

`JohansenResult` returns the eigenvalues `eig`, the trace and max-eigenvalue
statistics `lr1` / `lr2`, and their 90 / 95 / 99 % critical-value tables `cvt` /
`cvm`; reject a given rank when the statistic exceeds the column-95 % critical
value. [`VecmResults`] reports `alpha`, the normalized `beta`, the stacked
short-run matrix `gamma`, the deterministic pieces `det_coef` / `det_coef_coint`,
the ML residual covariance `sigma_u`, and the log-likelihood `llf`.

## Module reference

### Models

| Name | Description |
| --- | --- |
| `Var` | Reduced-form VAR(`p`); `new`, `with_trend`, `fit(p)`. |
| `Svar` | Recursively (Cholesky) identified structural VAR; `new`, `with_trend`, `fit(p)`. |
| `Vecm` | Vector error-correction model via Johansen ML; `new`, `with_deterministic`, `fit`. |

### Results

| Name | Description |
| --- | --- |
| `VarResults` | Coefficients, residual covariances, likelihood, AIC/BIC/HQIC/FPE, and coefficient inference. |
| `SvarResults` | Structural matrices `a` and `b`, `sigma_u_mle`, and the underlying `var` fit. |
| `VecmResults` | `alpha`, `beta`, `gamma`, deterministic coefficients, `sigma_u`, and `llf`. |
| `JohansenResult` | Eigenvalues, trace / max-eigenvalue statistics, and their critical-value tables. |

### Functions

| Name | Description |
| --- | --- |
| `coint_johansen` | Johansen trace and maximum-eigenvalue cointegration-rank test. |
| `c_sja` | Maximum-eigenvalue critical values for `n` common trends and deterministic order `p`. |
| `c_sjt` | Trace critical values for `n` common trends and deterministic order `p`. |

### Enums

| Name | Description |
| --- | --- |
| `Trend` | Deterministic term in the VAR design: `N` (none) or `C` (constant). |
| `Deterministic` | VECM deterministic spec: `None`, `ConstantOutside` (`"co"`), `ConstantInside` (`"ci"`). |

Full API: see the generated rustdoc for `solow-var`.

## References

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
  Springer. (VAR estimation and information criteria, ch. 3–4; VECM and
  cointegration, ch. 6–7.)
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector
  Autoregressive Models*. Oxford University Press.
- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors."
  *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
  (Chapters 10–11 and 19–20.)

[`solow-var`]: https://github.com/solow-rs/solow
[`Var`]: ./var.md
[`VarResults`]: ./var.md
[`Svar`]: ./var.md
[`Vecm`]: ./var.md
[`VecmResults`]: ./var.md
[`coint_johansen`]: ./var.md
