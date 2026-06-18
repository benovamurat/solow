# Time series by state space methods

The `solow-statespace` crate fits linear-Gaussian state-space models and the
time-series estimators built on top of them. At its core is a time-invariant
Kalman filter and smoother (`StateSpace` for a scalar observation, `MvStateSpace`
for a vector observation); on top of that sit three maximum-likelihood
estimators — `Sarimax` (seasonal ARIMA), `UnobservedComponents` (structural
level / trend / seasonal models), and `DynamicFactor` (a single common factor
driving a panel). Every estimator evaluates the *exact* Gaussian log-likelihood
by the prediction-error decomposition and optimizes it with BFGS over an
unconstrained reparametrization.

> The crate covers the linear-Gaussian core of the reference's state-space
> machinery. It does not (yet) expose a generic user-specified state-space
> builder, multi-step forecasting with confidence intervals, simulation
> smoothing, or exogenous regressors; the "fitted values" exposed are
> one-step-ahead in-sample predictions.

## Background

A time-invariant linear-Gaussian state-space model writes an observed series in
terms of an unobserved state vector \\( \alpha_t \in \mathbb{R}^m \\):

\\[
\begin{aligned}
\alpha_t &= T\,\alpha_{t-1} + R\,\eta_t, & \eta_t &\sim N(0, Q),\\\\
y_t &= Z\,\alpha_t + \varepsilon_t, & \varepsilon_t &\sim N(0, H).
\end{aligned}
\\]

Here \\( T \\) is the transition matrix, \\( R \\) the selection matrix, \\( Q \\)
the state-disturbance covariance, \\( Z \\) the design, and \\( H \\) the
measurement-noise covariance. In `StateSpace` the observation \\( y_t \\) is a
scalar (so \\( Z \\) is a row and \\( H \\) a scalar); in `MvStateSpace` it is a
\\( p \\)-vector.

**Kalman filter.** Given the prediction \\( a_{t\mid t-1} = \mathbb{E}[\alpha_t \mid y_{1:t-1}] \\)
and its covariance \\( P_{t\mid t-1} \\), the filter forms the one-step-ahead
prediction error and its variance,

\\[
v_t = y_t - Z\,a_{t\mid t-1}, \qquad
F_t = Z\,P_{t\mid t-1}\,Z^{\top} + H,
\\]

updates to the filtered state with the Kalman gain \\( K_t = P_{t\mid t-1} Z^{\top} F_t^{-1} \\),

\\[
a_{t\mid t} = a_{t\mid t-1} + K_t\,v_t, \qquad
P_{t\mid t} = P_{t\mid t-1} - K_t\,Z\,P_{t\mid t-1},
\\]

and predicts ahead via \\( a_{t+1\mid t} = T\,a_{t\mid t} \\) and
\\( P_{t+1\mid t} = T\,P_{t\mid t}\,T^{\top} + R\,Q\,R^{\top} \\).

**Log-likelihood (prediction-error decomposition).** The exact Gaussian
log-likelihood is the sum of the conditional densities of each prediction error,

\\[
\log L = \sum_{t} \log p(y_t \mid y_{1:t-1})
= -\tfrac{1}{2}\sum_{t}\Big( p\log 2\pi + \log\lvert F_t\rvert + v_t^{\top} F_t^{-1} v_t \Big),
\\]

with \\( p = 1 \\) in the scalar case. The `filter` method returns this total as
`loglike` (and the per-observation contributions as `loglike_obs`), along with
the prediction errors `forecast_error` (\\( v_t \\)) and their variances
`forecast_error_cov` (\\( F_t \\)). A leading `loglike_burn` observations can be
excluded from the total, which is how differenced and diffuse-initialized models
drop their uninformative start-up terms.

**Smoother.** `StateSpace::smooth` runs a fixed-interval (Rauch–Tung–Striebel)
backward pass, returning the smoothed states \\( a_{t\mid n} \\) using all
observations. Its gain is \\( J_t = P_{t\mid t}\,T^{\top}\,P_{t+1\mid t}^{-1} \\)
and the recursion is \\( a_{t\mid n} = a_{t\mid t} + J_t\,(a_{t+1\mid n} - a_{t+1\mid t}) \\).

**SARIMAX.** A SARIMAX\\( (p,d,q)(P,D,Q,s) \\) model first applies non-seasonal
and seasonal differencing (the *simple-differencing* convention) and then casts
the resulting stationary ARMA into the Harvey companion-form state space, where
\\( T \\) holds the AR coefficients and \\( R \\) holds the MA coefficients. The
state is initialized from its stationary distribution — the solution of the
discrete Lyapunov equation \\( P = T P T^{\top} + R Q R^{\top} \\). During
estimation the AR and MA polynomials are kept stationary and invertible through
the Monahan reparametrization, and \\( \sigma^2 \\) is mapped through a square, so
the unconstrained BFGS iterates always map to a valid model.

**Unobserved components.** A structural model decomposes \\( y_t \\) into a level
(and optionally a slope and a stochastic seasonal) plus an irregular term. The
local-level model is

\\[
y_t = \mu_t + \varepsilon_t, \qquad
\mu_t = \mu_{t-1} + \xi_t,
\\]

and the local-linear-trend model adds a random-walk slope
\\( \beta_t = \beta_{t-1} + \zeta_t \\) so that \\( \mu_t = \mu_{t-1} + \beta_{t-1} + \xi_t \\).
The nonstationary states use an approximate-diffuse initialization (a large
multiple of the identity) and their start-up observations are excluded from the
log-likelihood.

## Example

A short worked example fits a non-seasonal ARIMA(1, 1, 1) to a small series and
inspects the fitted result.

```rust
use ndarray::Array1;
use solow_statespace::{Sarimax, SarimaxOrder};

// A small upward-drifting series.
let y = Array1::from_vec(vec![
    1.0, 1.4, 1.1, 1.9, 2.2, 2.0, 2.7, 3.1, 2.9, 3.6,
    3.4, 4.1, 4.5, 4.2, 5.0, 5.3, 5.1, 5.9, 6.2, 6.0,
]);

// ARIMA(1, 1, 1): first-difference, then an AR(1) + MA(1) on the differences.
let order = SarimaxOrder::new(1, 1, 1);
let res = Sarimax::new(y, order).unwrap().fit().unwrap();

// Parameters are ordered [ar.L1, ma.L1, sigma2].
assert_eq!(res.params.len(), 3);
println!("params    = {:?}", res.params);
println!("loglike   = {:.4}", res.llf);
println!("AIC / BIC = {:.3} / {:.3}", res.aic, res.bic);
println!("converged = {}", res.converged);
```

`Sarimax::new` binds the model to the series; `fit` differences the data,
optimizes the exact log-likelihood, and returns a `SarimaxResults`. Beyond
`params`, that result exposes the inference battery `bse`, `zvalues`, `pvalues`
(standard errors from the inverse negative Hessian, z-statistics, and two-sided
normal p-values), the information criteria `aic`, `bic`, `hqic`, the
one-step-ahead in-sample `fittedvalues` and `resid`, the effective sample size
`nobs` (after differencing), and `converged`. The exact numbers depend on the
optimizer, so they are not reproduced here.

For a seasonal model, use `SarimaxOrder::seasonal`:

```rust
use solow_statespace::SarimaxOrder;

// (1, 1, 1)(1, 0, 0, 4): a non-seasonal ARIMA(1, 1, 1) plus a seasonal AR(1)
// at period 4.
let order = SarimaxOrder::seasonal(1, 1, 1, 1, 0, 0, 4);
```

**Structural (unobserved-components) model.** The same series can be fit as a
local-linear-trend structural model:

```rust
use ndarray::Array1;
use solow_statespace::{Level, UcSpec, UnobservedComponents};

let y = Array1::from_vec(vec![
    1.0, 1.4, 1.1, 1.9, 2.2, 2.0, 2.7, 3.1, 2.9, 3.6,
    3.4, 4.1, 4.5, 4.2, 5.0, 5.3, 5.1, 5.9, 6.2, 6.0,
]);

let spec = UcSpec::new(Level::LocalLinearTrend);
let res = UnobservedComponents::new(y, spec).unwrap().fit().unwrap();

// Variance parameters, ordered [sigma2.irregular, sigma2.level, sigma2.trend].
println!("variances = {:?}", res.params);
println!("loglike   = {:.4}", res.llf);
```

Add a stochastic seasonal with `UcSpec::new(Level::LocalLevel).with_seasonal(4)`.
The fitted `UcResults` carries the estimated variances in `params`, the
log-likelihood `llf`, the information criteria `aic` / `bic` / `hqic`, `nobs`,
and `converged`.

**Direct Kalman filtering.** If you have your own state-space matrices, build a
`StateSpace` and filter a series directly. The example below is the scalar AR(1)
model \\( \alpha_t = \phi\,\alpha_{t-1} + \eta_t \\), \\( y_t = \alpha_t \\):

```rust
use ndarray::{array, Array1};
use solow_statespace::StateSpace;

let phi = 0.5;
let sigma2 = 1.3;
let y: Array1<f64> = array![0.5, -0.2, 1.1, 0.3, -0.9, 0.7];

let ss = StateSpace {
    transition: array![[phi]],
    selection: array![[1.0]],
    state_cov: array![[sigma2]],
    design: array![1.0],
    obs_cov: 0.0,
    init_state: array![0.0],
    // Stationary AR(1) initial variance.
    init_cov: array![[sigma2 / (1.0 - phi * phi)]],
};

let out = ss.filter(&y, 0);
println!("loglike        = {:.6}", out.loglike);
println!("filtered state = {:?}", out.filtered_state);

// Backward smoothing pass over the same series.
let smoothed = ss.smooth(&y, 0).unwrap();
println!("smoothed state = {:?}", smoothed);
```

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `Sarimax` | Seasonal ARIMA estimator, fit by exact-likelihood ML. |
| `UnobservedComponents` | Structural level / trend / seasonal model, fit by ML. |
| `DynamicFactor` | Single common-factor model for a multivariate panel. |
| `StateSpace` | Time-invariant linear-Gaussian model with scalar observations; provides `filter` and `smooth`. |
| `MvStateSpace` | Time-invariant linear-Gaussian model with vector observations; provides `loglike` and `forecasts`. |

**Specifications**

| Name | Description |
| --- | --- |
| `SarimaxOrder` | Order \\( (p,d,q)(P,D,Q,s) \\); constructors `new` and `seasonal`. |
| `UcSpec` | Structural-model spec; `new`, `with_seasonal`. |

**Results**

| Name | Description |
| --- | --- |
| `SarimaxResults` | `params`, `bse`, `zvalues`, `pvalues`, `llf`, `aic`, `bic`, `hqic`, `fittedvalues`, `resid`, `nobs`, `converged`. |
| `UcResults` | `params`, `llf`, `aic`, `bic`, `hqic`, `nobs`, `converged`. |
| `DynamicFactorResults` | `params`, `llf`, `aic`, `bic`, `hqic`, `nobs`, `converged`. |
| `FilterOutput` | Kalman-filter output: `loglike`, `loglike_obs`, predicted/filtered states and covariances, `forecast_error`, `forecast_error_cov`. |

**Enums**

| Name | Description |
| --- | --- |
| `Level` | Level specification: `LocalLevel` or `LocalLinearTrend`. |

Full API: see the generated rustdoc for `solow-statespace`.

## References

- Durbin, J., and Koopman, S. J. (2012). *Time Series Analysis by State Space
  Methods*, 2nd ed. Oxford University Press.
- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
  Problems." *Journal of Basic Engineering*, 82(1), 35–45.
- Monahan, J. F. (1984). "A note on enforcing stationarity in autoregressive –
  moving average models." *Biometrika*, 71(2), 403–404.
