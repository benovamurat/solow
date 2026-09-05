# Markov switching and regime models

The `solow-regime` crate fits time-series models whose parameters follow a
hidden, first-order Markov chain over a finite set of `k` *regimes* (states).
Two estimators are provided: `MarkovRegression`, a switching regression in which
the intercept (and any exogenous coefficients) and, optionally, the error
variance switch across regimes; and `MarkovAutoregression`, a switching
autoregression of a given `order` in which the mean, the autoregressive
coefficients, and optionally the variance switch. Both are estimated by maximum
likelihood through the **Hamilton filter**, and both return a `MarkovResults`
exposing the transition matrix, the steady-state distribution, expected regime
durations, and the filtered and smoothed regime probabilities (the latter via
the **Kim smoother**).

## Background

Let \\( S_t \in \\{0, 1, \dots, k-1\\} \\) be an unobserved regime that evolves
as a first-order Markov chain with a time-invariant, left-stochastic transition
matrix \\( P \\),

\\[
P_{ij} = \Pr(S_t = i \mid S_{t-1} = j), \qquad \sum_{i=0}^{k-1} P_{ij} = 1 .
\\]

Each column \\( j \\) is the conditional distribution of the next regime given
the previous one. The chain's ergodic (steady-state) distribution
\\( \pi \\) solves \\( (I - P)\,\pi = 0 \\) subject to \\( \mathbf{1}^\top \pi = 1 \\)
and is used as the initial regime distribution.

**Switching regression.** Conditional on the regime, the response is Gaussian,

\\[
y_t = x_t^\top \beta_{S_t} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}\!\big(0,\ \sigma^2_{S_t}\big),
\\]

where \\( x_t \\) is the design row (a leading column of ones gives a switching
intercept). When `switching_variance` is `false`, a single \\( \sigma^2 \\) is
shared across regimes.

**Switching autoregression.** With autoregressive order \\( p \\),

\\[
y_t = a_{S_t} + \sum_{j=1}^{p} \phi_{j,S_t}\,\big(y_{t-j} - a_{S_{t-j}}\big)
      + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal{N}\!\big(0,\ \sigma^2_{S_t}\big),
\\]

a mean-adjusted (Hamilton) parameterization. The model conditions on the first
\\( p \\) observations, so estimation uses \\( \text{nobs} = T - p \\) points.

**Likelihood (Hamilton filter).** Because \\( S_t \\) is latent, the regimes are
integrated out one step at a time. Writing \\( Y_t = (y_1, \dots, y_t) \\), the
filter alternates a *prediction* step,

\\[
\Pr(S_t = i \mid Y_{t-1}) = \sum_{j} P_{ij}\,\Pr(S_{t-1} = j \mid Y_{t-1}),
\\]

and an *update* step that incorporates the conditional density
\\( f(y_t \mid S_t = i) \\),

\\[
\Pr(S_t = i \mid Y_t) =
\frac{f(y_t \mid S_t = i)\,\Pr(S_t = i \mid Y_{t-1})}
     {\sum_{l} f(y_t \mid S_t = l)\,\Pr(S_t = l \mid Y_{t-1})} .
\\]

The denominator is the one-step predictive likelihood \\( f(y_t \mid Y_{t-1}) \\),
and the log-likelihood maximized by the optimizer is

\\[
\ell = \sum_{t} \log f(y_t \mid Y_{t-1}) .
\\]

For an autoregression of order \\( p \\) the filter is carried over the joint
state \\( (S_t, S_{t-1}, \dots, S_{t-p}) \\), i.e. \\( k^{p+1} \\) combinations.
All quantities are evaluated in log space for numerical stability.

**Smoothing (Kim smoother).** A backward pass refines the filtered
probabilities into *smoothed* probabilities
\\( \Pr(S_t = i \mid Y_T) \\) that condition on the whole sample.

**Estimation.** The optimizer runs unconstrained. The transition columns are
mapped through a logistic (softmax) transform, the variances through a square
map, and the AR coefficients through the Monahan partial-autocorrelation
stationarity transform; a BFGS phase is followed by a damped-Newton polish.
The **expected duration** of regime \\( i \\) is reported as
\\( 1 / (1 - P_{ii}) \\).

## Example

The following fits a two-regime switching regression with a switching intercept
and a switching variance to a short series that visibly shifts level partway
through. A constant column is added internally, so you pass only the response.

```rust
use ndarray::{array, Array1};
use solow_regime::MarkovRegression;

// A series that sits near 0 for the first half and near 5 for the second.
let y: Array1<f64> = array![
    0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3, 0.1,
    5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.2,
];

// 2 regimes, switching intercept (added internally) and switching variance.
let model = MarkovRegression::new(y, 2, true).unwrap();
let res = model.fit().unwrap();

println!("converged          = {}", res.converged);
println!("log-likelihood     = {:.3}", res.llf);
println!("AIC / BIC          = {:.2} / {:.2}", res.aic, res.bic);
println!("param names        = {:?}", res.param_names);
println!("params             = {:?}", res.params);
println!("transition P       =\n{:?}", res.transition);
println!("steady-state pi    = {:?}", res.initial_probabilities);
println!("expected durations = {:?}", res.expected_durations);

// Probability of being in each regime at time t.
println!("filtered  Pr(S_t)  =\n{:?}", res.filtered_marginal_probabilities);
println!("smoothed  Pr(S_t)  =\n{:?}", res.smoothed_marginal_probabilities);
```

The `param_names` vector labels the constrained estimates in
`res.params`: the transition block first (named `p[j->i]`, e.g. `p[0->0]` and
`p[1->0]` for two regimes), then the per-regime intercepts (`const[0]`,
`const[1]`), then the variance(s) (`sigma2[0]`, `sigma2[1]` here, or a single
`sigma2` when the variance does not switch). The `transition` field is the
left-stochastic matrix \\( P \\) (columns sum to one), and
`filtered_marginal_probabilities` / `smoothed_marginal_probabilities` are
`(nobs, k)` arrays whose rows sum to one.

*Illustrative output.* For a clean level shift like the one above you should
expect the two recovered intercepts to be near `0` and `5`, the two diagonal
transition probabilities to be high (regimes are persistent, so expected
durations are well above one), and the smoothed probabilities to switch sharply
from regime to regime around the break. Exact numbers depend on the optimizer
path; inspect the fitted fields rather than hard-coding values.

For a switching autoregression, build the model with an explicit `order` and the
two switching flags (`switching_ar`, `switching_variance`); the full series is
passed and the first `order` observations are used only as lags:

```rust
use ndarray::{array, Array1};
use solow_regime::MarkovAutoregression;

let y: Array1<f64> = array![
    0.0, 0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7,
    3.0, 3.4, 3.1, 3.5, 3.2, 3.6, 3.3, 3.7,
];

// 2 regimes, AR order 1, switching AR coefficient, non-switching variance.
let model = MarkovAutoregression::new(y, 2, 1, true, false).unwrap();
let res = model.fit().unwrap();

println!("nobs = {} (= T - order)", res.nobs);
println!("param names = {:?}", res.param_names);  // includes ar.L1[0], ar.L1[1]
println!("smoothed probabilities shape = {:?}",
    res.smoothed_marginal_probabilities.dim());
```

To fit from a custom starting point in constrained parameter space, use
`fit_from(Some(start))` instead of `fit()`.

## Module reference

| Kind | Name | Description |
| --- | --- | --- |
| Model | `MarkovRegression` | First-order `k`-regime switching regression; switching intercept/exog and optional switching variance. Constructors `new` (switching intercept) and `with_exog` (explicit design matrix). |
| Model | `MarkovAutoregression` | `k`-regime switching autoregression of a given `order`; switching mean and optional switching AR coefficients and variance. Constructor `new`. |
| Result | `MarkovResults` | Fitted output: `params`, `param_names`, `llf`, `nobs`, `k_params`, `aic`, `bic`, `converged`, `transition`, `initial_probabilities`, `filtered_marginal_probabilities`, `smoothed_marginal_probabilities`, `expected_durations`. |

Both models expose `fit()` and `fit_from(Some(start))`, returning a
`MarkovResults`. The Hamilton filter and Kim smoother that power them are
internal; you interact with them only through the probability arrays on the
result.

Full API: see the generated rustdoc for `solow-regime`.

### Honest scope

This crate covers the two most-used regime-switching estimators (switching
regression and switching autoregression) with Gaussian conditional densities and
a fixed transition matrix. It does not (yet) include the broader catalogue the
reference offers — for example time-varying transition probabilities,
dynamic-factor or state-space Markov-switching models, or parameter standard
errors and confidence intervals from the estimated information matrix.

## References

- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2),
  357–384.
- Kim, C.-J. (1994). "Dynamic Linear Models with Markov-Switching." *Journal of
  Econometrics*, 60(1–2), 1–22.
- Kim, C.-J., and Nelson, C. R. (1999). *State-Space Models with Regime
  Switching: Classical and Gibbs-Sampling Approaches with Applications.* MIT
  Press.
- Hamilton, J. D. (1994). *Time Series Analysis*, Chapter 22. Princeton
  University Press.
