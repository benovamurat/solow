# Bayesian methods

The `solow-bayes` crate fits **Bayesian generalized linear mixed models**
(GLMMs) with a random-effects design. The single estimator type,
`BayesMixedGlm`, supports two families — binomial (logit link) and Poisson (log
link) — and offers two deterministic posterior approximations: mean-field
**variational Bayes** via `BayesMixedGlm::fit_vb` (a factored-Gaussian posterior
maximizing the evidence lower bound), and **maximum a posteriori** estimation
via `BayesMixedGlm::fit_map` (the posterior mode). Both fits are fully
deterministic — there is no Monte Carlo sampling.

> **Scope, honestly.** This crate is narrower than the reference's full Bayesian
> surface: it covers the mixed-GLM variational/MAP estimators only. There is no
> MCMC sampler (no Gibbs or Metropolis), no posterior-draw machinery, and no
> credible-interval helper. Uncertainty is summarized by the variational
> posterior standard deviations returned alongside the means.

## Background

For observation \\(i\\) with fixed-effects design row \\(x_i\\) and random-effects
design row \\(z_i\\), the linear predictor is

\\[
\eta_i = x_i^\top \beta + z_i^\top u,
\\]

where \\(\beta\\) are the fixed-effects coefficients (`exog`, length `k_fep`) and
\\(u\\) are the random-effect realizations (`exog_vc`, length `k_vc`). The mean is
\\(\mu_i = g^{-1}(\eta_i)\\): the logistic function for the binomial family and
\\(\exp(\cdot)\\) for the Poisson family. The likelihood is the corresponding GLM
density with its normalizing constant,

\\[
\log p(y_i \mid \eta_i) =
\begin{cases}
y_i \eta_i - \log\!\big(1 + e^{\eta_i}\big), & \text{binomial},\\[4pt]
y_i \eta_i - e^{\eta_i} - \log\Gamma(y_i + 1), & \text{Poisson}.
\end{cases}
\\]

Each random effect \\(u_j\\) is a priori Gaussian with mean zero and standard
deviation \\(\exp(\theta_{\,\mathrm{ident}[j]})\\), where the **variance-component
parameters** \\(\theta\\) (`vcp`, the log standard deviations) are shared across
columns that map to the same `ident` value. The priors are

\\[
u_j \sim \mathcal{N}\!\big(0,\ e^{2\theta_{\mathrm{ident}[j]}}\big), \qquad
\theta_k \sim \mathcal{N}(0,\ \texttt{vcp\_p}^2), \qquad
\beta_k \sim \mathcal{N}(0,\ \texttt{fe\_p}^2).
\\]

**Variational Bayes.** `fit_vb` approximates the posterior over
\\((\beta, \theta, u)\\) by a fully factored Gaussian \\(q\\) with independent
coordinate means \\(m\\) and standard deviations \\(s\\), and maximizes the
evidence lower bound

\\[
\mathrm{ELBO}(q) = \mathbb{E}_q\big[\log p(y, \beta, \theta, u)\big] + \mathcal{H}(q).
\\]

The intractable expectation of the family term over the Gaussian linear
predictor is evaluated by ten-point Gauss–Legendre quadrature. The analytic
ELBO and its gradient are exposed as `vb_elbo` and `vb_elbo_grad`; the optimizer
works on the reparameterized vector \\([m;\ \log s]\\) and minimizes
\\(-\mathrm{ELBO}\\) by BFGS. Because the maximized ELBO is the model's
variational log-likelihood, `BayesMixedGlmResults::llf` returns it as an alias
for the `elbo` field.

**MAP.** `fit_map` instead locates the posterior **mode** — the
\\((\beta, \theta, u)\\) maximizing the joint log-density
\\(\log p(y, \beta, \theta, u)\\), exposed as `log_posterior` with gradient
`log_posterior_grad`. This is a point estimate in the same `dim()`-dimensional
space as a single variational-mean vector, with no posterior spread attached.

## Example

A binomial random-intercept model: six groups of four observations each, one
fixed-effects slope plus an intercept column, and a per-group random intercept
(`exog_vc` is a group-indicator matrix, with a single shared variance component,
so every `ident` entry is `0`).

```rust
use ndarray::{Array1, Array2};
use solow_bayes::{BayesMixedGlm, BinomialBayesMixedGLM, Family};

let n_groups = 6usize;
let per = 4usize;
let n = n_groups * per;

// Fixed effects: intercept column + one centered covariate.
let mut exog = Array2::<f64>::zeros((n, 2));
// Random effects: one indicator column per group (the random intercepts).
let mut exog_vc = Array2::<f64>::zeros((n, n_groups));
for i in 0..n {
    exog[[i, 0]] = 1.0;
    exog[[i, 1]] = -1.5 + 3.0 * i as f64 / (n as f64 - 1.0);
    exog_vc[[i, i / per]] = 1.0;
}

let endog: Array1<f64> = ndarray::array![
    0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1.,
    1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1.
];

// All six random-intercept columns share one variance component.
let ident = vec![0usize; n_groups];

// Prior sds: vcp_p for the log-sd parameter, fe_p for the fixed effects.
let model = BinomialBayesMixedGLM::new(
    endog, exog, exog_vc, ident, /* vcp_p */ 0.5, /* fe_p */ 2.0,
)
.unwrap();

// Variational Bayes: default start (None, None), generous iteration budget.
let res = model.fit_vb(None, None, 100_000, 1e-10).unwrap();

println!("converged   = {}", res.converged);
println!("fe_mean     = {:?}", res.fe_mean);   // posterior means of [intercept, slope]
println!("fe_sd       = {:?}", res.fe_sd);     // posterior sds of the fixed effects
println!("vcp_mean    = {:?}", res.vcp_mean);  // log-sd of the random intercept
println!("vc_mean     = {:?}", res.vc_mean);   // per-group random-intercept means
println!("ELBO (llf)  = {:.4}", res.llf());
```

`BinomialBayesMixedGLM::new` is a thin named constructor that forwards to
`BayesMixedGlm::new(Family::Binomial, ..)`; use `PoissonBayesMixedGLM::new` (or
pass `Family::Poisson` directly) for count responses. The returned
`BayesMixedGlmResults` carries the posterior means `fe_mean`, `vcp_mean`,
`vc_mean` and the matching posterior standard deviations `fe_sd`, `vcp_sd`,
`vc_sd`, plus `elbo`, `converged`, `iters`, and `grad_norm`.

The same model can be fit at its posterior mode instead:

```rust
use ndarray::Array1;
# use solow_bayes::BayesMixedGlm;
# fn demo(model: &BayesMixedGlm) {
// MAP: maximize the joint log-density from the default deterministic start.
let map = model.fit_map(None, 5_000, 1e-10).unwrap();

println!("fe (mode)     = {:?}", map.fe);
println!("vcp (mode)    = {:?}", map.vcp);
println!("vc (mode)     = {:?}", map.vc);
println!("log-posterior = {:.4}", map.logposterior);

// The full stacked [fep, vcp, vc] vector is also available:
let stacked: &Array1<f64> = &map.params;
# let _ = stacked;
# }
```

*Illustrative output:* the binomial fit converges to a near-zero ELBO gradient
and prints a two-element `fe_mean` (intercept and slope), a single `vcp_mean`
log-standard-deviation for the shared random intercept, and six `vc_mean`
per-group offsets; `llf()` reports the maximized ELBO. The exact printed numbers
depend on the optimizer tolerance, so they are not reproduced here — run the
example to see them.

## Module reference

**Models**

| Name | Description |
| --- | --- |
| `BayesMixedGlm` | Bayesian mixed GLM; construct with `new`, fit with `fit_vb` or `fit_map`. |
| `BinomialBayesMixedGLM` | Named constructor for a binomial (logit-link) mixed GLM. |
| `PoissonBayesMixedGLM` | Named constructor for a Poisson (log-link) mixed GLM. |

**Results**

| Name | Description |
| --- | --- |
| `BayesMixedGlmResults` | Variational posterior summary: `fe_mean`/`vcp_mean`/`vc_mean`, `fe_sd`/`vcp_sd`/`vc_sd`, `elbo`, `converged`, `iters`, `grad_norm`. |
| `MapResult` | Posterior-mode summary: `params`, `fe`, `vcp`, `vc`, `logposterior`, `converged`, `iters`, `grad_norm`. |

**Functions (methods on `BayesMixedGlm`)**

| Name | Description |
| --- | --- |
| `new` | Build a model from `family`, `endog`, `exog`, `exog_vc`, `ident`, `vcp_p`, `fe_p`. |
| `fit_vb` | Mean-field variational Bayes fit; returns `BayesMixedGlmResults`. |
| `fit_map` | Maximum a posteriori (posterior-mode) fit; returns `MapResult`. |
| `vb_elbo` | Evidence lower bound at a `(mean, sd)` pair. |
| `vb_elbo_grad` | Gradient of the ELBO with respect to `(mean, sd)`. |
| `log_posterior` | Joint log-density at a stacked `[fep, vcp, vc]` vector. |
| `log_posterior_grad` | Gradient of the joint log-density. |
| `llf` | (On `BayesMixedGlmResults`) the maximized ELBO, as a variational log-likelihood. |

**Enums**

| Name | Description |
| --- | --- |
| `Family` | The GLM family: `Family::Binomial` (logit) or `Family::Poisson` (log). |

Full API: see the generated rustdoc for `solow-bayes`.

## References

- A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and D. B.
  Rubin. *Bayesian Data Analysis*, 3rd ed. Chapman & Hall/CRC, 2013.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational Inference: A
  Review for Statisticians." *Journal of the American Statistical Association*,
  112(518):859–877, 2017.
- C. M. Bishop. *Pattern Recognition and Machine Learning*, ch. 10
  (Approximate Inference). Springer, 2006.
- C. E. McCulloch and S. R. Searle. *Generalized, Linear, and Mixed Models*,
  2nd ed. Wiley, 2008.
