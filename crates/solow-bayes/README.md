# solow-bayes

Bayesian generalized linear mixed models fit by **mean-field variational
Bayes** (deterministic; no MCMC). Two families are provided with a
random-effects design (`exog_vc`): [`BinomialBayesMixedGLM`] (logit link) and
[`PoissonBayesMixedGLM`] (log link). Both are fit by [`BayesMixedGlm::fit_vb`],
which maximizes the evidence lower bound (ELBO) of a factored Gaussian
variational posterior over the fixed effects, the variance-component
parameters (log standard deviations of the random effects), and the random
effect realizations.

The parameterization, ELBO and its gradient mirror the canonical
statistical-computing reference's `genmod.bayes_mixed_glm` module exactly, so
the posterior means ([`BayesMixedGlmResults::fe_mean`],
[`BayesMixedGlmResults::vcp_mean`], [`BayesMixedGlmResults::vc_mean`]) and the
ELBO ([`BayesMixedGlmResults::elbo`] / [`BayesMixedGlmResults::llf`]) agree to
optimizer tolerance.

## Model

For observation `i` the linear predictor is
`eta_i = x_i · fe + z_i · vc`, where `x` is `exog` (fixed effects) and `z` is
`exog_vc` (random effects). Each random effect realization `vc_j` is
Gaussian with mean zero and standard deviation `exp(vcp[ident[j]])`; the
`vcp` parameters (log standard deviations) have a `N(0, vcp_p²)` prior and the
fixed effects a `N(0, fe_p²)` prior.

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-bayes) · License: BSD-3-Clause
