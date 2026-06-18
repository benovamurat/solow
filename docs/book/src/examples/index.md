# Examples gallery

A tour of the major models in Solow, each illustrated end to end: a small,
**fully deterministic** example dataset, the call that fits the model, the
printed results summary, and a rendered plot. Every page corresponds to a
runnable program in the `solow-gallery` crate, so you can reproduce any figure
and summary on your own machine.

The gallery is modeled on the canonical examples index of the reference Python
statistics library: one self-contained vignette per modeling family.

## Running the examples

The gallery lives in its own crate (`crates/solow-gallery`) that is *not* part
of the root workspace — it depends on the Solow crates by path. Run any example
by name:

```text
cargo run --manifest-path crates/solow-gallery/Cargo.toml --bin ols
```

Each program prints its summary to standard output and writes its plot SVG into
`docs/book/src/examples/img/`, which is exactly where the images embedded on
these pages come from.

## The examples

| Example | Model | Plot |
| --- | --- | --- |
| [Ordinary least squares](./ols.md) | `LinearModel::ols` | scatter + fitted line |
| [Weighted & generalized least squares](./wls_gls.md) | `LinearModel::wls` / `gls` | OLS vs WLS lines under heteroskedasticity |
| [Generalized linear models](./glm_poisson_logit.md) | `Glm` (Poisson, Logit) | fitted mean / probability curves |
| [Generalized additive models](./gam_smooth.md) | `GlmGam` (penalized B-spline) | data with the fitted smooth curve |
| [Generalized estimating equations](./gee_marginal.md) | `Gee` (exchangeable) | clustered data with the marginal fit |
| [Linear mixed effects](./mixed_ranef.md) | `MixedLm` (random intercepts) | grouped data with partial-pooling levels |
| [Robust regression](./robust.md) | `Rlm` (Huber's T) | OLS vs robust fit with outliers |
| [Beta regression](./beta_reg.md) | `BetaModel` | (0,1) response with the fitted mean curve |
| [Time series](./time_series.md) | `acf` / `pacf` / `AutoReg` / `seasonal_decompose` | ACF, PACF, AR fit, seasonal component |
| [Forecasting service (case study)](./case_forecasting.md) | `AutoReg` (+ backtest) | demand forecast with a 95% prediction band |
| [Vector autoregression](./var_forecast.md) | `Var` | two series with an iterated forecast |
| [State space & the Kalman filter](./state_space.md) | `UnobservedComponents` | observed series vs filtered state |
| [Markov switching regimes](./regime.md) | `MarkovRegression` | smoothed regime probability over time |
| [Survival analysis](./survival.md) | `SurvfuncRight` (Kaplan-Meier) | survival step function with bands |
| [Copula dependence](./copula_density.md) | `ClaytonCopula` | copula density over the unit square |
| [Empirical likelihood](./emplike_ratio.md) | `DescStat` | −2 log-ELR profile with the χ² threshold |
| [Principal component analysis](./pca.md) | `Pca` | scree plot + PC1/PC2 scores |
| [Q-Q plot diagnostics](./graphics_qq.md) | `ProbPlot` | normal Q-Q plot with reference line |
| [Multiple imputation](./mice_convergence.md) | `conditional_mean_impute` / `combine` | imputation spread and the pooled estimate |
| [Bayesian posterior](./bayes_posterior.md) | `BayesMixedGlm` (variational Bayes) | posterior means with credible bands |
| [Optimizer convergence](./optim_convergence.md) | `minimize_bfgs` / `newton_stationary` | gradient norm vs iteration (log scale) |
| [The formula interface](./formula.md) | `solow_fit::ols` / `poisson` | observed vs fitted |

Every figure on these pages is produced by `solow-viz`, Solow's dependency-light
SVG plotting backend.
