# Quickstart

This chapter fits an ordinary least squares (OLS) regression end to end: build
a design matrix, estimate the model, read off the results, and render a summary
table. It is the smallest complete example that touches the pieces you will use
everywhere in Solow.

## Set up the crate

```toml
[dependencies]
solow-core = "0.1"
solow-regression = "0.1"
solow-summary = "0.1"
ndarray = "0.16"
```

## Fit an OLS model

Solow does **not** add an intercept for you (matching the convention of the
reference stack). You add the constant column explicitly with `add_constant`.

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

fn main() {
    // Predictor (one column) and response.
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
    let y: Array1<f64> = array![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];

    // Prepend a constant column → design matrix with [const, x].
    let design = add_constant(&x, true, HasConstant::Add).unwrap();

    // Estimate by ordinary least squares.
    let res = LinearModel::ols(y, design).unwrap().fit().unwrap();

    // Inspect the results.
    println!("params      = {:?}", res.params);      // [intercept, slope]
    println!("std errors  = {:?}", res.bse);
    println!("t-values    = {:?}", res.tvalues);
    println!("p-values    = {:?}", res.pvalues);
    println!("R-squared   = {:.4}", res.rsquared);
    println!("adj R-sq    = {:.4}", res.rsquared_adj);
    println!("F statistic = {:.2} (p = {:.3e})", res.fvalue, res.f_pvalue);
    println!("AIC / BIC   = {:.2} / {:.2}", res.aic, res.bic);
}
```

The fitted [`LinearResults`] struct carries the full battery of quantities you
would expect: `params`, `bse`, `tvalues`, `pvalues`, `conf_int(alpha)`,
`fittedvalues`, `resid`, `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`,
`aic`, `bic`, `llf`, `scale`, `df_model`, `df_resid`, and the coefficient
covariance `cov_params`.

## Confidence intervals

```rust
# use ndarray::{array, Array1};
# use solow_core::tools::{add_constant, HasConstant};
# use solow_regression::LinearModel;
# let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
# let y: Array1<f64> = array![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
# let design = add_constant(&x, true, HasConstant::Add).unwrap();
# let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
// 95% confidence interval: a (k, 2) matrix of [lower, upper] per coefficient.
let ci = res.conf_int(0.05);
for (i, row) in ci.rows().into_iter().enumerate() {
    println!("beta[{i}] in [{:.3}, {:.3}]", row[0], row[1]);
}
```

## Predict on new data

```rust
# use ndarray::{array, Array1};
# use solow_core::tools::{add_constant, HasConstant};
# use solow_regression::LinearModel;
# let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
# let y: Array1<f64> = array![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
# let design = add_constant(&x, true, HasConstant::Add).unwrap();
let model = LinearModel::ols(y, design).unwrap();
let res = model.fit().unwrap();

let new_x = array![[1.0, 7.0], [1.0, 8.0]]; // include the constant column
let yhat = model.predict(&res.params, &new_x);
println!("predictions = {:?}", yhat);
```

## Render a summary table

The `solow-summary` crate turns plain result values into the familiar
two-block summary. It is intentionally model-agnostic, so you feed it values
rather than a model object:

```rust
# use ndarray::{array, Array1};
# use solow_core::tools::{add_constant, HasConstant};
# use solow_regression::LinearModel;
use solow_summary::{HeaderStats, RegressionSummary};

# let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
# let y: Array1<f64> = array![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
# let design = add_constant(&x, true, HasConstant::Add).unwrap();
# let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
let names = ["const", "x1"];
let ci = res.conf_int(0.05);
let conf: Vec<(f64, f64)> =
    (0..names.len()).map(|i| (ci[[i, 0]], ci[[i, 1]])).collect();

let header = HeaderStats {
    model: Some("OLS".into()),
    nobs: Some(res.nobs),
    rsquared: Some(res.rsquared),
    aic: Some(res.aic),
    bic: Some(res.bic),
    ..HeaderStats::new()
};

let summary = RegressionSummary::new(
    &names,
    res.params.as_slice().unwrap(),
    res.bse.as_slice().unwrap(),
    res.tvalues.as_slice().unwrap(),
    res.pvalues.as_slice().unwrap(),
    &conf,
    header,
);

println!("{summary}");
```

## Where to next

- [Linear regression](./regression.md) — WLS, GLS, and robust covariances.
- [The formula interface](./formula.md) — build the design matrix from a
  formula string instead of by hand.
- [Generalized linear models](./glm.md) and [Discrete choice](./discrete.md)
  for non-Gaussian responses.

[`LinearResults`]: ./regression.md
