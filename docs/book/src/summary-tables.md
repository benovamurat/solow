# Summary tables

The [`solow-summary`] crate renders the familiar two-block results table — a
header of key statistics on top, a coefficient table below. It is deliberately
**model-agnostic**: a [`RegressionSummary`] is built from plain values
(parameter names, estimates, standard errors, test statistics, p-values, and
confidence intervals), so any estimator's results can be rendered without the
crate depending on that estimator. Only the displayed numbers are cross-checked
against a reference; the visual layout is Solow's own.

## Building a summary

`RegressionSummary::new` takes the coefficient columns plus a [`HeaderStats`]
struct for the header block:

```rust
use solow_summary::{HeaderStats, RegressionSummary};

let names = ["const", "x1"];
let params = [1.5, -2.0];
let bse = [0.07, 0.08];
let tvalues = [21.4, -25.0];
let pvalues = [1e-20, 1e-22];
let conf_int = [(1.35, 1.65), (-2.16, -1.84)];

let header = HeaderStats {
    model: Some("OLS".into()),
    nobs: Some(40.0),
    rsquared: Some(0.95),
    aic: Some(45.9),
    bic: Some(51.0),
    ..HeaderStats::new()
};

let summary =
    RegressionSummary::new(&names, &params, &bse, &tvalues, &pvalues, &conf_int, header);

let text = summary.to_string();
assert!(text.contains("R-squared:"));
assert!(text.contains("const"));
println!("{text}");
```

`HeaderStats::new()` gives an all-empty header; fill in only the fields you have
and let `..HeaderStats::new()` default the rest.

## Wiring a fitted model in

The coefficient inputs come straight off a fitted result. The confidence
interval is an `(k, 2)` matrix, which you reshape into a `Vec<(f64, f64)>`:

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;
use solow_summary::{HeaderStats, RegressionSummary};

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
let y: Array1<f64> = array![2.1, 3.9, 6.2, 7.8, 10.1, 12.0];
let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design).unwrap().fit().unwrap();

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

## Z statistics instead of t

For models whose inference uses the normal distribution (GLM, discrete choice,
SARIMAX), switch the test-statistic column label from `t` to `z` with
`with_stat_kind`:

```rust
use solow_summary::{HeaderStats, RegressionSummary, StatKind};

let names = ["const", "x1"];
let params = [0.4, 1.1];
let bse = [0.2, 0.3];
let zvalues = [2.0, 3.67];
let pvalues = [0.045, 0.0002];
let conf_int = [(0.01, 0.79), (0.51, 1.69)];

let summary = RegressionSummary::new(
    &names, &params, &bse, &zvalues, &pvalues, &conf_int, HeaderStats::new(),
)
.with_stat_kind(StatKind::Z)
.with_title("Logit Regression Results")
.with_alpha(0.05);

println!("{summary}");
```

## Lower-level tables

For ad-hoc tables outside the regression layout, use [`SummaryTable`] directly:
it lays out titled, optionally-headered, row-based tables with per-column
[`Align`] control. The formatting helpers `format_fixed`, `format_g`, and
`format_pvalue` produce the same number strings the summary uses.

[`solow-summary`]: https://github.com/solow-rs/solow
[`RegressionSummary`]: ./summary-tables.md
[`HeaderStats`]: ./summary-tables.md
[`SummaryTable`]: ./summary-tables.md
[`Align`]: ./summary-tables.md
