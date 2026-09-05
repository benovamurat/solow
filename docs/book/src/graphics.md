# Statistical graphics

The `solow-graphics` crate provides the diagnostic and exploratory plots that
accompany a fitted model: normal probability (Q-Q) plots, autocorrelation and
partial-autocorrelation plots, residuals-versus-fitted scatters, and the
regression-influence family (leverage, studentized residuals, Cook's distance,
DFFITS). Every routine computes the underlying arrays *and* renders them to an
SVG `Figure` through the `solow-viz` backend, so the numerics can be inspected
and tested independently of the (intentionally not pixel-exact) drawing. The
main entry points are the `ProbPlot` builder, the `Influence` diagnostics
struct, and the free `plot_*` / `acf` / `pacf_yw` functions.

## Background

**Normal probability plot.** For a sample of size \\( n \\), sort the data to
obtain the order statistics \\( x_{(1)} \le \cdots \le x_{(n)} \\). Each is
paired with a *plotting position*

\\[
p_i = \frac{i - a}{n + 1 - 2a}, \qquad i = 1, \dots, n,
\\]

where \\( a \\) selects the convention (\\( a = 0 \\) Weibull — the default,
\\( a = 3/8 \\) Blom, \\( a = 1/2 \\) Hazen). The *theoretical quantiles* are
the standard-normal inverse CDF \\( \Phi^{-1}(p_i) \\), and the Q-Q plot scatters
\\( \big(\Phi^{-1}(p_i),\, x_{(i)}\big) \\). A reference line summarizes the
relationship; three are offered — an OLS regression of sample on theoretical
quantiles, a standardized line with slope equal to the population standard
deviation and intercept the sample mean, and a quartile line through the first
and third quartiles.

**Autocorrelation.** With \\( \bar{x} \\) the sample mean, the biased
autocovariance and autocorrelation at lag \\( k \\) are

\\[
\hat{\gamma}_k = \frac{1}{n} \sum_{t=k}^{n-1} (x_t - \bar{x})(x_{t-k} - \bar{x}),
\qquad
\hat{\rho}_k = \frac{\hat{\gamma}_k}{\hat{\gamma}_0}.
\\]

The partial autocorrelation at lag \\( k \\) is the last coefficient
\\( \phi_{kk} \\) of the order-\\( k \\) autoregression, obtained by solving the
Yule-Walker equations \\( R\,\phi = r \\) on autocovariances estimated with the
adjusted divisor \\( n - k \\). Under a white-noise null the estimates are
approximately \\( \mathcal{N}(0, 1/n) \\), giving the symmetric confidence band

\\[
\pm\, \frac{z_{1 - \alpha/2}}{\sqrt{n}}.
\\]

**Regression influence.** Let \\( H = X (X'X)^{-1} X' \\) be the hat matrix of a
fitted OLS model, \\( h_i = H_{ii} \\) the leverage, \\( e_i \\) the residual,
\\( k \\) the number of parameters, and \\( s^2 = \mathrm{SSR}/(n-k) \\) the
residual variance. The internally and externally (leave-one-out) studentized
residuals are

\\[
r_i = \frac{e_i}{\sqrt{s^2 (1 - h_i)}},
\qquad
t_i = \frac{e_i}{\sqrt{s_{(i)}^2 (1 - h_i)}},
\qquad
s_{(i)}^2 = \frac{(n-k)\,s^2 - e_i^2/(1 - h_i)}{n - k - 1},
\\]

and the two standard global-influence measures are Cook's distance and DFFITS,

\\[
D_i = \frac{r_i^2}{k}\,\frac{h_i}{1 - h_i},
\qquad
\mathrm{DFFITS}_i = t_i \sqrt{\frac{h_i}{1 - h_i}}.
\\]

The influence plot scatters \\( t_i \\) against \\( h_i \\) with marker area
scaled by \\( D_i \\).

## Example

Build a small design and response, fit an OLS model with `solow-regression`,
then compute influence diagnostics and render a residuals-versus-fitted plot.
The design's first column is the explicit intercept.

```rust
use ndarray::{array, Array1, Array2};
use solow_regression::LinearModel;
use solow_graphics::{Influence, influence_plot, plot_resid_fitted, ProbPlot, acf, pacf_yw, plot_acf};

// Design matrix (intercept + one regressor) and response.
let x: Array2<f64> = array![
    [1.0, 0.1],
    [1.0, 0.9],
    [1.0, 2.1],
    [1.0, 3.2],
    [1.0, 3.9],
    [1.0, 5.0],
];
let y: Array1<f64> = array![1.0, 2.1, 2.9, 4.2, 4.8, 6.1];

let res = LinearModel::ols(y, x.clone()).unwrap().fit().unwrap();

// Per-observation influence diagnostics.
let inf = Influence::new(&res, &x);
println!("leverage (hat_diag)   = {:?}", inf.hat_diag);
println!("studentized (ext)     = {:?}", inf.resid_studentized_external);
println!("Cook's distance       = {:?}", inf.cooks_distance);
println!("DFFITS                = {:?}", inf.dffits);

// Render diagnostics to SVG figures.
let (inf_fig, _inf) = influence_plot(&res, &x);
let resid_fitted = plot_resid_fitted(
    res.fittedvalues.as_slice().unwrap(),
    res.resid.as_slice().unwrap(),
);
assert!(inf_fig.to_svg().starts_with("<svg"));
assert!(resid_fitted.to_svg().starts_with("<svg"));
```

The `hat_diag` entries lie in \\( (0, 1) \\) and sum to the model rank (here
\\( 2 \\)); observations with large leverage *and* a large externally
studentized residual produce the biggest Cook's distance and the largest bubble
in the influence plot. The `influence_plot` and `plot_resid_fitted` calls each
return a `Figure` whose `to_svg()` begins with `<svg` and closes with `</svg>`.

A normal probability plot and the ACF/PACF of a sequence follow the same shape —
construct, inspect arrays, then render:

```rust
use solow_graphics::{ProbPlot, acf, pacf_yw, plot_acf};

let sample = [-1.2, 0.3, 0.1, 1.4, -0.7, 2.1, -0.2, 0.9];
let pp = ProbPlot::new(&sample);            // a = 0.0 (Weibull) by default
let line = pp.qqline_regression();           // QqLine { slope, intercept }
println!("theoretical quantiles = {:?}", pp.theoretical_quantiles());
println!("sample quantiles      = {:?}", pp.sample_quantiles());
println!("Q-Q reference line: y = {:.3} x + {:.3}", line.slope, line.intercept);

let series = [1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0];
let r = acf(&series, 4);                      // index 0 is exactly 1.0
let p = pacf_yw(&series, 4);                  // Yule-Walker PACF, index 0 is 1.0
println!("acf  = {:?}", r);
println!("pacf = {:?}", p);

// The plotting wrapper returns (Figure, AcfResult) with a white-noise band.
let (fig, acf_res) = plot_acf(&series, 4, 0.05);
println!("conf_band half-width = {:.4}", acf_res.conf_band);
assert!(fig.to_svg().starts_with("<svg"));
```

*(Illustrative description, not literal output.)* `acf` always returns
`1.0` at lag 0, every value stays in \\( [-1, 1] \\), and `acf_res.conf_band`
equals \\( z_{0.975}/\sqrt{n} \approx 1.95996/\sqrt{10} \\). For
`conf_band(100, 0.05)` the half-width is exactly `0.1959963984540054`.

## Module reference

**Models / builders**

| Name | Description |
| --- | --- |
| `ProbPlot` | Normal probability (Q-Q) plot builder; holds sorted data and the plotting-position parameter `a`. |
| `Influence` | Per-observation OLS influence diagnostics built from a fitted `LinearResults` and its design. |

**Results**

| Name | Description |
| --- | --- |
| `QqLine` | A Q-Q reference line with public fields `slope` and `intercept`. |
| `AcfResult` | Correlation `values` (lag `0..=nlags`) plus the `conf_band` half-width. |
| `MosaicData` | Normalized `row_widths` and `cell_heights` behind a mosaic plot. |

**Functions**

| Name | Description |
| --- | --- |
| `qqplot` | Convenience wrapper: build a `ProbPlot` and render its Q-Q `Figure`, returning `(Figure, ProbPlot)`. |
| `acf` | Biased autocorrelation for lags `0..=nlags`. |
| `pacf_yw` | Yule-Walker partial autocorrelation (adjusted autocovariances). |
| `plot_acf` | Compute the ACF with a white-noise band and render a stem plot; returns `(Figure, AcfResult)`. |
| `plot_pacf` | As `plot_acf` for the Yule-Walker PACF. |
| `conf_band` | Symmetric white-noise band half-width \\( z_{1-\alpha/2}/\sqrt{n} \\). |
| `plot_resid_fitted` | Residuals-versus-fitted diagnostic scatter with a zero reference line. |
| `influence_plot` | Externally studentized residual vs. leverage, bubble-sized by Cook's distance; returns `(Figure, Influence)`. |
| `plot_fit` | Observed and fitted response plotted against one regressor column. |
| `plot_regress_exog` | Residual-versus-regressor diagnostic panel for one regressor. |
| `mosaic` | Mosaic plot of a 2-D contingency table; returns `(Figure, MosaicData)`. |

`ProbPlot` also exposes `with_a`, `nobs`, `theoretical_percentiles`,
`theoretical_quantiles`, `sample_quantiles`, `qqline_regression`,
`qqline_standardized`, `qqline_quartile`, and `qqplot`. This crate is a focused
subset of the reference's graphics module: added-variable plots are present only
as the single residual-versus-regressor panel of `plot_regress_exog` (not the
full 2×2 grid), and there is no general interaction or factor-plot machinery.

Full API: see the generated rustdoc for `solow-graphics`.

## References

- Chambers, J. M., Cleveland, W. S., Kleiner, B., and Tukey, P. A. *Graphical
  Methods for Data Analysis.* Wadsworth, 1983.
- Cook, R. D., and Weisberg, S. *Residuals and Influence in Regression.*
  Chapman and Hall, 1982.
- Belsley, D. A., Kuh, E., and Welsch, R. E. *Regression Diagnostics:
  Identifying Influential Data and Sources of Collinearity.* Wiley, 1980.
- Box, G. E. P., Jenkins, G. M., and Reinsel, G. C. *Time Series Analysis:
  Forecasting and Control,* 4th ed. Wiley, 2008.
