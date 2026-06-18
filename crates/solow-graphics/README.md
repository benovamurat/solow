# solow-graphics

Statistical-graphics helpers that compute the data behind a plot and render
it through the [`solow_viz`] SVG backend. Every routine returns *both* the
rendered [`Figure`] and the computed arrays, so the numerics can be tested
independently of the (intentionally un-pixel-exact) SVG output.

Provided:

* [`ProbPlot`] / [`qqplot`] — theoretical vs. sample quantiles of a
  probability plot, plus the fitted reference line ([`QqLine`]).
* [`plot_acf`] / [`plot_pacf`] — the (biased) autocorrelation and the
  Yule-Walker partial autocorrelation, with a white-noise confidence band.
* [`plot_resid_fitted`] — a residuals-vs-fitted diagnostic scatter.

```
use solow_graphics::ProbPlot;
let data = [-1.2, 0.3, 0.1, 1.4, -0.7, 2.1, -0.2, 0.9];
let pp = ProbPlot::new(&data);
assert_eq!(pp.sample_quantiles().len(), data.len());
let line = pp.qqline_regression();
let svg = pp.qqplot().to_svg();
assert!(svg.starts_with("<svg"));
let _ = line.slope;
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-graphics) · License: BSD-3-Clause
