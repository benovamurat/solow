# solow-stats

Statistical diagnostics and hypothesis tests, validated against an
authoritative reference implementation.

The crate provides the standard battery of regression-diagnostic and
normality tests, simple weighted descriptive statistics, two-sample
location tests, autocorrelation tests, and multiple-testing corrections.
Every public quantity is cross-validated to tight tolerances against the
reference (see `tests/reference.rs`).

```
use ndarray::array;
use solow_stats::durbin_watson;

let resid = array![0.1, -0.2, 0.05, 0.15, -0.1];
let dw = durbin_watson(&resid);
assert!((0.0..=4.0).contains(&dw));
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-stats) · License: BSD-3-Clause
