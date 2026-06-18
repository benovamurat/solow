# solow-nonparametric

Nonparametric smoothers for the Solow statistical-computing stack.

This crate provides three families of nonparametric tools, each verified
against the canonical Python reference implementation to tight tolerances:

* [`lowess`] — LOWESS (locally-weighted scatterplot smoothing), a robust
  locally-weighted linear regression following Cleveland (1979).
* Rule-of-thumb bandwidth selectors: [`bw_silverman`], [`bw_scott`] and
  [`bw_normal_reference`].
* [`KdeUnivariate`] — a univariate Gaussian kernel density estimator.

All routines operate on [`ndarray`] `Array1<f64>` data.

## Example

```
use ndarray::array;
use solow_nonparametric::lowess;

let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![1.0, 1.9, 3.2, 3.8, 5.1];
let fit = lowess(&y, &x, Default::default()).unwrap();
assert_eq!(fit.x.len(), 5);
assert_eq!(fit.fitted.len(), 5);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-nonparametric) · License: BSD-3-Clause
