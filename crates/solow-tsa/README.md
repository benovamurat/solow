# solow-tsa

Time-series analysis primitives and models for the Solow statistical stack,
validated against an authoritative reference.

The crate provides:

- Sample second-moment estimators: [`acovf`], [`acf`], [`pacf`], [`ccf`].
- The Ljung-Box [`q_stat`] portmanteau statistic.
- Design helpers [`lagmat`] and [`add_trend`].
- The augmented Dickey-Fuller unit-root test [`adfuller`].
- The autoregressive estimator [`AutoReg`].

```
use ndarray::Array1;
use solow_tsa::{acf, AutoReg, Trend};

// A short AR(1)-like series.
let y = Array1::from_vec(vec![
    0.0, 0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8,
]);
let a = acf(&y, 3, false).unwrap();
assert!((a[0] - 1.0).abs() < 1e-12);

let res = AutoReg::new(y, 1, Trend::C).unwrap().fit().unwrap();
assert_eq!(res.params.len(), 2); // const + 1 lag
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-tsa) · License: BSD-3-Clause
