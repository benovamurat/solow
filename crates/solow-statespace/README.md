# solow-statespace

Linear-Gaussian state-space models and SARIMAX estimation.

The crate provides two layers:

* [`kalman`] — a time-invariant [`StateSpace`] with an
  exact Kalman [`filter`](kalman::StateSpace::filter) (returning the Gaussian
  log-likelihood and the filtered/predicted state sequences) and a
  fixed-interval [`smoother`](kalman::StateSpace::smooth).
* [`sarimax`] — a [`Sarimax`] seasonal ARIMA estimator
  built on the Kalman filter and fit by maximum likelihood. The AR/MA and
  seasonal polynomials are mapped into the Harvey companion-form state space,
  stationarity/invertibility are enforced by the Monahan reparametrization,
  and the model is optimized with BFGS over `-loglike`.

```
use ndarray::Array1;
use solow_statespace::{Sarimax, SarimaxOrder};

// A short AR(1)-like series.
let y = Array1::from_vec(vec![
    0.2, 0.5, 0.1, -0.3, 0.4, 0.8, 0.3, -0.1, 0.0, 0.6, 0.9, 0.2, -0.4, 0.1,
]);
let model = Sarimax::new(y, SarimaxOrder::new(1, 0, 0)).unwrap();
let res = model.fit().unwrap();
assert_eq!(res.params.len(), 2); // [ar.L1, sigma2]
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-statespace) · License: BSD-3-Clause
