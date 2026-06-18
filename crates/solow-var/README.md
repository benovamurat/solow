# solow-var

Vector autoregression (VAR) for the Solow statistical stack, validated
against an authoritative reference.

The crate fits a VAR(`p`) model by equation-by-equation ordinary least
squares on the stacked lag design with an optional constant term, exposing
the per-equation coefficients, both the maximum-likelihood (`/T`) and
degrees-of-freedom-adjusted (`/(T-Kp-1)`) residual covariances, the Gaussian
log-likelihood, the AIC/BIC/HQIC/FPE information criteria, and per-coefficient
standard errors, t-statistics and p-values.

```
use ndarray::array;
use solow_var::Var;

let y = array![
    [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
    [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
    [0.9, 0.8], [1.3, 0.5],
];
let res = Var::new(y).unwrap().fit(1).unwrap();
assert_eq!(res.neqs, 2);
assert_eq!(res.coefs.len(), 1);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-var) · License: BSD-3-Clause
