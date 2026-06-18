# solow-discrete

Discrete-choice and count regression models estimated by maximum likelihood:
[`Logit`], [`Probit`], and [`Poisson`]. Each model is fit with a full Newton
step using analytic log-likelihood, score, and Hessian, converging to the true
optimum so that results agree with the canonical reference to machine precision.

```
use ndarray::{array, Array2};
use solow_discrete::Logit;

let mut x = Array2::<f64>::ones((5, 2));
x.column_mut(1)
    .assign(&array![0.1, -0.4, 1.2, 0.7, -1.1]);
let y = array![0.0, 0.0, 1.0, 1.0, 0.0];
let res = Logit::new(y, x).unwrap().fit().unwrap();
assert!(res.converged);
assert_eq!(res.params.len(), 2);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-discrete) · License: BSD-3-Clause
