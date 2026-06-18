# solow-othermod

Models that fall outside the core regression and GLM families. Currently this
provides [`BetaModel`], a maximum-likelihood **beta regression** with separate
linear predictors (and links) for the conditional mean and the precision.

```
use ndarray::{array, Array2};
use solow_othermod::BetaModel;

let y = array![0.3, 0.55, 0.62, 0.48, 0.71, 0.4];
let x = Array2::from_shape_vec((6, 2), vec![
    1.0, -1.0, 1.0, -0.3, 1.0, 0.2, 1.0, 0.0, 1.0, 0.8, 1.0, -0.5,
]).unwrap();
let z = Array2::from_shape_vec((6, 1), vec![1.0; 6]).unwrap();
let res = BetaModel::new(y, x, z).unwrap().fit().unwrap();
assert!(res.converged);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-othermod) · License: BSD-3-Clause
