# solow-regression

Linear regression models estimated by least squares, with the full standard
battery of results and inference statistics. Validated against an
authoritative reference.

```
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];
let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design).unwrap().fit().unwrap();
assert!((res.rsquared - 0.997).abs() < 0.01);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-regression) · License: BSD-3-Clause
