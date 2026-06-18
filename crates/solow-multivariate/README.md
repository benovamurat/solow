# solow-multivariate

Multivariate statistical analysis for the Solow stack. This crate provides
[`Pca`] (principal component analysis), [`Factor`] (principal-axis factor
analysis), [`Manova`] (multivariate analysis of variance with the four
classical test statistics) and [`CanCorr`] (canonical correlation analysis),
all matching the conventions of the reference implementation
(`multivariate.pca`, `.factor`, `.manova`, `.cancorr`).

```
use ndarray::array;
use solow_multivariate::Pca;

let data = array![
    [1.0, 2.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
    [4.0, 1.0, 0.0],
];
let pca = Pca::new(data).fit().unwrap();
assert_eq!(pca.eigenvals.len(), 3);
// Eigenvalues are sorted in descending order.
assert!(pca.eigenvals[0] >= pca.eigenvals[1]);
```

---

Part of **[Solow](https://github.com/benovamurat/solow)** — a complete statistical-modeling, econometrics & data-visualization toolkit for Rust. · [Docs](https://docs.rs/solow-multivariate) · License: BSD-3-Clause
