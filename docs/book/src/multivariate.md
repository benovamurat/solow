# Multivariate analysis

The [`solow-multivariate`] crate provides the classical multivariate methods,
matching the conventions of the reference's `multivariate` module: principal
component analysis, principal-axis factor analysis (with rotation), multivariate
analysis of variance, and canonical correlation.

Data are `n × p` matrices (rows = observations, columns = variables).

## Principal component analysis (PCA)

`Pca::new(data).fit()` returns the eigenvalues, loadings, factor scores, and
projections:

```rust
use ndarray::array;
use solow_multivariate::Pca;

let data = array![
    [1.0, 2.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
    [4.0, 1.0, 0.0],
    [2.5, 1.5, 1.0],
];

let pca = Pca::new(data).fit().unwrap();
println!("eigenvalues       = {:?}", pca.eigenvals);   // descending
println!("first PC loadings = {:?}", pca.loadings.column(0));
```

`PcaResults` exposes `eigenvals` (variance explained per component, in
descending order), `loadings` (the eigenvectors), `scores`, and `factors`.

## Factor analysis

`Factor::from_data(&data, n_factor, smc)` builds a principal-axis factor model
with `n_factor` common factors; `smc = true` seeds the communalities with
squared multiple correlations. `.fit(maxiter, tol)` iterates to convergence:

```rust
use ndarray::array;
use solow_multivariate::Factor;

let data = array![
    [1.0, 2.0, 0.5],
    [2.0, 1.0, 1.5],
    [3.0, 0.0, 2.5],
    [4.0, 1.0, 0.0],
    [2.5, 1.5, 1.0],
];

let fa = Factor::from_data(&data, 1, true).fit(50, 1e-8).unwrap();
println!("factor loadings = {:?}", fa.loadings);
```

You can also start from a correlation matrix with `Factor::from_corr`. To
rotate an extracted loading matrix for interpretability, use [`rotate_factors`]
with a [`RotationMethod`] (varimax, quartimax, oblimin, and others).

## MANOVA

[`Manova`] computes the four classical multivariate test statistics — Wilks'
lambda, Pillai's trace, the Hotelling–Lawley trace, and Roy's greatest root —
for a multivariate linear hypothesis.

## Canonical correlation

[`CanCorr`] finds the linear combinations of two variable sets with maximal
correlation and reports the canonical correlations together with the associated
significance tests.

[`solow-multivariate`]: https://github.com/solow-rs/solow
[`rotate_factors`]: ./multivariate.md
[`RotationMethod`]: ./multivariate.md
[`Manova`]: ./multivariate.md
[`CanCorr`]: ./multivariate.md
