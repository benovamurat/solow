# solow-cross-decomposition

**Partial least squares and canonical correlation for Rust.** PLS regression, PLS canonical, PLS SVD, and canonical correlation analysis via the NIPALS solver.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-cross-decomposition = "0.7"
```

## Quick example

```rust
use solow_cross_decomposition::PLSRegression;

let pls = PLSRegression::new(2).fit(&x, &y)?;
let y_pred = pls.predict(&x_test)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `PLSRegression` | Predictive PLS: finds latent components that maximise covariance between X and Y. Used for regression with correlated predictors and multi-output targets. |
| `PLSCanonical` | Symmetric PLS with mode-A deflation, giving equally-scaled X and Y projections. |
| `PLSSVD` | Direct SVD of the cross-covariance matrix. |
| `CCA` | Canonical correlation analysis via NIPALS with mode-B deflation. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
