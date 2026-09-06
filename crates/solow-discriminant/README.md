# solow-discriminant

**Discriminant analysis for Rust.** Linear discriminant analysis (Fisher 1936) with pooled covariance, and quadratic discriminant analysis with per-class covariance.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-discriminant = "0.7"
```

## Quick example

```rust
use solow_discriminant::LinearDiscriminantAnalysis;

let lda = LinearDiscriminantAnalysis::new().fit(&x, &y)?;
let pred = lda.predict(&x_test)?;
let proba = lda.predict_proba(&x_test)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `LinearDiscriminantAnalysis` | Fisher LDA with a shared pooled covariance solved by Cholesky with diagonal regularization. Assumes equal class covariance. |
| `QuadraticDiscriminantAnalysis` | QDA with per-class covariance. Handles heteroscedastic classes at the cost of more parameters. |

Both estimators expose `predict`, `predict_proba`, and `decision_function`.

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
