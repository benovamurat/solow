# Discriminant analysis

`solow-discriminant` provides linear discriminant analysis (Fisher 1936) with pooled covariance and quadratic discriminant analysis with per-class covariance.

## Estimators

| Estimator | What it does |
|---|---|
| `LinearDiscriminantAnalysis` | Fisher LDA with a shared pooled covariance solved by Cholesky with diagonal regularization. Assumes equal class covariance. |
| `QuadraticDiscriminantAnalysis` | QDA with per-class covariance. Handles heteroscedastic classes at the cost of more parameters. |

Both expose `predict`, `predict_proba`, and `decision_function`.

## Example

```rust
use solow_discriminant::LinearDiscriminantAnalysis;

let lda = LinearDiscriminantAnalysis::new().fit(&x, &y)?;
let pred = lda.predict(&x_test)?;
let proba = lda.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics 7(2), 179-188.
