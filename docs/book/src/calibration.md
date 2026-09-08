# Probability calibration

`solow-calibration` provides cross-validated post-hoc calibration with Platt sigmoid or isotonic regression around any base classifier.

## Estimators

| Estimator | What it does |
|---|---|
| `CalibratedClassifierCV` | Cross-validated post-hoc probability calibration around any base classifier. Fits `k` base models on `k-1`-fold splits, then calibrates on the held-out fold and averages calibrators. |
| `Method::Sigmoid` | Platt scaling: parametric sigmoid fit by Newton-Raphson with safe target labels. |
| `Method::Isotonic` | Non-parametric monotone calibration via pool-adjacent-violators. |

## Example

```rust
use solow_calibration::{CalibratedClassifierCV, Method};

let cal = CalibratedClassifierCV::new(base_estimator, Method::Sigmoid)
    .cv(5)
    .fit(&x, &y)?;

let proba = cal.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Platt, J. C. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. In Advances in Large Margin Classifiers.
- Zadrozny, B., & Elkan, C. (2002). *Transforming classifier scores into accurate multiclass probability estimates*. KDD.
