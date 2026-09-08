# Naive Bayes

`solow-naive-bayes` provides five naive-Bayes variants covering continuous, count, binary, imbalanced-text, and categorical features. All expose `predict`, `predict_proba`, and `predict_log_proba` via log-sum-exp for numerical stability.

## Variants

| Estimator | Best for |
|---|---|
| `GaussianNB` | Continuous features. Uses Welford one-pass mean and variance with variance smoothing. |
| `MultinomialNB` | Count features (bag-of-words). Laplace or Lidstone smoothing. |
| `BernoulliNB` | Binary or thresholded features. Configurable `binarize` threshold. |
| `ComplementNB` | Multinomial variant designed for imbalanced text (Rennie et al. 2003), with optional row normalization. |
| `CategoricalNB` | Categorical features with per-column category counts. |

## Example

```rust
use solow_naive_bayes::GaussianNB;

let nb = GaussianNB::new().fit(&x, &y)?;
let proba = nb.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). *Tackling the poor assumptions of naive Bayes text classifiers*. ICML.
