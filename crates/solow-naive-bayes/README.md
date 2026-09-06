# solow-naive-bayes

**Naive Bayes classifiers for Rust.** Gaussian, multinomial, Bernoulli, complement, and categorical variants with Laplace or Lidstone smoothing.

All five expose `predict`, `predict_proba`, and `predict_log_proba` via log-sum-exp for numerical stability.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-naive-bayes = "0.7"
```

## Quick example

```rust
use solow_naive_bayes::GaussianNB;

let nb = GaussianNB::new().fit(&x, &y)?;
let proba = nb.predict_proba(&x_test)?;
```

## What is inside

| Estimator | Best for |
|---|---|
| `GaussianNB` | Continuous features. Uses Welford one-pass mean and variance with variance smoothing. |
| `MultinomialNB` | Count features (bag-of-words). Laplace or Lidstone smoothing. |
| `BernoulliNB` | Binary or thresholded features. Configurable `binarize` threshold. |
| `ComplementNB` | Multinomial variant designed for imbalanced text (Rennie et al. 2003), with optional row normalization. |
| `CategoricalNB` | Categorical features with per-column category counts. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Log-sum-exp everywhere to avoid underflow on high-dimensional data.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
